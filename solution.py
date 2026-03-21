"""
solution.py
Sentio Mind · Project 5 · Behavioral Anomaly & Early Distress Detection

Run: python solution.py
Produces: alert_feed.json  and  alert_digest.html
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime, date
from collections import defaultdict, Counter

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR   = Path("sample_data")
REPORT_OUT = Path("alert_digest.html")
FEED_OUT   = Path("alert_feed.json")
SCHOOL     = "Demo School"

THRESHOLDS = {
    "sudden_drop_delta":          20,
    "sudden_drop_high_std_delta": 30,   # used when baseline_std > 15
    "sustained_low_score":        45,
    "sustained_low_days":          3,
    "social_withdrawal_delta":    25,
    "hyperactivity_delta":        40,
    "regression_recover_days":     3,
    "regression_drop":            15,
    "gaze_avoidance_days":         3,
    "absence_days":                2,
    "baseline_window":             3,
    "high_std_baseline":          15,
}


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_daily_data(folder: Path) -> dict:
    """
    Read all analysis_DayN.json files from folder (sorted by name).
    Return: { "YYYY-MM-DD": { "PERSON_ID": { wellbeing, social_engagement,
              physical_energy, movement_energy, gaze_direction, seen_in_video,
              person_info }, ... } }
    """
    daily = {}
    for fp in sorted(folder.glob("analysis_Day*.json")):
        with open(fp) as f:
            records = json.load(f)
        for r in records:
            date_str = r["date"]
            pid      = r["person_id"]
            daily.setdefault(date_str, {})[pid] = {
                "wellbeing":         r["wellbeing"],
                "social_engagement": r.get("social_engagement", 50),
                "physical_energy":   r.get("physical_energy",   50),
                "movement_energy":   r.get("movement_energy",   50),
                "gaze_direction":    r.get("gaze", "forward"),
                "seen_in_video":     r.get("seen_in_video", True),
                "person_info": {
                    "name":              r.get("person_name", pid),
                    "profile_image_b64": "",
                },
            }
    return daily


# ---------------------------------------------------------------------------
# BASELINE
# ---------------------------------------------------------------------------

def compute_baseline(history: list) -> dict:
    """
    history: list of daily dicts, oldest first.
    Uses first THRESHOLDS['baseline_window'] days.
    Returns wellbeing_mean, wellbeing_std, baseline_social, baseline_energy,
            trait_means, avg_gaze.
    """
    window = history[: THRESHOLDS["baseline_window"]]
    if not window:
        return {
            "wellbeing_mean":  50.0, "wellbeing_std": 10.0,
            "baseline_social": 50.0, "baseline_energy": 100.0,
            "trait_means": {}, "avg_gaze": "forward",
        }

    wbs    = [h["wellbeing"]                                     for h in window]
    socs   = [h.get("social_engagement", 50)                     for h in window]
    energy = [h.get("physical_energy",50)+h.get("movement_energy",50) for h in window]
    gazes  = [h.get("gaze_direction", "forward")                 for h in window]

    trait_keys = ["social_engagement", "physical_energy", "movement_energy"]
    trait_means = {k: float(np.mean([h.get(k, 50) for h in window])) for k in trait_keys}

    return {
        "wellbeing_mean":  float(np.mean(wbs)),
        "wellbeing_std":   float(np.std(wbs)),
        "baseline_social": float(np.mean(socs)),
        "baseline_energy": float(np.mean(energy)),
        "trait_means":     trait_means,
        "avg_gaze":        Counter(gazes).most_common(1)[0][0],
    }


# ---------------------------------------------------------------------------
# ANOMALY DETECTORS
# ---------------------------------------------------------------------------

def detect_sudden_drop(today: dict, baseline: dict) -> dict | None:
    """Drop >= threshold vs personal baseline. If baseline_std > 15, use 1.5x threshold."""
    std = baseline["wellbeing_std"]
    thr = (THRESHOLDS["sudden_drop_high_std_delta"]
           if std > THRESHOLDS["high_std_baseline"]
           else THRESHOLDS["sudden_drop_delta"])
    delta_mag = baseline["wellbeing_mean"] - today["wellbeing"]
    if delta_mag >= thr:
        sev = "urgent" if delta_mag > 35 else "monitor"
        return {
            "category": "SUDDEN_DROP", "severity": sev,
            "title": "Sudden wellbeing drop detected",
            "description": (
                f"Wellbeing fell from baseline {baseline['wellbeing_mean']:.0f} "
                f"to {today['wellbeing']} — a {delta_mag:.0f}-pt drop."
            ),
        }
    return None


def detect_sustained_low(history: list) -> dict | None:
    """Wellbeing < 45 for 3+ consecutive days (ending today)."""
    n   = THRESHOLDS["sustained_low_days"]
    thr = THRESHOLDS["sustained_low_score"]
    if len(history) < n:
        return None
    recent = history[-n:]
    if not all(r["wellbeing"] < thr for r in recent):
        return None
    consec = sum(1 for _ in (r for r in reversed(history) if r["wellbeing"] < thr)
                 for __ in [None] if True)   # simple count via loop below
    consec = 0
    for r in reversed(history):
        if r["wellbeing"] < thr:
            consec += 1
        else:
            break
    return {
        "category": "SUSTAINED_LOW", "severity": "urgent",
        "title": "Sustained low wellbeing",
        "description": (
            f"Wellbeing below {thr} for {consec} consecutive days "
            f"(current: {history[-1]['wellbeing']})."
        ),
    }


def detect_social_withdrawal(today: dict, baseline: dict) -> dict | None:
    """social_engagement < (baseline_social - 25) AND gaze == 'down'."""
    drop = baseline["baseline_social"] - today.get("social_engagement", 50)
    gaze = today.get("gaze_direction", "forward")
    if drop >= THRESHOLDS["social_withdrawal_delta"] and gaze == "down":
        return {
            "category": "SOCIAL_WITHDRAWAL", "severity": "urgent",
            "title": "Social withdrawal signs detected",
            "description": (
                f"Social engagement dropped {drop:.0f} pts from baseline "
                f"({baseline['baseline_social']:.0f} -> {today.get('social_engagement',50)}). "
                f"Gaze: {gaze}."
            ),
        }
    return None


def detect_hyperactivity_spike(today: dict, baseline: dict) -> dict | None:
    """(today_physical + today_movement) > baseline_energy + 40."""
    combined = today.get("physical_energy", 0) + today.get("movement_energy", 0)
    spike    = combined - baseline["baseline_energy"]
    if spike >= THRESHOLDS["hyperactivity_delta"]:
        return {
            "category": "HYPERACTIVITY_SPIKE", "severity": "monitor",
            "title": "Hyperactivity spike detected",
            "description": (
                f"Combined physical+movement score {combined} is "
                f"{spike:.0f} pts above baseline ({baseline['baseline_energy']:.0f})."
            ),
        }
    return None


def detect_regression(history: list) -> dict | None:
    """Wellbeing strictly increasing for 3+ days, then drops > 15 pts today."""
    recover = THRESHOLDS["regression_recover_days"]  # 3
    drop_t  = THRESHOLDS["regression_drop"]           # 15
    if len(history) < recover + 2:
        return None
    recent = history[-(recover + 2):]   # 5 entries: 4 pre + today
    pre    = recent[:-1]
    today_wb = recent[-1]["wellbeing"]
    if not all(pre[i+1]["wellbeing"] > pre[i]["wellbeing"] for i in range(len(pre)-1)):
        return None
    peak = pre[-1]["wellbeing"]
    if peak - today_wb > drop_t:
        return {
            "category": "REGRESSION", "severity": "urgent",
            "title": "Regression after recovery period",
            "description": (
                f"After {recover} days of improvement (peak: {peak}), "
                f"wellbeing dropped {peak - today_wb} pts to {today_wb} today."
            ),
        }
    return None


def detect_gaze_avoidance(history: list) -> dict | None:
    """Gaze 'down' or 'away' for 3+ consecutive days ending today."""
    n = THRESHOLDS["gaze_avoidance_days"]
    if len(history) < n:
        return None
    if not all(h.get("gaze_direction","forward") in ("down","away","side")
               for h in history[-n:]):
        return None
    consec = 0
    for h in reversed(history):
        if h.get("gaze_direction","forward") in ("down","away","side"):
            consec += 1
        else:
            break
    return {
        "category": "GAZE_AVOIDANCE", "severity": "monitor",
        "title": "Persistent gaze avoidance",
        "description": f"No eye contact for {consec} consecutive days "
                       f"(gaze: {history[-1].get('gaze_direction','?')}).",
    }


# ---------------------------------------------------------------------------
# ANALYSE ONE PERSON  — alerts for the LATEST day only
# ---------------------------------------------------------------------------

def analyse_person(person_id: str, sorted_days: dict, info: dict) -> list:
    """
    sorted_days: { "YYYY-MM-DD": person_data_dict } in date order.
    Returns a list of fully-populated alert dicts for the latest day.
    days_flagged_consecutively is computed from full history.
    """
    name = info.get("name", person_id)

    history = []
    for date_str, d in sorted_days.items():
        history.append({
            "date":              date_str,
            "wellbeing":         d.get("wellbeing",         50),
            "social_engagement": d.get("social_engagement", 50),
            "physical_energy":   d.get("physical_energy",   50),
            "movement_energy":   d.get("movement_energy",   50),
            "gaze_direction":    d.get("gaze_direction",    "forward"),
            "seen_in_video":     d.get("seen_in_video",     True),
        })

    if not history:
        return []

    baseline = compute_baseline(history)

    # ── Compute days_flagged_consecutively via full history ──────────────────
    flagged_dates: set = set()
    for i in range(len(history)):
        h = history[: i + 1]
        t = h[-1]
        fired = any(fn is not None for fn in [
            detect_sudden_drop(t, baseline),
            detect_sustained_low(h),
            detect_social_withdrawal(t, baseline),
            detect_hyperactivity_spike(t, baseline),
            detect_regression(h),
            detect_gaze_avoidance(h),
        ])
        if fired:
            flagged_dates.add(t["date"])

    consecutive = 0
    for h in reversed(history):
        if h["date"] in flagged_dates:
            consecutive += 1
        else:
            break

    # ── Generate alerts for the LATEST day only ──────────────────────────────
    today   = history[-1]
    trend   = [h["wellbeing"] for h in history[-5:]]
    traits  = {
        "social_engagement": today["social_engagement"],
        "physical_energy":   today["physical_energy"],
        "movement_energy":   today["movement_energy"],
    }
    lowest_trait = min(traits, key=traits.get)

    detectors = [
        detect_sudden_drop(today, baseline),
        detect_sustained_low(history),
        detect_social_withdrawal(today, baseline),
        detect_hyperactivity_spike(today, baseline),
        detect_regression(history),
        detect_gaze_avoidance(history),
    ]

    alerts = []
    idx = 0
    for partial in detectors:
        if partial is None:
            continue
        idx += 1
        partial.update({
            "alert_id":                   f"ALT_{person_id}_{today['date']}_{idx}",
            "person_id":                  person_id,
            "person_name":                name,
            "date":                       today["date"],
            "baseline_wellbeing":         round(baseline["wellbeing_mean"], 1),
            "today_wellbeing":            today["wellbeing"],
            "delta":                      round(today["wellbeing"] - baseline["wellbeing_mean"], 1),
            "days_flagged_consecutively": consecutive,
            "trend_last_5_days":          trend,
            "lowest_trait":               lowest_trait,
            "lowest_trait_value":         traits[lowest_trait],
            "recommended_action":         "Schedule pastoral check-in today",
            "profile_image_b64":          "",
        })
        alerts.append(partial)

    return alerts


# ---------------------------------------------------------------------------
# SVG SPARKLINE
# ---------------------------------------------------------------------------

def _svg_sparkline(trend: list, severity: str) -> str:
    """Inline SVG bar chart — no JS, no CDN."""
    W, H = 90, 36
    if not trend:
        return f'<svg width="{W}" height="{H}"></svg>'
    n   = len(trend)
    bw  = max(1, (W - (n - 1) * 2) // n)
    col = "#e74c3c" if severity == "urgent" else "#f39c12"
    rects = ""
    for i, v in enumerate(trend):
        bh = max(2, int(v / 100 * H))
        x  = i * (bw + 2)
        y  = H - bh
        rects += (f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" '
                  f'fill="{col}" rx="1"><title>{v}</title></rect>')
    return (f'<svg width="{W}" height="{H}" '
            f'style="display:block;overflow:visible" '
            f'xmlns="http://www.w3.org/2000/svg">{rects}</svg>')


# ---------------------------------------------------------------------------
# HTML REPORT
# ---------------------------------------------------------------------------

def generate_alert_digest(alerts: list, absence_flags: list,
                           school_summary: dict, output_path: Path):
    """
    Fully offline HTML — inline CSS, SVG sparklines, zero CDN.
    Section 1: Header + School Summary bar
    Section 2: Action Required Today (alert cards)
    Section 3: Absence Flags
    Section 4: Chronic Cases (days_flagged_consecutively >= 3)
    """
    today_str    = str(date.today())
    today_alerts = [a for a in alerts if a.get("date") == today_str]
    today_alerts.sort(key=lambda a: (
        0 if a.get("severity") == "urgent" else 1,
        a.get("delta", 0),
    ))

    chronic = [a for a in alerts if a.get("days_flagged_consecutively", 0) >= 3]
    seen_c: dict = {}
    for a in chronic:
        k = f"{a['person_id']}_{a['category']}"
        if k not in seen_c or a.get("days_flagged_consecutively", 0) > seen_c[k].get("days_flagged_consecutively", 0):
            seen_c[k] = a
    chronic_list = sorted(seen_c.values(),
                          key=lambda a: a.get("days_flagged_consecutively", 0), reverse=True)

    # ── CSS ──────────────────────────────────────────────────────────────────
    css = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;
     background:#ecf0f1;color:#2c3e50;padding:24px}
h1{font-size:22px;font-weight:700;color:#2c3e50}
.subtitle{font-size:12px;color:#7f8c8d;margin-bottom:24px;margin-top:4px}
h2{font-size:16px;font-weight:600;margin:28px 0 12px;padding-left:10px;
   border-left:4px solid #3498db}
.stats{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:24px}
.stat-box{background:#fff;border-radius:10px;padding:14px 22px;min-width:130px;
          border:1px solid #dce1e7;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.stat-num{font-size:30px;font-weight:700}
.stat-label{font-size:11px;color:#7f8c8d;margin-top:3px}
.card{background:#fff;border-radius:10px;padding:16px 18px;margin-bottom:10px;
      border-left:5px solid #bdc3c7;box-shadow:0 1px 4px rgba(0,0,0,.06)}
.card.urgent{border-left-color:#e74c3c;background:#fff8f8}
.card.monitor{border-left-color:#f39c12;background:#fffbf2}
.card.absence{border-left-color:#e74c3c;background:#fff0f0}
.card-top{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px}
.name{font-size:15px;font-weight:600}
.badge{font-size:10px;font-weight:700;padding:2px 8px;border-radius:10px;
       text-transform:uppercase;letter-spacing:.4px}
.badge-urgent{background:#e74c3c;color:#fff}
.badge-monitor{background:#f39c12;color:#fff}
.badge-absence{background:#c0392b;color:#fff}
.cat{font-size:10px;background:#eaf0fb;color:#2c5282;padding:2px 8px;border-radius:10px}
.desc{font-size:13px;color:#555;margin:5px 0}
.meta{font-size:11px;color:#888;display:flex;gap:14px;flex-wrap:wrap;margin-top:5px}
.spark{margin-top:8px}
.action{font-size:11px;color:#e74c3c;margin-top:7px;font-weight:600}
.chronic-label{font-size:11px;color:#8e44ad;font-weight:700;
               background:#f5eef8;padding:2px 8px;border-radius:8px}
.no-data{color:#7f8c8d;font-size:13px;padding:14px 0}
table{width:100%;border-collapse:collapse;background:#fff;border-radius:10px;
      overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.06)}
th{background:#2c3e50;color:#fff;text-align:left;padding:10px 14px;font-size:12px}
td{padding:10px 14px;font-size:12px;border-bottom:1px solid #f0f0f0}
tr:last-child td{border-bottom:none}
"""

    # ── helpers ───────────────────────────────────────────────────────────────
    def card(a):
        sev   = a.get("severity", "monitor")
        spark = _svg_sparkline(a.get("trend_last_5_days", []), sev)
        chron = (' <span class="chronic-label">Chronic — immediate intervention</span>'
                 if a.get("days_flagged_consecutively", 0) >= 3 else "")
        return f"""
<div class="card {sev}">
  <div class="card-top">
    <span class="name">{a['person_name']}</span>
    <span class="badge badge-{sev}">{sev}</span>
    <span class="cat">{a.get('category','')}</span>
    {chron}
  </div>
  <p class="desc">{a.get('description','')}</p>
  <div class="meta">
    <span>Baseline: <b>{a.get('baseline_wellbeing')}</b></span>
    <span>Today: <b>{a.get('today_wellbeing')}</b></span>
    <span>Delta: <b>{a.get('delta')}</b></span>
    <span>Days flagged: <b>{a.get('days_flagged_consecutively',1)}</b></span>
    <span>Lowest: <b>{a.get('lowest_trait')}</b> ({a.get('lowest_trait_value')})</span>
  </div>
  <div class="spark">{spark}</div>
  <p class="action">&#128276; {a.get('recommended_action','')}</p>
</div>"""

    # Section 2
    s2 = "".join(card(a) for a in today_alerts) if today_alerts else '<p class="no-data">No alerts today.</p>'

    # Section 3 — absence
    if absence_flags:
        rows = "".join(
            f"<tr><td>{f['person_name']}</td><td>{f['person_id']}</td>"
            f"<td>{f['last_seen_date']}</td><td><b>{f['days_absent']}</b></td>"
            f"<td style='color:#e74c3c'>{f['recommended_action']}</td></tr>"
            for f in absence_flags
        )
        s3 = f"""<table>
<tr><th>Name</th><th>Person ID</th><th>Last Seen</th><th>Days Absent</th><th>Action</th></tr>
{rows}</table>"""
    else:
        s3 = '<p class="no-data">No absence flags.</p>'

    # Section 4 — chronic
    if chronic_list:
        rows = "".join(
            f"<tr><td>{a['person_name']}</td><td>{a['person_id']}</td>"
            f"<td><span class='badge badge-{a['severity']}'>{a['severity']}</span></td>"
            f"<td>{a['category']}</td><td><b>{a['days_flagged_consecutively']}</b></td>"
            f"<td>{a['description']}</td></tr>"
            for a in chronic_list
        )
        s4 = f"""<table>
<tr><th>Name</th><th>ID</th><th>Severity</th><th>Category</th><th>Days</th><th>Description</th></tr>
{rows}</table>"""
    else:
        s4 = '<p class="no-data">No chronic cases at this time.</p>'

    # School summary stat boxes
    ss   = school_summary
    s1   = f"""
<div class="stats">
  <div class="stat-box">
    <div class="stat-num">{ss['total_persons_tracked']}</div>
    <div class="stat-label">Persons Tracked</div>
  </div>
  <div class="stat-box">
    <div class="stat-num" style="color:#e74c3c">{ss['persons_flagged_today']}</div>
    <div class="stat-label">Flagged Today</div>
  </div>
  <div class="stat-box">
    <div class="stat-num" style="color:#e74c3c">{len(absence_flags)}</div>
    <div class="stat-label">Absence Flags</div>
  </div>
  <div class="stat-box">
    <div class="stat-num" style="color:#27ae60">{ss['school_avg_wellbeing_today']}</div>
    <div class="stat-label">Avg Wellbeing Today</div>
  </div>
  <div class="stat-box">
    <div class="stat-num" style="font-size:14px;padding-top:8px;color:#8e44ad">{ss['most_common_anomaly_this_week']}</div>
    <div class="stat-label">Top Anomaly This Week</div>
  </div>
</div>"""

    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sentio Mind | Alert Digest</title>
<style>{css}</style>
</head>
<body>
<h1>Sentio Mind | Daily Counsellor Alert Digest</h1>
<p class="subtitle">{SCHOOL} &nbsp;&bull;&nbsp; Generated: {ts} &nbsp;&bull;&nbsp; Date: {today_str}</p>

{s1}

<h2>Action Required Today</h2>
{s2}

<h2>Absence Flags</h2>
{s3}

<h2>Chronic Cases (3+ consecutive days flagged)</h2>
{s4}
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    daily_data = load_daily_data(DATA_DIR)
    all_dates  = sorted(daily_data.keys())
    print(f"Loaded {len(daily_data)} days: {all_dates}")

    person_days: dict = defaultdict(dict)
    person_info: dict = {}
    for d, persons in daily_data.items():
        for pid, pdata in persons.items():
            person_days[pid][d] = pdata
            if pid not in person_info:
                person_info[pid] = pdata.get("person_info", {"name": pid, "profile_image_b64": ""})

    all_alerts:    list = []
    absence_flags: list = []

    for pid, days in person_days.items():
        sorted_days   = dict(sorted(days.items()))
        person_alerts = analyse_person(pid, sorted_days, person_info.get(pid, {}))
        all_alerts.extend(person_alerts)

        # Absence detection via seen_in_video field
        absent = 0
        for d in reversed(all_dates):
            if d in days and not days[d].get("seen_in_video", True):
                absent += 1
            else:
                break
        if absent >= THRESHOLDS["absence_days"]:
            last_seen = next(
                (d for d in reversed(all_dates) if d in days and days[d].get("seen_in_video", True)),
                "unknown"
            )
            absence_flags.append({
                "person_id":          pid,
                "person_name":        person_info.get(pid, {}).get("name", pid),
                "last_seen_date":     last_seen,
                "days_absent":        absent,
                "recommended_action": "Welfare check — contact family if absent again tomorrow",
            })

    # Sort: urgent first, then by delta ascending (largest drop first)
    all_alerts.sort(key=lambda a: (
        0 if a.get("severity") == "urgent" else 1,
        a.get("delta", 0),
    ))

    today_str = all_dates[-1] if all_dates else str(date.today())

    flagged_today = len(set(a["person_id"] for a in all_alerts))
    flagged_yest  = len(set(a["person_id"] for a in all_alerts
                            if a.get("days_flagged_consecutively", 0) >= 2))

    cat_counter  = Counter(a.get("category") for a in all_alerts)
    top_category = cat_counter.most_common(1)[0][0] if cat_counter else "none"

    today_wbs    = [pdata["wellbeing"] for pdata in daily_data.get(today_str, {}).values()]
    avg_wb_today = int(np.mean(today_wbs)) if today_wbs else 0

    school_summary = {
        "total_persons_tracked":         len(person_days),
        "persons_flagged_today":         flagged_today,
        "persons_flagged_yesterday":     flagged_yest,
        "most_common_anomaly_this_week": top_category,
        "school_avg_wellbeing_today":    avg_wb_today,
    }

    feed = {
        "source":        "p5_anomaly_detection",
        "generated_at":  datetime.now().isoformat(),
        "school":        SCHOOL,
        "alert_summary": {
            "total_alerts":  len(all_alerts),
            "urgent":        sum(1 for a in all_alerts if a.get("severity") == "urgent"),
            "monitor":       sum(1 for a in all_alerts if a.get("severity") == "monitor"),
            "informational": 0,
        },
        "alerts":         all_alerts,
        "absence_flags":  absence_flags,
        "school_summary": school_summary,
    }

    with open(FEED_OUT, "w") as f:
        json.dump(feed, f, indent=2)

    generate_alert_digest(all_alerts, absence_flags, school_summary, REPORT_OUT)

    # Print summary
    print()
    print("=" * 58)
    print(f"  Alerts:        {feed['alert_summary']['total_alerts']} total  "
          f"({feed['alert_summary']['urgent']} urgent, "
          f"{feed['alert_summary']['monitor']} monitor)")
    print(f"  Absence flags: {len(absence_flags)}")
    print(f"  Report  ->  {REPORT_OUT}")
    print(f"  JSON    ->  {FEED_OUT}")
    print("=" * 58)
    print()
    print("Categories fired:")
    for cat, cnt in sorted(cat_counter.items()):
        persons = [a["person_name"] for a in all_alerts if a["category"] == cat]
        print(f"  {cat:<25} ({cnt}x)  {', '.join(set(persons))}")
    if absence_flags:
        print(f"  {'ABSENCE_FLAG':<25} ({len(absence_flags)}x)  "
              f"{', '.join(f['person_name'] for f in absence_flags)}")
