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
# CONFIG — adjust thresholds here, nowhere else
# ---------------------------------------------------------------------------
DATA_DIR   = Path("sample_data")
REPORT_OUT = Path("alert_digest.html")
FEED_OUT   = Path("alert_feed.json")
SCHOOL     = "Demo School"

THRESHOLDS = {
    "sudden_drop_delta":           20,   # baseline - today >= this → SUDDEN_DROP
    "sudden_drop_high_std_delta":  30,   # used when baseline_std > 15
    "sustained_low_score":         45,   # below this = low
    "sustained_low_days":           3,   # consecutive days below threshold
    "social_withdrawal_delta":     25,   # social_engagement drop
    "hyperactivity_delta":         40,   # combined energy spike
    "regression_recover_days":      3,   # days improving before regression counts
    "regression_drop":             15,   # drop after recovery
    "gaze_avoidance_days":          3,   # consecutive days no eye contact
    "absence_days":                 2,   # days not detected
    "baseline_window":              3,   # days used for baseline
    "high_std_baseline":           15,   # if std above this, use relaxed threshold
}


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_daily_data(folder: Path) -> dict:
    """
    Read all analysis_*.json files from folder.
    Return: { "YYYY-MM-DD": { "PERSON_ID": { wellbeing, gaze_direction, seen_in_video,
                                              social_engagement, physical_energy,
                                              movement_energy, person_info }, ... } }
    """
    daily = {}
    for fp in sorted(folder.glob("*.json")):
        with open(fp) as f:
            records = json.load(f)
        for r in records:
            date_str = r["date"]
            pid      = r["person_id"]
            if date_str not in daily:
                daily[date_str] = {}
            daily[date_str][pid] = {
                "wellbeing":        r["wellbeing"],
                "social_engagement": r.get("social_engagement", 50),
                "physical_energy":   r.get("physical_energy",   50),
                "movement_energy":   r.get("movement_energy",   50),
                "gaze_direction":    r.get("gaze", "forward"),
                "seen_in_video":     r.get("seen_in_video", True),
                "person_info": {
                    "name":             r.get("person_name", pid),
                    "profile_image_b64": "",
                },
            }
    return daily


# ---------------------------------------------------------------------------
# BASELINE
# ---------------------------------------------------------------------------

def compute_baseline(history: list) -> dict:
    """
    history: list of daily dicts (oldest first).
    Uses first THRESHOLDS['baseline_window'] days.
    Returns { wellbeing_mean, wellbeing_std, trait_means: {}, avg_gaze }
    """
    window = history[: THRESHOLDS["baseline_window"]]
    if not window:
        return {
            "wellbeing_mean": 50.0,
            "wellbeing_std":  10.0,
            "trait_means":    {},
            "avg_gaze":       "forward",
        }

    wellbeings = [h["wellbeing"] for h in window]
    trait_keys = ["social_engagement", "physical_energy", "movement_energy"]
    trait_means = {k: float(np.mean([h.get(k, 50) for h in window])) for k in trait_keys}
    gazes       = [h.get("gaze_direction", "forward") for h in window]

    return {
        "wellbeing_mean": float(np.mean(wellbeings)),
        "wellbeing_std":  float(np.std(wellbeings)),
        "trait_means":    trait_means,
        "avg_gaze":       Counter(gazes).most_common(1)[0][0],
    }


# ---------------------------------------------------------------------------
# ANOMALY DETECTORS  — each returns a partial alert dict or None
# ---------------------------------------------------------------------------

def detect_sudden_drop(today: dict, baseline: dict) -> dict | None:
    """
    today['wellbeing'] vs baseline['wellbeing_mean'].
    If baseline_std > 15, raise threshold to sudden_drop_high_std_delta.
    Severity: delta > 35 = urgent, else monitor.
    """
    std       = baseline["wellbeing_std"]
    threshold = (THRESHOLDS["sudden_drop_high_std_delta"]
                 if std > THRESHOLDS["high_std_baseline"]
                 else THRESHOLDS["sudden_drop_delta"])
    mag = baseline["wellbeing_mean"] - today["wellbeing"]
    if mag >= threshold:
        severity = "urgent" if mag > 35 else "monitor"
        return {
            "category": "SUDDEN_DROP",
            "severity": severity,
            "title":    "Sudden wellbeing drop detected",
            "description": (
                f"{today.get('_name','Student')}'s wellbeing dropped from a baseline of "
                f"{baseline['wellbeing_mean']:.0f} to {today['wellbeing']} today — "
                f"a {mag:.0f}-point fall. Gaze: {today.get('gaze_direction','?')}."
            ),
            "_days_consec": 1,
        }
    return None


def detect_sustained_low(history: list) -> dict | None:
    """
    Last sustained_low_days entries all have wellbeing < sustained_low_score → alert.
    Severity: urgent.
    """
    n         = THRESHOLDS["sustained_low_days"]
    threshold = THRESHOLDS["sustained_low_score"]
    if len(history) < n:
        return None
    recent = history[-n:]
    if not all(r["wellbeing"] < threshold for r in recent):
        return None
    # Count consecutive low days going back from today
    consec = 0
    for r in reversed(history):
        if r["wellbeing"] < threshold:
            consec += 1
        else:
            break
    return {
        "category":    "SUSTAINED_LOW",
        "severity":    "urgent",
        "title":       "Sustained low wellbeing",
        "description": (
            f"Wellbeing has been below {threshold} for {consec} consecutive days "
            f"(current: {history[-1]['wellbeing']})."
        ),
        "_days_consec": consec,
    }


def detect_social_withdrawal(today: dict, baseline: dict) -> dict | None:
    """
    social_engagement dropped >= social_withdrawal_delta AND
    today's gaze_direction is 'down', 'side', or 'away'.
    Severity: urgent.
    """
    delta_thr    = THRESHOLDS["social_withdrawal_delta"]
    base_social  = baseline["trait_means"].get("social_engagement", 50)
    drop         = base_social - today.get("social_engagement", 50)
    gaze         = today.get("gaze_direction", "forward")
    if drop >= delta_thr and gaze in ("down", "side", "away"):
        return {
            "category":    "SOCIAL_WITHDRAWAL",
            "severity":    "urgent",
            "title":       "Social withdrawal signs detected",
            "description": (
                f"Social engagement dropped {drop:.0f} pts from baseline "
                f"({base_social:.0f} → {today.get('social_engagement',50)}). "
                f"Dominant gaze: {gaze}."
            ),
            "_days_consec": 1,
        }
    return None


def detect_hyperactivity_spike(today: dict, baseline: dict) -> dict | None:
    """
    (today.physical_energy + today.movement_energy) minus
    (baseline.physical_energy_mean + baseline.movement_energy_mean) >= hyperactivity_delta.
    Severity: monitor.
    """
    today_combined    = today.get("physical_energy", 0) + today.get("movement_energy", 0)
    baseline_combined = (
        baseline["trait_means"].get("physical_energy",   0) +
        baseline["trait_means"].get("movement_energy",   0)
    )
    spike = today_combined - baseline_combined
    if spike >= THRESHOLDS["hyperactivity_delta"]:
        return {
            "category":    "HYPERACTIVITY_SPIKE",
            "severity":    "monitor",
            "title":       "Hyperactivity spike detected",
            "description": (
                f"Combined physical+movement score {today_combined} is "
                f"{spike:.0f} pts above baseline ({baseline_combined:.0f})."
            ),
            "_days_consec": 1,
        }
    return None


def detect_regression(history: list) -> dict | None:
    """
    Last regression_recover_days + 1 entries (before today) all improving,
    then today drops > regression_drop.
    Severity: urgent.
    """
    recover_days  = THRESHOLDS["regression_recover_days"]   # 3
    drop_thr      = THRESHOLDS["regression_drop"]            # 15
    # Need: recover_days improvements (= recover_days+1 points) + today = recover_days+2
    if len(history) < recover_days + 2:
        return None
    recent   = history[-(recover_days + 2):]   # e.g. last 5 entries
    pre      = recent[:-1]                      # 4 points → 3 pairwise checks
    today_wb = recent[-1]["wellbeing"]
    improving = all(
        pre[i + 1]["wellbeing"] > pre[i]["wellbeing"]
        for i in range(len(pre) - 1)
    )
    if not improving:
        return None
    peak = pre[-1]["wellbeing"]
    if peak - today_wb > drop_thr:
        return {
            "category":    "REGRESSION",
            "severity":    "urgent",
            "title":       "Regression after recovery",
            "description": (
                f"After {recover_days} days of improving scores (peak: {peak}), "
                f"wellbeing dropped {peak - today_wb} pts today ({today_wb})."
            ),
            "_days_consec": 1,
        }
    return None


def detect_gaze_avoidance(history: list) -> dict | None:
    """
    Last gaze_avoidance_days entries all have gaze 'down', 'away', or 'side'.
    Severity: monitor.
    """
    n = THRESHOLDS["gaze_avoidance_days"]
    if len(history) < n:
        return None
    recent = history[-n:]
    if not all(r.get("gaze_direction", "forward") in ("down", "away", "side") for r in recent):
        return None
    # Count total consecutive gaze-avoidance days
    consec = 0
    for r in reversed(history):
        if r.get("gaze_direction", "forward") in ("down", "away", "side"):
            consec += 1
        else:
            break
    return {
        "category":    "GAZE_AVOIDANCE",
        "severity":    "monitor",
        "title":       "Persistent gaze avoidance",
        "description": (
            f"No eye contact detected for {consec} consecutive days "
            f"(gaze: {history[-1].get('gaze_direction','?')})."
        ),
        "_days_consec": consec,
    }


# ---------------------------------------------------------------------------
# ANALYSE ONE PERSON
# ---------------------------------------------------------------------------

def analyse_person(person_id: str, sorted_days: dict, info: dict) -> list:
    """
    sorted_days: { "YYYY-MM-DD": person_data_dict } — keys in date order
    info: { name, profile_image_b64, ... }

    Builds history, computes baseline, runs all detectors for every day.
    Returns list of fully-populated alert dicts.
    """
    name = info.get("name", person_id)

    # Build flat history list
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
            "_name":             name,
        })

    baseline = compute_baseline(history)

    alerts      = []
    alert_index = 0
    # Per-category streak counters
    streak: dict[str, int] = defaultdict(int)

    for i, today in enumerate(history):
        current_history = history[: i + 1]

        # Run all detectors
        candidates = [
            detect_sudden_drop(today, baseline),
            detect_sustained_low(current_history),
            detect_social_withdrawal(today, baseline),
            detect_hyperactivity_spike(today, baseline),
            detect_regression(current_history),
            detect_gaze_avoidance(current_history),
        ]

        fired_cats = set()
        for partial in candidates:
            if partial is None:
                continue

            cat = partial["category"]
            fired_cats.add(cat)
            streak[cat] += 1

            # Trend: last 5 wellbeing values up to and including today
            trend = [h["wellbeing"] for h in history[max(0, i - 4): i + 1]]

            # Lowest trait
            traits = {
                "social_engagement": today["social_engagement"],
                "physical_energy":   today["physical_energy"],
                "movement_energy":   today["movement_energy"],
            }
            lowest_trait = min(traits, key=traits.get)

            alert_index += 1
            partial.update({
                "alert_id":               f"ALT_{person_id}_{today['date']}_{alert_index}",
                "person_id":              person_id,
                "person_name":            name,
                "date":                   today["date"],
                "baseline_wellbeing":     round(baseline["wellbeing_mean"], 1),
                "today_wellbeing":        today["wellbeing"],
                "delta":                  round(today["wellbeing"] - baseline["wellbeing_mean"], 1),
                "days_flagged_consecutively": partial.get("_days_consec", streak[cat]),
                "trend_last_5_days":      trend,
                "lowest_trait":           lowest_trait,
                "lowest_trait_value":     traits[lowest_trait],
                "recommended_action":     "Schedule pastoral check-in today",
                "profile_image_b64":      "",
            })
            # Remove internal helper key
            partial.pop("_days_consec", None)
            alerts.append(partial)

        # Reset streaks for categories that didn't fire today
        for cat in list(streak.keys()):
            if cat not in fired_cats:
                streak[cat] = 0

    return alerts


# ---------------------------------------------------------------------------
# HTML REPORT
# ---------------------------------------------------------------------------

def _sparkline(trend: list, severity: str) -> str:
    """Inline CSS bar-chart sparkline — no JS, no CDN."""
    if not trend:
        return ""
    max_val = max(trend) if max(trend) > 0 else 1
    bar_color = "#E74C3C" if severity == "urgent" else "#F39C12"
    bars = ""
    for v in trend:
        pct = int(v / 100 * 100)
        tip = str(v)
        bars += (
            f'<div title="{tip}" style="width:14px;height:{pct}%;'
            f'background:{bar_color};border-radius:2px 2px 0 0;'
            f'display:inline-block;vertical-align:bottom;margin:0 1px;"></div>'
        )
    return (
        f'<div style="display:inline-flex;align-items:flex-end;'
        f'height:36px;background:#F5F5F5;padding:3px 4px;'
        f'border-radius:4px;gap:1px;">{bars}</div>'
    )


def generate_alert_digest(alerts: list, absence_flags: list,
                           school_summary: dict, output_path: Path):
    """
    Self-contained HTML — no CDN, inline CSS only.
    Section 1: Today's alerts (red = urgent, orange = monitor).
    Section 2: School summary stat boxes.
    Section 3: Chronic flags (3+ consecutive days).
    """
    today_str     = str(date.today())
    today_alerts  = [a for a in alerts if a.get("date") == today_str]
    today_alerts.sort(key=lambda a: 0 if a.get("severity") == "urgent" else 1)

    chronic = [a for a in alerts if a.get("days_flagged_consecutively", 0) >= 3]
    # Deduplicate chronic by person+category, keep latest
    seen_chronic: dict[str, dict] = {}
    for a in chronic:
        key = f"{a['person_id']}_{a['category']}"
        seen_chronic[key] = a
    chronic_list = sorted(seen_chronic.values(),
                          key=lambda a: a.get("days_flagged_consecutively", 0), reverse=True)

    # ── CSS ──────────────────────────────────────────────────────────────────
    css = """
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', Arial, sans-serif; background: #F0F2F5;
           color: #2C3E50; padding: 24px; }
    h1   { font-size: 24px; font-weight: 700; color: #1A252F; margin-bottom: 4px; }
    .subtitle { font-size: 13px; color: #7F8C8D; margin-bottom: 28px; }
    h2   { font-size: 18px; font-weight: 600; margin: 28px 0 14px; border-left: 4px solid #3498DB;
           padding-left: 10px; }
    .card { background: #FFF; border-radius: 10px; padding: 16px 20px;
            margin-bottom: 12px; border: 2px solid #ddd; box-shadow: 0 1px 4px rgba(0,0,0,.06); }
    .card.urgent  { border-color: #E74C3C; background: #FFF8F8; }
    .card.monitor { border-color: #F39C12; background: #FFFBF2; }
    .card-header  { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
    .name   { font-size: 16px; font-weight: 600; }
    .badge  { font-size: 11px; font-weight: 700; padding: 3px 8px;
              border-radius: 12px; text-transform: uppercase; letter-spacing: .5px; }
    .badge-urgent  { background: #E74C3C; color: #fff; }
    .badge-monitor { background: #F39C12; color: #fff; }
    .cat-badge { font-size: 11px; background: #EAF0FB; color: #2C5282;
                 padding: 3px 8px; border-radius: 12px; }
    .desc   { font-size: 13px; color: #555; margin: 6px 0; }
    .meta   { font-size: 12px; color: #888; display: flex; gap: 16px; flex-wrap: wrap; margin-top: 8px; }
    .spark-wrap { margin-top: 10px; }
    .spark-label { font-size: 11px; color: #999; margin-bottom: 4px; }
    .stats  { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 10px; }
    .stat-box { background: #fff; border-radius: 10px; padding: 16px 24px; min-width: 140px;
                border: 1px solid #E0E0E0; box-shadow: 0 1px 4px rgba(0,0,0,.05); }
    .stat-num   { font-size: 32px; font-weight: 700; color: #2C3E50; }
    .stat-label { font-size: 12px; color: #7F8C8D; margin-top: 4px; }
    .no-alerts { color: #7F8C8D; font-size: 14px; padding: 20px 0; }
    table { width: 100%; border-collapse: collapse; background: #fff;
            border-radius: 10px; overflow: hidden; }
    th { background: #3498DB; color: #fff; text-align: left; padding: 10px 14px; font-size: 13px; }
    td { padding: 10px 14px; font-size: 13px; border-bottom: 1px solid #F0F0F0; }
    tr:last-child td { border-bottom: none; }
    """

    # ── helpers ───────────────────────────────────────────────────────────────
    def alert_card(a: dict) -> str:
        sev   = a.get("severity", "monitor")
        spark = _sparkline(a.get("trend_last_5_days", []), sev)
        return f"""
        <div class="card {sev}">
          <div class="card-header">
            <span class="name">{a['person_name']}</span>
            <span class="badge badge-{sev}">{sev}</span>
            <span class="cat-badge">{a.get('category','')}</span>
          </div>
          <p class="desc">{a.get('description','')}</p>
          <div class="meta">
            <span>Baseline: <b>{a.get('baseline_wellbeing','?')}</b></span>
            <span>Today: <b>{a.get('today_wellbeing','?')}</b></span>
            <span>Delta: <b>{a.get('delta','?')}</b></span>
            <span>Days flagged: <b>{a.get('days_flagged_consecutively', 1)}</b></span>
            <span>Lowest trait: <b>{a.get('lowest_trait','?')}</b>
                  ({a.get('lowest_trait_value','?')})</span>
          </div>
          <div class="spark-wrap">
            <div class="spark-label">5-day wellbeing trend</div>
            {spark}
          </div>
          <p style="font-size:12px;color:#E74C3C;margin-top:8px;">
            &#128276; {a.get('recommended_action','')}
          </p>
        </div>"""

    # ── Section 1 ────────────────────────────────────────────────────────────
    if today_alerts:
        s1_html = "".join(alert_card(a) for a in today_alerts)
    else:
        s1_html = '<p class="no-alerts">No alerts for today.</p>'

    # ── Section 2 ────────────────────────────────────────────────────────────
    ss = school_summary
    s2_html = f"""
    <div class="stats">
      <div class="stat-box">
        <div class="stat-num">{ss['total_persons_tracked']}</div>
        <div class="stat-label">Persons tracked</div>
      </div>
      <div class="stat-box">
        <div class="stat-num" style="color:#E74C3C">{ss['persons_flagged_today']}</div>
        <div class="stat-label">Alerts today</div>
      </div>
      <div class="stat-box">
        <div class="stat-num" style="color:#F39C12">{len(absence_flags)}</div>
        <div class="stat-label">Absence flags</div>
      </div>
      <div class="stat-box">
        <div class="stat-num" style="color:#27AE60">{ss['school_avg_wellbeing_today']}</div>
        <div class="stat-label">Avg wellbeing today</div>
      </div>
      <div class="stat-box">
        <div class="stat-num" style="font-size:18px;color:#8E44AD">{ss['most_common_anomaly_this_week']}</div>
        <div class="stat-label">Top anomaly this week</div>
      </div>
    </div>"""

    # Absence flags table
    if absence_flags:
        rows = "".join(
            f"<tr><td>{f['person_name']}</td><td>{f['person_id']}</td>"
            f"<td>{f['last_seen_date']}</td><td><b>{f['days_absent']}</b></td>"
            f"<td style='color:#E74C3C'>{f['recommended_action']}</td></tr>"
            for f in absence_flags
        )
        s2_html += f"""
        <h3 style="margin:20px 0 10px;font-size:15px;">Absence Flags</h3>
        <table>
          <tr><th>Name</th><th>ID</th><th>Last Seen</th><th>Days Absent</th><th>Action</th></tr>
          {rows}
        </table>"""

    # ── Section 3 ────────────────────────────────────────────────────────────
    if chronic_list:
        rows = "".join(
            f"<tr><td>{a['person_name']}</td><td>{a['person_id']}</td>"
            f"<td><span class='badge badge-{a['severity']}'>{a['severity']}</span></td>"
            f"<td>{a['category']}</td>"
            f"<td><b>{a['days_flagged_consecutively']}</b></td>"
            f"<td>{a['description']}</td></tr>"
            for a in chronic_list
        )
        s3_html = f"""
        <table>
          <tr><th>Name</th><th>ID</th><th>Severity</th><th>Category</th>
              <th>Days</th><th>Description</th></tr>
          {rows}
        </table>"""
    else:
        s3_html = '<p class="no-alerts">No chronic flags at this time.</p>'

    # ── Assemble ──────────────────────────────────────────────────────────────
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sentio Mind — Alert Digest · {SCHOOL}</title>
  <style>{css}</style>
</head>
<body>
  <h1>&#128270; Sentio Mind — Counsellor Alert Digest</h1>
  <p class="subtitle">{SCHOOL} &nbsp;|&nbsp; Generated: {generated_at} &nbsp;|&nbsp; Report date: {today_str}</p>

  <h2>&#9888;&#65039; Section 1: Today's Action Required</h2>
  {s1_html}

  <h2>&#127982; Section 2: School Summary</h2>
  {s2_html}

  <h2>&#128200; Section 3: Chronic Flags (3+ consecutive days)</h2>
  {s3_html}
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

    # Build per-person history
    person_days: dict = defaultdict(dict)
    person_info: dict = {}
    for d, persons in daily_data.items():
        for pid, pdata in persons.items():
            person_days[pid][d] = pdata
            if pid not in person_info:
                person_info[pid] = pdata.get("person_info", {"name": pid, "profile_image_b64": ""})

    all_alerts    = []
    absence_flags = []

    for pid, days in person_days.items():
        sorted_days   = dict(sorted(days.items()))
        person_alerts = analyse_person(pid, sorted_days, person_info.get(pid, {}))
        all_alerts.extend(person_alerts)

        # Absence detection
        present = set(days.keys())
        absent  = 0
        for d in reversed(all_dates):
            if d not in present:
                absent += 1
            else:
                break
        if absent >= THRESHOLDS["absence_days"]:
            last_seen = sorted(present)[-1] if present else "unknown"
            absence_flags.append({
                "person_id":          pid,
                "person_name":        person_info.get(pid, {}).get("name", pid),
                "last_seen_date":     last_seen,
                "days_absent":        absent,
                "recommended_action": "Welfare check — contact family if absent again tomorrow",
            })

    sev_order = {"urgent": 0, "monitor": 1, "informational": 2}
    all_alerts.sort(key=lambda a: sev_order.get(a.get("severity", "informational"), 3))

    today_str      = str(date.today())
    flagged_today  = sum(1 for a in all_alerts if a.get("date") == today_str)
    cat_counter    = Counter(a.get("category") for a in all_alerts)
    top_category   = cat_counter.most_common(1)[0][0] if cat_counter else "none"

    # School average wellbeing for today
    today_wellbeings = [
        pdata["wellbeing"]
        for pdata in daily_data.get(today_str, {}).values()
    ]
    avg_wb_today = int(np.mean(today_wellbeings)) if today_wellbeings else 0

    school_summary = {
        "total_persons_tracked":         len(person_days),
        "persons_flagged_today":         flagged_today,
        "persons_flagged_yesterday":     0,
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
            "informational": sum(1 for a in all_alerts if a.get("severity") == "informational"),
        },
        "alerts":         all_alerts,
        "absence_flags":  absence_flags,
        "school_summary": school_summary,
    }

    with open(FEED_OUT, "w") as f:
        json.dump(feed, f, indent=2)

    generate_alert_digest(all_alerts, absence_flags, school_summary, REPORT_OUT)

    print()
    print("=" * 56)
    print(f"  Alerts:        {feed['alert_summary']['total_alerts']} total  "
          f"({feed['alert_summary']['urgent']} urgent, "
          f"{feed['alert_summary']['monitor']} monitor)")
    print(f"  Absence flags: {len(absence_flags)}")
    print(f"  Report  ->  {REPORT_OUT}")
    print(f"  JSON    ->  {FEED_OUT}")
    print("=" * 56)
