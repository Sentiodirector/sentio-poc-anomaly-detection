import json
import numpy as np
from pathlib import Path
from datetime import datetime, date, timedelta
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

_alert_counter = [0]


def _next_id():
    _alert_counter[0] += 1
    return f"ALT_{_alert_counter[0]:03d}"


def _get_trait(d, key):
    return d.get(key, d.get("traits", {}).get(key, 0))


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
def load_daily_data(folder: Path) -> dict:
    """
    Read all analysis_*.json files from folder.
    Return: { "YYYY-MM-DD": { "PERSON_ID": { wellbeing, traits, gaze, name, ... }, ... }, ... }
    """
    daily = {}
    for fp in sorted(folder.glob("*.json")):
        with open(fp) as f:
            raw = json.load(f)
        date_str = raw.get("date")
        if not date_str:
            continue
        daily[date_str] = {}
        for pid, pdata in raw.get("students", {}).items():
            flat = dict(pdata)
            # Flatten trait sub-keys to top level for uniform access
            for tk in ["social_engagement", "physical_energy", "movement_energy"]:
                if tk not in flat:
                    flat[tk] = flat.get("traits", {}).get(tk, 0)
            daily[date_str][pid] = flat
    return daily


# ---------------------------------------------------------------------------
# BASELINE
# ---------------------------------------------------------------------------
def compute_baseline(history: list) -> dict:
    """
    history: list of daily dicts (oldest first), each has at minimum:
      { wellbeing: int, traits: {}, gaze_direction: str }

    Use first THRESHOLDS['baseline_window'] days.

    Returns:
        { wellbeing_mean, wellbeing_std, trait_means: {}, avg_gaze }
    """
    window = history[: THRESHOLDS["baseline_window"]]
    if not window:
        return {"wellbeing_mean": 50.0, "wellbeing_std": 10.0, "trait_means": {}, "avg_gaze": "forward"}

    wb_vals = [d["wellbeing"] for d in window]
    wellbeing_mean = float(np.mean(wb_vals))
    wellbeing_std  = float(np.std(wb_vals)) if len(wb_vals) > 1 else 0.0

    trait_keys  = ["social_engagement", "physical_energy", "movement_energy"]
    trait_means = {tk: float(np.mean([_get_trait(d, tk) for d in window])) for tk in trait_keys}
    # trait_means = {tk: float(np.mean([d[tk] for d in window])) for tk in trait_keys}


    gazes    = [d.get("gaze_direction", "forward") for d in window]
    avg_gaze = Counter(gazes).most_common(1)[0][0]

    return {"wellbeing_mean": wellbeing_mean, "wellbeing_std": wellbeing_std,
            "trait_means": trait_means, "avg_gaze": avg_gaze}


# ---------------------------------------------------------------------------
# SHARED ALERT BUILDER
# ---------------------------------------------------------------------------

def _build_alert(category, severity, person_id, person_name, date_str,
                 baseline, today, trend, title, description, action, info,
                 delta_override=None, days_consec=1):
    wb   = today["wellbeing"]
    bwb  = round(baseline.get("wellbeing_mean", 0.0), 1)
    traits = {k: _get_trait(today, k) for k in ["social_engagement", "physical_energy", "movement_energy"]}
    lt   = min(traits, key=traits.get)
    delta = delta_override if delta_override is not None else round(wb - bwb, 1)
    return {
        "alert_id":               _next_id(),
        "person_id":              person_id,
        "person_name":            person_name,
        "date":                   date_str,
        "severity":               severity,
        "category":               category,
        "title":                  title,
        "description":            description,
        "baseline_wellbeing":     bwb,
        "today_wellbeing":        wb,
        "delta":                  delta,
        "days_flagged_consecutively": days_consec,
        "trend_last_5_days":      trend,
        "lowest_trait":           lt,
        "lowest_trait_value":     traits[lt],
        "recommended_action":     action,
        "profile_image_b64":      info.get("profile_image_b64", ""),
    }


# ---------------------------------------------------------------------------
# ANOMALY DETECTORS  — each returns an alert dict or None
# ---------------------------------------------------------------------------

def detect_sudden_drop(today: dict, baseline: dict, **ctx) -> dict | None:
    """
    today['wellbeing'] vs baseline['wellbeing_mean'].
    If baseline_std > 15, raise threshold to sudden_drop_high_std_delta.
    Severity: delta > 35 = urgent, else monitor.
    """
    wb    = today["wellbeing"]
    bm    = baseline["wellbeing_mean"]
    bs    = baseline["wellbeing_std"]
    thr   = THRESHOLDS["sudden_drop_high_std_delta"] if bs > THRESHOLDS["high_std_baseline"] \
            else THRESHOLDS["sudden_drop_delta"]
    delta = bm - wb
    if delta < thr:
        return None

    sev   = "urgent" if delta > 35 else "monitor"
    name  = ctx["person_name"]
    gaze  = today.get("gaze_direction", "forward")
    traits = {k: _get_trait(today, k) for k in ["social_engagement", "physical_energy", "movement_energy"]}
    lt    = min(traits, key=traits.get)
    desc  = (f"{name}'s wellbeing dropped from a baseline of {bm:.0f} to {wb} today "
             f"— a {delta:.0f}-point fall. Lowest trait: {lt} at {traits[lt]}. "
             f"Dominant gaze: {gaze}.")
    action = "Schedule pastoral check-in today" if sev == "urgent" else "Monitor closely for next 2 days"
    return _build_alert("SUDDEN_DROP", sev, ctx["person_id"], name, ctx["date_str"],
                        baseline, today, ctx["trend"],
                        "Sudden wellbeing drop detected", desc, action, ctx["info"],
                        delta_override=round(-delta, 1))


def detect_sustained_low(history: list, **ctx) -> dict | None:
    """
    Check the last sustained_low_days entries in history.
    If all have wellbeing < sustained_low_score → alert.
    Severity: urgent.
    """
    n = THRESHOLDS["sustained_low_days"]
    if len(history) < n:
        return None
    last_n = history[-n:]
    if not all(d["wellbeing"] < THRESHOLDS["sustained_low_score"] for d in last_n):
        return None

    today = last_n[-1]
    name  = ctx["person_name"]
    desc  = (f"{name} has had wellbeing below {THRESHOLDS['sustained_low_score']} "
             f"for {n} consecutive days. Today's score: {today['wellbeing']}.")
    return _build_alert("SUSTAINED_LOW", "urgent", ctx["person_id"], name, ctx["date_str"],
                     ctx["baseline"], today, ctx["trend"],
                     f"Wellbeing below {THRESHOLDS['sustained_low_score']} for {n}+ consecutive days",
                     desc, "Immediate counsellor referral", ctx["info"], days_consec=n)
    

def detect_social_withdrawal(today: dict, baseline: dict, **ctx) -> dict | None:
    """
    social_engagement dropped >= social_withdrawal_delta AND
    today's gaze_direction is 'down' or 'side'.
    Severity: monitor.
    """
    se_today = _get_trait(today, "social_engagement")
    se_base  = baseline["trait_means"].get("social_engagement", 50)
    se_drop  = se_base - se_today
    gaze     = today.get("gaze_direction", "forward")
    if se_drop < THRESHOLDS["social_withdrawal_delta"] or gaze not in ("down", "side"):
        return None

    name = ctx["person_name"]
    desc = (f"{name}'s social engagement dropped {se_drop:.0f} pts from baseline "
            f"({se_base:.0f} → {se_today}). Dominant gaze: {gaze}.")
    return _build_alert("SOCIAL_WITHDRAWAL", "monitor", ctx["person_id"], name, ctx["date_str"],
                        baseline, today, ctx["trend"],
                        "Social withdrawal detected", desc,
                        "Check in with student; observe social interactions", ctx["info"],
                        delta_override=round(se_today - se_base, 1))


def detect_hyperactivity_spike(today: dict, baseline: dict, **ctx) -> dict | None:
    """
    (today.physical_energy + today.movement_energy) minus
    (baseline.physical_energy_mean + baseline.movement_energy_mean) >= hyperactivity_delta.
    Severity: monitor.
    """
    pe      = _get_trait(today, "physical_energy")
    me      = _get_trait(today, "movement_energy")
    pe_b    = baseline["trait_means"].get("physical_energy", 50)
    me_b    = baseline["trait_means"].get("movement_energy", 50)
    cdelta  = (pe + me) - (pe_b + me_b)
    if cdelta < THRESHOLDS["hyperactivity_delta"]:
        return None

    name = ctx["person_name"]
    desc = (f"{name}'s combined energy (physical + movement) spiked {cdelta:.0f} pts "
            f"above baseline. Physical: {pe}, Movement: {me}.")
    return _build_alert("HYPERACTIVITY_SPIKE", "monitor", ctx["person_id"], name, ctx["date_str"],
                        baseline, today, ctx["trend"],
                        "Hyperactivity spike detected", desc,
                        "Observe for anxiety/manic signs; consider structured activity", ctx["info"],
                        delta_override=round(cdelta, 1))


def detect_regression(history: list, **ctx) -> dict | None:
    """
    Find if the last regression_recover_days entries were all improving (each > previous),
    then today dropped > regression_drop.
    Severity: monitor.
    """
    n = THRESHOLDS["regression_recover_days"]
    if len(history) < n + 1:
        return None
    recent   = history[-(n + 1):]
    recovery = recent[:n]
    today_e  = recent[n]
    improving = all(recovery[i]["wellbeing"] > recovery[i - 1]["wellbeing"] for i in range(1, n))
    if not improving:
        return None
    drop = recovery[-1]["wellbeing"] - today_e["wellbeing"]
    if drop <= THRESHOLDS["regression_drop"]:
        return None

    name    = ctx["person_name"]
    fake_bl = {"wellbeing_mean": float(recovery[-1]["wellbeing"]),
               "wellbeing_std": 0.0,
               "trait_means": ctx["baseline"]["trait_means"]}
    desc    = (f"{name} was recovering for {n} days (scores improving each day), "
               f"then dropped {drop:.0f} pts today to {today_e['wellbeing']}.")
    return _build_alert("REGRESSION", "monitor", ctx["person_id"], name, ctx["date_str"],
                        fake_bl, today_e, ctx["trend"],
                        "Regression after recovery period", desc,
                        "Revisit support plan — relapse after improvement", ctx["info"],
                        delta_override=round(-drop, 1))


def detect_gaze_avoidance(history: list, **ctx) -> dict | None:
    """
    Last gaze_avoidance_days entries all have eye_contact == False (or missing).
    Severity: monitor.
    """
    n = THRESHOLDS["gaze_avoidance_days"]
    if len(history) < n:
        return None
    last_n = history[-n:]
    if not all(not d.get("eye_contact", True) for d in last_n):
        return None

    today = last_n[-1]
    name  = ctx["person_name"]
    desc  = (f"{name} has shown no eye contact for {n} consecutive days. "
             f"This may indicate social anxiety or withdrawal behaviour.")
    return _build_alert("GAZE_AVOIDANCE", "monitor", ctx["person_id"], name, ctx["date_str"],
                     ctx["baseline"], today, ctx["trend"],
                     f"No eye contact for {n}+ consecutive days", desc,
                     "Gentle check-in; consider counsellor referral", ctx["info"],
                     days_consec=n)
    

# ---------------------------------------------------------------------------
# Peer comparison anomaly
# ---------------------------------------------------------------------------

def detect_peer_outlier(today: dict, school_avg: float, school_std: float,
                        person_id: str, person_name: str, date_str: str,
                        trend: list, info: dict) -> dict | None:
    wb = today["wellbeing"]
    if school_std > 0 and (school_avg - wb) > 2 * school_std:
        traits = {k: _get_trait(today, k) for k in ["social_engagement", "physical_energy", "movement_energy"]}
        lt = min(traits, key=traits.get)
        return {
            "alert_id":   _next_id(),
            "person_id":  person_id, "person_name": person_name, "date": date_str,
            "severity":   "monitor",
            "category":   "SUDDEN_DROP",
            "title":      "Wellbeing significantly below class average",
            "description": (f"{person_name}'s wellbeing ({wb}) is more than 2 SD below "
                            f"the class average ({school_avg:.0f}). Peer comparison flag."),
            "baseline_wellbeing": round(school_avg, 1),
            "today_wellbeing": wb,
            "delta": round(wb - school_avg, 1),
            "days_flagged_consecutively": 1,
            "trend_last_5_days": trend,
            "lowest_trait": lt, "lowest_trait_value": traits[lt],
            "recommended_action": "Check in — significantly below class peers today",
            "profile_image_b64": info.get("profile_image_b64", ""),
        }
    return None


# ---------------------------------------------------------------------------
# ANALYSE ONE PERSON
# ---------------------------------------------------------------------------
def analyse_person(person_id: str, sorted_days: dict, info: dict) -> list:
    """
    sorted_days: { "YYYY-MM-DD": person_data_dict } — keys in date order
    info: { name, profile_image_b64, ... }
    """
    person_name = info.get("name", person_id)
    history     = []
    alerts      = []

    for d_str in sorted_days:
        entry = dict(sorted_days[d_str])
        history.append(entry)

        baseline = compute_baseline(history)
        trend    = ([0] * max(0, 5 - len(history))) + [h["wellbeing"] for h in history[-5:]]
        ctx_base = dict(person_id=person_id, person_name=person_name, date_str=d_str,
                        trend=trend, info=info)
        ctx_hist = dict(**ctx_base, baseline=baseline)

        # Point-in-time detectors (require established baseline first)
        if len(history) > THRESHOLDS["baseline_window"]:
            for fn in (detect_sudden_drop, detect_social_withdrawal, detect_hyperactivity_spike):
                a = fn(entry, baseline, **ctx_base)
                if a:
                    alerts.append(a)

        # History-based detectors
        for fn in (detect_sustained_low, detect_regression, detect_gaze_avoidance):
            a = fn(history, **ctx_hist)
            if a:
                alerts.append(a)

    return alerts


# ---------------------------------------------------------------------------
# HTML REPORT
# ---------------------------------------------------------------------------
def _sparkline(trend: list, w=110, h=30) -> str:
    vals  = [v if v else 0 for v in trend]
    lo, hi = min(vals), max(vals)
    rng   = hi - lo if hi != lo else 1
    n     = len(vals)
    pts   = []
    for i, v in enumerate(vals):
        x = int(i * (w - 6) / max(n - 1, 1)) + 3
        y = h - 4 - int((v - lo) / rng * (h - 10))
        pts.append(f"{x},{y}")
    pts_str = " ".join(pts)
    last    = vals[-1] if vals else 50
    color   = "#e74c3c" if last < 45 else "#f39c12" if last < 60 else "#27ae60"
    lx, ly  = pts[-1].split(",") if pts else ("55", "15")
    return (f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">'
            f'<polyline points="{pts_str}" fill="none" stroke="{color}" stroke-width="2.2" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
            f'<circle cx="{lx}" cy="{ly}" r="3.5" fill="{color}"/>'
            f'</svg>')


def generate_alert_digest(alerts: list, absence_flags: list,
                          school_summary: dict, output_path: Path):
    SEV_BADGE = {
        "urgent":        '<span style="background:#e74c3c;color:#fff;padding:2px 10px;border-radius:12px;font-size:11px;font-weight:bold;letter-spacing:.5px">URGENT</span>',
        "monitor":       '<span style="background:#f39c12;color:#fff;padding:2px 10px;border-radius:12px;font-size:11px;font-weight:bold;letter-spacing:.5px">MONITOR</span>',
        "informational": '<span style="background:#3498db;color:#fff;padding:2px 10px;border-radius:12px;font-size:11px;font-weight:bold;letter-spacing:.5px">INFO</span>',
    }
    CAT_COLOR = {
        "SUDDEN_DROP":        "#e74c3c", "SUSTAINED_LOW":     "#c0392b",
        "SOCIAL_WITHDRAWAL":  "#8e44ad", "HYPERACTIVITY_SPIKE":"#e67e22",
        "REGRESSION":         "#f39c12", "GAZE_AVOIDANCE":    "#16a085",
        "ABSENCE_FLAG":       "#2c3e50",
    }

    today_str    = str(date.today())
    # Use latest data date for "today" section
    all_dates    = sorted({a["date"] for a in alerts})
    display_date = all_dates[-1] if all_dates else today_str
    today_alerts = [a for a in alerts if a.get("date") == display_date]

    # Persistent (flagged 3+ consecutive days)
    flagged_days = defaultdict(set)
    for a in alerts:
        flagged_days[a["person_id"]].add(a["date"])
    persistent = []
    for pid, days in flagged_days.items():
        sorted_d = sorted(days)
        consec = max_consec = 1
        for i in range(1, len(sorted_d)):
            d1 = datetime.strptime(sorted_d[i-1], "%Y-%m-%d").date()
            d2 = datetime.strptime(sorted_d[i],   "%Y-%m-%d").date()
            consec = consec + 1 if (d2 - d1).days == 1 else 1
            max_consec = max(max_consec, consec)
        if max_consec >= 3:
            name = next((a["person_name"] for a in alerts if a["person_id"] == pid), pid)
            persistent.append({"name": name, "days": max_consec})

    # Build alert cards
    cards = ""
    for a in today_alerts:
        cat   = a.get("category", "")
        cc    = CAT_COLOR.get(cat, "#7f8c8d")
        badge = SEV_BADGE.get(a.get("severity", "informational"), "")
        spark = _sparkline(a.get("trend_last_5_days", [0]*5))
        cards += f"""
        <div style="background:#fff;border-left:5px solid {cc};border-radius:8px;padding:16px 18px;
                    margin-bottom:12px;box-shadow:0 1px 5px rgba(0,0,0,0.08)">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
            <div>
              <strong style="font-size:15px">{a['person_name']}</strong>
              <span style="margin-left:8px;background:{cc};color:#fff;padding:2px 8px;
                           border-radius:10px;font-size:11px">{cat}</span>
              <span style="margin-left:6px">{badge}</span>
            </div>
            <div title="5-day wellbeing trend">{spark}</div>
          </div>
          <p style="margin:4px 0;color:#555;font-size:13px;line-height:1.5">{a.get('description','')}</p>
          <p style="margin:8px 0 0;font-size:12px;color:#888">
            &#128203; <em>{a.get('recommended_action','')}</em>
          </p>
        </div>"""

    if not cards:
        cards = '<p style="color:#888;padding:8px 0">No alerts for today.</p>'

    # Absence table rows
    ab_rows = ""
    for af in absence_flags:
        ab_rows += f"""
        <tr>
          <td style="padding:9px 12px">{af['person_name']}</td>
          <td style="padding:9px 12px;text-align:center;color:#e74c3c;font-weight:bold">{af['days_absent']}</td>
          <td style="padding:9px 12px">{af['last_seen_date']}</td>
          <td style="padding:9px 12px;color:#e74c3c">{af['recommended_action']}</td>
        </tr>"""
    if not ab_rows:
        ab_rows = '<tr><td colspan="4" style="padding:12px;color:#aaa;text-align:center">No absence flags</td></tr>'

    # Persistent list
    pers_html = "".join(
        f'<li><strong>{p["name"]}</strong> — flagged <strong>{p["days"]}</strong> consecutive days</li>'
        for p in persistent
    ) or '<li style="color:#aaa">No persistent flags this period</li>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sentio Mind — Alert Digest</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
        background:#f0f2f5;color:#2c3e50;padding:28px 20px}}
  .wrap{{max-width:920px;margin:0 auto}}
  .header{{margin-bottom:24px}}
  .header h1{{font-size:22px;font-weight:700}}
  .header p{{color:#7f8c8d;font-size:13px;margin-top:4px}}
  .card{{background:#fff;border-radius:12px;padding:22px;margin-bottom:20px;
         box-shadow:0 1px 6px rgba(0,0,0,0.08)}}
  .card h2{{font-size:15px;font-weight:600;margin-bottom:16px;padding-bottom:10px;
            border-bottom:2px solid #f0f2f5;color:#2c3e50}}
  .stat-row{{display:flex;gap:14px;flex-wrap:wrap}}
  .stat{{background:#f8f9fa;border-radius:10px;padding:14px 18px;flex:1;min-width:130px}}
  .stat .v{{font-size:26px;font-weight:700;color:#2c3e50}}
  .stat .l{{font-size:11px;color:#95a5a6;margin-top:3px;text-transform:uppercase;letter-spacing:.4px}}
  table{{width:100%;border-collapse:collapse}}
  th{{background:#f8f9fa;padding:10px 12px;text-align:left;font-size:11px;
      color:#95a5a6;text-transform:uppercase;letter-spacing:.5px}}
  tr:nth-child(even) td{{background:#fafafa}}
  ul{{padding-left:22px}}
  li{{margin-bottom:8px;font-size:14px;line-height:1.5}}
</style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <h1>&#129504; Sentio Mind &mdash; Alert Digest</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp; School: {SCHOOL}</p>
  </div>

  <div class="card">
    <h2>&#128202; School Summary</h2>
    <div class="stat-row">
      <div class="stat">
        <div class="v">{school_summary.get('total_persons_tracked',0)}</div>
        <div class="l">Students Tracked</div>
      </div>
      <div class="stat">
        <div class="v" style="color:#e74c3c">{school_summary.get('persons_flagged_today',0)}</div>
        <div class="l">Flagged Today</div>
      </div>
      <div class="stat">
        <div class="v">{school_summary.get('persons_flagged_yesterday',0)}</div>
        <div class="l">Flagged Yesterday</div>
      </div>
      <div class="stat">
        <div class="v" style="color:#27ae60">{school_summary.get('school_avg_wellbeing_today',0):.0f}</div>
        <div class="l">Avg Wellbeing Today</div>
      </div>
      <div class="stat">
        <div class="v" style="font-size:13px;padding-top:8px">{school_summary.get('most_common_anomaly_this_week','—')}</div>
        <div class="l">Top Anomaly This Week</div>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>&#128680; Today&apos;s Alerts &mdash; {display_date}</h2>
    {cards}
  </div>

  <div class="card">
    <h2>&#128683; Absence Flags</h2>
    <table>
      <thead><tr>
        <th>Student</th><th>Days Absent</th><th>Last Seen</th><th>Recommended Action</th>
      </tr></thead>
      <tbody>{ab_rows}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>&#128260; Persistent Alerts (3+ Consecutive Days)</h2>
    <ul>{pers_html}</ul>
  </div>
</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    daily_data = load_daily_data(DATA_DIR)
    all_dates  = sorted(daily_data.keys())
    print(f"Loaded {len(daily_data)} days: {all_dates}")

    # Build per-person history
    person_days = defaultdict(dict)
    person_info = {}
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

        # Check absence
        present = set(days.keys())
        absent = 0
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

    # peer-comparison for latest date
    if all_dates:
        latest     = all_dates[-1]
        today_data = daily_data.get(latest, {})
        wb_vals    = [v["wellbeing"] for v in today_data.values()]
        if wb_vals:
            s_avg = float(np.mean(wb_vals))
            s_std = float(np.std(wb_vals))
            for pid, pdata in today_data.items():
                pinfo = person_info.get(pid, {"name": pid, "profile_image_b64": ""})
                hist  = list(dict(sorted(person_days[pid].items())).values())
                trend = ([0] * max(0, 5 - len(hist))) + [h["wellbeing"] for h in hist[-5:]]
                pa    = detect_peer_outlier(pdata, s_avg, s_std, pid,
                                            pinfo.get("name", pid), latest, trend, pinfo)
                if pa:
                    all_alerts.append(pa)

    sev_order = {"urgent": 0, "monitor": 1, "informational": 2}
    all_alerts.sort(key=lambda a: sev_order.get(a.get("severity", "informational"), 3))

    latest          = all_dates[-1] if all_dates else str(date.today())
    flagged_latest  = len({a["person_id"] for a in all_alerts if a["date"] == latest})
    cat_counter     = Counter(a.get("category") for a in all_alerts)
    top_category    = cat_counter.most_common(1)[0][0] if cat_counter else "none"
    today_wb        = [daily_data[latest][p]["wellbeing"] for p in daily_data.get(latest, {})]
    avg_wb_today    = round(float(np.mean(today_wb)), 1) if today_wb else 0.0

    school_summary = {
        "total_persons_tracked":        len(person_days),
        "persons_flagged_today":        flagged_latest,
        "persons_flagged_yesterday":    0,
        "most_common_anomaly_this_week": top_category,
        "school_avg_wellbeing_today":   avg_wb_today,
    }

    feed = {
        "source":       "p5_anomaly_detection",
        "generated_at": datetime.now().isoformat(),
        "school":       SCHOOL,
        "alert_summary": {
            "total_alerts":  len(all_alerts),
            "urgent":        sum(1 for a in all_alerts if a.get("severity") == "urgent"),
            "monitor":       sum(1 for a in all_alerts if a.get("severity") == "monitor"),
            "informational": sum(1 for a in all_alerts if a.get("severity") == "informational"),
        },
        "alerts":        all_alerts,
        "absence_flags": absence_flags,
        "school_summary": school_summary,
    }

    with open(FEED_OUT, "w") as f:
        json.dump(feed, f, indent=2)

    generate_alert_digest(all_alerts, absence_flags, school_summary, REPORT_OUT)

    print()
    print("=" * 50)
    print(f"  Alerts:  {feed['alert_summary']['total_alerts']} total  "
          f"({feed['alert_summary']['urgent']} urgent, {feed['alert_summary']['monitor']} monitor)")
    print(f"  Absence flags: {len(absence_flags)}")
    print(f"  Report → {REPORT_OUT}")
    print(f"  JSON   → {FEED_OUT}")
    print("=" * 50)
