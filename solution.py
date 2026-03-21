"""
solution.py
Sentio Mind · Project 5 · Behavioral Anomaly & Early Distress Detection

"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime, date, timedelta
from collections import defaultdict, Counter

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR   = Path("sample_data")
REPORT_OUT = Path("alert_digest.html")
FEED_OUT   = Path("alert_feed.json")
SCHOOL     = "JAGRAN PUBLIC SCHOOL"

THRESHOLDS = {
    "sudden_drop_delta":           20,
    "sudden_drop_high_std_delta":  30,
    "sustained_low_score":         45,
    "sustained_low_days":           3,
    "social_withdrawal_delta":     25,
    "hyperactivity_delta":         40,
    "regression_recover_days":      3,
    "regression_drop":             15,
    "gaze_avoidance_days":          3,
    "absence_days":                 2,
    "baseline_window":              3,
    "high_std_baseline":           15,
}

_alert_counter = [0]

def _next_alert_id():
    _alert_counter[0] += 1
    return f"ALT_{_alert_counter[0]:03d}"


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_daily_data(folder: Path) -> dict:
    """
    Read all analysis_*.json files from folder.
    Return: { "YYYY-MM-DD": { "PERSON_ID": { wellbeing, gaze, ... } } }
    """
    daily = {}
    for fp in sorted(folder.glob("*.json")):
        with open(fp) as f:
            raw = json.load(f)
        date_str = raw.get("date", fp.stem)
        persons  = raw.get("persons", {})
        daily[date_str] = persons
    return daily


# ---------------------------------------------------------------------------
# BASELINE
# ---------------------------------------------------------------------------

def compute_baseline(history: list) -> dict:
    """
    Use first THRESHOLDS['baseline_window'] days (or all if fewer).
    Returns means/std for wellbeing, social, energy, and most common gaze.
    """
    window = history[:THRESHOLDS["baseline_window"]]
    if not window:
        return {
            "wellbeing_mean": 50.0, "wellbeing_std": 10.0,
            "social_mean": 50.0,
            "physical_energy_mean": 50.0, "movement_energy_mean": 50.0,
            "avg_gaze": "forward",
        }

    gazes = [d["gaze_direction"] for d in window]
    return {
        "wellbeing_mean":       float(np.mean([d["wellbeing"]        for d in window])),
        "wellbeing_std":        float(np.std( [d["wellbeing"]        for d in window])),
        "social_mean":          float(np.mean([d["social_engagement"] for d in window])),
        "physical_energy_mean": float(np.mean([d["physical_energy"]  for d in window])),
        "movement_energy_mean": float(np.mean([d["movement_energy"]  for d in window])),
        "avg_gaze":             Counter(gazes).most_common(1)[0][0],
    }


# ---------------------------------------------------------------------------
# ANOMALY DETECTORS
# ---------------------------------------------------------------------------

def detect_sudden_drop(today: dict, baseline: dict, person_id: str,
                       person_name: str, date_str: str, trend: list) -> dict | None:
    threshold = (
        THRESHOLDS["sudden_drop_high_std_delta"]
        if baseline["wellbeing_std"] > THRESHOLDS["high_std_baseline"]
        else THRESHOLDS["sudden_drop_delta"]
    )
    delta = baseline["wellbeing_mean"] - today["wellbeing"]
    if delta < threshold:
        return None
    severity = "urgent" if delta > 35 else "monitor"
    traits = {
        "social_engagement": today["social_engagement"],
        "physical_energy":   today["physical_energy"],
        "movement_energy":   today["movement_energy"],
    }
    lowest_trait     = min(traits, key=traits.get)
    lowest_trait_val = traits[lowest_trait]
    return {
        "alert_id": _next_alert_id(), "person_id": person_id,
        "person_name": person_name, "date": date_str, "severity": severity,
        "category": "SUDDEN_DROP", "title": "Sudden wellbeing drop detected",
        "description": (
            f"{person_name}'s wellbeing dropped from a baseline of "
            f"{baseline['wellbeing_mean']:.0f} to {today['wellbeing']} today "
            f"— a {delta:.0f}-point fall. "
            f"Lowest trait: {lowest_trait} at {lowest_trait_val}. "
            f"Dominant gaze: {today['gaze_direction']}."
        ),
        "baseline_wellbeing": round(baseline["wellbeing_mean"]),
        "today_wellbeing": today["wellbeing"], "delta": -round(delta),
        "days_flagged_consecutively": 1, "trend_last_5_days": trend,
        "lowest_trait": lowest_trait, "lowest_trait_value": lowest_trait_val,
        "recommended_action": "Schedule pastoral check-in today",
        "profile_image_b64": today.get("person_info", {}).get("profile_image_b64", ""),
    }


def detect_sustained_low(history: list, person_id: str,
                         person_name: str, date_str: str, trend: list) -> dict | None:
    n = THRESHOLDS["sustained_low_days"]
    if len(history) < n:
        return None
    recent = history[-n:]
    if not all(d["wellbeing"] < THRESHOLDS["sustained_low_score"] for d in recent):
        return None
    avg_wb = np.mean([d["wellbeing"] for d in recent])
    return {
        "alert_id": _next_alert_id(), "person_id": person_id,
        "person_name": person_name, "date": date_str, "severity": "urgent",
        "category": "SUSTAINED_LOW", "title": "Sustained low wellbeing",
        "description": (
            f"{person_name} has had wellbeing below "
            f"{THRESHOLDS['sustained_low_score']} for {n}+ consecutive days "
            f"(avg {avg_wb:.0f}). Immediate support recommended."
        ),
        "baseline_wellbeing": None, "today_wellbeing": history[-1]["wellbeing"],
        "delta": None, "days_flagged_consecutively": n, "trend_last_5_days": trend,
        "lowest_trait": "wellbeing", "lowest_trait_value": history[-1]["wellbeing"],
        "recommended_action": "Escalate to school counsellor immediately",
        "profile_image_b64": history[-1].get("person_info", {}).get("profile_image_b64", ""),
    }


def detect_social_withdrawal(today: dict, baseline: dict, person_id: str,
                              person_name: str, date_str: str, trend: list) -> dict | None:
    social_delta  = baseline["social_mean"] - today["social_engagement"]
    gaze_withdrawn = today["gaze_direction"] in ("down", "side")
    if social_delta < THRESHOLDS["social_withdrawal_delta"] or not gaze_withdrawn:
        return None
    return {
        "alert_id": _next_alert_id(), "person_id": person_id,
        "person_name": person_name, "date": date_str, "severity": "monitor",
        "category": "SOCIAL_WITHDRAWAL", "title": "Social withdrawal detected",
        "description": (
            f"{person_name}'s social engagement dropped "
            f"{social_delta:.0f} points vs baseline "
            f"(baseline {baseline['social_mean']:.0f} → today {today['social_engagement']}). "
            f"Gaze direction: {today['gaze_direction']}."
        ),
        "baseline_wellbeing": round(baseline["wellbeing_mean"]),
        "today_wellbeing": today["wellbeing"], "delta": -round(social_delta),
        "days_flagged_consecutively": 1, "trend_last_5_days": trend,
        "lowest_trait": "social_engagement", "lowest_trait_value": today["social_engagement"],
        "recommended_action": "Encourage peer interaction; monitor over next 2 days",
        "profile_image_b64": today.get("person_info", {}).get("profile_image_b64", ""),
    }


def detect_hyperactivity_spike(today: dict, baseline: dict, person_id: str,
                                person_name: str, date_str: str, trend: list) -> dict | None:
    today_combined    = today["physical_energy"] + today["movement_energy"]
    baseline_combined = baseline["physical_energy_mean"] + baseline["movement_energy_mean"]
    delta             = today_combined - baseline_combined
    if delta < THRESHOLDS["hyperactivity_delta"]:
        return None
    return {
        "alert_id": _next_alert_id(), "person_id": person_id,
        "person_name": person_name, "date": date_str, "severity": "monitor",
        "category": "HYPERACTIVITY_SPIKE", "title": "Hyperactivity spike detected",
        "description": (
            f"{person_name}'s combined energy score spiked by {delta:.0f} points "
            f"above baseline (baseline {baseline_combined:.0f} → today {today_combined}). "
            f"Physical: {today['physical_energy']}, Movement: {today['movement_energy']}."
        ),
        "baseline_wellbeing": round(baseline["wellbeing_mean"]),
        "today_wellbeing": today["wellbeing"], "delta": round(delta),
        "days_flagged_consecutively": 1, "trend_last_5_days": trend,
        "lowest_trait": "movement_energy", "lowest_trait_value": today["movement_energy"],
        "recommended_action": "Check in with student; may indicate anxiety or mania",
        "profile_image_b64": today.get("person_info", {}).get("profile_image_b64", ""),
    }


def detect_regression(history: list, person_id: str,
                      person_name: str, date_str: str, trend: list) -> dict | None:
    n = THRESHOLDS["regression_recover_days"]
    if len(history) < n + 2:
        return None
    recovery_window = history[-(n + 1):-1]
    today_wb        = history[-1]["wellbeing"]
    improving = all(
        recovery_window[i]["wellbeing"] > recovery_window[i - 1]["wellbeing"]
        for i in range(1, len(recovery_window))
    )
    if not improving:
        return None
    peak_wb = recovery_window[-1]["wellbeing"]
    drop    = peak_wb - today_wb
    if drop <= THRESHOLDS["regression_drop"]:
        return None
    return {
        "alert_id": _next_alert_id(), "person_id": person_id,
        "person_name": person_name, "date": date_str, "severity": "monitor",
        "category": "REGRESSION", "title": "Regression after recovery",
        "description": (
            f"{person_name} was improving for {n}+ days (peaked at {peak_wb}) "
            f"but dropped {drop} points today to {today_wb}. Possible relapse."
        ),
        "baseline_wellbeing": peak_wb, "today_wellbeing": today_wb,
        "delta": -drop, "days_flagged_consecutively": 1, "trend_last_5_days": trend,
        "lowest_trait": "wellbeing", "lowest_trait_value": today_wb,
        "recommended_action": "Follow up — recovery may have stalled",
        "profile_image_b64": history[-1].get("person_info", {}).get("profile_image_b64", ""),
    }


def detect_gaze_avoidance(history: list, person_id: str,
                           person_name: str, date_str: str, trend: list) -> dict | None:
    n = THRESHOLDS["gaze_avoidance_days"]
    if len(history) < n:
        return None
    recent = history[-n:]
    if not all(d.get("eye_contact") == False for d in recent):
        return None
    return {
        "alert_id": _next_alert_id(), "person_id": person_id,
        "person_name": person_name, "date": date_str, "severity": "monitor",
        "category": "GAZE_AVOIDANCE", "title": "Persistent gaze avoidance",
        "description": (
            f"{person_name} has had zero eye contact for {n} consecutive days. "
            f"Dominant gaze today: {history[-1]['gaze_direction']}."
        ),
        "baseline_wellbeing": None, "today_wellbeing": history[-1]["wellbeing"],
        "delta": None, "days_flagged_consecutively": n, "trend_last_5_days": trend,
        "lowest_trait": "eye_contact", "lowest_trait_value": 0,
        "recommended_action": "Observe for social anxiety; consider referral",
        "profile_image_b64": history[-1].get("person_info", {}).get("profile_image_b64", ""),
    }


# ---------------------------------------------------------------------------
# BONUS: PEER COMPARISON
# ---------------------------------------------------------------------------

def detect_peer_outlier(today: dict, class_mean: float, class_std: float,
                        person_id: str, person_name: str, date_str: str,
                        trend: list) -> dict | None:
    if class_std == 0:
        return None
    z_score = (today["wellbeing"] - class_mean) / class_std
    if z_score >= -2:
        return None
    return {
        "alert_id": _next_alert_id(), "person_id": person_id,
        "person_name": person_name, "date": date_str, "severity": "monitor",
        "category": "SUDDEN_DROP", "title": "Peer comparison outlier",
        "description": (
            f"{person_name}'s wellbeing ({today['wellbeing']}) is more than 2 std devs "
            f"below class average ({class_mean:.0f})."
        ),
        "baseline_wellbeing": round(class_mean), "today_wellbeing": today["wellbeing"],
        "delta": round(today["wellbeing"] - class_mean),
        "days_flagged_consecutively": 1, "trend_last_5_days": trend,
        "lowest_trait": "wellbeing", "lowest_trait_value": today["wellbeing"],
        "recommended_action": "Peer outlier — cross-check with personal baseline alerts",
        "profile_image_b64": today.get("person_info", {}).get("profile_image_b64", ""),
    }


# ---------------------------------------------------------------------------
# ANALYSE ONE PERSON
# ---------------------------------------------------------------------------

def analyse_person(person_id: str, sorted_days: dict, info: dict,
                   class_wellbeings: dict) -> list:
    alerts      = []
    history     = []
    person_name = info.get("name", person_id)
    dates       = sorted(sorted_days.keys())

    for date_str in dates:
        today = sorted_days[date_str]
        history.append(today)

        # Need at least 2 days (1 for baseline, 1 for today)
        if len(history) < 2:
            continue

        baseline = compute_baseline(history[:-1])
        trend    = [d["wellbeing"] for d in history[-5:]]

        for detector in [
            lambda: detect_sudden_drop(today, baseline, person_id, person_name, date_str, trend),
            lambda: detect_sustained_low(history, person_id, person_name, date_str, trend),
            lambda: detect_social_withdrawal(today, baseline, person_id, person_name, date_str, trend),
            lambda: detect_hyperactivity_spike(today, baseline, person_id, person_name, date_str, trend),
            lambda: detect_regression(history, person_id, person_name, date_str, trend),
            lambda: detect_gaze_avoidance(history, person_id, person_name, date_str, trend),
        ]:
            result = detector()
            if result:
                alerts.append(result)

        # Bonus peer comparison
        if date_str in class_wellbeings:
            others = [w for pid, w in class_wellbeings[date_str] if pid != person_id]
            if others:
                cm = float(np.mean(others))
                cs = float(np.std(others))
                result = detect_peer_outlier(today, cm, cs, person_id, person_name, date_str, trend)
                if result:
                    alerts.append(result)

    return alerts


# ---------------------------------------------------------------------------
# HTML REPORT
# ---------------------------------------------------------------------------

def _sparkline_svg(values: list, width=80, height=24) -> str:
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span   = hi - lo if hi != lo else 1
    n      = len(values)
    step   = width / max(n - 1, 1)
    points = []
    for i, v in enumerate(values):
        x = i * step
        y = height - ((v - lo) / span) * (height - 4) - 2
        points.append(f"{x:.1f},{y:.1f}")
    last_v = values[-1]
    color  = "#ef4444" if last_v < 45 else "#f59e0b" if last_v < 60 else "#22c55e"
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" style="display:inline-block;vertical-align:middle">'
        f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" '
        f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        f'</svg>'
    )


def _severity_badge(severity: str) -> str:
    colors = {
        "urgent":        ("background:#ef4444;color:#fff",  "URGENT"),
        "monitor":       ("background:#f59e0b;color:#fff",  "MONITOR"),
        "informational": ("background:#3b82f6;color:#fff",  "INFO"),
    }
    style, label = colors.get(severity, ("background:#6b7280;color:#fff", severity.upper()))
    return (
        f'<span style="{style};padding:2px 10px;border-radius:12px;'
        f'font-size:11px;font-weight:700;letter-spacing:0.5px">{label}</span>'
    )


def generate_alert_digest(alerts: list, absence_flags: list,
                           school_summary: dict, output_path: Path):
    today_str    = max(a.get("date", "") for a in alerts) if alerts else str(date.today())
    today_alerts = [a for a in alerts if a.get("date") == today_str]

    # Persistent: flagged 3+ consecutive days
    person_dates = defaultdict(list)
    for a in alerts:
        person_dates[a["person_id"]].append(a["date"])

    persistent = []
    for pid, adates in person_dates.items():
        unique_dates = sorted(set(adates))
        if len(unique_dates) < 3:
            continue
        dt_dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in unique_dates]
        streak, max_streak = 1, 1
        for i in range(1, len(dt_dates)):
            if (dt_dates[i] - dt_dates[i-1]).days == 1:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 1
        if max_streak >= 3:
            name = next((a["person_name"] for a in alerts if a["person_id"] == pid), pid)
            persistent.append({"name": name, "streak": max_streak, "pid": pid})

    # Alert cards
    alert_cards_html = ""
    if today_alerts:
        for a in today_alerts:
            trend_svg = _sparkline_svg(a.get("trend_last_5_days", []))
            badge     = _severity_badge(a["severity"])
            alert_cards_html += f"""
            <div style="border:1px solid #e5e7eb;border-radius:10px;padding:16px 20px;
                        margin-bottom:14px;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,0.06)">
              <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;flex-wrap:wrap">
                <span style="font-weight:700;font-size:15px;color:#1f2937">{a['person_name']}</span>
                {badge}
                <span style="font-size:12px;color:#6b7280;margin-left:auto">{a['category']}</span>
              </div>
              <p style="margin:0 0 10px;color:#374151;font-size:13px;line-height:1.5">{a['description']}</p>
              <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
                <span style="font-size:11px;color:#6b7280">5-day trend:</span>
                {trend_svg}
                <span style="font-size:11px;color:#6b7280;margin-left:auto">
                  &#128204; {a.get('recommended_action','')}
                </span>
              </div>
            </div>"""
    else:
        alert_cards_html = '<p style="color:#6b7280;font-style:italic">No alerts triggered today.</p>'

    # Absence cards
    absence_html = ""
    for ab in absence_flags:
        absence_html += f"""
        <div style="border:1px solid #fca5a5;border-radius:10px;padding:14px 18px;
                    margin-bottom:12px;background:#fff7f7">
          <strong style="color:#b91c1c">&#9888; {ab['person_name']}</strong>
          <span style="font-size:12px;color:#6b7280;margin-left:8px">
            Absent {ab['days_absent']} day(s) &nbsp;|&nbsp; Last seen: {ab['last_seen_date']}
          </span>
          <p style="margin:6px 0 0;font-size:12px;color:#374151">{ab['recommended_action']}</p>
        </div>"""
    if not absence_html:
        absence_html = '<p style="color:#6b7280;font-style:italic">No absences flagged.</p>'

    # Persistent section
    persistent_html = ""
    if persistent:
        for p in persistent:
            persistent_html += f"""
            <div style="display:flex;align-items:center;gap:10px;padding:10px 14px;
                        border-radius:8px;background:#fff;border:1px solid #e5e7eb;margin-bottom:8px">
              <span style="font-weight:600;color:#1f2937">{p['name']}</span>
              <span style="font-size:12px;color:#ef4444">
                Flagged {p['streak']} consecutive days
              </span>
            </div>"""
    else:
        persistent_html = '<p style="color:#6b7280;font-style:italic">No students flagged 3+ consecutive days.</p>'

    ss   = school_summary
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sentio Mind · Alert Digest · {today_str}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f3f4f6; color: #1f2937; min-height: 100vh; padding: 32px 16px;
  }}
  .container {{ max-width: 860px; margin: 0 auto; }}
  h1 {{ font-size: 22px; font-weight: 800; color: #111827; margin-bottom: 4px; }}
  h2 {{ font-size: 15px; font-weight: 700; color: #374151; margin: 28px 0 12px;
        border-bottom: 2px solid #e5e7eb; padding-bottom: 6px; }}
  .subtitle {{ font-size: 12px; color: #9ca3af; margin-bottom: 28px; }}
  .summary-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px; margin-bottom: 8px;
  }}
  .stat-card {{
    background: #fff; border-radius: 10px; padding: 16px; text-align: center;
    border: 1px solid #e5e7eb; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }}
  .stat-val {{ font-size: 28px; font-weight: 800; color: #111827; }}
  .stat-lbl {{ font-size: 11px; color: #6b7280; margin-top: 4px; }}
  footer {{ text-align:center; font-size:11px; color:#9ca3af; margin-top:40px; }}
</style>
</head>
<body>
<div class="container">
  <h1>&#128276; Sentio Mind &middot; Alert Digest</h1>
  <p class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp; School: {SCHOOL}</p>

  <h2>School Summary</h2>
  <div class="summary-grid">
    <div class="stat-card">
      <div class="stat-val">{ss['total_persons_tracked']}</div>
      <div class="stat-lbl">Students tracked</div>
    </div>
    <div class="stat-card">
      <div class="stat-val" style="color:#ef4444">{ss['persons_flagged_today']}</div>
      <div class="stat-lbl">Flagged today</div>
    </div>
    <div class="stat-card">
      <div class="stat-val" style="color:#f59e0b">{ss['persons_flagged_yesterday']}</div>
      <div class="stat-lbl">Flagged yesterday</div>
    </div>
    <div class="stat-card">
      <div class="stat-val" style="font-size:14px;padding-top:6px">{ss['most_common_anomaly_this_week']}</div>
      <div class="stat-lbl">Top anomaly this week</div>
    </div>
    <div class="stat-card">
      <div class="stat-val">{ss['school_avg_wellbeing_today']}</div>
      <div class="stat-lbl">Avg wellbeing today</div>
    </div>
  </div>

  <h2>Today's Alerts ({len(today_alerts)})</h2>
  {alert_cards_html}

  <h2>Absence Flags ({len(absence_flags)})</h2>
  {absence_html}

  <h2>Persistent Alerts &mdash; Flagged 3+ Consecutive Days</h2>
  {persistent_html}

  <footer>Sentio Mind &copy; 2026 &nbsp;|&nbsp; Automated digest &mdash; not a substitute for professional assessment</footer>
</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Report → {output_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    daily_data = load_daily_data(DATA_DIR)
    all_dates  = sorted(daily_data.keys())
    print(f"Loaded {len(daily_data)} days: {all_dates}")

    person_days      = defaultdict(dict)
    person_info      = {}
    class_wellbeings = defaultdict(list)

    for d, persons in daily_data.items():
        for pid, pdata in persons.items():
            person_days[pid][d] = pdata
            if pid not in person_info:
                person_info[pid] = pdata.get("person_info", {"name": pid, "profile_image_b64": ""})
            class_wellbeings[d].append((pid, pdata["wellbeing"]))

    all_alerts    = []
    absence_flags = []

    for pid, days in person_days.items():
        sorted_days   = dict(sorted(days.items()))
        person_alerts = analyse_person(pid, sorted_days, person_info.get(pid, {}), class_wellbeings)
        all_alerts.extend(person_alerts)

        # Absence check
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

    today_str     = all_dates[-1]
    yesterday_str = str((datetime.strptime(today_str, "%Y-%m-%d") - timedelta(days=1)).date())
    flagged_today     = len(set(a["person_id"] for a in all_alerts if a.get("date") == today_str))
    flagged_yesterday = len(set(a["person_id"] for a in all_alerts if a.get("date") == yesterday_str))
    cat_counter       = Counter(a.get("category") for a in all_alerts)
    top_category      = cat_counter.most_common(1)[0][0] if cat_counter else "none"
    today_wbs         = [wb for _, wb in class_wellbeings.get(today_str, [])]
    avg_wb_today      = round(float(np.mean(today_wbs))) if today_wbs else 0

    school_summary = {
        "total_persons_tracked":         len(person_days),
        "persons_flagged_today":         flagged_today,
        "persons_flagged_yesterday":     flagged_yesterday,
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
    print("=" * 52)
    print(f"  Total alerts : {feed['alert_summary']['total_alerts']}")
    print(f"  Urgent       : {feed['alert_summary']['urgent']}")
    print(f"  Monitor      : {feed['alert_summary']['monitor']}")
    print(f"  Absence flags: {len(absence_flags)}")
    print(f"  Report → {REPORT_OUT}")
    print(f"  JSON   → {FEED_OUT}")
    print("=" * 52)