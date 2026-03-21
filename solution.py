import json
import numpy as np
from pathlib import Path
from datetime import datetime, date
from collections import defaultdict, Counter

DATA_DIR   = Path("sample_data")
REPORT_OUT = Path("alert_digest.html")
FEED_OUT   = Path("alert_feed.json")
SCHOOL     = "Demo School"

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


def load_daily_data(folder: Path) -> dict:
    daily = {}
    for fp in sorted(folder.glob("*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            raw = json.load(f)
        day_date = raw.get("date", fp.stem)
        persons = raw.get("persons", {})
        daily[day_date] = persons
    return daily


def compute_baseline(history: list) -> dict:
    window = THRESHOLDS["baseline_window"]
    baseline_days = history[:window]
    if not baseline_days:
        return {
            "wellbeing_mean": 50.0,
            "wellbeing_std":  10.0,
            "trait_means":    {},
            "avg_gaze":       "forward",
        }
    wb_scores = [d["wellbeing"] for d in baseline_days]
    wellbeing_mean = float(np.mean(wb_scores))
    wellbeing_std  = float(np.std(wb_scores))
    all_trait_keys = set()
    for d in baseline_days:
        all_trait_keys.update(d.get("traits", {}).keys())
    trait_means = {}
    for trait in all_trait_keys:
        vals = [d.get("traits", {}).get(trait, 0) for d in baseline_days]
        trait_means[trait] = float(np.mean(vals))
    gazes = [d.get("gaze_direction", "forward") for d in baseline_days]
    gaze_counter = Counter(gazes)
    avg_gaze = gaze_counter.most_common(1)[0][0]
    return {
        "wellbeing_mean": wellbeing_mean,
        "wellbeing_std":  wellbeing_std,
        "trait_means":    trait_means,
        "avg_gaze":       avg_gaze,
    }


def detect_sudden_drop(today: dict, baseline: dict) -> dict | None:
    bm = baseline["wellbeing_mean"]
    bs = baseline["wellbeing_std"]
    tw = today["wellbeing"]
    if bs > THRESHOLDS["high_std_baseline"]:
        threshold = THRESHOLDS["sudden_drop_high_std_delta"]
    else:
        threshold = THRESHOLDS["sudden_drop_delta"]
    delta = bm - tw
    if delta >= threshold:
        severity = "urgent" if delta > 35 else "monitor"
        return {
            "category":    "SUDDEN_DROP",
            "severity":    severity,
            "title":       "Sudden wellbeing drop detected",
            "delta":       -delta,
            "baseline_wellbeing": round(bm),
            "today_wellbeing":    tw,
        }
    return None


def detect_sustained_low(history: list) -> dict | None:
    n = THRESHOLDS["sustained_low_days"]
    if len(history) < n:
        return None
    recent = history[-n:]
    if all(d["wellbeing"] < THRESHOLDS["sustained_low_score"] for d in recent):
        scores = [d["wellbeing"] for d in recent]
        return {
            "category":    "SUSTAINED_LOW",
            "severity":    "urgent",
            "title":       f"Sustained low wellbeing for {n}+ days",
            "delta":       0,
            "baseline_wellbeing": 0,
            "today_wellbeing":    recent[-1]["wellbeing"],
        }
    return None


def detect_social_withdrawal(today: dict, baseline: dict) -> dict | None:
    today_se   = today.get("traits", {}).get("social_engagement", 0)
    baseline_se = baseline.get("trait_means", {}).get("social_engagement", 0)
    gaze       = today.get("gaze_direction", "forward")
    drop = baseline_se - today_se
    if drop >= THRESHOLDS["social_withdrawal_delta"] and gaze in ("down", "side"):
        return {
            "category":    "SOCIAL_WITHDRAWAL",
            "severity":    "monitor",
            "title":       "Social withdrawal pattern detected",
            "delta":       -drop,
            "baseline_wellbeing": round(baseline["wellbeing_mean"]),
            "today_wellbeing":    today["wellbeing"],
        }
    return None


def detect_hyperactivity_spike(today: dict, baseline: dict) -> dict | None:
    today_pe = today.get("traits", {}).get("physical_energy", 0)
    today_me = today.get("traits", {}).get("movement_energy", 0)
    base_pe  = baseline.get("trait_means", {}).get("physical_energy", 0)
    base_me  = baseline.get("trait_means", {}).get("movement_energy", 0)
    today_combined    = today_pe + today_me
    baseline_combined = base_pe + base_me
    spike = today_combined - baseline_combined
    if spike >= THRESHOLDS["hyperactivity_delta"]:
        return {
            "category":    "HYPERACTIVITY_SPIKE",
            "severity":    "monitor",
            "title":       "Hyperactivity spike detected",
            "delta":       spike,
            "baseline_wellbeing": round(baseline["wellbeing_mean"]),
            "today_wellbeing":    today["wellbeing"],
        }
    return None


def detect_regression(history: list) -> dict | None:
    n = THRESHOLDS["regression_recover_days"]
    if len(history) < n + 1:
        return None
    today = history[-1]
    recovery_window = history[-(n+1):-1]
    improving = True
    for i in range(1, len(recovery_window)):
        if recovery_window[i]["wellbeing"] <= recovery_window[i-1]["wellbeing"]:
            improving = False
            break
    if improving:
        last_recovery_score = recovery_window[-1]["wellbeing"]
        drop = last_recovery_score - today["wellbeing"]
        if drop > THRESHOLDS["regression_drop"]:
            return {
                "category":    "REGRESSION",
                "severity":    "monitor",
                "title":       "Regression after recovery detected",
                "delta":       -drop,
                "baseline_wellbeing": last_recovery_score,
                "today_wellbeing":    today["wellbeing"],
            }
    return None


def detect_gaze_avoidance(history: list) -> dict | None:
    n = THRESHOLDS["gaze_avoidance_days"]
    if len(history) < n:
        return None
    recent = history[-n:]
    if all(not d.get("eye_contact", False) for d in recent):
        return {
            "category":    "GAZE_AVOIDANCE",
            "severity":    "monitor",
            "title":       f"No eye contact for {n}+ consecutive days",
            "delta":       0,
            "baseline_wellbeing": 0,
            "today_wellbeing":    recent[-1]["wellbeing"],
        }
    return None


def analyse_person(person_id: str, sorted_days: dict, info: dict) -> list:
    alerts = []
    dates = sorted(sorted_days.keys())
    if not dates:
        return alerts
    history = [sorted_days[d] for d in dates]
    baseline = compute_baseline(history)
    today      = history[-1]
    today_date = dates[-1]
    person_name = info.get("name", person_id)
    profile_img = info.get("profile_image_b64", "")
    trend = [h["wellbeing"] for h in history[-5:]]
    traits_today = today.get("traits", {})
    if traits_today:
        lowest_trait = min(traits_today, key=traits_today.get)
        lowest_trait_val = traits_today[lowest_trait]
    else:
        lowest_trait = "none"
        lowest_trait_val = 0
    alert_counter = [0]

    def _make_alert(detection_result: dict) -> dict:
        alert_counter[0] += 1
        cat = detection_result["category"]
        sev = detection_result["severity"]
        bw  = detection_result.get("baseline_wellbeing", round(baseline["wellbeing_mean"]))
        tw  = detection_result.get("today_wellbeing", today["wellbeing"])
        delta = detection_result.get("delta", 0)
        desc_line = (
            f"{person_name}'s wellbeing is {tw} today "
            f"(baseline {bw}). "
            f"Lowest trait: {lowest_trait} at {lowest_trait_val}. "
            f"Gaze: {today.get('gaze_direction', 'unknown')}."
        )
        recommended = {
            "urgent":        "Schedule pastoral check-in today",
            "monitor":       "Keep monitoring -- follow up if pattern persists",
            "informational": "No immediate action required",
        }
        return {
            "alert_id":               f"ALT_{person_id}_{alert_counter[0]:03d}",
            "person_id":              person_id,
            "person_name":            person_name,
            "date":                   today_date,
            "severity":               sev,
            "category":               cat,
            "title":                  detection_result["title"],
            "description":            desc_line,
            "baseline_wellbeing":     bw,
            "today_wellbeing":        tw,
            "delta":                  delta,
            "days_flagged_consecutively": 1,
            "trend_last_5_days":      trend,
            "lowest_trait":           lowest_trait,
            "lowest_trait_value":     lowest_trait_val,
            "recommended_action":     recommended.get(sev, "Monitor"),
            "profile_image_b64":      profile_img,
        }

    r = detect_sudden_drop(today, baseline)
    if r:
        alerts.append(_make_alert(r))
    r = detect_sustained_low(history)
    if r:
        a = _make_alert(r)
        n = THRESHOLDS["sustained_low_days"]
        recent_scores = [h["wellbeing"] for h in history[-n:]]
        a["baseline_wellbeing"] = round(baseline["wellbeing_mean"])
        alerts.append(a)
    r = detect_social_withdrawal(today, baseline)
    if r:
        alerts.append(_make_alert(r))
    r = detect_hyperactivity_spike(today, baseline)
    if r:
        alerts.append(_make_alert(r))
    r = detect_regression(history)
    if r:
        alerts.append(_make_alert(r))
    r = detect_gaze_avoidance(history)
    if r:
        alerts.append(_make_alert(r))
    return alerts


def generate_alert_digest(alerts: list, absence_flags: list,
                           school_summary: dict, output_path: Path):

    def sparkline_svg(values, width=120, height=30):
        if not values or len(values) < 2:
            return ""
        mn, mx = min(values), max(values)
        rng = mx - mn if mx != mn else 1
        points = []
        step = width / (len(values) - 1)
        for i, v in enumerate(values):
            x = round(i * step, 1)
            y = round(height - ((v - mn) / rng) * (height - 4) - 2, 1)
            points.append(f"{x},{y}")
        polyline = " ".join(points)
        color = "#b5271d" if values[-1] < values[0] else "#4a7c3f"
        return (
            f'<svg width="{width}" height="{height}" style="vertical-align:middle;">'
            f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2" />'
            f'</svg>'
        )

    severity_colors = {
        "urgent":        ("#b5271d", "#fff"),
        "monitor":       ("#c8860a", "#fff"),
        "informational": ("#5b7a8a", "#fff"),
    }

    alert_cards_html = ""
    for a in alerts:
        sev = a.get("severity", "informational")
        bg, fg = severity_colors.get(sev, ("#5b7a8a", "#fff"))
        trend = a.get("trend_last_5_days", [])
        spark = sparkline_svg(trend)
        trend_text = " -> ".join(str(v) for v in trend)
        alert_cards_html += f"""
        <div class="alert-card" style="border-left: 5px solid {bg};">
            <div class="card-header">
                <span class="person-name">{a.get('person_name', 'Unknown')}</span>
                <span class="badge" style="background:{bg};color:{fg};">{sev.upper()}</span>
                <span class="category-tag">{a.get('category', '')}</span>
            </div>
            <p class="description">{a.get('description', '')}</p>
            <div class="card-details">
                <div class="detail-item">
                    <span class="detail-label">Baseline</span>
                    <span class="detail-value">{a.get('baseline_wellbeing', 0)}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Today</span>
                    <span class="detail-value">{a.get('today_wellbeing', 0)}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Delta</span>
                    <span class="detail-value">{a.get('delta', 0)}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Lowest Trait</span>
                    <span class="detail-value">{a.get('lowest_trait', '')} ({a.get('lowest_trait_value', 0)})</span>
                </div>
            </div>
            <div class="sparkline-row">
                <span class="detail-label">5-Day Trend:</span> {spark}
                <span class="trend-nums">{trend_text}</span>
            </div>
            <p class="action">⚡ {a.get('recommended_action', '')}</p>
        </div>
        """

    absence_html = ""
    for af in absence_flags:
        absence_html += f"""
        <div class="alert-card" style="border-left: 5px solid #b5271d;">
            <div class="card-header">
                <span class="person-name">{af.get('person_name', 'Unknown')}</span>
                <span class="badge" style="background:#b5271d;color:#fff;">ABSENT</span>
            </div>
            <p class="description">
                Not detected for <strong>{af.get('days_absent', 0)}</strong> day(s).
                Last seen: {af.get('last_seen_date', 'unknown')}.
            </p>
            <p class="action">⚡ {af.get('recommended_action', '')}</p>
        </div>
        """

    persistent_html = ""
    persistent_persons = [a for a in alerts if a.get("days_flagged_consecutively", 0) >= 3]
    if persistent_persons:
        for a in persistent_persons:
            persistent_html += f"""
            <div class="persistent-item">
                <strong>{a.get('person_name', '')}</strong> — {a.get('category', '')}
                for {a.get('days_flagged_consecutively', 0)} consecutive days
            </div>
            """
    else:
        persistent_html = "<p class='no-data'>No persons flagged for 3+ consecutive days today.</p>"

    ss = school_summary
    urgent_count = sum(1 for a in alerts if a.get("severity") == "urgent")
    monitor_count = sum(1 for a in alerts if a.get("severity") == "monitor")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sentio Mind — Alert Digest</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Georgia', 'Segoe UI', serif;
    background: #f5f0e6;
    color: #3d2e1f;
    padding: 20px;
    line-height: 1.6;
  }}
  .container {{ max-width: 960px; margin: 0 auto; }}
  h1 {{
    text-align: center;
    color: #8b1a1a;
    font-size: 2em;
    margin-bottom: 5px;
  }}
  .subtitle {{
    text-align: center;
    color: #7a6b5d;
    margin-bottom: 30px;
    font-size: 0.95em;
  }}
  h2 {{
    color: #8b1a1a;
    font-size: 1.3em;
    margin: 30px 0 15px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid #d4c5b0;
  }}
  .summary-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 15px;
    margin-bottom: 25px;
  }}
  .summary-card {{
    background: #fefcf7;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    border: 1px solid #e0d5c5;
    overflow: hidden;
    word-break: break-word;
    box-shadow: 0 2px 6px rgba(61,46,31,0.08);
  }}
  .summary-card .number {{
    font-size: 2.2em;
    font-weight: 700;
    display: block;
    margin-bottom: 5px;
  }}
  .summary-card .label {{ color: #7a6b5d; font-size: 0.85em; }}
  .number.urgent {{ color: #b5271d; }}
  .number.monitor {{ color: #c8860a; }}
  .number.tracked {{ color: #5b7a8a; }}
  .number.anomaly {{ color: #8b1a1a; font-size: 1.1em; }}
  .alert-card {{
    background: #fefcf7;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 15px;
    border: 1px solid #e0d5c5;
    transition: transform 0.15s;
    box-shadow: 0 2px 6px rgba(61,46,31,0.06);
  }}
  .alert-card:hover {{ transform: translateX(4px); }}
  .card-header {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
    flex-wrap: wrap;
  }}
  .person-name {{
    font-size: 1.15em;
    font-weight: 600;
    color: #2d1f12;
  }}
  .badge {{
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72em;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }}
  .category-tag {{
    color: #7a6b5d;
    font-size: 0.82em;
    font-family: monospace;
    background: #efe8db;
    padding: 2px 8px;
    border-radius: 4px;
  }}
  .description {{
    color: #5a4a3a;
    font-size: 0.92em;
    margin-bottom: 12px;
  }}
  .card-details {{
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    margin-bottom: 10px;
  }}
  .detail-item {{ display: flex; flex-direction: column; }}
  .detail-label {{
    font-size: 0.75em;
    color: #9a8b7a;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .detail-value {{
    font-size: 1.05em;
    font-weight: 600;
    color: #3d2e1f;
  }}
  .sparkline-row {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
  }}
  .trend-nums {{
    font-size: 0.8em;
    color: #9a8b7a;
    font-family: monospace;
  }}
  .action {{
    color: #b5271d;
    font-size: 0.88em;
    font-weight: 500;
  }}
  .persistent-item {{
    background: #fef8f0;
    padding: 12px 16px;
    border-radius: 8px;
    margin-bottom: 8px;
    border-left: 3px solid #b5271d;
  }}
  .no-data {{ color: #9a8b7a; font-style: italic; }}
  .footer {{
    text-align: center;
    color: #9a8b7a;
    margin-top: 40px;
    font-size: 0.82em;
    padding-top: 20px;
    border-top: 1px solid #d4c5b0;
  }}
</style>
</head>
<body>
<div class="container">
  <h1>🧠 Sentio Mind — Alert Digest</h1>
  <p class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | School: {SCHOOL}</p>

  <h2>📊 School Summary</h2>
  <div class="summary-grid">
    <div class="summary-card">
      <span class="number tracked">{ss.get('total_persons_tracked', 0)}</span>
      <span class="label">Persons Tracked</span>
    </div>
    <div class="summary-card">
      <span class="number urgent">{urgent_count}</span>
      <span class="label">Urgent Alerts</span>
    </div>
    <div class="summary-card">
      <span class="number monitor">{monitor_count}</span>
      <span class="label">Monitor Alerts</span>
    </div>
    <div class="summary-card">
      <span class="number anomaly">{ss.get('most_common_anomaly_this_week', 'N/A')}</span>
      <span class="label">Top Anomaly This Week</span>
    </div>
  </div>

  <h2>🚨 Today's Alerts (sorted by severity)</h2>
  {alert_cards_html if alert_cards_html else '<p class="no-data">No alerts today.</p>'}

  <h2>🚫 Absence Flags</h2>
  {absence_html if absence_html else '<p class="no-data">No absence flags.</p>'}

  <h2>⚠️ Persistent Alerts (3+ consecutive days)</h2>
  {persistent_html}

  <div class="footer">
    Sentio Mind · Behavioral Anomaly &amp; Early Distress Detection · 2026
  </div>
</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    daily_data = load_daily_data(DATA_DIR)
    all_dates  = sorted(daily_data.keys())
    print(f"Loaded {len(daily_data)} days: {all_dates}")

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

        present   = set(days.keys())
        absent    = 0
        for d in reversed(all_dates):
            if d not in present:
                absent += 1
            else:
                break
        if absent >= THRESHOLDS["absence_days"]:
            last_seen = sorted(present)[-1] if present else "unknown"
            absence_flags.append({
                "person_id":        pid,
                "person_name":      person_info.get(pid, {}).get("name", pid),
                "last_seen_date":   last_seen,
                "days_absent":      absent,
                "recommended_action": "Welfare check -- contact family if absent again tomorrow",
            })

    sev_order = {"urgent": 0, "monitor": 1, "informational": 2}
    all_alerts.sort(key=lambda a: sev_order.get(a.get("severity", "informational"), 3))

    latest_date = all_dates[-1] if all_dates else str(date.today())
    flagged_today = len(set(a.get("person_id") for a in all_alerts if a.get("date") == latest_date))
    cat_counter   = Counter(a.get("category") for a in all_alerts)
    top_category  = cat_counter.most_common(1)[0][0] if cat_counter else "none"

    last_date = all_dates[-1] if all_dates else None
    school_avg = 0
    if last_date and last_date in daily_data:
        wbs = [p.get("wellbeing", 0) for p in daily_data[last_date].values()]
        school_avg = round(float(np.mean(wbs)), 1) if wbs else 0

    school_summary = {
        "total_persons_tracked":       len(person_days),
        "persons_flagged_today":       flagged_today,
        "persons_flagged_yesterday":   0,
        "most_common_anomaly_this_week": top_category,
        "school_avg_wellbeing_today":  school_avg,
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
    print(f"  Report -> {REPORT_OUT}")
    print(f"  JSON   -> {FEED_OUT}")
    print("=" * 50)
