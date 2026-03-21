"""
anomaly_detection.py
Sentio Mind · Project 5 · Behavioral Anomaly & Early Distress Detection

Copy this file to solution.py and fill in every TODO block.
Do not rename any function. No OpenCV needed — pure data analysis.
Run: python solution.py
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
    Return: { "YYYY-MM-DD": { "PERSON_ID": { wellbeing, traits, gaze, name, ... }, ... }, ... }

    Each Sentio Mind daily file has the structure from the README.
    Parse it and flatten into the above format.
    If your dataset uses a different format, adapt the parsing here.

    TODO: implement
    """
    daily = {}
    for fp in sorted(folder.glob("*.json")):
        # TODO: load and parse each file
        with open(fp, "r") as f:
            data = json.load(f)
            
        date = data["date"]
        students = data["persons"]
        daily[date] = students
        # print(daily)
    return daily


# ---------------------------------------------------------------------------
# BASELINE
# ---------------------------------------------------------------------------

def compute_baseline(history: list) -> dict:
    """
    history: list of daily dicts (oldest first), each has at minimum:
      { wellbeing: int, traits: {}, gaze_direction: str }

    Use first THRESHOLDS['baseline_window'] days.
    Return:
      { wellbeing_mean, wellbeing_std, trait_means: {}, avg_gaze: str }

    TODO: implement
    """
    # Use first 3 days for baseline
    baseline_window = THRESHOLDS['baseline_window']
    if len(history) >= baseline_window:
        baseline_days = history[:baseline_window]
    else:
        baseline_days = history

    # If no data, return defaults
    if not baseline_days:
        return {
            "wellbeing_mean": 50.0,
            "wellbeing_std":  10.0,
            "trait_means":    {},
            "avg_gaze":       "forward",
        }

    # Get all wellbeing scores
    wellbeing_scores = []
    for day in baseline_days:
        wellbeing_scores.append(day['wellbeing'])

    # Calculate mean and std
    wellbeing_mean = np.mean(wellbeing_scores)
    wellbeing_std = np.std(wellbeing_scores)

    # Calculate trait means
    trait_means = {}
    if 'traits' in baseline_days[0]:
        trait_names = baseline_days[0]['traits'].keys()
        for trait_name in trait_names:
            trait_values = []
            for day in baseline_days:
                if 'traits' in day:
                    trait_values.append(day['traits'].get(trait_name, 0))
            if trait_values:
                trait_means[trait_name] = np.mean(trait_values)

    # Find most common gaze
    gaze_list = []
    for day in baseline_days:
        gaze_list.append(day.get('gaze_direction', 'forward'))
    gaze_counter = Counter(gaze_list)
    avg_gaze = gaze_counter.most_common(1)[0][0] if gaze_counter else "forward"

    return {
        "wellbeing_mean": float(wellbeing_mean),
        "wellbeing_std":  float(wellbeing_std),
        "trait_means":    {k: float(v) for k, v in trait_means.items()},
        "avg_gaze":       avg_gaze,
    }


# ---------------------------------------------------------------------------
# ANOMALY DETECTORS  — each returns an alert dict or None
# ---------------------------------------------------------------------------

def detect_sudden_drop(today: dict, baseline: dict) -> dict | None:
    """
    today['wellbeing'] vs baseline['wellbeing_mean'].
    If baseline_std > 15, raise threshold to sudden_drop_high_std_delta.
    Severity: delta > 35 = urgent, else monitor.
    TODO: implement
    """
    baseline_mean = baseline['wellbeing_mean']
    baseline_std = baseline['wellbeing_std']
    today_wellbeing = today['wellbeing']

    # Calculate the drop
    delta = baseline_mean - today_wellbeing

    # Choose threshold based on baseline std
    if baseline_std > THRESHOLDS['high_std_baseline']:
        threshold = THRESHOLDS['sudden_drop_high_std_delta']
    else:
        threshold = THRESHOLDS['sudden_drop_delta']

    # Check if drop is significant
    if delta >= threshold:
        # Determine severity
        if delta > 35:
            severity = "urgent"
        else:
            severity = "monitor"

        return {
            "category": "SUDDEN_DROP",
            "severity": severity,
            "baseline_wellbeing": baseline_mean,
            "today_wellbeing": today_wellbeing,
            "delta": -delta
        }

    return None


def detect_sustained_low(history: list) -> dict | None:
    """
    Check the last sustained_low_days entries in history.
    If all have wellbeing < sustained_low_score → alert.
    Severity: urgent.
    TODO: implement
    """
    days_needed = THRESHOLDS['sustained_low_days']
    low_threshold = THRESHOLDS['sustained_low_score']

    # Check if we have enough days
    if len(history) < days_needed:
        return None

    # Get last N days
    last_days = history[-days_needed:]

    # Check if all are below threshold
    all_low = True
    for day in last_days:
        if day['wellbeing'] >= low_threshold:
            all_low = False
            break

    if all_low:
        return {
            "category": "SUSTAINED_LOW",
            "severity": "urgent",
            "days_low": days_needed
        }

    return None


def detect_social_withdrawal(today: dict, baseline: dict) -> dict | None:
    """
    social_engagement dropped >= social_withdrawal_delta AND
    today's gaze_direction is "down" or "side".
    Severity: monitor.
    TODO: implement
    """
    # Get baseline social engagement
    if 'social_engagement' not in baseline.get('trait_means', {}):
        return None

    baseline_social = baseline['trait_means']['social_engagement']
    today_social = today.get('traits', {}).get('social_engagement', 0)
    gaze = today.get('gaze_direction', 'forward')

    # Calculate drop
    drop = baseline_social - today_social

    # Check conditions
    if drop >= THRESHOLDS['social_withdrawal_delta'] and (gaze == "down" or gaze == "side"):
        return {
            "category": "SOCIAL_WITHDRAWAL",
            "severity": "monitor",
            "social_engagement_drop": drop,
            "gaze_direction": gaze
        }

    return None


def detect_hyperactivity_spike(today: dict, baseline: dict) -> dict | None:
    """
    (today.physical_energy + today.movement_energy) minus
    (baseline.physical_energy_mean + baseline.movement_energy_mean) >= hyperactivity_delta.
    Severity: monitor.
    TODO: implement
    """
    # Check if traits exist
    trait_means = baseline.get('trait_means', {})
    if 'physical_energy' not in trait_means or 'movement_energy' not in trait_means:
        return None

    # Get baseline values
    baseline_physical = trait_means['physical_energy']
    baseline_movement = trait_means['movement_energy']

    # Get today values
    today_traits = today.get('traits', {})
    today_physical = today_traits.get('physical_energy', 0)
    today_movement = today_traits.get('movement_energy', 0)

    # Calculate combined energy
    baseline_combined = baseline_physical + baseline_movement
    today_combined = today_physical + today_movement

    # Check spike
    spike = today_combined - baseline_combined

    if spike >= THRESHOLDS['hyperactivity_delta']:
        return {
            "category": "HYPERACTIVITY_SPIKE",
            "severity": "monitor",
            "energy_spike": spike
        }

    return None


def detect_regression(history: list) -> dict | None:
    """
    Find if the last regression_recover_days entries were all improving (each > previous),
    then today dropped > regression_drop.
    Severity: monitor.
    TODO: implement
    """
    recover_days = THRESHOLDS['regression_recover_days']
    drop_threshold = THRESHOLDS['regression_drop']

    # Need at least recover_days + 1 (for today)
    if len(history) < recover_days + 1:
        return None

    # Get the recovery period (before today)
    recovery_period = history[-(recover_days + 1):-1]
    today = history[-1]

    # Check if all days were improving
    was_improving = True
    for i in range(len(recovery_period) - 1):
        if recovery_period[i + 1]['wellbeing'] <= recovery_period[i]['wellbeing']:
            was_improving = False
            break

    # Check if today dropped
    last_recovery_day = recovery_period[-1]
    drop = last_recovery_day['wellbeing'] - today['wellbeing']

    if was_improving and drop > drop_threshold:
        return {
            "category": "REGRESSION",
            "severity": "monitor",
            "drop_amount": drop
        }

    return None


def detect_gaze_avoidance(history: list) -> dict | None:
    """
    Last gaze_avoidance_days entries all have eye_contact == False (or missing).
    Severity: monitor.
    TODO: implement
    """
    days_needed = THRESHOLDS['gaze_avoidance_days']

    # Check if we have enough days
    if len(history) < days_needed:
        return None

    # Get last N days
    last_days = history[-days_needed:]

    # Check if all have no eye contact
    all_no_contact = True
    for day in last_days:
        if day.get('eye_contact', False) == True:
            all_no_contact = False
            break

    if all_no_contact:
        return {
            "category": "GAZE_AVOIDANCE",
            "severity": "monitor",
            "days_no_contact": days_needed
        }

    return None


# ---------------------------------------------------------------------------
# ANALYSE ONE PERSON
# ---------------------------------------------------------------------------

def analyse_person(person_id: str, sorted_days: dict, info: dict) -> list:
    """
    sorted_days: { "YYYY-MM-DD": person_data_dict } — keys in date order
    info: { name, profile_image_b64, ... }

    Build history list, compute baseline, run all detectors.
    Return list of alert dicts. Each alert must include person_id, person_name, date,
    and all fields from anomaly_detection.json schema.

    TODO: implement
    """
    # Build history list
    history = []
    dates_list = []
    for date_str, person_data in sorted_days.items():
        history.append(person_data)
        dates_list.append(date_str)

    # If no data, return empty
    if not history:
        return []

    # Compute baseline
    baseline = compute_baseline(history)

    # Get today (last day)
    today = history[-1]
    today_date = dates_list[-1]

    # Get person name
    person_name = info.get('name', person_id)

    # Get last 5 days of wellbeing for trend
    trend_scores = [day['wellbeing'] for day in history[-5:]]

    alerts = []

    # Run all detectors
    alert = detect_sudden_drop(today, baseline)
    if alert:
        alert['person_id'] = person_id
        alert['person_name'] = person_name
        alert['date'] = today_date
        alert['trend_last_5_days'] = trend_scores
        alerts.append(alert)

    alert = detect_sustained_low(history)
    if alert:
        alert['person_id'] = person_id
        alert['person_name'] = person_name
        alert['date'] = today_date
        alert['trend_last_5_days'] = trend_scores
        alerts.append(alert)

    alert = detect_social_withdrawal(today, baseline)
    if alert:
        alert['person_id'] = person_id
        alert['person_name'] = person_name
        alert['date'] = today_date
        alert['trend_last_5_days'] = trend_scores
        alerts.append(alert)

    alert = detect_hyperactivity_spike(today, baseline)
    if alert:
        alert['person_id'] = person_id
        alert['person_name'] = person_name
        alert['date'] = today_date
        alert['trend_last_5_days'] = trend_scores
        alerts.append(alert)

    alert = detect_regression(history)
    if alert:
        alert['person_id'] = person_id
        alert['person_name'] = person_name
        alert['date'] = today_date
        alert['trend_last_5_days'] = trend_scores
        alerts.append(alert)

    alert = detect_gaze_avoidance(history)
    if alert:
        alert['person_id'] = person_id
        alert['person_name'] = person_name
        alert['date'] = today_date
        alert['trend_last_5_days'] = trend_scores
        alerts.append(alert)

    return alerts


# ---------------------------------------------------------------------------
# HTML REPORT
# ---------------------------------------------------------------------------

def generate_alert_digest(alerts: list, absence_flags: list,
                           school_summary: dict, output_path: Path):
    """
    Self-contained HTML — no CDN, inline CSS only.

    Section 1: Today's alerts sorted by severity.
      Each card: person name, badge (urgent=red, monitor=amber), description,
      5-day sparkline (inline SVG or just coloured squares — keep it simple).

    Section 2: School summary numbers.

    Section 3: Persons flagged 3+ consecutive days.

    TODO: implement
    """
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Sentio Mind - Alert Digest</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            margin-top: 30px;
        }
        .alert-card {
            border: 1px solid #ddd;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            background: #fafafa;
        }
        .badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 3px;
            color: white;
            font-weight: bold;
            font-size: 12px;
        }
        .badge-urgent {
            background-color: #f44336;
        }
        .badge-monitor {
            background-color: #ff9800;
        }
        .summary-box {
            background: #e3f2fd;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .summary-item {
            display: inline-block;
            margin-right: 30px;
        }
        .absence-card {
            border-left: 4px solid #f44336;
            background: #ffebee;
            padding: 10px;
            margin: 10px 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        th, td {
            text-align: left;
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f0f0f0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Sentio Mind - Student Wellbeing Alert Digest</h1>
        <p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>

        <h2>School Summary</h2>
        <div class="summary-box">
            <div class="summary-item"><strong>Total Students Tracked:</strong> """ + str(school_summary.get('total_persons_tracked', 0)) + """</div>
            <div class="summary-item"><strong>Alerts Today:</strong> """ + str(len(alerts)) + """</div>
            <div class="summary-item"><strong>Urgent Alerts:</strong> """ + str(sum(1 for a in alerts if a.get('severity') == 'urgent')) + """</div>
            <div class="summary-item"><strong>Most Common Issue:</strong> """ + school_summary.get('most_common_anomaly_this_week', 'None') + """</div>
        </div>

        <h2>Today's Alerts</h2>
"""

    if not alerts:
        html += "<p>No alerts detected today. All students appear to be doing well!</p>"
    else:
        for alert in alerts:
            severity = alert.get('severity', 'monitor')
            category = alert.get('category', 'UNKNOWN')
            person_name = alert.get('person_name', 'Unknown')

            badge_class = 'badge-urgent' if severity == 'urgent' else 'badge-monitor'

            html += f"""
        <div class="alert-card">
            <h3>{person_name} <span class="badge {badge_class}">{severity.upper()}</span></h3>
            <p><strong>Type:</strong> {category}</p>
"""

            if category == "SUDDEN_DROP":
                delta_value = abs(alert.get('delta', 0))
                html += f"<p>Wellbeing dropped from {alert.get('baseline_wellbeing', 'N/A'):.1f} to {alert.get('today_wellbeing', 'N/A')} (change: {delta_value:.1f} points)</p>"
            elif category == "SUSTAINED_LOW":
                html += f"<p>Wellbeing has been low for {alert.get('days_low', 'N/A')} consecutive days</p>"
            elif category == "SOCIAL_WITHDRAWAL":
                html += f"<p>Social engagement dropped by {alert.get('social_engagement_drop', 'N/A'):.1f} points, gaze: {alert.get('gaze_direction', 'N/A')}</p>"
            elif category == "HYPERACTIVITY_SPIKE":
                html += f"<p>Energy levels spiked by {alert.get('energy_spike', 'N/A'):.1f} points above baseline</p>"
            elif category == "REGRESSION":
                html += f"<p>After improving, wellbeing dropped by {alert.get('drop_amount', 'N/A'):.1f} points</p>"
            elif category == "GAZE_AVOIDANCE":
                html += f"<p>No eye contact detected for {alert.get('days_no_contact', 'N/A')} consecutive days</p>"

            # Add 5-day sparkline
            trend = alert.get('trend_last_5_days', [])
            if trend:
                html += "<p><strong>5-Day Trend:</strong> "
                for score in trend:
                    if score >= 70:
                        color = "green"
                    elif score >= 45:
                        color = "orange"
                    else:
                        color = "red"
                    html += f'<span style="display:inline-block;width:30px;height:20px;background-color:{color};margin:2px;border-radius:3px;text-align:center;color:white;font-size:11px;line-height:20px;">{score}</span>'
                html += "</p>"

            html += """
        </div>
"""

    html += """
        <h2>Absence Flags</h2>
"""

    if not absence_flags:
        html += "<p>All students have been present recently.</p>"
    else:
        for absence in absence_flags:
            html += f"""
        <div class="absence-card">
            <strong>{absence.get('person_name', 'Unknown')}</strong> - Last seen: {absence.get('last_seen_date', 'Unknown')}<br>
            Days absent: {absence.get('days_absent', 0)}<br>
            <em>{absence.get('recommended_action', 'Follow up required')}</em>
        </div>
"""

    html += """
    </div>
</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
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
                "recommended_action": "Welfare check - contact family if absent again tomorrow",
            })

    # Add alert IDs
    for i, alert in enumerate(all_alerts, 1):
        alert['alert_id'] = f"ALT_{i:03d}"

    sev_order = {"urgent": 0, "monitor": 1, "informational": 2}
    all_alerts.sort(key=lambda a: sev_order.get(a.get("severity", "informational"), 3))

    today_str = str(date.today())
    flagged_today = sum(1 for a in all_alerts if a.get("date") == today_str)
    cat_counter   = Counter(a.get("category") for a in all_alerts)
    top_category  = cat_counter.most_common(1)[0][0] if cat_counter else "none"

    school_summary = {
        "total_persons_tracked":       len(person_days),
        "persons_flagged_today":       flagged_today,
        "persons_flagged_yesterday":   0,   # extend if you have yesterday's run stored
        "most_common_anomaly_this_week": top_category,
        "school_avg_wellbeing_today":  0,   # compute from daily_data[today_str] if available
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
