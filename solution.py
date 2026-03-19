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
        date = fp.stem.split('_')[1]
        try:
            with open(fp, 'r', encoding='utf8') as f:
                content = json.load(f)
                daily[date] = content
            
        except (IndexError, json.JSONDecodeError) as e:
            print(f"Skipping {fp.name}: {e}")
            continue
        pass
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
    # TODO
    n = THRESHOLDS["baseline_window"]

    wellbeing_list = []
    gaze = []
    baseline_days = history[:min(n, len(history))]
    
    for i in range(min(n, len(history))):

        day = history[i]

        wellbeing_list.append(day.get("wellbeing", 0))
        
    trait_sums = defaultdict(list)

    for day in baseline_days:
        traits = day.get("traits", {})

        for key, value in traits.items():
            if isinstance(value, (int, float)):
                trait_sums[key].append(value)
    gazes = [
        day.get("gaze_direction", "forward")
        for day in baseline_days
    ]

    trait_means = {
        key: float(np.mean(values))
        for key, values in trait_sums.items()
        if values
    }

    wellbeing_mean = float(np.mean(wellbeing_list))
    wellbeing_std  = float(np.std(wellbeing_list))

    gaze_dict = {}
    for item in gazes:
        gaze_dict[item] = gaze_dict.get(item, 0) + 1

    avg_gaze = Counter(gazes).most_common(1)[0][0] if gazes else "forward"

    return {
        "wellbeing_mean": wellbeing_mean,
        "wellbeing_std":  wellbeing_std,
        "trait_means":    trait_means,
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
    # TODO
    today_score=today.get("wellbeing",0)
    baseline_mean=baseline.get("wellbeing_mean",0)
    baseline_std=baseline.get("wellbeing_std",0)
    
    threshold=THRESHOLDS["sudden_drop_delta"]
    if baseline_std > THRESHOLDS["high_std_baseline"]:
        threshold=THRESHOLDS["sudden_drop_high_std_delta"]
    
    drop = baseline_mean - today_score
    
    if drop >= threshold:
        if drop > 35:
            severity="urgent"
        else:
            severity="monitor"
            
        return {
            "category": "SUDDEN_DROP",
            "severity": severity,
            "delta": float(drop),
            "description": f"Wellbeing dropped by {drop:.1f} points from baseline"
        }
    
    return None


def detect_sustained_low(history: list) -> dict | None:
    """
    Check the last sustained_low_days entries in history.
    If all have wellbeing < sustained_low_score → alert.
    Severity: urgent.
    TODO: implement
    """
    # TODO
    required_days = THRESHOLDS["sustained_low_days"]
    threshold = THRESHOLDS["sustained_low_score"]
 
    if len(history) < required_days:
        return None

    recent_days = history[-required_days:]
 
    for day in recent_days:
        if day.get("wellbeing", 100) >= threshold:
            return None
   
    return {
        "category": "SUSTAINED_LOW",
        "severity": "urgent",
        "description": f"Wellbeing below {threshold} for {required_days} consecutive days"
    }


def detect_social_withdrawal(today: dict, baseline: dict) -> dict | None:
    """
    social_engagement dropped >= social_withdrawal_delta AND
    today's gaze_direction is "down" or "side".
    Severity: monitor.
    TODO: implement
    """
    # TODO
    today_score=today.get("social_engagement",0)
    baseline_score = baseline.get("trait_means", {}).get("social_engagement", 0)
    gaze = today.get("gaze_direction","")
    
    drop=baseline_score - today_score
    if drop >= THRESHOLDS["social_withdrawal_delta"] and gaze in ["down","side"]:
        return {
            "category": "SOCIAL_WITHDRAWAL",
            "severity": "monitor",
            "description": f"Social engagement dropped by {int(drop)} points with downward/side gaze"
            }
    return None


def detect_hyperactivity_spike(today: dict, baseline: dict) -> dict | None:
    """
    (today.physical_energy + today.movement_energy) minus
    (baseline.physical_energy_mean + baseline.movement_energy_mean) >= hyperactivity_delta.
    Severity: monitor.
    TODO: implement
    """
    # TODO
    
    today_energy = (
    today.get("traits", {}).get("physical_energy", 0) +
    today.get("traits", {}).get("movement_energy", 0)
)

    baseline_energy = (
    baseline.get("trait_means", {}).get("physical_energy", 0) +
    baseline.get("trait_means", {}).get("movement_energy", 0)
)
    drop = today_energy - baseline_energy
    
    if drop >= THRESHOLDS["hyperactivity_delta"]:
        return {
            "category": "HYPERACTIVITY_SPIKE",
            "severity": "monitor",
            "description": f"Energy spike of {drop:.1f} points above baseline"
        }
    return None


def detect_regression(history: list) -> dict | None:
    """
    Find if the last regression_recover_days entries were all improving (each > previous),
    then today dropped > regression_drop.
    Severity: monitor.
    TODO: implement
    """
    # TODO
    recover_days=THRESHOLDS["regression_recover_days"]
    drop_threshold=THRESHOLDS["regression_drop"]
    
    if len(history) < recover_days + 1:
        return None
    
    recent_days=history[-(recover_days+1):]
    improve=True
    for i in range(recover_days-1):
        if recent_days[i]["wellbeing"] >= recent_days[i + 1]["wellbeing"]:
            improve = False
            break
    
    if not improve:
        return None

    last_good = recent_days[recover_days - 1]["wellbeing"]
    today = recent_days[recover_days]["wellbeing"]

    drop = last_good - today
    # print(last_good, today ,drop)
    if drop >= drop_threshold:
        return {
            "category": "REGRESSION",
            "severity": "monitor",
            "description": f"Wellbeing dropped by {drop} after improvement streak"
        }

    return None


def detect_gaze_avoidance(history: list) -> dict | None:
    """
    Last gaze_avoidance_days entries all have eye_contact == False (or missing).
    Severity: monitor.
    TODO: implement
    """
    # TODO
    n=THRESHOLDS["gaze_avoidance_days"]
    if len(history) < n:
        return None
    recent_days=history[-n:]
    
    for i in recent_days:
        if i["traits"].get("eye_contact", True):
            return None

    return {
        "category": "GAZE_AVOIDANCE",
        "severity": "monitor",
        "description": f"No eye contact detected for {n} consecutive days"
    }
    


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
    
    alert_counter=1
    alerts = []
    # TODO
    alerts = []

    person_name = info.get("name", person_id)
    profile_img = info.get("profile_image_b64", "")

    dates = list(sorted_days.keys())
    history = [sorted_days[d] for d in dates]

    baseline = compute_baseline(history)

    titles = {
        "SUDDEN_DROP": "Sudden wellbeing drop detected",
        "SUSTAINED_LOW": "Sustained low wellbeing",
        "SOCIAL_WITHDRAWAL": "Social withdrawal detected",
        "HYPERACTIVITY_SPIKE": "Hyperactivity spike detected",
        "REGRESSION": "Regression after improvement",
        "GAZE_AVOIDANCE": "Gaze avoidance pattern detected"
    }


    streak_tracker = defaultdict(int)

    for i, d in enumerate(dates):
        today = history[i]
        current_history = history[:i+1]

        current_categories = set()

        results = [
            detect_sudden_drop(today, baseline),
            detect_sustained_low(current_history),
            detect_social_withdrawal(today, baseline),
            detect_hyperactivity_spike(today, baseline),
            detect_regression(current_history),
            detect_gaze_avoidance(current_history),
        ]

        today_wellbeing = today.get("wellbeing", 0)
        baseline_wellbeing = baseline.get("wellbeing_mean", 0)
        delta = round(today_wellbeing - baseline_wellbeing, 2)

        trend = [d_.get("wellbeing", 0) for d_ in history[max(0, i-4):i+1]]

        traits = today.get("traits", {})

        numeric_traits = {
            k: v for k, v in traits.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }

        if numeric_traits:
            lowest_trait = min(numeric_traits, key=numeric_traits.get)
            lowest_value = numeric_traits[lowest_trait]
        else:
            lowest_trait = ""
            lowest_value = 0

        gaze = today.get("gaze_direction", "unknown")


        for res in results:
            if res is None:
                continue

            category = res.get("category")
            severity = res.get("severity", "monitor")

            current_categories.add(category)

            streak_tracker[category] += 1


            if severity == "urgent":
                action = "Schedule immediate counselling session"
            else:
                action = "Monitor and follow up if persists"

            alert = {
                "alert_id": f"{person_id}_{d}_{category}",
                "person_id": person_id,
                "person_name": person_name,
                "date": d,
                "severity": severity,
                "_note_severity": "one of: urgent / monitor / informational",
                "category": category,
                "_note_category": "one of: SUDDEN_DROP / SUSTAINED_LOW / SOCIAL_WITHDRAWAL / HYPERACTIVITY_SPIKE / REGRESSION / GAZE_AVOIDANCE / ABSENCE_FLAG",
                "title": titles.get(category, "Behavioral anomaly detected"),
                "description": (
                    f"{person_name}'s {res.get('description')}. "
                    f"Lowest trait: {lowest_trait} ({lowest_value}). "
                    f"Gaze direction: {gaze}."
                ),
                "baseline_wellbeing": round(baseline_wellbeing, 2),
                "today_wellbeing": today_wellbeing,
                "delta": delta,
                "days_flagged_consecutively": streak_tracker[category],
                "trend_last_5_days": trend,
                "lowest_trait": lowest_trait,
                "lowest_trait_value": lowest_value,
                "recommended_action": action,
                "profile_image_b64": profile_img
            }

            alerts.append(alert)

 
        for cat in list(streak_tracker.keys()):
            if cat not in current_categories:
                streak_tracker[cat] = 0

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
    # TODO
    def severity_color(sev):
        if sev == "urgent":
            return "#ff4d4d"
        elif sev == "monitor":
            return "#ffb84d"
        return "#cccccc"

    today_str = sorted([a.get("date") for a in alerts])[-1] if alerts else ""

    todays_alerts = [a for a in alerts if a.get("date") == today_str]

    sev_order = {"urgent": 0, "monitor": 1, "informational": 2}
    todays_alerts.sort(key=lambda a: sev_order.get(a.get("severity", ""), 3))

    person_alert_days = defaultdict(list)

    for a in alerts:
        person_alert_days[a["person_id"]].append(a["date"])

    persistent_people = []

    for pid, dates in person_alert_days.items():
        unique_dates = sorted(set(dates))
        if len(unique_dates) >= 3:
            persistent_people.append(pid)


    html = f"""
    <html>
    <head>
        <title>Alert Digest</title>
        <style>
             body {{
                font-family: Arial;
                padding: 20px;
                background: #f5f5f5;
            }}
            .card {{
                background: white;
                padding: 15px;
                margin-bottom: 10px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .badge {{
                padding: 4px 8px;
                border-radius: 5px;
                color: white;
                font-size: 12px;
                margin-left: 10px;
            }}
            .section {{
                margin-top: 30px;
            }}
        </style>
    </head>
    <body>

    <h1> Alert Digest</h1>

    <div class="section">
        <h2> Today's Alerts</h2>
    """

    if not todays_alerts:
        html += "<p>No alerts today.</p>"
    else:
        for a in todays_alerts:
            color = severity_color(a.get("severity"))
            html += f"""
            <div class="card">
                <b>{a.get("person_name")}</b>
                <span class="badge" style="background:{color}">
                    {a.get("severity").upper()}
                </span>
                <p><b>{a.get("category")}</b>: {a.get("description")}</p>
                <p> {a.get("date")}</p>
            </div>
            """

    html += f"""
    </div>

    <div class="section">
        <h2> School Summary</h2>
        <div class="card">
            <p>Total Persons: {school_summary.get("total_persons_tracked")}</p>
            <p>Flagged Today: {school_summary.get("persons_flagged_today")}</p>
            <p>Flagged Yesterday: {school_summary.get("persons_flagged_yesterday")}</p>
            <p>Most Common Anomaly: {school_summary.get("most_common_anomaly_this_week")}</p>
        </div>
    </div>
    """
   
    html += """
    <div class="section">
        <h2> Persistent Alerts (3+ days)</h2>
    """

    if not persistent_people:
        html += "<p>No persistent alerts.</p>"
    else:
        for pid in persistent_people:
            html += f"""
            <div class="card">
                <b>{pid}</b> flagged for multiple consecutive days
            </div>
            """


    html += """
    <div class="section">
        <h2> Absence Flags</h2>
    """

    if not absence_flags:
        html += "<p>No absence flags.</p>"
    else:
        for a in absence_flags:
            html += f"""
            <div class="card">
                <b>{a.get("person_name")}</b><br>
                Last Seen: {a.get("last_seen_date")} <br>
                Days Absent: {a.get("days_absent")} <br>
                Action: {a.get("recommended_action")}
            </div>
            """

    html += """
    </div>

    </body>
    </html>
    """

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    pass


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
                "recommended_action": "Welfare check — contact family if absent again tomorrow",
            })

    sev_order = {"urgent": 0, "monitor": 1, "informational": 2}
    all_alerts.sort(key=lambda a: sev_order.get(a.get("severity", "informational"), 3))

    today_str = all_dates[-1] if all_dates else ""
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
    print(f"  Report → {REPORT_OUT}")
    print(f"  JSON   → {FEED_OUT}")
    print("=" * 50)
