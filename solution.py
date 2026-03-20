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
    from collections import defaultdict
    daily = defaultdict(dict)
    
    for fp in sorted(folder.glob("*.json")):
        try:
            with open(fp, "r") as f:
                data = json.load(f)
                for record in data:
                    date_str = record.get("date")
                    pid = record.get("person_id")
                    if date_str and pid and record.get("seen_in_video", True):
                        daily[date_str][pid] = record
        except Exception as e:
            print(f"Error reading {fp}: {e}")
            
    return dict(daily)

# ---------------------------------------------------------------------------
# BASELINE
# ---------------------------------------------------------------------------

def compute_baseline(history: list) -> dict:
    if not history:
        return {"wellbeing_mean": 0.0, "wellbeing_std": 0.0, "trait_means": {}, "avg_gaze": "forward"}

    # Use first 3 days of data (or all if < 3)
    window = THRESHOLDS['baseline_window']
    base_records = history[:window]
    
    wellbeings = [r.get('wellbeing', 0) for r in base_records]
    
    trait_keys = ['social_engagement', 'physical_energy', 'movement_energy']
    trait_means = {}
    for trait in trait_keys:
        vals = [r.get(trait, 0) for r in base_records if trait in r]
        trait_means[trait] = float(np.mean(vals)) if vals else 0.0
        
    gazes = [r.get('gazed', 'forward') for r in base_records]
    common_gaze = Counter(gazes).most_common(1)[0][0] if gazes else "forward"

    return {
        "wellbeing_mean": float(np.mean(wellbeings)) if wellbeings else 0.0,
        "wellbeing_std": float(np.std(wellbeings)) if wellbeings else 0.0,
        "trait_means": trait_means,
        "avg_gaze": common_gaze,
    }

# ---------------------------------------------------------------------------
# ANOMALY DETECTORS  — each returns an alert dict or None
# ---------------------------------------------------------------------------

def detect_sudden_drop(today: dict, baseline: dict) -> dict | None:
    current_wb = today.get('wellbeing', 0)
    base_wb = baseline.get('wellbeing_mean', 0)
    drop = base_wb - current_wb
    
    # Check for high variance in baseline
    threshold = THRESHOLDS['sudden_drop_delta']
    if baseline.get('wellbeing_std', 0) > THRESHOLDS['high_std_baseline']:
        threshold = THRESHOLDS['sudden_drop_high_std_delta']
        
    if drop >= threshold:
        severity = "urgent" if drop > 35 else "monitor"
        return {
            "category": "SUDDEN_DROP",
            "severity": severity,
            "title": "Sudden wellbeing drop",
            "description": f"Dropped {drop:.1f} pts from baseline {base_wb:.1f}.",
            "delta": -drop
        }
    return None

def detect_sustained_low(history: list) -> dict | None:
    days_req = THRESHOLDS['sustained_low_days']
    if len(history) < days_req:
        return None
        
    recent = history[-days_req:]
    if all(r.get('wellbeing', 100) < THRESHOLDS['sustained_low_score'] for r in recent):
        return {
            "category": "SUSTAINED_LOW",
            "severity": "urgent",
            "title": "Sustained low wellbeing",
            "description": f"Wellbeing below {THRESHOLDS['sustained_low_score']} for {days_req}+ days.",
        }
    return None

def detect_social_withdrawal(today: dict, baseline: dict) -> dict | None:
    base_social = baseline.get('trait_means', {}).get('social_engagement', 0)
    current_social = today.get('social_engagement', 0)
    drop = base_social - current_social
    gaze = today.get('gaze', '').lower()
    
    if drop >= THRESHOLDS['social_withdrawal_delta'] and gaze in ['down', 'side']:
        return {
            "category": "SOCIAL_WITHDRAWAL",
            "severity": "monitor",
            "title": "Social withdrawal",
            "description": f"Social engagement dropped {drop:.1f} pts. Gaze is {gaze}.",
        }
    return None

def detect_hyperactivity_spike(today: dict, baseline: dict) -> dict | None:
    current_energy = today.get('physical_energy', 0) + today.get('movement_energy', 0)
    base_energy = baseline.get('trait_means', {}).get('physical_energy', 0) + baseline.get('trait_means', {}).get('movement_energy', 0)
    
    if (current_energy - base_energy) >= THRESHOLDS['hyperactivity_delta']:
        return {
            "category": "HYPERACTIVITY_SPIKE",
            "severity": "monitor",
            "title": "Hyperactivity spike",
            "description": f"Combined energy is {(current_energy - base_energy):.1f} pts above baseline.",
        }
    return None

def detect_regression(history: list) -> dict | None:
    days_req = THRESHOLDS['regression_recover_days']
    if len(history) < days_req + 1:
        return None
        
    recovery_period = history[-(days_req+1):-1]
    is_recovering = all(recovery_period[i].get('wellbeing', 0) < recovery_period[i+1].get('wellbeing', 0) for i in range(len(recovery_period)-1))
    
    if is_recovering:
        peak_wb = recovery_period[-1].get('wellbeing', 0)
        today_wb = history[-1].get('wellbeing', 0)
        drop = peak_wb - today_wb
        
        if drop > THRESHOLDS['regression_drop']:
            return {
                "category": "REGRESSION",
                "severity": "monitor",
                "title": "Progress regression",
                "description": f"Wellbeing dropped {drop:.1f} pts after {days_req} days of recovery.",
            }
    return None

def detect_gaze_avoidance(history: list) -> dict | None:
    days_req = THRESHOLDS['gaze_avoidance_days']
    if len(history) < days_req:
        return None
        
    recent = history[-days_req:]
    
    # Check if 'eye_contact' is explicitly False or missing (which defaults to False)
    if all(r.get('eye_contact', False) == False for r in recent):
        return {
            "category": "GAZE_AVOIDANCE",
            "severity": "monitor",
            "title": "Gaze avoidance",
            "description": f"No direct eye contact detected for {days_req}+ consecutive days.",
        }
    return None


# ---------------------------------------------------------------------------
# ANALYSE ONE PERSON
# ---------------------------------------------------------------------------
def analyse_person(person_id: str, sorted_days: dict, info: dict) -> list:
    history = list(sorted_days.values())
    if not history:
        return []
        
    baseline = compute_baseline(history)
    today = history[-1]
    today_date = today.get('date', '')
    consecutive_days = 0
    for i in range(len(history), 0, -1):
        sub_history = history[:i]
        sub_baseline = compute_baseline(sub_history)
        sub_today = sub_history[-1]
        
        fired = any([
            detect_sudden_drop(sub_today, sub_baseline),
            detect_sustained_low(sub_history),
            detect_social_withdrawal(sub_today, sub_baseline),
            detect_hyperactivity_spike(sub_today, sub_baseline),
            detect_regression(sub_history),
            detect_gaze_avoidance(sub_history)
        ])
        
        if fired:
            consecutive_days += 1
        else:
            break 
            
    consecutive_days = max(1, consecutive_days) 
    detectors = [
        detect_sudden_drop(today, baseline),
        detect_sustained_low(history),
        detect_social_withdrawal(today, baseline),
        detect_hyperactivity_spike(today, baseline),
        detect_regression(history),
        detect_gaze_avoidance(history)
    ]
    
    alerts = []
    trend = [r.get('wellbeing', 0) for r in history[-5:]]
    
    for i, alert_data in enumerate(detectors):
        if alert_data:
            traits = {
                "social_engagement": today.get("social_engagement", 100),
                "physical_energy": today.get("physical_energy", 100),
                "movement_energy": today.get("movement_energy", 100)
            }
            lowest_t = min(traits, key=traits.get)
            lowest_v = traits[lowest_t]

            alert = {
                "alert_id": f"ALT_{person_id}_{today_date}_{i}",
                "person_id": person_id,
                "person_name": info.get("name", person_id),
                "date": today_date,
                "severity": alert_data["severity"],
                "category": alert_data["category"],
                "title": alert_data["title"],
                "description": alert_data["description"],
                "baseline_wellbeing": baseline.get("wellbeing_mean", 0),
                "today_wellbeing": today.get("wellbeing", 0),
                "delta": alert_data.get("delta", 0),
                "days_flagged_consecutively": consecutive_days, 
                "trend_last_5_days": trend,
                "lowest_trait": lowest_t,     
                "lowest_trait_value": lowest_v, 
                "recommended_action": "Schedule pastoral check-in today",
                "profile_image_b64": info.get("profile_image_b64", "")
            }
            alerts.append(alert)
            
    return alerts
# ---------------------------------------------------------------------------
# HTML REPORT
# ---------------------------------------------------------------------------

def generate_alert_digest(alerts: list, absence_flags: list, school_summary: dict, output_path: Path):
    # Sort alerts so 'urgent' comes before 'monitor'
    sorted_alerts = sorted(alerts, key=lambda x: 0 if x.get('severity') == 'urgent' else 1)
    
    # Filter for persons flagged 3+ consecutive days
    chronic_alerts = [a for a in sorted_alerts if a.get('days_flagged_consecutively', 0) >= 3]

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Sentio Mind Alert Digest</title>
        <style>
            body {{ font-family: system-ui, sans-serif; margin: 40px; background: #f9f9f9; color: #333; }}
            .card {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 15px; border-left: 5px solid #ccc; }}
            .urgent {{ border-left-color: #e74c3c; }}
            .monitor {{ border-left-color: #f39c12; }}
            .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; color: white; margin-left: 10px; }}
            .badge-urgent {{ background: #e74c3c; }}
            .badge-monitor {{ background: #f39c12; }}
            .sparkline {{ display: flex; align-items: flex-end; height: 40px; gap: 3px; margin-top: 10px; }}
            .bar {{ width: 15px; background: #3498db; border-radius: 2px 2px 0 0; }}
            .summary-box {{ display: flex; gap: 20px; margin-bottom: 30px; margin-top: 20px; }}
            .stat {{ background: white; padding: 20px; border-radius: 8px; flex: 1; text-align: center; border: 1px solid #ddd; }}
            .stat h3 {{ margin: 0; font-size: 24px; color: #2c3e50; }}
            .stat p {{ margin: 5px 0 0 0; color: #7f8c8d; font-size: 14px; }}
        </style>
    </head>
    <body>
        <h1>Daily Counsellor Alert Digest</h1>
        
        <h2>1. Today's Action Required</h2>
    """
    
    # Render all alerts
    for alert in sorted_alerts:
        sev = alert.get('severity', 'monitor')
        bars_html = "".join([f'<div class="bar" style="height: {val}%" title="{val}"></div>' for val in alert.get('trend_last_5_days', [])])
        
        html += f"""
        <div class="card {sev}">
            <h3 style="margin-top:0;">{alert['person_name']} <span class="badge badge-{sev}">{sev.upper()}</span></h3>
            <strong>{alert['title']}</strong>
            <p>{alert['description']}</p>
            <div class="sparkline">{bars_html}</div>
        </div>
        """
        
    for flag in absence_flags:
        html += f"""
        <div class="card urgent">
            <h3 style="margin-top:0;">{flag['person_name']} <span class="badge badge-urgent">URGENT</span></h3>
            <strong>ABSENCE FLAG</strong>
            <p>Not seen for {flag['days_absent']} consecutive days. {flag['recommended_action']}</p>
        </div>
        """

    # SECTION 2: School summary numbers
    html += f"""
        <h2>2. School Summary Numbers</h2>
        <div class="summary-box">
            <div class="stat"><h3>{school_summary.get('persons_flagged_today', 0)}</h3><p>Flagged Today</p></div>
            <div class="stat"><h3>{len(absence_flags)}</h3><p>Absence Flags</p></div>
            <div class="stat"><h3>{school_summary.get('most_common_anomaly_this_week', 'none')}</h3><p>Top Anomaly</p></div>
        </div>
    """

    # SECTION 3: Persons flagged 3+ consecutive days
    html += "<h2>3. Chronic Flags (3+ Consecutive Days)</h2>"
    if chronic_alerts:
        seen_persons = set()
        for alert in chronic_alerts:
            if alert['person_id'] not in seen_persons:
                html += f"""
                <div class="card urgent">
                    <h3 style="margin-top:0;">{alert['person_name']}</h3>
                    <p>Flagged consecutively for {alert['days_flagged_consecutively']} days. Immediate intervention recommended.</p>
                </div>
                """
                seen_persons.add(alert['person_id'])
    else:
        html += "<p>No students flagged for 3+ consecutive days.</p>"

    html += "</body></html>"
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

    today_str = all_dates[-1] if all_dates else str(date.today())
    today_alerts = [a for a in all_alerts if a.get("date") == today_str]
    flagged_today_count = len(set(a.get("person_id") for a in today_alerts))
    cat_counter   = Counter(a.get("category") for a in all_alerts)
    top_category  = cat_counter.most_common(1)[0][0] if cat_counter else "none"
    today_wb_scores = [p.get("wellbeing", 0) for p in daily_data.get(today_str, {}).values()]
    school_avg_wb = round(sum(today_wb_scores) / len(today_wb_scores)) if today_wb_scores else 0

    school_summary = {
        "total_persons_tracked":       len(person_days),
        "persons_flagged_today":       flagged_today_count,
        "persons_flagged_yesterday":   0,   
        "most_common_anomaly_this_week": top_category,
        "school_avg_wellbeing_today":  school_avg_wb,     
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
