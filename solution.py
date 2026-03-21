"""solution.py - Arun Kumar 21f3003030"""
import json
import numpy as np
from pathlib import Path
from datetime import datetime, date
from collections import defaultdict, Counter

DATA_DIR = Path("sample_data")
REPORT_OUT = Path("alert_digest.html")
FEED_OUT = Path("alert_feed.json")
SCHOOL = "Demo School"

THRESHOLDS = {"sudden_drop_delta": 20, "sudden_drop_high_std_delta": 30, "sustained_low_score": 45, "sustained_low_days": 3, "social_withdrawal_delta": 25, "hyperactivity_delta": 40, "regression_recover_days": 3, "regression_drop": 15, "gaze_avoidance_days": 3, "absence_days": 2, "baseline_window": 3, "high_std_baseline": 15}

def load_daily_data(folder: Path) -> dict:
    daily = {}
    for fp in sorted(folder.glob("*.json")):
        with open(fp) as f:
            data = json.load(f)
            if "date" in data:
                daily[data["date"]] = data.get("persons", {})
    return daily

def compute_baseline(history: list) -> dict:
    w = min(len(history), THRESHOLDS["baseline_window"])
    base = history[:w]
    wb = [d.get("wellbeing", 50) for d in base]
    traits = defaultdict(list)
    for d in base:
        for k, v in d.get("traits", {}).items():
            traits[k].append(v)
    gazes = [d.get("gaze_direction", "forward") for d in base]
    return {"wellbeing_mean": np.mean(wb), "wellbeing_std": np.std(wb), "trait_means": {k: np.mean(v) for k, v in traits.items()}, "avg_gaze": max(set(gazes), key=gazes.count)}

def detect_sudden_drop(today, baseline):
    tw = today.get("wellbeing", 50)
    bm = baseline["wellbeing_mean"]
    bs = baseline["wellbeing_std"]
    thresh = THRESHOLDS["sudden_drop_high_std_delta"] if bs > THRESHOLDS["high_std_baseline"] else THRESHOLDS["sudden_drop_delta"]
    delta = bm - tw
    if delta >= thresh:
        return {"category": "SUDDEN_DROP", "severity": "urgent" if delta > 35 else "monitor", "title": "Sudden wellbeing drop detected", "description": f"Wellbeing dropped from {bm:.1f} to {tw} — {delta:.1f}-point fall.", "baseline_wellbeing": round(bm, 1), "today_wellbeing": tw, "delta": round(-delta, 1), "recommended_action": "Schedule pastoral check-in today"}
    return None

def detect_sustained_low(history):
    if len(history) < THRESHOLDS["sustained_low_days"]:
        return None
    if all(d.get("wellbeing", 50) < THRESHOLDS["sustained_low_score"] for d in history[-THRESHOLDS["sustained_low_days"]:]):
        return {"category": "SUSTAINED_LOW", "severity": "urgent", "title": "Sustained low wellbeing", "description": f"Wellbeing below {THRESHOLDS['sustained_low_score']} for {THRESHOLDS['sustained_low_days']} consecutive days.", "days_flagged_consecutively": THRESHOLDS["sustained_low_days"], "recommended_action": "Immediate counsellor intervention required"}
    return None

def detect_social_withdrawal(today, baseline):
    ts = today.get("traits", {}).get("social_engagement", 50)
    bs = baseline["trait_means"].get("social_engagement", 50)
    g = today.get("gaze_direction", "forward")
    delta = bs - ts
    if delta >= THRESHOLDS["social_withdrawal_delta"] and g in ["down", "side"]:
        return {"category": "SOCIAL_WITHDRAWAL", "severity": "monitor", "title": "Social withdrawal detected", "description": f"Social engagement dropped {delta:.1f} points with {g} gaze.", "recommended_action": "Check-in with peer group"}
    return None

def detect_hyperactivity_spike(today, baseline):
    te = today.get("traits", {}).get("physical_energy", 50) + today.get("traits", {}).get("movement_energy", 50)
    be = baseline["trait_means"].get("physical_energy", 50) + baseline["trait_means"].get("movement_energy", 50)
    delta = te - be
    if delta >= THRESHOLDS["hyperactivity_delta"]:
        return {"category": "HYPERACTIVITY_SPIKE", "severity": "monitor", "title": "Hyperactivity spike", "description": f"Energy {delta:.1f} points above baseline.", "recommended_action": "Observe for ADHD patterns"}
    return None

def detect_regression(history):
    if len(history) < THRESHOLDS["regression_recover_days"] + 1:
        return None
    rec = history[-(THRESHOLDS["regression_recover_days"] + 1):]
    improving = all(rec[i].get("wellbeing", 50) < rec[i + 1].get("wellbeing", 50) for i in range(len(rec) - 2))
    if improving:
        drop = rec[-2].get("wellbeing", 50) - rec[-1].get("wellbeing", 50)
        if drop > THRESHOLDS["regression_drop"]:
            return {"category": "REGRESSION", "severity": "monitor", "title": "Regression after recovery", "description": f"Dropped {drop:.1f} points after {THRESHOLDS['regression_recover_days']} days recovery.", "recommended_action": "Investigate trigger"}
    return None

def detect_gaze_avoidance(history):
    if len(history) < THRESHOLDS["gaze_avoidance_days"]:
        return None
    if all(not d.get("eye_contact", True) for d in history[-THRESHOLDS["gaze_avoidance_days"]:]):
        return {"category": "GAZE_AVOIDANCE", "severity": "monitor", "title": "Prolonged gaze avoidance", "description": f"No eye contact for {THRESHOLDS['gaze_avoidance_days']} days.", "recommended_action": "Screen for social anxiety"}
    return None

def analyse_person(person_id, sorted_days, info):
    alerts = []
    dates = list(sorted_days.keys())
    history = [sorted_days[d] for d in dates]
    if not history:
        return alerts
    baseline = compute_baseline(history)
    today = history[-1]
    for det in [detect_sudden_drop, detect_social_withdrawal, detect_hyperactivity_spike]:
        a = det(today, baseline)
        if a:
            a.update({"alert_id": f"ALT_{person_id}_{len(alerts)+1}", "person_id": person_id, "person_name": info.get("name", person_id), "date": dates[-1], "trend_last_5_days": [sorted_days[d].get("wellbeing", 50) for d in dates[-5:]], "profile_image_b64": ""})
            alerts.append(a)
    for det in [detect_sustained_low, detect_regression, detect_gaze_avoidance]:
        a = det(history)
        if a:
            a.update({"alert_id": f"ALT_{person_id}_{len(alerts)+1}", "person_id": person_id, "person_name": info.get("name", person_id), "date": dates[-1], "trend_last_5_days": [sorted_days[d].get("wellbeing", 50) for d in dates[-5:]], "profile_image_b64": ""})
            alerts.append(a)
    return alerts

def generate_alert_digest(alerts, absence_flags, school_summary, output_path):
    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Alert Digest</title><style>body{{font-family:sans-serif;margin:20px;background:#f5f5f5}}.container{{max-width:1200px;margin:auto}}.summary{{background:white;padding:20px;border-radius:8px;margin-bottom:20px}}.alert-card{{background:white;padding:20px;border-radius:8px;margin-bottom:15px;border-left:4px solid #ccc}}.alert-card.urgent{{border-left-color:#f44336}}.alert-card.monitor{{border-left-color:#ff9800}}.badge{{padding:4px 12px;border-radius:12px;font-size:0.85em;font-weight:bold}}.badge.urgent{{background:#ffebee;color:#c62828}}.badge.monitor{{background:#fff3e0;color:#e65100}}</style></head><body><div class="container"><h1>Alert Digest - {SCHOOL}</h1><p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p><div class="summary"><h2>Summary</h2><p>Total Students: {school_summary['total_persons_tracked']} | Flagged Today: {school_summary['persons_flagged_today']} | Total Alerts: {len(alerts)}</p></div><h2>Alerts</h2>"""
    for a in alerts:
        html += f"""<div class="alert-card {a['severity']}"><span class="badge {a['severity']}">{a['severity']}</span> <strong>{a['person_name']}</strong> · {a['date']}<h3>{a['title']}</h3><p>{a['description']}</p><p><strong>Action:</strong> {a['recommended_action']}</p></div>"""
    html += "</div></body></html>"
    with open(output_path, 'w') as f:
        f.write(html)

if __name__ == "__main__":
    daily_data = load_daily_data(DATA_DIR)
    all_dates = sorted(daily_data.keys())
    print(f"Loaded {len(daily_data)} days: {all_dates}")
    person_days = defaultdict(dict)
    person_info = {}
    for d, persons in daily_data.items():
        for pid, pdata in persons.items():
            person_days[pid][d] = pdata
            if pid not in person_info:
                person_info[pid] = {"name": pdata.get("name", pid), "profile_image_b64": ""}
    all_alerts = []
    absence_flags = []
    for pid, days in person_days.items():
        sorted_days = dict(sorted(days.items()))
        all_alerts.extend(analyse_person(pid, sorted_days, person_info.get(pid, {})))
        present = set(days.keys())
        absent = 0
        for d in reversed(all_dates):
            if d not in present:
                absent += 1
            else:
                break
        if absent >= THRESHOLDS["absence_days"]:
            absence_flags.append({"person_id": pid, "person_name": person_info.get(pid, {}).get("name", pid), "last_seen_date": sorted(present)[-1] if present else "unknown", "days_absent": absent, "recommended_action": "Welfare check"})
    all_alerts.sort(key=lambda a: {"urgent": 0, "monitor": 1}.get(a.get("severity"), 2))
    school_summary = {"total_persons_tracked": len(person_days), "persons_flagged_today": len(set(a["person_id"] for a in all_alerts)), "persons_flagged_yesterday": 0, "most_common_anomaly_this_week": Counter(a["category"] for a in all_alerts).most_common(1)[0][0] if all_alerts else "none", "school_avg_wellbeing_today": 0, "school": SCHOOL}
    feed = {"source": "p5_anomaly_detection", "generated_at": datetime.now().isoformat(), "school": SCHOOL, "alert_summary": {"total_alerts": len(all_alerts), "urgent": sum(1 for a in all_alerts if a.get("severity") == "urgent"), "monitor": sum(1 for a in all_alerts if a.get("severity") == "monitor"), "informational": 0}, "alerts": all_alerts, "absence_flags": absence_flags, "school_summary": school_summary}
    with open(FEED_OUT, 'w') as f:
        json.dump(feed, f, indent=2)
    generate_alert_digest(all_alerts, absence_flags, school_summary, REPORT_OUT)
    print(f"\n{'='*50}\nAlerts: {len(all_alerts)} total ({feed['alert_summary']['urgent']} urgent, {feed['alert_summary']['monitor']} monitor)\nAbsence flags: {len(absence_flags)}\nReport → {REPORT_OUT}\nJSON → {FEED_OUT}\n{'='*50}")
