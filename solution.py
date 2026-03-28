"""
anomaly_detection.py
Sentio Mind - Project 5 - Behavioral Anomaly and Early Distress Detection

Copy this file to solution.py and fill in every TODO block.
Do not rename any function. No OpenCV needed - pure data analysis.
Run: python solution.py
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime, date, timedelta
from collections import defaultdict, Counter
from typing import Dict, Any, List, Optional, Tuple

# ---------------------------------------------------------------------------
# CONFIG - adjust thresholds here, nowhere else
# ---------------------------------------------------------------------------
DATA_DIR = Path("sample_data")
REPORT_OUT = Path("alert_digest.html")
FEED_OUT = Path("alert_feed.json")
SCHOOL = "Demo School"

THRESHOLDS = {
    "sudden_drop_delta": 20,
    "sudden_drop_high_std_delta": 30,
    "sustained_low_score": 45,
    "sustained_low_days": 3,
    "social_withdrawal_delta": 25,
    "hyperactivity_delta": 40,
    "regression_recover_days": 3,
    "regression_drop": 15,
    "gaze_avoidance_days": 3,
    "absence_days": 2,
    "baseline_window": 3,
    "high_std_baseline": 15,
}

SEVERITY_ORDER = {"urgent": 0, "monitor": 1, "informational": 2}


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_date_string(value: str) -> Optional[str]:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _date_from_filename(fp: Path) -> Optional[str]:
    stem = fp.stem
    for token in stem.replace("_", "-").split("-"):
        if len(token) == 10:
            parsed = _parse_date_string(token)
            if parsed:
                return parsed
    return None


def _normalize_gaze(value: Any) -> str:
    if not value:
        return "forward"
    gaze = str(value).strip().lower()
    if gaze in {"down", "side", "forward", "up"}:
        return gaze
    if "down" in gaze:
        return "down"
    if "side" in gaze:
        return "side"
    return "forward"


def _normalize_person_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    person_id = record.get("person_id") or record.get("id") or record.get("student_id")
    if not person_id:
        return None

    person_name = record.get("person_name") or record.get("name") or record.get("full_name") or person_id
    profile_image_b64 = record.get("profile_image_b64", "")

    wellbeing = _safe_float(record.get("wellbeing"))
    social_engagement = _safe_float(record.get("social_engagement"))

    traits = {}
    if isinstance(record.get("traits"), dict):
        traits.update(record.get("traits"))

    for key in ("physical_energy", "movement_energy", "fidgeting", "movement_speed", "restlessness_index"):
        if key in record:
            traits[key] = _safe_float(record.get(key))

    gaze_direction = _normalize_gaze(record.get("gaze_direction") or record.get("dominant_gaze") or record.get("gaze"))

    eye_contact = record.get("eye_contact")
    if eye_contact is None:
        if "eye_contact_detected" in record:
            eye_contact = bool(record.get("eye_contact_detected"))
        elif "eye_contact_duration" in record:
            eye_contact = _safe_float(record.get("eye_contact_duration")) > 0
        else:
            eye_contact = True

    person_detected = bool(record.get("person_detected", True))

    return {
        "person_id": person_id,
        "person_name": person_name,
        "profile_image_b64": profile_image_b64,
        "wellbeing": wellbeing,
        "social_engagement": social_engagement,
        "traits": traits,
        "gaze_direction": gaze_direction,
        "eye_contact": bool(eye_contact),
        "person_detected": person_detected,
        "person_info": {"name": person_name, "profile_image_b64": profile_image_b64},
    }


def _sparkline_svg(values: List[float], width: int = 140, height: int = 24) -> str:
    if not values:
        return ""
    min_val = min(values)
    max_val = max(values)
    span = max_val - min_val if max_val > min_val else 1.0
    points = []
    for i, val in enumerate(values):
        x = (i / max(1, len(values) - 1)) * width
        y = height - ((val - min_val) / span) * height
        points.append(f"{x:.1f},{y:.1f}")
    return f"<svg viewBox='0 0 {width} {height}' width='{width}' height='{height}'><polyline points='{ ' '.join(points) }' fill='none' stroke='#2563eb' stroke-width='2'/></svg>"


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
    """
    daily: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for fp in sorted(folder.glob("*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            payload = json.load(f)

        file_date = None
        if isinstance(payload, dict):
            file_date = _parse_date_string(payload.get("date") or payload.get("analysis_date"))
        if not file_date:
            file_date = _date_from_filename(fp)
        if not file_date:
            file_date = fp.stem

        persons: List[Dict[str, Any]] = []
        if isinstance(payload, list):
            persons = payload
        elif isinstance(payload, dict):
            for key in ("persons", "students", "people", "data"):
                if isinstance(payload.get(key), list):
                    persons = payload.get(key)
                    break
            if not persons:
                candidates = []
                for key, value in payload.items():
                    if key in {"date", "analysis_date", "school"}:
                        continue
                    if isinstance(value, dict):
                        if "person_id" not in value:
                            value = {**value, "person_id": key}
                        candidates.append(value)
                persons = candidates

        daily.setdefault(file_date, {})
        for record in persons:
            normalized = _normalize_person_record(record)
            if not normalized:
                continue
            daily[file_date][normalized["person_id"]] = normalized

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
    """
    window = history[:THRESHOLDS["baseline_window"]]
    wellbeing_vals = [h.get("wellbeing", 0) for h in window]
    wellbeing_mean = float(np.mean(wellbeing_vals)) if wellbeing_vals else 0.0
    wellbeing_std = float(np.std(wellbeing_vals, ddof=1)) if len(wellbeing_vals) > 1 else 0.0

    trait_means: Dict[str, float] = {}
    trait_keys = set()
    for h in window:
        trait_keys.update(h.get("traits", {}).keys())

    for key in trait_keys:
        vals = [_safe_float(h.get("traits", {}).get(key)) for h in window]
        trait_means[key] = float(np.mean(vals)) if vals else 0.0

    gaze_vals = [h.get("gaze_direction", "forward") for h in window if h.get("gaze_direction")]
    avg_gaze = Counter(gaze_vals).most_common(1)[0][0] if gaze_vals else "forward"

    return {
        "wellbeing_mean": wellbeing_mean,
        "wellbeing_std": wellbeing_std,
        "trait_means": trait_means,
        "avg_gaze": avg_gaze,
    }


# ---------------------------------------------------------------------------
# ANOMALY DETECTORS - each returns an alert dict or None
# ---------------------------------------------------------------------------

def detect_sudden_drop(today: dict, baseline: dict) -> Optional[dict]:
    """
    today['wellbeing'] vs baseline['wellbeing_mean'].
    If baseline_std > 15, raise threshold to sudden_drop_high_std_delta.
    Severity: delta > 35 = urgent, else monitor.
    """
    baseline_mean = baseline.get("wellbeing_mean", 0.0)
    baseline_std = baseline.get("wellbeing_std", 0.0)
    drop = baseline_mean - today.get("wellbeing", 0.0)

    threshold = THRESHOLDS["sudden_drop_high_std_delta"] if baseline_std > THRESHOLDS["high_std_baseline"] else THRESHOLDS["sudden_drop_delta"]

    if drop >= threshold:
        severity = "urgent" if drop > 35 else "monitor"
        return {
            "category": "SUDDEN_DROP",
            "severity": severity,
            "title": "Sudden wellbeing drop detected",
            "delta": -float(drop),
        }
    return None


def detect_sustained_low(history: list) -> Optional[dict]:
    """
    Check the last sustained_low_days entries in history.
    If all have wellbeing < sustained_low_score - alert.
    Severity: urgent.
    """
    window = history[-THRESHOLDS["sustained_low_days"]:]
    if len(window) < THRESHOLDS["sustained_low_days"]:
        return None
    if all(h.get("wellbeing", 0) < THRESHOLDS["sustained_low_score"] for h in window):
        return {
            "category": "SUSTAINED_LOW",
            "severity": "urgent",
            "title": "Sustained low wellbeing",
            "delta": 0.0,
        }
    return None


def detect_social_withdrawal(today: dict, baseline: dict) -> Optional[dict]:
    """
    social_engagement dropped >= social_withdrawal_delta AND
    today's gaze_direction is "down" or "side".
    Severity: monitor.
    """
    baseline_engagement = _safe_float(baseline.get("trait_means", {}).get("social_engagement", baseline.get("social_engagement_mean", 0)))
    today_engagement = _safe_float(today.get("social_engagement", 0))
    drop = baseline_engagement - today_engagement
    gaze = today.get("gaze_direction", "forward")
    if drop >= THRESHOLDS["social_withdrawal_delta"] and gaze in {"down", "side"}:
        return {
            "category": "SOCIAL_WITHDRAWAL",
            "severity": "monitor",
            "title": "Social withdrawal detected",
            "delta": -float(drop),
        }
    return None


def detect_hyperactivity_spike(today: dict, baseline: dict) -> Optional[dict]:
    """
    (today.physical_energy + today.movement_energy) minus
    (baseline.physical_energy_mean + baseline.movement_energy_mean) >= hyperactivity_delta.
    Severity: monitor.
    """
    traits = today.get("traits", {})
    today_energy = _safe_float(traits.get("physical_energy")) + _safe_float(traits.get("movement_energy"))
    base_traits = baseline.get("trait_means", {})
    baseline_energy = _safe_float(base_traits.get("physical_energy")) + _safe_float(base_traits.get("movement_energy"))

    if (today_energy - baseline_energy) >= THRESHOLDS["hyperactivity_delta"]:
        return {
            "category": "HYPERACTIVITY_SPIKE",
            "severity": "monitor",
            "title": "Hyperactivity spike detected",
            "delta": float(today_energy - baseline_energy),
        }
    return None


def detect_regression(history: list) -> Optional[dict]:
    """
    Find if the last regression_recover_days entries were all improving (each > previous),
    then today dropped > regression_drop.
    Severity: monitor.
    """
    recover_days = THRESHOLDS["regression_recover_days"]
    if len(history) < recover_days + 1:
        return None

    prev = history[-(recover_days + 1):-1]
    today = history[-1]
    improving = all(prev[i].get("wellbeing", 0) < prev[i + 1].get("wellbeing", 0) for i in range(len(prev) - 1))
    drop = prev[-1].get("wellbeing", 0) - today.get("wellbeing", 0)

    if improving and drop > THRESHOLDS["regression_drop"]:
        return {
            "category": "REGRESSION",
            "severity": "monitor",
            "title": "Regression after recovery",
            "delta": -float(drop),
        }
    return None


def detect_gaze_avoidance(history: list) -> Optional[dict]:
    """
    Last gaze_avoidance_days entries all have eye_contact == False (or missing).
    Severity: monitor.
    """
    window = history[-THRESHOLDS["gaze_avoidance_days"]:]
    if len(window) < THRESHOLDS["gaze_avoidance_days"]:
        return None
    if all(not h.get("eye_contact", True) for h in window):
        return {
            "category": "GAZE_AVOIDANCE",
            "severity": "monitor",
            "title": "Gaze avoidance detected",
            "delta": 0.0,
        }
    return None


# ---------------------------------------------------------------------------
# ANALYSE ONE PERSON
# ---------------------------------------------------------------------------

def analyse_person(person_id: str, sorted_days: dict, info: dict) -> list:
    """
    sorted_days: { "YYYY-MM-DD": person_data_dict } - keys in date order
    info: { name, profile_image_b64, ... }

    Build history list, compute baseline, run all detectors.
    Return list of alert dicts. Each alert must include person_id, person_name, date,
    and all fields from anomaly_detection.json schema.
    """
    alerts: List[Dict[str, Any]] = []
    history: List[Dict[str, Any]] = []

    dates = list(sorted_days.keys())
    if not dates:
        return alerts

    baseline = compute_baseline([sorted_days[d] for d in dates])

    consecutive_by_category: Dict[str, Tuple[Optional[str], int]] = {}

    for day in dates:
        today = sorted_days[day]
        history.append(today)
        if len(history) < THRESHOLDS["baseline_window"]:
            continue

        candidates = [
            detect_sudden_drop(today, baseline),
            detect_sustained_low(history),
            detect_social_withdrawal(today, baseline),
            detect_hyperactivity_spike(today, baseline),
            detect_regression(history),
            detect_gaze_avoidance(history),
        ]

        for result in [c for c in candidates if c]:
            category = result["category"]
            severity = result["severity"]
            title = result["title"]

            last_date, count = consecutive_by_category.get(category, (None, 0))
            if last_date and _parse_date_string(last_date) and _parse_date_string(day):
                last_dt = datetime.strptime(last_date, "%Y-%m-%d")
                curr_dt = datetime.strptime(day, "%Y-%m-%d")
                if (curr_dt - last_dt).days == 1:
                    count += 1
                else:
                    count = 1
            else:
                count = 1
            consecutive_by_category[category] = (day, count)

            trend = [h.get("wellbeing", 0) for h in history[-5:]]

            traits = today.get("traits", {})
            trait_values = {k: _safe_float(v) for k, v in traits.items()}
            trait_values["social_engagement"] = _safe_float(today.get("social_engagement", 0))
            if trait_values:
                lowest_trait = min(trait_values, key=trait_values.get)
                lowest_trait_value = trait_values[lowest_trait]
            else:
                lowest_trait = "wellbeing"
                lowest_trait_value = _safe_float(today.get("wellbeing", 0))

            baseline_wellbeing = baseline.get("wellbeing_mean", 0.0)
            today_wellbeing = _safe_float(today.get("wellbeing", 0))

            description = _build_description(
                category=category,
                person_name=info.get("name", person_id),
                baseline_wellbeing=baseline_wellbeing,
                today_wellbeing=today_wellbeing,
                lowest_trait=lowest_trait,
                lowest_trait_value=lowest_trait_value,
                gaze=today.get("gaze_direction", "forward"),
                delta=result.get("delta", 0.0),
            )

            alerts.append({
                "alert_id": "",
                "person_id": person_id,
                "person_name": info.get("name", person_id),
                "date": day,
                "severity": severity,
                "category": category,
                "title": title,
                "description": description,
                "baseline_wellbeing": round(float(baseline_wellbeing), 2),
                "today_wellbeing": round(float(today_wellbeing), 2),
                "delta": round(float(result.get("delta", today_wellbeing - baseline_wellbeing)), 2),
                "days_flagged_consecutively": count,
                "trend_last_5_days": trend,
                "lowest_trait": lowest_trait,
                "lowest_trait_value": round(float(lowest_trait_value), 2),
                "recommended_action": _recommended_action(category, severity),
                "profile_image_b64": info.get("profile_image_b64", ""),
            })

    return alerts


# ---------------------------------------------------------------------------
# HTML REPORT
# ---------------------------------------------------------------------------

def generate_alert_digest(alerts: list, absence_flags: list,
                           school_summary: dict, output_path: Path):
    """
    Self-contained HTML - no CDN, inline CSS only.

    Section 1: Today's alerts sorted by severity.
      Each card: person name, badge (urgent=red, monitor=amber), description,
      5-day sparkline (inline SVG or just coloured squares - keep it simple).

    Section 2: School summary numbers.

    Section 3: Persons flagged 3+ consecutive days.
    """
    today = school_summary.get("today", str(date.today()))
    todays_alerts = [a for a in alerts if a.get("date") == today]
    todays_alerts.sort(key=lambda a: SEVERITY_ORDER.get(a.get("severity", "monitor"), 9))

    persistent = [a for a in alerts if a.get("days_flagged_consecutively", 0) >= 3]

    def alert_card(a: Dict[str, Any]) -> str:
        severity = a.get("severity", "monitor")
        spark = _sparkline_svg([float(x) for x in a.get("trend_last_5_days", [])])
        return (
            f"<div class='alert-card {severity}'>"
            f"<div class='alert-header'><span class='name'>{a.get('person_name')}</span>"
            f"<span class='badge {severity}'>{severity}</span></div>"
            f"<div class='meta'>{a.get('category')}</div>"
            f"<div class='desc'>{a.get('description')}</div>"
            f"<div class='spark'>{spark}</div>"
            f"</div>"
        )

    today_html = "".join(alert_card(a) for a in todays_alerts) or "<div class='empty'>No alerts for today.</div>"
    persistent_html = "".join(alert_card(a) for a in persistent) or "<div class='empty'>No persistent alerts.</div>"

    summary_html = (
        f"<div class='summary-grid'>"
        f"<div class='summary-item'><div class='label'>Persons flagged today</div><div class='value'>{school_summary.get('persons_flagged_today', 0)}</div></div>"
        f"<div class='summary-item'><div class='label'>Persons flagged yesterday</div><div class='value'>{school_summary.get('persons_flagged_yesterday', 0)}</div></div>"
        f"<div class='summary-item'><div class='label'>Most common anomaly</div><div class='value'>{school_summary.get('most_common_anomaly_this_week', 'none')}</div></div>"
        f"<div class='summary-item'><div class='label'>School avg wellbeing today</div><div class='value'>{school_summary.get('school_avg_wellbeing_today', 0)}</div></div>"
        f"</div>"
    )

    html = f"""<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='UTF-8'>
<meta name='viewport' content='width=device-width, initial-scale=1.0'>
<title>Sentio Mind Alert Digest</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; color: #222; background: #f5f6f8; }}
.header {{ background: #0f172a; color: #fff; padding: 16px 20px; border-radius: 8px; }}
.section {{ margin-top: 20px; }}
.section h2 {{ margin-bottom: 10px; }}
.alert-card {{ background: #fff; border: 1px solid #e5e7eb; border-left: 4px solid #f59e0b; padding: 12px; border-radius: 6px; margin-bottom: 10px; }}
.alert-card.urgent {{ border-left-color: #dc2626; }}
.alert-card.monitor {{ border-left-color: #f59e0b; }}
.alert-header {{ display: flex; justify-content: space-between; align-items: center; }}
.badge {{ padding: 2px 8px; border-radius: 10px; font-size: 12px; text-transform: uppercase; }}
.badge.urgent {{ background: #dc2626; color: #fff; }}
.badge.monitor {{ background: #f59e0b; color: #fff; }}
.meta {{ color: #475569; font-size: 12px; margin-top: 4px; }}
.desc {{ margin-top: 6px; }}
.spark {{ margin-top: 8px; }}
.summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }}
.summary-item {{ background: #fff; padding: 10px; border-radius: 6px; border: 1px solid #e5e7eb; }}
.label {{ color: #64748b; font-size: 12px; }}
.value {{ font-size: 18px; font-weight: bold; }}
.empty {{ background: #fff; padding: 12px; border-radius: 6px; border: 1px dashed #cbd5f5; color: #64748b; }}
</style>
</head>
<body>
  <div class='header'>
    <h1>Sentio Mind Alert Digest</h1>
    <div>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
  </div>

  <div class='section'>
    <h2>Today's alerts</h2>
    {today_html}
  </div>

  <div class='section'>
    <h2>School summary</h2>
    {summary_html}
  </div>

  <div class='section'>
    <h2>Persistent alerts (3+ days)</h2>
    {persistent_html}
  </div>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _recommended_action(category: str, severity: str) -> str:
    if category == "ABSENCE_FLAG":
        return "Welfare check - contact family if absent again tomorrow"
    if category == "SUSTAINED_LOW":
        return "Schedule pastoral check-in today"
    if severity == "urgent":
        return "Schedule pastoral check-in today"
    return "Monitor and follow up if pattern persists"


def _build_description(category: str, person_name: str, baseline_wellbeing: float,
                       today_wellbeing: float, lowest_trait: str,
                       lowest_trait_value: float, gaze: str, delta: float) -> str:
    if category == "SUDDEN_DROP":
        drop = abs(int(round(delta)))
        return (f"{person_name}'s wellbeing dropped from a baseline of {int(round(baseline_wellbeing))} "
                f"to {int(round(today_wellbeing))} today - a {drop}-point fall. "
                f"Lowest trait: {lowest_trait} at {int(round(lowest_trait_value))}. Dominant gaze: {gaze}.")
    if category == "SUSTAINED_LOW":
        return f"Wellbeing has stayed below {THRESHOLDS['sustained_low_score']} for {THRESHOLDS['sustained_low_days']} consecutive days."
    if category == "SOCIAL_WITHDRAWAL":
        return f"Social engagement dropped sharply and dominant gaze is {gaze}."
    if category == "HYPERACTIVITY_SPIKE":
        return "Energy traits are significantly above baseline today."
    if category == "REGRESSION":
        return "After several days of improvement, wellbeing dropped sharply today."
    if category == "GAZE_AVOIDANCE":
        return f"No eye contact detected for {THRESHOLDS['gaze_avoidance_days']} consecutive days."
    if category == "ABSENCE_FLAG":
        return "Person not detected for multiple consecutive days."
    return "Alert detected based on recent behavior changes."


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    daily_data = load_daily_data(DATA_DIR)
    all_dates = sorted(daily_data.keys())
    print(f"Loaded {len(daily_data)} days: {all_dates}")

    person_days = defaultdict(dict)
    person_info = {}

    for day, persons in daily_data.items():
        for pid, pdata in persons.items():
            person_days[pid][day] = pdata
            if pid not in person_info:
                person_info[pid] = pdata.get("person_info", {"name": pid, "profile_image_b64": ""})

    all_alerts: List[Dict[str, Any]] = []
    absence_flags: List[Dict[str, Any]] = []

    for pid, days in person_days.items():
        sorted_days = dict(sorted(days.items()))
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
                "person_id": pid,
                "person_name": person_info.get(pid, {}).get("name", pid),
                "last_seen_date": last_seen,
                "days_absent": absent,
                "recommended_action": "Welfare check - contact family if absent again tomorrow",
            })

            baseline = compute_baseline(list(sorted_days.values()))
            baseline_wellbeing = baseline.get("wellbeing_mean", 0.0)
            absence_alert = {
                "alert_id": "",
                "person_id": pid,
                "person_name": person_info.get(pid, {}).get("name", pid),
                "date": all_dates[-1] if all_dates else str(date.today()),
                "severity": "urgent",
                "category": "ABSENCE_FLAG",
                "title": "Absence flag",
                "description": f"Person not detected for {absent} consecutive days. Last seen {last_seen}.",
                "baseline_wellbeing": round(float(baseline_wellbeing), 2),
                "today_wellbeing": 0.0,
                "delta": round(0.0 - float(baseline_wellbeing), 2),
                "days_flagged_consecutively": absent,
                "trend_last_5_days": [h.get("wellbeing", 0) for h in list(sorted_days.values())[-5:]],
                "lowest_trait": "wellbeing",
                "lowest_trait_value": 0.0,
                "recommended_action": "Welfare check - contact family if absent again tomorrow",
                "profile_image_b64": person_info.get(pid, {}).get("profile_image_b64", ""),
            }
            all_alerts.append(absence_alert)

    # Assign alert IDs
    for idx, alert in enumerate(all_alerts, start=1):
        alert["alert_id"] = f"ALT_{idx:03d}"

    all_alerts.sort(key=lambda a: SEVERITY_ORDER.get(a.get("severity", "informational"), 3))

    today_str = all_dates[-1] if all_dates else str(date.today())
    yesterday_str = None
    if all_dates:
        try:
            dt = datetime.strptime(today_str, "%Y-%m-%d")
            yesterday_str = (dt - timedelta(days=1)).strftime("%Y-%m-%d")
        except ValueError:
            yesterday_str = None

    flagged_today = set(a.get("person_id") for a in all_alerts if a.get("date") == today_str)
    flagged_yesterday = set(a.get("person_id") for a in all_alerts if a.get("date") == (yesterday_str or ""))

    week_start = None
    try:
        week_start = datetime.strptime(today_str, "%Y-%m-%d") - timedelta(days=6)
    except ValueError:
        week_start = None

    cat_counter = Counter()
    if week_start:
        for a in all_alerts:
            try:
                adt = datetime.strptime(a.get("date"), "%Y-%m-%d")
                if adt >= week_start:
                    cat_counter[a.get("category")] += 1
            except (TypeError, ValueError):
                continue

    top_category = cat_counter.most_common(1)[0][0] if cat_counter else "none"

    school_avg_today = 0.0
    if today_str in daily_data and daily_data[today_str]:
        school_avg_today = float(np.mean([p.get("wellbeing", 0) for p in daily_data[today_str].values()]))

    school_summary = {
        "total_persons_tracked": len(person_days),
        "persons_flagged_today": len(flagged_today),
        "persons_flagged_yesterday": len(flagged_yesterday),
        "most_common_anomaly_this_week": top_category,
        "school_avg_wellbeing_today": round(school_avg_today, 2),
        "today": today_str,
    }

    feed = {
        "source": "p5_anomaly_detection",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "school": SCHOOL,
        "alert_summary": {
            "total_alerts": len(all_alerts),
            "urgent": sum(1 for a in all_alerts if a.get("severity") == "urgent"),
            "monitor": sum(1 for a in all_alerts if a.get("severity") == "monitor"),
            "informational": sum(1 for a in all_alerts if a.get("severity") == "informational"),
        },
        "alerts": all_alerts,
        "absence_flags": absence_flags,
        "school_summary": {
            "total_persons_tracked": school_summary["total_persons_tracked"],
            "persons_flagged_today": school_summary["persons_flagged_today"],
            "persons_flagged_yesterday": school_summary["persons_flagged_yesterday"],
            "most_common_anomaly_this_week": school_summary["most_common_anomaly_this_week"],
            "school_avg_wellbeing_today": school_summary["school_avg_wellbeing_today"],
        },
    }

    with open(FEED_OUT, "w", encoding="utf-8") as f:
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
