"""
solution.py
Sentio Mind - Project 5 - Behavioral Anomaly & Early Distress Detection

Run: python solution.py
"""

import json
import html
import math
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Any

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


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _coalesce(record: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _std(values: List[float]) -> float:
    if not values:
        return 0.0
    mu = _mean(values)
    variance = sum((v - mu) ** 2 for v in values) / len(values)
    return float(math.sqrt(variance))


def _normalize_person_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    person_info = raw.get("person_info", {}) if isinstance(raw.get("person_info"), dict) else {}

    person_id = _coalesce(raw, ["person_id", "id", "student_id", "uid"], "UNKNOWN")
    name = _coalesce(raw, ["name", "person_name"], None) or person_info.get("name") or person_id
    profile_image_b64 = person_info.get("profile_image_b64", "")

    wellbeing = _safe_float(
        _coalesce(raw, ["wellbeing", "wellbeing_score", "overall_wellbeing", "score"], 0)
    )

    traits = raw.get("traits", {}) if isinstance(raw.get("traits"), dict) else {}
    if not traits:
        traits = {
            "social_engagement": _safe_float(raw.get("social_engagement", 0)),
            "physical_energy": _safe_float(raw.get("physical_energy", 0)),
            "movement_energy": _safe_float(raw.get("movement_energy", 0)),
        }
    else:
        traits = {k: _safe_float(v, 0.0) for k, v in traits.items()}

    social_engagement = _safe_float(
        _coalesce(raw, ["social_engagement"], traits.get("social_engagement", 0.0))
    )
    traits.setdefault("social_engagement", social_engagement)
    traits.setdefault("physical_energy", _safe_float(traits.get("physical_energy", 0.0)))
    traits.setdefault("movement_energy", _safe_float(traits.get("movement_energy", 0.0)))

    gaze_obj = raw.get("gaze", {}) if isinstance(raw.get("gaze"), dict) else {}
    gaze_direction = str(
        _coalesce(raw, ["gaze_direction", "dominant_gaze"], gaze_obj.get("dominant_direction", "forward"))
    ).lower()

    eye_contact_raw = _coalesce(raw, ["eye_contact", "eye_contact_detected"], None)
    if eye_contact_raw is None:
        eye_contact_ratio = _safe_float(_coalesce(raw, ["eye_contact_ratio"], gaze_obj.get("eye_contact_ratio", 0.0)))
        eye_contact = eye_contact_ratio > 0.0
    else:
        eye_contact = bool(eye_contact_raw)

    detected_raw = _coalesce(raw, ["detected", "person_detected", "is_present"], True)
    detected = bool(detected_raw)

    return {
        "person_id": person_id,
        "person_info": {"name": name, "profile_image_b64": profile_image_b64},
        "wellbeing": float(wellbeing),
        "traits": traits,
        "gaze_direction": gaze_direction,
        "eye_contact": eye_contact,
        "detected": detected,
    }


def load_daily_data(folder: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Read all .json files from folder and return:
      { "YYYY-MM-DD": { "PERSON_ID": person_data, ... }, ... }
    """
    daily: Dict[str, Dict[str, Dict[str, Any]]] = {}

    if not folder.exists():
        return daily

    for fp in sorted(folder.glob("*.json")):
        with fp.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        day_key = str(_coalesce(payload, ["date", "day", "analysis_date"], fp.stem))

        persons_obj = _coalesce(payload, ["persons", "students", "data"], [])
        people_for_day: Dict[str, Dict[str, Any]] = {}

        if isinstance(persons_obj, dict):
            for pid, pdata in persons_obj.items():
                if isinstance(pdata, dict):
                    normalized = _normalize_person_record({"person_id": pid, **pdata})
                    people_for_day[normalized["person_id"]] = normalized
        elif isinstance(persons_obj, list):
            for person in persons_obj:
                if isinstance(person, dict):
                    normalized = _normalize_person_record(person)
                    people_for_day[normalized["person_id"]] = normalized

        daily[day_key] = people_for_day

    return dict(sorted(daily.items(), key=lambda x: x[0]))


# ---------------------------------------------------------------------------
# BASELINE
# ---------------------------------------------------------------------------

def compute_baseline(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Use first baseline_window days from history to compute personal baseline.
    """
    if not history:
        return {
            "wellbeing_mean": 0.0,
            "wellbeing_std": 0.0,
            "trait_means": {},
            "avg_gaze": "forward",
        }

    window = history[: THRESHOLDS["baseline_window"]]

    wellbeing_values = [_safe_float(d.get("wellbeing", 0.0)) for d in window]
    wellbeing_mean = _mean(wellbeing_values)
    wellbeing_std = _std(wellbeing_values)

    all_trait_keys = set()
    for d in window:
        all_trait_keys.update((d.get("traits") or {}).keys())

    trait_means: Dict[str, float] = {}
    for key in all_trait_keys:
        vals = [_safe_float((d.get("traits") or {}).get(key, 0.0)) for d in window]
        trait_means[key] = _mean(vals)

    gazes = [str(d.get("gaze_direction", "forward")).lower() for d in window]
    avg_gaze = Counter(gazes).most_common(1)[0][0] if gazes else "forward"

    return {
        "wellbeing_mean": wellbeing_mean,
        "wellbeing_std": wellbeing_std,
        "trait_means": trait_means,
        "avg_gaze": avg_gaze,
    }


# ---------------------------------------------------------------------------
# ANOMALY DETECTORS - each returns an alert payload or None
# ---------------------------------------------------------------------------

def detect_sudden_drop(today: Dict[str, Any], baseline: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    base = _safe_float(baseline.get("wellbeing_mean", 0.0))
    today_wb = _safe_float(today.get("wellbeing", 0.0))
    drop = base - today_wb

    threshold = THRESHOLDS["sudden_drop_delta"]
    if _safe_float(baseline.get("wellbeing_std", 0.0)) > THRESHOLDS["high_std_baseline"]:
        threshold = THRESHOLDS["sudden_drop_high_std_delta"]

    if drop >= threshold:
        severity = "urgent" if drop > 35 else "monitor"
        return {
            "category": "SUDDEN_DROP",
            "severity": severity,
            "title": "Sudden wellbeing drop detected",
            "description": (
                f"Wellbeing dropped from baseline {int(round(base))} to {int(round(today_wb))} "
                f"({int(round(drop))}-point fall)."
            ),
            "recommended_action": "Schedule pastoral check-in today",
        }
    return None


def detect_sustained_low(history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    n = THRESHOLDS["sustained_low_days"]
    if len(history) < n:
        return None

    tail = history[-n:]
    if all(_safe_float(d.get("wellbeing", 0.0)) < THRESHOLDS["sustained_low_score"] for d in tail):
        return {
            "category": "SUSTAINED_LOW",
            "severity": "urgent",
            "title": "Sustained low wellbeing",
            "description": f"Wellbeing stayed below {THRESHOLDS['sustained_low_score']} for {n} consecutive days.",
            "recommended_action": "Escalate to counsellor for same-day follow-up",
        }
    return None


def detect_social_withdrawal(today: Dict[str, Any], baseline: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    today_social = _safe_float((today.get("traits") or {}).get("social_engagement", 0.0))
    baseline_social = _safe_float((baseline.get("trait_means") or {}).get("social_engagement", 0.0))
    drop = baseline_social - today_social
    gaze = str(today.get("gaze_direction", "")).lower()

    if drop >= THRESHOLDS["social_withdrawal_delta"] and gaze in {"down", "side"}:
        return {
            "category": "SOCIAL_WITHDRAWAL",
            "severity": "monitor",
            "title": "Social withdrawal signal",
            "description": (
                f"Social engagement dropped by {int(round(drop))} points and gaze was mostly {gaze}."
            ),
            "recommended_action": "Arrange supportive peer/counsellor conversation",
        }
    return None


def detect_hyperactivity_spike(today: Dict[str, Any], baseline: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    t_traits = today.get("traits") or {}
    b_traits = baseline.get("trait_means") or {}

    today_combined = _safe_float(t_traits.get("physical_energy", 0.0)) + _safe_float(t_traits.get("movement_energy", 0.0))
    base_combined = _safe_float(b_traits.get("physical_energy", 0.0)) + _safe_float(b_traits.get("movement_energy", 0.0))
    delta = today_combined - base_combined

    if delta >= THRESHOLDS["hyperactivity_delta"]:
        return {
            "category": "HYPERACTIVITY_SPIKE",
            "severity": "monitor",
            "title": "Hyperactivity spike",
            "description": f"Combined energy is {int(round(delta))} points above personal baseline.",
            "recommended_action": "Observe classroom regulation and trigger calming support",
        }
    return None


def detect_regression(history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    recover_days = THRESHOLDS["regression_recover_days"]
    if len(history) < recover_days + 1:
        return None

    segment = history[-(recover_days + 1):]
    recover_chain = segment[:-1]

    improving = True
    for idx in range(1, len(recover_chain)):
        prev_w = _safe_float(recover_chain[idx - 1].get("wellbeing", 0.0))
        curr_w = _safe_float(recover_chain[idx].get("wellbeing", 0.0))
        if curr_w <= prev_w:
            improving = False
            break

    if not improving:
        return None

    yesterday = _safe_float(segment[-2].get("wellbeing", 0.0))
    today = _safe_float(segment[-1].get("wellbeing", 0.0))
    drop = yesterday - today

    if drop > THRESHOLDS["regression_drop"]:
        return {
            "category": "REGRESSION",
            "severity": "monitor",
            "title": "Regression after recovery",
            "description": (
                f"After {recover_days} improving days, wellbeing fell by {int(round(drop))} points today."
            ),
            "recommended_action": "Review potential trigger events with student",
        }
    return None


def detect_gaze_avoidance(history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    n = THRESHOLDS["gaze_avoidance_days"]
    if len(history) < n:
        return None

    tail = history[-n:]
    if all(not bool(day.get("eye_contact", False)) for day in tail):
        return {
            "category": "GAZE_AVOIDANCE",
            "severity": "monitor",
            "title": "Gaze avoidance pattern",
            "description": f"No eye contact detected for {n} consecutive days.",
            "recommended_action": "Use low-pressure check-in and monitor engagement",
        }
    return None


def detect_absence_flag(all_dates: List[str], present_dates: List[str]) -> Optional[Dict[str, Any]]:
    if not all_dates:
        return None

    present = set(present_dates)
    absent = 0
    for d in reversed(all_dates):
        if d not in present:
            absent += 1
        else:
            break

    if absent >= THRESHOLDS["absence_days"]:
        last_seen = sorted(present)[-1] if present else "unknown"
        return {
            "category": "ABSENCE_FLAG",
            "severity": "urgent",
            "title": "Repeated absence detected",
            "description": f"No detection for {absent} consecutive days.",
            "last_seen_date": last_seen,
            "days_absent": absent,
            "recommended_action": "Welfare check - contact family if absent again tomorrow",
        }
    return None


# ---------------------------------------------------------------------------
# ANALYSE ONE PERSON
# ---------------------------------------------------------------------------

def analyse_person(person_id: str, sorted_days: Dict[str, Dict[str, Any]], info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Run all detectors for each available day for a person and return alerts.
    """
    alerts: List[Dict[str, Any]] = []

    if not sorted_days:
        return alerts

    day_items = sorted(sorted_days.items(), key=lambda x: x[0])
    full_history = [entry for _, entry in day_items]
    baseline = compute_baseline(full_history)

    category_streak: Dict[str, int] = defaultdict(int)

    for idx, (day_key, today) in enumerate(day_items):
        history = full_history[: idx + 1]
        triggered: List[Dict[str, Any]] = []

        detectors = [
            detect_sudden_drop(today, baseline),
            detect_sustained_low(history),
            detect_social_withdrawal(today, baseline),
            detect_hyperactivity_spike(today, baseline),
            detect_regression(history),
            detect_gaze_avoidance(history),
        ]

        for maybe_alert in detectors:
            if maybe_alert:
                triggered.append(maybe_alert)

        triggered_categories = {t["category"] for t in triggered}

        for cat in [
            "SUDDEN_DROP",
            "SUSTAINED_LOW",
            "SOCIAL_WITHDRAWAL",
            "HYPERACTIVITY_SPIKE",
            "REGRESSION",
            "GAZE_AVOIDANCE",
        ]:
            if cat in triggered_categories:
                category_streak[cat] += 1
            else:
                category_streak[cat] = 0

        trend = [int(round(_safe_float(h.get("wellbeing", 0.0)))) for h in history[-5:]]
        traits = today.get("traits") or {}

        if traits:
            lowest_trait, lowest_trait_value = min(traits.items(), key=lambda kv: _safe_float(kv[1], 0.0))
            lowest_trait = str(lowest_trait)
            lowest_trait_value = int(round(_safe_float(lowest_trait_value, 0.0)))
        else:
            lowest_trait = "n/a"
            lowest_trait_value = 0

        for detail in triggered:
            alert_id = f"ALT_{len(alerts) + 1:03d}_{person_id}_{day_key.replace('-', '')}_{detail['category']}"
            baseline_w = int(round(_safe_float(baseline.get("wellbeing_mean", 0.0))))
            today_w = int(round(_safe_float(today.get("wellbeing", 0.0))))
            delta = today_w - baseline_w

            alert = {
                "alert_id": alert_id,
                "person_id": person_id,
                "person_name": info.get("name") or today.get("person_info", {}).get("name", person_id),
                "date": day_key,
                "severity": detail["severity"],
                "category": detail["category"],
                "title": detail["title"],
                "description": detail["description"],
                "baseline_wellbeing": baseline_w,
                "today_wellbeing": today_w,
                "delta": delta,
                "days_flagged_consecutively": category_streak[detail["category"]],
                "trend_last_5_days": trend,
                "lowest_trait": lowest_trait,
                "lowest_trait_value": lowest_trait_value,
                "recommended_action": detail["recommended_action"],
                "profile_image_b64": info.get("profile_image_b64", ""),
            }
            alerts.append(alert)

    return alerts


# ---------------------------------------------------------------------------
# HTML REPORT
# ---------------------------------------------------------------------------

def _sparkline_svg(values: List[int], width: int = 120, height: int = 34) -> str:
    if not values:
        return ""
    if len(values) == 1:
        values = [values[0], values[0]]

    min_v = min(values)
    max_v = max(values)
    span = (max_v - min_v) or 1
    step = width / (len(values) - 1)

    points = []
    for i, v in enumerate(values):
        x = round(i * step, 2)
        y = round(height - ((v - min_v) / span) * (height - 4) - 2, 2)
        points.append(f"{x},{y}")

    points_str = " ".join(points)
    return (
        f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' "
        f"xmlns='http://www.w3.org/2000/svg'>"
        f"<polyline fill='none' stroke='#0b5fff' stroke-width='2' points='{points_str}'/>"
        f"</svg>"
    )


def generate_alert_digest(
    alerts: List[Dict[str, Any]],
    absence_flags: List[Dict[str, Any]],
    school_summary: Dict[str, Any],
    output_path: Path,
):
    """
    Build an offline HTML counsellor digest with three sections.
    """
    sev_order = {"urgent": 0, "monitor": 1, "informational": 2}
    latest_date = max((a.get("date", "") for a in alerts), default="")
    todays_alerts = [a for a in alerts if a.get("date") == latest_date]
    todays_alerts.sort(key=lambda a: sev_order.get(a.get("severity", "informational"), 3))

    cards_html = []
    for a in todays_alerts:
        badge_class = "badge-urgent" if a["severity"] == "urgent" else "badge-monitor"
        spark = _sparkline_svg([int(x) for x in a.get("trend_last_5_days", [])])
        cards_html.append(
            """
            <div class='card'>
              <div class='row'>
                <h3>{person}</h3>
                <span class='badge {badge_class}'>{severity}</span>
              </div>
              <div class='meta'>{category}</div>
              <p>{desc}</p>
              <div class='spark'>{spark}</div>
            </div>
            """.format(
                person=html.escape(str(a.get("person_name", "Unknown"))),
                badge_class=badge_class,
                severity=html.escape(str(a.get("severity", "monitor")).upper()),
                category=html.escape(str(a.get("category", ""))),
                desc=html.escape(str(a.get("description", ""))),
                spark=spark,
            )
        )

    if not cards_html:
        cards_html = ["<p>No alerts in the latest day of data.</p>"]

    persistent_candidates: Dict[str, int] = {}
    for a in alerts:
        if int(a.get("days_flagged_consecutively", 0)) >= 3:
            name = str(a.get("person_name", "Unknown"))
            persistent_candidates[name] = max(persistent_candidates.get(name, 0), int(a.get("days_flagged_consecutively", 0)))

    persistent_html = []
    for name, streak in sorted(persistent_candidates.items(), key=lambda x: -x[1]):
        persistent_html.append(f"<li>{html.escape(name)} - flagged for {streak} consecutive days</li>")

    if not persistent_html:
        persistent_html = ["<li>No persistent 3+ day alert streaks.</li>"]

    absence_html = []
    for a in absence_flags:
        absence_html.append(
            "<li>{name} - absent {days} days (last seen {last_seen})</li>".format(
                name=html.escape(str(a.get("person_name", "Unknown"))),
                days=int(a.get("days_absent", 0)),
                last_seen=html.escape(str(a.get("last_seen_date", "unknown"))),
            )
        )

    if not absence_html:
        absence_html = ["<li>No active absence flags.</li>"]

    html_doc = f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width,initial-scale=1'>
  <title>Alert Digest</title>
  <style>
    :root {{
      --bg: #f6f8fc;
      --panel: #ffffff;
      --ink: #1d2733;
      --muted: #617083;
      --urgent: #c81e1e;
      --monitor: #b45309;
      --line: #d7dfeb;
    }}
    body {{ margin: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(145deg, #f6f8fc 0%, #eef2f9 60%, #e6edf8 100%); color: var(--ink); }}
    .wrap {{ max-width: 980px; margin: 0 auto; padding: 24px; }}
    .hero {{ background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 18px 20px; box-shadow: 0 8px 24px rgba(0,0,0,0.04); }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    .sub {{ color: var(--muted); font-size: 14px; }}
    .section {{ margin-top: 20px; background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 16px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid var(--line); border-radius: 10px; padding: 12px; background: #fbfdff; }}
    .row {{ display: flex; justify-content: space-between; align-items: center; gap: 8px; }}
    .meta {{ color: var(--muted); font-size: 12px; margin-top: 2px; }}
    .badge {{ font-size: 11px; font-weight: 700; padding: 4px 8px; border-radius: 999px; color: #fff; }}
    .badge-urgent {{ background: var(--urgent); }}
    .badge-monitor {{ background: var(--monitor); }}
    p {{ margin: 8px 0 10px; line-height: 1.4; }}
    .spark {{ min-height: 34px; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; }}
    .metric {{ border: 1px solid var(--line); border-radius: 10px; padding: 12px; background: #fbfdff; }}
    .metric .k {{ font-size: 12px; color: var(--muted); }}
    .metric .v {{ font-size: 22px; margin-top: 6px; font-weight: 700; }}
    ul {{ margin: 8px 0 0; padding-left: 18px; }}
  </style>
</head>
<body>
  <div class='wrap'>
    <div class='hero'>
      <h1>Counsellor Alert Digest</h1>
      <div class='sub'>School: {html.escape(str(SCHOOL))} | Latest data date: {html.escape(latest_date or 'n/a')}</div>
    </div>

    <div class='section'>
      <h2>Today's Alerts</h2>
      <div class='cards'>
        {''.join(cards_html)}
      </div>
    </div>

    <div class='section'>
      <h2>School Summary</h2>
      <div class='summary-grid'>
        <div class='metric'><div class='k'>Persons Tracked</div><div class='v'>{int(school_summary.get('total_persons_tracked', 0))}</div></div>
        <div class='metric'><div class='k'>Flagged Today</div><div class='v'>{int(school_summary.get('persons_flagged_today', 0))}</div></div>
        <div class='metric'><div class='k'>Flagged Yesterday</div><div class='v'>{int(school_summary.get('persons_flagged_yesterday', 0))}</div></div>
        <div class='metric'><div class='k'>Most Common Anomaly</div><div class='v' style='font-size:14px'>{html.escape(str(school_summary.get('most_common_anomaly_this_week', 'none')))}</div></div>
        <div class='metric'><div class='k'>Avg Wellbeing Today</div><div class='v'>{int(round(_safe_float(school_summary.get('school_avg_wellbeing_today', 0.0))))}</div></div>
      </div>
    </div>

    <div class='section'>
      <h2>Persistent Alerts (3+ Consecutive Days)</h2>
      <ul>
        {''.join(persistent_html)}
      </ul>
    </div>

    <div class='section'>
      <h2>Absence Flags</h2>
      <ul>
        {''.join(absence_html)}
      </ul>
    </div>
  </div>
</body>
</html>
"""

    output_path.write_text(html_doc, encoding="utf-8")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    daily_data = load_daily_data(DATA_DIR)
    all_dates = sorted(daily_data.keys())
    print(f"Loaded {len(daily_data)} days: {all_dates}")

    person_days: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    person_info: Dict[str, Dict[str, str]] = {}

    for d, persons in daily_data.items():
        for pid, pdata in persons.items():
            if not pdata.get("detected", True):
                continue
            person_days[pid][d] = pdata
            if pid not in person_info:
                person_info[pid] = pdata.get("person_info", {"name": pid, "profile_image_b64": ""})

    all_alerts: List[Dict[str, Any]] = []
    absence_flags: List[Dict[str, Any]] = []

    for pid, days in person_days.items():
        sorted_days = dict(sorted(days.items(), key=lambda x: x[0]))
        p_alerts = analyse_person(pid, sorted_days, person_info.get(pid, {}))
        all_alerts.extend(p_alerts)

        absence_alert = detect_absence_flag(all_dates, list(sorted_days.keys()))
        if absence_alert:
            absence_flags.append(
                {
                    "person_id": pid,
                    "person_name": person_info.get(pid, {}).get("name", pid),
                    "last_seen_date": absence_alert["last_seen_date"],
                    "days_absent": absence_alert["days_absent"],
                    "recommended_action": absence_alert["recommended_action"],
                }
            )

    sev_order = {"urgent": 0, "monitor": 1, "informational": 2}
    all_alerts.sort(key=lambda a: (a.get("date", ""), sev_order.get(a.get("severity", "informational"), 3), a.get("person_id", "")))

    latest_date = max(all_dates) if all_dates else ""
    prev_date = all_dates[-2] if len(all_dates) >= 2 else ""

    flagged_today_people = {a.get("person_id") for a in all_alerts if a.get("date") == latest_date}
    flagged_yesterday_people = {a.get("person_id") for a in all_alerts if a.get("date") == prev_date}

    if all_dates and daily_data.get(latest_date):
        wb_today = [_safe_float(p.get("wellbeing", 0.0)) for p in daily_data[latest_date].values()]
        avg_wb_today = _mean(wb_today)
    else:
        avg_wb_today = 0.0

    if latest_date:
        latest_dt = datetime.fromisoformat(latest_date)
        recent_cutoff = latest_dt.toordinal() - 6
        recent_alerts = [
            a for a in all_alerts if datetime.fromisoformat(a.get("date", latest_date)).toordinal() >= recent_cutoff
        ]
    else:
        recent_alerts = all_alerts

    cat_counter = Counter(a.get("category") for a in recent_alerts)
    top_category = cat_counter.most_common(1)[0][0] if cat_counter else "none"

    school_summary = {
        "total_persons_tracked": len(person_days),
        "persons_flagged_today": len(flagged_today_people),
        "persons_flagged_yesterday": len(flagged_yesterday_people),
        "most_common_anomaly_this_week": top_category,
        "school_avg_wellbeing_today": round(avg_wb_today, 2),
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
        "school_summary": school_summary,
    }

    with FEED_OUT.open("w", encoding="utf-8") as f:
        json.dump(feed, f, indent=2)

    generate_alert_digest(all_alerts, absence_flags, school_summary, REPORT_OUT)

    print()
    print("=" * 52)
    print(
        f" Alerts: {feed['alert_summary']['total_alerts']} total "
        f"({feed['alert_summary']['urgent']} urgent, {feed['alert_summary']['monitor']} monitor)"
    )
    print(f" Absence flags: {len(absence_flags)}")
    print(f" Report -> {REPORT_OUT}")
    print(f" JSON   -> {FEED_OUT}")
    print("=" * 52)
