
import json
from pathlib import Path
from datetime import datetime, date
from collections import defaultdict, Counter
import re
from html import escape
import statistics

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


def _to_float(value, default=0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _pick_first(d: dict, keys: list, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _normalize_person_record(person: dict) -> tuple[str, dict] | None:
    pid = _pick_first(person, ["person_id", "id", "student_id"])
    if not pid:
        return None

    traits_raw = person.get("traits", {}) if isinstance(person.get("traits", {}), dict) else {}
    social = _to_float(_pick_first(person, ["social_engagement"]), _to_float(traits_raw.get("social_engagement"), 0))
    physical = _to_float(_pick_first(person, ["physical_energy", "physical_activity", "energy"]), _to_float(traits_raw.get("physical_energy"), 0))
    movement = _to_float(_pick_first(person, ["movement_energy", "activity", "activity_level"]), _to_float(traits_raw.get("movement_energy"), 0))
    wellbeing = _to_float(_pick_first(person, ["wellbeing", "well_being", "wellbeing_score"]), 0)

    gaze = str(_pick_first(person, ["gaze_direction", "gaze", "gaze_contact"], "forward")).strip().lower()
    eye_contact_raw = _pick_first(person, ["eye_contact", "eye_contact_detected"], None)
    if eye_contact_raw is None:
        eye_contact_raw = gaze in {"forward", "center", "direct"}
    eye_contact = bool(eye_contact_raw)

    detected = _pick_first(person, ["detected", "detection_flag", "is_detected"], True)
    detected = bool(detected)

    name = _pick_first(person, ["person_name", "name"], str(pid))
    profile = str(_pick_first(person, ["profile_image_b64"], ""))
    person_info = person.get("person_info", {}) if isinstance(person.get("person_info", {}), dict) else {}
    if "name" not in person_info:
        person_info["name"] = name
    if "profile_image_b64" not in person_info:
        person_info["profile_image_b64"] = profile

    record = {
        "wellbeing": wellbeing,
        "traits": {
            "social_engagement": social,
            "physical_energy": physical,
            "movement_energy": movement,
        },
        "gaze_direction": gaze,
        "eye_contact": eye_contact,
        "detected": detected,
        "person_info": person_info,
    }
    return str(pid), record


def _infer_date_from_filename(path: Path) -> str:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", path.stem)
    if m:
        return m.group(1)
    try:
        return date.fromtimestamp(path.stat().st_mtime).isoformat()
    except OSError:
        return str(date.today())


def _build_alert_base(person_id: str, person_name: str, profile_image_b64: str,
                      day: dict, history: list, category: str, severity: str,
                      title: str, description: str, delta: float,
                      recommended_action: str) -> dict:
    trait_vals = day.get("traits", {})
    lowest_trait = "social_engagement"
    lowest_trait_value = _to_float(trait_vals.get("social_engagement"), 0)
    for t in ["physical_energy", "movement_energy", "social_engagement"]:
        v = _to_float(trait_vals.get(t), 0)
        if v < lowest_trait_value:
            lowest_trait = t
            lowest_trait_value = v

    trend = [int(round(_to_float(h.get("wellbeing"), 0))) for h in history[-5:]]
    return {
        "alert_id": "",
        "person_id": person_id,
        "person_name": person_name,
        "date": day.get("date", ""),
        "severity": severity,
        "_note_severity": "one of: urgent / monitor / informational",
        "category": category,
        "_note_category": "one of: SUDDEN_DROP / SUSTAINED_LOW / SOCIAL_WITHDRAWAL / HYPERACTIVITY_SPIKE / REGRESSION / GAZE_AVOIDANCE / ABSENCE_FLAG",
        "title": title,
        "description": description,
        "baseline_wellbeing": int(round(_to_float(day.get("baseline_wellbeing", 0)))),
        "today_wellbeing": int(round(_to_float(day.get("wellbeing", 0)))),
        "delta": int(round(delta)),
        "days_flagged_consecutively": 1,
        "trend_last_5_days": trend,
        "lowest_trait": lowest_trait,
        "lowest_trait_value": int(round(lowest_trait_value)),
        "recommended_action": recommended_action,
        "profile_image_b64": profile_image_b64,
    }


def _sparkline_svg(points: list[int]) -> str:
    if not points:
        return ""
    if len(points) == 1:
        points = [points[0], points[0]]

    w, h, pad = 120, 30, 2
    mn, mx = min(points), max(points)
    span = max(mx - mn, 1)
    coords = []
    for i, p in enumerate(points):
        x = pad + i * ((w - 2 * pad) / (len(points) - 1))
        y = h - pad - ((p - mn) / span) * (h - 2 * pad)
        coords.append(f"{x:.2f},{y:.2f}")
    poly = " ".join(coords)
    return (
        f"<svg viewBox='0 0 {w} {h}' width='{w}' height='{h}' aria-label='trend'>"
        f"<polyline fill='none' stroke='#1f6feb' stroke-width='2' points='{poly}' />"
        "</svg>"
    )


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
    if not folder.exists():
        return daily

    for fp in sorted(folder.glob("*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            payload = json.load(f)

        day_str = _pick_first(payload if isinstance(payload, dict) else {}, ["date", "analysis_date"], None)
        if not day_str:
            day_str = _infer_date_from_filename(fp)
        day_str = str(day_str)
        daily.setdefault(day_str, {})

        people = []
        if isinstance(payload, dict):
            if isinstance(payload.get("persons"), list):
                people = payload["persons"]
            elif isinstance(payload.get("students"), list):
                people = payload["students"]
            elif isinstance(payload.get("persons"), dict):
                people = list(payload["persons"].values())
            elif isinstance(payload.get("students"), dict):
                people = list(payload["students"].values())
            else:
                possible = []
                for _, v in payload.items():
                    if isinstance(v, dict) and _pick_first(v, ["person_id", "id", "student_id"]):
                        possible.append(v)
                people = possible
        elif isinstance(payload, list):
            people = payload

        for person in people:
            if not isinstance(person, dict):
                continue
            normalized = _normalize_person_record(person)
            if not normalized:
                continue
            pid, record = normalized
            if record.get("detected", True):
                daily[day_str][pid] = record

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
    if not history:
        return {
            "wellbeing_mean": 50.0,
            "wellbeing_std": 10.0,
            "trait_means": {
                "social_engagement": 50.0,
                "physical_energy": 50.0,
                "movement_energy": 50.0,
            },
            "avg_gaze": "forward",
        }

    window = history[:THRESHOLDS["baseline_window"]]
    wellbeing_arr = [_to_float(d.get("wellbeing"), 0) for d in window]
    trait_names = ["social_engagement", "physical_energy", "movement_energy"]
    trait_means = {}
    for t in trait_names:
        vals = [_to_float(d.get("traits", {}).get(t), 0) for d in window]
        trait_means[t] = float(statistics.fmean(vals)) if vals else 0.0

    gazes = [str(d.get("gaze_direction", "forward")).lower() for d in window]
    avg_gaze = Counter(gazes).most_common(1)[0][0] if gazes else "forward"

    return {
        "wellbeing_mean": float(statistics.fmean(wellbeing_arr)) if wellbeing_arr else 50.0,
        "wellbeing_std": float(statistics.pstdev(wellbeing_arr)) if len(wellbeing_arr) > 1 else 0.0,
        "trait_means": trait_means,
        "avg_gaze": avg_gaze,
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
    base = _to_float(baseline.get("wellbeing_mean"), 0)
    today_w = _to_float(today.get("wellbeing"), 0)
    std = _to_float(baseline.get("wellbeing_std"), 0)
    threshold = THRESHOLDS["sudden_drop_high_std_delta"] if std > THRESHOLDS["high_std_baseline"] else THRESHOLDS["sudden_drop_delta"]
    drop = base - today_w
    if drop >= threshold:
        severity = "urgent" if drop > 35 else "monitor"
        return {
            "category": "SUDDEN_DROP",
            "severity": severity,
            "title": "Sudden wellbeing drop detected",
            "description": f"Wellbeing dropped from baseline {base:.1f} to {today_w:.1f} ({drop:.1f} points).",
            "delta": today_w - base,
            "recommended_action": "Schedule pastoral check-in today" if severity == "urgent" else "Monitor closely over next 48 hours",
        }
    return None


def detect_sustained_low(history: list) -> dict | None:
    """
    Check the last sustained_low_days entries in history.
    If all have wellbeing < sustained_low_score → alert.
    Severity: urgent.
    TODO: implement
    """
    n = THRESHOLDS["sustained_low_days"]
    if len(history) < n:
        return None
    recent = history[-n:]
    lows = [_to_float(d.get("wellbeing"), 0) < THRESHOLDS["sustained_low_score"] for d in recent]
    if all(lows):
        return {
            "category": "SUSTAINED_LOW",
            "severity": "urgent",
            "title": "Sustained low wellbeing pattern",
            "description": f"Wellbeing has remained below {THRESHOLDS['sustained_low_score']} for {n} consecutive days.",
            "delta": _to_float(recent[-1].get("wellbeing"), 0) - THRESHOLDS["sustained_low_score"],
            "recommended_action": "Immediate counsellor follow-up recommended",
        }
    return None


def detect_social_withdrawal(today: dict, baseline: dict) -> dict | None:
    """
    social_engagement dropped >= social_withdrawal_delta AND
    today's gaze_direction is "down" or "side".
    Severity: monitor.
    TODO: implement
    """
    base_social = _to_float(baseline.get("trait_means", {}).get("social_engagement"), 0)
    today_social = _to_float(today.get("traits", {}).get("social_engagement"), 0)
    social_drop = base_social - today_social
    gaze = str(today.get("gaze_direction", "forward")).lower()
    if social_drop >= THRESHOLDS["social_withdrawal_delta"] and gaze in {"down", "downward", "side"}:
        return {
            "category": "SOCIAL_WITHDRAWAL",
            "severity": "monitor",
            "title": "Social withdrawal pattern detected",
            "description": f"Social engagement dropped by {social_drop:.1f} points and gaze was mostly {gaze}.",
            "delta": today_social - base_social,
            "recommended_action": "Use low-pressure peer interaction support",
        }
    return None


def detect_hyperactivity_spike(today: dict, baseline: dict) -> dict | None:
    """
    (today.physical_energy + today.movement_energy) minus
    (baseline.physical_energy_mean + baseline.movement_energy_mean) >= hyperactivity_delta.
    Severity: monitor.
    TODO: implement
    """
    today_combined = (
        _to_float(today.get("traits", {}).get("physical_energy"), 0)
        + _to_float(today.get("traits", {}).get("movement_energy"), 0)
    )
    base_combined = (
        _to_float(baseline.get("trait_means", {}).get("physical_energy"), 0)
        + _to_float(baseline.get("trait_means", {}).get("movement_energy"), 0)
    )
    spike = today_combined - base_combined
    if spike >= THRESHOLDS["hyperactivity_delta"]:
        return {
            "category": "HYPERACTIVITY_SPIKE",
            "severity": "monitor",
            "title": "Hyperactivity spike detected",
            "description": f"Combined activity rose by {spike:.1f} points above baseline.",
            "delta": spike,
            "recommended_action": "Use structured movement breaks and re-check tomorrow",
        }
    return None


def detect_regression(history: list) -> dict | None:
    """
    Find if the last regression_recover_days entries were all improving (each > previous),
    then today dropped > regression_drop.
    Severity: monitor.
    TODO: implement
    """
    recover = THRESHOLDS["regression_recover_days"]
    if len(history) < recover + 2:
        return None

    window = history[-(recover + 2):]
    vals = [_to_float(d.get("wellbeing"), 0) for d in window]
    improving = all(vals[i] < vals[i + 1] for i in range(recover))
    drop = vals[recover] - vals[recover + 1]
    if improving and drop > THRESHOLDS["regression_drop"]:
        return {
            "category": "REGRESSION",
            "severity": "monitor",
            "title": "Regression after recovery",
            "description": f"After {recover} improving days, wellbeing dropped by {drop:.1f} points in one day.",
            "delta": -drop,
            "recommended_action": "Review recent stressors and reinforce previous supports",
        }
    return None


def detect_gaze_avoidance(history: list) -> dict | None:
    """
    Last gaze_avoidance_days entries all have eye_contact == False (or missing).
    Severity: monitor.
    TODO: implement
    """
    n = THRESHOLDS["gaze_avoidance_days"]
    if len(history) < n:
        return None
    recent = history[-n:]
    no_eye_contact = [not bool(d.get("eye_contact", False)) for d in recent]
    if all(no_eye_contact):
        return {
            "category": "GAZE_AVOIDANCE",
            "severity": "monitor",
            "title": "Gaze avoidance pattern",
            "description": f"No eye contact detected for {n} consecutive days.",
            "delta": 0,
            "recommended_action": "Use trust-building one-on-one engagement",
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
    alerts = []
    if not sorted_days:
        return alerts

    person_name = info.get("name", person_id)
    profile_image_b64 = info.get("profile_image_b64", "")

    items = []
    for d, pdata in sorted(sorted_days.items()):
        row = dict(pdata)
        row["date"] = d
        items.append(row)

    baseline = compute_baseline(items)
    baseline_w = _to_float(baseline.get("wellbeing_mean"), 0)
    baseline_window = min(len(items), THRESHOLDS["baseline_window"])

    for i in range(baseline_window, len(items)):
        today = dict(items[i])
        history = items[: i + 1]
        today["baseline_wellbeing"] = baseline_w

        detectors = [
            detect_sudden_drop(today, baseline),
            detect_sustained_low(history),
            detect_social_withdrawal(today, baseline),
            detect_hyperactivity_spike(today, baseline),
            detect_regression(history),
            detect_gaze_avoidance(history),
        ]

        for det in detectors:
            if not det:
                continue
            alert = _build_alert_base(
                person_id=person_id,
                person_name=person_name,
                profile_image_b64=profile_image_b64,
                day=today,
                history=history,
                category=det["category"],
                severity=det["severity"],
                title=det["title"],
                description=det["description"],
                delta=_to_float(det["delta"]),
                recommended_action=det["recommended_action"],
            )
            alerts.append(alert)

    alerts.sort(key=lambda a: a.get("date", ""))
    day_set = set()
    consec = 0
    prev_date = None
    for a in alerts:
        d = a["date"]
        if prev_date == d:
            a["days_flagged_consecutively"] = consec
            continue
        if prev_date is not None:
            prev_dt = datetime.fromisoformat(prev_date).date()
            curr_dt = datetime.fromisoformat(d).date()
            if (curr_dt - prev_dt).days == 1:
                consec += 1
            else:
                consec = 1
        else:
            consec = 1
        prev_date = d
        day_set.add(d)
        a["days_flagged_consecutively"] = consec

    for idx, alert in enumerate(alerts, start=1):
        alert["alert_id"] = f"ALT_{idx:04d}_{person_id}"

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
    sev_order = {"urgent": 0, "monitor": 1, "informational": 2}
    sorted_alerts = sorted(
        alerts,
        key=lambda a: (a.get("date", ""), sev_order.get(a.get("severity", "informational"), 3)),
        reverse=True,
    )
    latest_date = max((a.get("date", "") for a in alerts), default="")
    todays_alerts = [a for a in sorted_alerts if a.get("date", "") == latest_date]

    cards = []
    for a in todays_alerts:
        badge_class = "urgent" if a.get("severity") == "urgent" else "monitor"
        spark = _sparkline_svg(a.get("trend_last_5_days", []))
        cards.append(
            "<article class='card'>"
            f"<div class='row'><h3>{escape(a.get('person_name', 'Unknown'))}</h3>"
            f"<span class='badge {badge_class}'>{escape(a.get('category', ''))}</span></div>"
            f"<p class='date'>{escape(a.get('date', ''))}</p>"
            f"<p>{escape(a.get('description', ''))}</p>"
            f"<div class='spark'>{spark}</div>"
            "</article>"
        )

    persistent = [a for a in alerts if int(a.get("days_flagged_consecutively", 1)) >= 3]
    persistent_html = "".join(
        (
            "<li>"
            f"{escape(a.get('person_name', 'Unknown'))} - {escape(a.get('category', ''))} "
            f"({int(a.get('days_flagged_consecutively', 1))} days)"
            "</li>"
        )
        for a in persistent
    ) or "<li>No persistent alerts yet.</li>"

    absence_html = "".join(
        (
            "<li>"
            f"{escape(x.get('person_name', 'Unknown'))} absent for {int(x.get('days_absent', 0))} days "
            f"(last seen {escape(x.get('last_seen_date', 'unknown'))})"
            "</li>"
        )
        for x in absence_flags
    ) or "<li>No active absence flags.</li>"

    html = f"""<!doctype html>
<html lang='en'>
<head>
    <meta charset='utf-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1'>
    <title>Student Alert Report</title>
    <style>
        :root {{ --bg:#f8fbff; --ink:#0b1f33; --card:#ffffff; --accent:#1f6feb; --urgent:#b42318; --monitor:#b54708; --muted:#667085; }}
        body {{ margin:0; font-family:Segoe UI, Tahoma, sans-serif; color:var(--ink); background:radial-gradient(circle at 10% 10%, #dbeafe 0%, #f8fbff 45%, #ecfdf3 100%); }}
        main {{ max-width:980px; margin:0 auto; padding:24px; }}
        h1 {{ margin:0 0 8px; }}
        h2 {{ margin:28px 0 12px; }}
        .sub {{ color:var(--muted); margin:0; }}
        .grid {{ display:grid; gap:12px; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); }}
        .card {{ background:var(--card); border-radius:12px; padding:14px; box-shadow:0 8px 24px rgba(16,24,40,.08); }}
        .row {{ display:flex; align-items:center; justify-content:space-between; gap:8px; }}
        .badge {{ border-radius:999px; padding:3px 10px; font-size:12px; font-weight:600; color:#fff; }}
        .badge.urgent {{ background:var(--urgent); }}
        .badge.monitor {{ background:var(--monitor); }}
        .date {{ margin:4px 0 8px; color:var(--muted); font-size:13px; }}
        .stats {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:10px; }}
        .stat {{ background:#eef4ff; border-radius:10px; padding:12px; }}
        ul {{ margin:0; padding-left:18px; }}
        .spark {{ min-height:34px; }}
    </style>
</head>
<body>
    <main>
        <h1>Student Alert Report</h1>
        <p class='sub'>Latest date in data: {escape(latest_date or 'n/a')}</p>

        <h2>Today's Alerts</h2>
        <div class='grid'>
            {''.join(cards) if cards else "<article class='card'><p>No alerts for latest day.</p></article>"}
        </div>

        <h2>School Summary</h2>
        <div class='stats'>
            <div class='stat'><strong>{int(school_summary.get('total_persons_tracked', 0))}</strong><br>Total persons tracked</div>
            <div class='stat'><strong>{int(school_summary.get('persons_flagged_today', 0))}</strong><br>Persons flagged today</div>
            <div class='stat'><strong>{int(school_summary.get('persons_flagged_yesterday', 0))}</strong><br>Persons flagged yesterday</div>
            <div class='stat'><strong>{escape(str(school_summary.get('most_common_anomaly_this_week', 'none')))}</strong><br>Most common anomaly this week</div>
        </div>

        <h2>Persistent Alerts (3+ Days)</h2>
        <ul>{persistent_html}</ul>

        <h2>Absence Flags</h2>
        <ul>{absence_html}</ul>
    </main>
</body>
</html>
"""
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
    flagged_today = sum(1 for a in all_alerts if a.get("date") == today_str)
    day_before = all_dates[-2] if len(all_dates) > 1 else ""
    flagged_yesterday = sum(1 for a in all_alerts if a.get("date") == day_before)
    cat_counter   = Counter(a.get("category") for a in all_alerts)
    top_category  = cat_counter.most_common(1)[0][0] if cat_counter else "none"

    today_people = daily_data.get(today_str, {}) if today_str else {}
    school_avg_today = 0
    if today_people:
        vals = [_to_float(v.get("wellbeing"), 0) for v in today_people.values()]
        school_avg_today = int(round(float(statistics.fmean(vals)))) if vals else 0

    school_summary = {
        "total_persons_tracked":       len(person_days),
        "persons_flagged_today":       flagged_today,
        "persons_flagged_yesterday":   flagged_yesterday,
        "most_common_anomaly_this_week": top_category,
        "school_avg_wellbeing_today":  school_avg_today,
    }

    feed = {
        "_readme": "Your alert_feed.json must match this structure exactly. It is returned by the /get_alerts Flask endpoint in Sentio Mind. Do not add, remove, or rename any top-level key.",
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
