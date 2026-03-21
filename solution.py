"""
solution.py
Sentio Mind · Project 5 · Behavioral Anomaly & Early Distress Detection

Complete implementation of the anomaly_detection.py template.
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
    """
    daily = {}
    for fp in sorted(folder.glob("*.json")):
        try:
            raw = json.loads(fp.read_text())
        except Exception as e:
            print(f"  [WARN] Could not parse {fp.name}: {e}")
            continue

        # Support two file layouts:
        # Layout A: { "date": "YYYY-MM-DD", "persons": { pid: {...} } }
        # Layout B: { "YYYY-MM-DD": { pid: {...} } }  (flat date-keyed)
        if "date" in raw and "persons" in raw:
            day_str  = raw["date"]
            persons  = raw["persons"]
            daily[day_str] = persons
        else:
            # Try to infer date from filename or top-level keys
            for key, val in raw.items():
                # If key looks like a date, treat it as Layout B
                try:
                    datetime.strptime(key, "%Y-%m-%d")
                    daily[key] = val
                except ValueError:
                    # Not a date key — skip
                    pass

    return daily


# ---------------------------------------------------------------------------
# BASELINE
# ---------------------------------------------------------------------------

def compute_baseline(history: list) -> dict:
    """
    history: list of daily dicts (oldest first), each has at minimum:
      { wellbeing: int, traits: {}, gaze_direction: str }

    Use first THRESHOLDS['baseline_window'] days (or all if fewer available).
    Return:
      { wellbeing_mean, wellbeing_std, trait_means: {}, avg_gaze: str,
        social_engagement_mean, physical_energy_mean, movement_energy_mean }
    """
    window = history[: THRESHOLDS["baseline_window"]]
    if not window:
        return {
            "wellbeing_mean": 50.0,
            "wellbeing_std":  10.0,
            "trait_means":    {},
            "avg_gaze":       "forward",
            "social_engagement_mean": 50.0,
            "physical_energy_mean":   50.0,
            "movement_energy_mean":   50.0,
        }

    wbs = [d.get("wellbeing", 50) for d in window]
    wellbeing_mean = float(np.mean(wbs))
    wellbeing_std  = float(np.std(wbs)) if len(wbs) > 1 else 0.0

    # Trait means
    all_trait_keys = set()
    for d in window:
        all_trait_keys.update(d.get("traits", {}).keys())

    trait_means = {}
    for tk in all_trait_keys:
        vals = [d["traits"][tk] for d in window if tk in d.get("traits", {})]
        trait_means[tk] = float(np.mean(vals)) if vals else 50.0

    # Most common gaze direction
    gazes   = [d.get("gaze_direction", "forward") for d in window]
    avg_gaze = Counter(gazes).most_common(1)[0][0]

    return {
        "wellbeing_mean":          wellbeing_mean,
        "wellbeing_std":           wellbeing_std,
        "trait_means":             trait_means,
        "avg_gaze":                avg_gaze,
        "social_engagement_mean":  trait_means.get("social_engagement", 50.0),
        "physical_energy_mean":    trait_means.get("physical_energy",   50.0),
        "movement_energy_mean":    trait_means.get("movement_energy",   50.0),
    }


# ---------------------------------------------------------------------------
# ANOMALY DETECTORS  — each returns an alert dict or None
# ---------------------------------------------------------------------------

def detect_sudden_drop(today: dict, baseline: dict) -> dict | None:
    """
    today['wellbeing'] vs baseline['wellbeing_mean'].
    If baseline_std > 15, raise threshold to sudden_drop_high_std_delta.
    Severity: delta > 35 = urgent, else monitor.
    """
    wb_today    = today.get("wellbeing", 50)
    wb_baseline = baseline["wellbeing_mean"]
    std         = baseline["wellbeing_std"]

    threshold = (
        THRESHOLDS["sudden_drop_high_std_delta"]
        if std > THRESHOLDS["high_std_baseline"]
        else THRESHOLDS["sudden_drop_delta"]
    )

    delta = wb_baseline - wb_today
    if delta < threshold:
        return None

    severity = "urgent" if delta > 35 else "monitor"

    # Find lowest trait today
    traits = today.get("traits", {})
    lowest_trait       = min(traits, key=traits.get) if traits else "unknown"
    lowest_trait_value = traits.get(lowest_trait, 0)

    return {
        "category":           "SUDDEN_DROP",
        "severity":           severity,
        "title":              "Sudden wellbeing drop detected",
        "description": (
            f"Wellbeing dropped from a baseline of {wb_baseline:.0f} to {wb_today} today "
            f"— a {delta:.0f}-point fall. Lowest trait: {lowest_trait} at {lowest_trait_value}. "
            f"Dominant gaze: {today.get('gaze_direction', 'unknown')}."
        ),
        "baseline_wellbeing": round(wb_baseline),
        "today_wellbeing":    wb_today,
        "delta":              -round(delta),
        "lowest_trait":       lowest_trait,
        "lowest_trait_value": lowest_trait_value,
        "recommended_action": "Schedule pastoral check-in today",
    }


def detect_sustained_low(history: list) -> dict | None:
    """
    Check the last sustained_low_days entries in history.
    If all have wellbeing < sustained_low_score → alert.
    Severity: urgent.
    """
    n = THRESHOLDS["sustained_low_days"]
    if len(history) < n:
        return None

    recent = history[-n:]
    threshold = THRESHOLDS["sustained_low_score"]
    if not all(d.get("wellbeing", 100) < threshold for d in recent):
        return None

    scores = [d.get("wellbeing", 0) for d in recent]
    avg    = sum(scores) / len(scores)

    traits      = history[-1].get("traits", {})
    lowest_trait       = min(traits, key=traits.get) if traits else "unknown"
    lowest_trait_value = traits.get(lowest_trait, 0)

    return {
        "category":           "SUSTAINED_LOW",
        "severity":           "urgent",
        "title":              f"Sustained low wellbeing for {n}+ days",
        "description": (
            f"Wellbeing has remained below {threshold} for the last {n} consecutive days "
            f"(scores: {', '.join(str(s) for s in scores)}). "
            f"Average: {avg:.0f}. Lowest trait today: {lowest_trait} at {lowest_trait_value}."
        ),
        "baseline_wellbeing": 0,
        "today_wellbeing":    history[-1].get("wellbeing", 0),
        "delta":              0,
        "lowest_trait":       lowest_trait,
        "lowest_trait_value": lowest_trait_value,
        "recommended_action": "Urgent: arrange wellbeing support meeting",
    }


def detect_social_withdrawal(today: dict, baseline: dict) -> dict | None:
    """
    social_engagement dropped >= social_withdrawal_delta AND
    today's gaze_direction is "down" or "side".
    Severity: monitor.
    """
    se_today    = today.get("traits", {}).get("social_engagement", 50)
    se_baseline = baseline.get("social_engagement_mean", 50.0)
    delta       = se_baseline - se_today
    gaze        = today.get("gaze_direction", "forward")

    if delta < THRESHOLDS["social_withdrawal_delta"]:
        return None
    if gaze not in ("down", "side"):
        return None

    traits      = today.get("traits", {})
    lowest_trait       = min(traits, key=traits.get) if traits else "unknown"
    lowest_trait_value = traits.get(lowest_trait, 0)

    return {
        "category":           "SOCIAL_WITHDRAWAL",
        "severity":           "monitor",
        "title":              "Social withdrawal detected",
        "description": (
            f"Social engagement dropped {delta:.0f} points below baseline "
            f"(baseline: {se_baseline:.0f}, today: {se_today}). "
            f"Gaze is predominantly '{gaze}', suggesting avoidance."
        ),
        "baseline_wellbeing": round(baseline["wellbeing_mean"]),
        "today_wellbeing":    today.get("wellbeing", 0),
        "delta":              -round(delta),
        "lowest_trait":       lowest_trait,
        "lowest_trait_value": lowest_trait_value,
        "recommended_action": "Observe in group settings; consider gentle outreach",
    }


def detect_hyperactivity_spike(today: dict, baseline: dict) -> dict | None:
    """
    (today.physical_energy + today.movement_energy) minus
    (baseline.physical_energy_mean + baseline.movement_energy_mean) >= hyperactivity_delta.
    Severity: monitor.
    """
    pe_today = today.get("traits", {}).get("physical_energy", 50)
    me_today = today.get("traits", {}).get("movement_energy", 50)
    pe_base  = baseline.get("physical_energy_mean", 50.0)
    me_base  = baseline.get("movement_energy_mean", 50.0)

    combined_today    = pe_today + me_today
    combined_baseline = pe_base + me_base
    delta = combined_today - combined_baseline

    if delta < THRESHOLDS["hyperactivity_delta"]:
        return None

    traits      = today.get("traits", {})
    lowest_trait       = min(traits, key=traits.get) if traits else "unknown"
    lowest_trait_value = traits.get(lowest_trait, 0)

    return {
        "category":           "HYPERACTIVITY_SPIKE",
        "severity":           "monitor",
        "title":              "Hyperactivity spike detected",
        "description": (
            f"Combined physical + movement energy is {delta:.0f} points above baseline "
            f"(baseline: {combined_baseline:.0f}, today: {combined_today}). "
            f"Physical energy: {pe_today}, movement energy: {me_today}."
        ),
        "baseline_wellbeing": round(baseline["wellbeing_mean"]),
        "today_wellbeing":    today.get("wellbeing", 0),
        "delta":              round(delta),
        "lowest_trait":       lowest_trait,
        "lowest_trait_value": lowest_trait_value,
        "recommended_action": "Monitor for impulsivity; check sleep/diet patterns",
    }


def detect_regression(history: list) -> dict | None:
    """
    Find if the last regression_recover_days entries were all improving (each > previous),
    then today dropped > regression_drop.
    Severity: monitor.
    """
    n = THRESHOLDS["regression_recover_days"]
    # Need at least n+2 entries: n recovery days + 1 preceding + today
    if len(history) < n + 2:
        return None

    today_score = history[-1].get("wellbeing", 50)
    prev_score  = history[-2].get("wellbeing", 50)
    drop = prev_score - today_score

    if drop <= THRESHOLDS["regression_drop"]:
        return None

    # Check that the n days before today were all improving (ascending)
    recovery_window = history[-(n + 2): -1]   # n+1 entries up to and including prev
    scores = [d.get("wellbeing", 50) for d in recovery_window]
    if not all(scores[i] < scores[i + 1] for i in range(len(scores) - 1)):
        return None

    traits      = history[-1].get("traits", {})
    lowest_trait       = min(traits, key=traits.get) if traits else "unknown"
    lowest_trait_value = traits.get(lowest_trait, 0)

    return {
        "category":           "REGRESSION",
        "severity":           "monitor",
        "title":              "Regression after recovery",
        "description": (
            f"After {n}+ days of improving wellbeing (scores: "
            f"{', '.join(str(s) for s in scores)}), "
            f"today dropped {drop} points to {today_score}. "
            f"Risk of deeper relapse."
        ),
        "baseline_wellbeing": scores[0],
        "today_wellbeing":    today_score,
        "delta":              -drop,
        "lowest_trait":       lowest_trait,
        "lowest_trait_value": lowest_trait_value,
        "recommended_action": "Follow up — relapse after recovery needs support",
    }


def detect_gaze_avoidance(history: list) -> dict | None:
    """
    Last gaze_avoidance_days entries all have eye_contact == False (or missing).
    Severity: monitor.
    """
    n = THRESHOLDS["gaze_avoidance_days"]
    if len(history) < n:
        return None

    recent = history[-n:]
    # eye_contact should be False for all; treat missing as False too
    if not all(d.get("eye_contact", False) is False for d in recent):
        return None

    traits      = history[-1].get("traits", {})
    lowest_trait       = min(traits, key=traits.get) if traits else "unknown"
    lowest_trait_value = traits.get(lowest_trait, 0)

    gazes = [d.get("gaze_direction", "down") for d in recent]

    return {
        "category":           "GAZE_AVOIDANCE",
        "severity":           "monitor",
        "title":              f"No eye contact for {n}+ consecutive days",
        "description": (
            f"No eye contact detected for the past {n} days. "
            f"Gaze directions: {', '.join(gazes)}. "
            f"May indicate withdrawal, anxiety or low confidence."
        ),
        "baseline_wellbeing": 0,
        "today_wellbeing":    history[-1].get("wellbeing", 0),
        "delta":              0,
        "lowest_trait":       lowest_trait,
        "lowest_trait_value": lowest_trait_value,
        "recommended_action": "Gentle 1-on-1 conversation recommended",
    }


# ---------------------------------------------------------------------------
# ANALYSE ONE PERSON
# ---------------------------------------------------------------------------

def analyse_person(person_id: str, sorted_days: dict, info: dict) -> list:
    """
    sorted_days: { "YYYY-MM-DD": person_data_dict } — keys in date order
    info: { name, profile_image_b64, ... }

    Build history list, compute baseline, run all detectors.
    Return list of alert dicts matching anomaly_detection.json schema.
    """
    alerts      = []
    date_keys   = sorted(sorted_days.keys())
    history     = [sorted_days[d] for d in date_keys]
    person_name = info.get("name", person_id)

    if len(history) == 0:
        return alerts

    baseline = compute_baseline(history)
    today    = history[-1]
    today_str = date_keys[-1]

    # Track consecutive flagged days per category
    def consecutive_days_flagged(category: str) -> int:
        """Count how many of the last N days would trigger this category."""
        count = 0
        for d in reversed(date_keys[:-1]):
            h_slice = history[: date_keys.index(d) + 1]
            # Quick heuristic: check wellbeing trend
            if category in ("SUDDEN_DROP", "SUSTAINED_LOW"):
                if sorted_days[d].get("wellbeing", 100) < THRESHOLDS["sustained_low_score"]:
                    count += 1
                else:
                    break
            elif category == "GAZE_AVOIDANCE":
                if not sorted_days[d].get("eye_contact", True):
                    count += 1
                else:
                    break
            else:
                break
        return count + 1

    def get_trend() -> list:
        """Last up-to-5 wellbeing scores."""
        scores = [d.get("wellbeing", 0) for d in history]
        return scores[-5:]

    alert_counter = [0]

    def build_alert(det: dict) -> dict:
        alert_counter[0] += 1
        cat  = det["category"]
        days = consecutive_days_flagged(cat)
        return {
            "alert_id":                 f"ALT_{alert_counter[0]:03d}_{person_id[-4:]}",
            "person_id":                person_id,
            "person_name":              person_name,
            "date":                     today_str,
            "severity":                 det["severity"],
            "category":                 cat,
            "title":                    det["title"],
            "description":              det["description"],
            "baseline_wellbeing":       det.get("baseline_wellbeing", round(baseline["wellbeing_mean"])),
            "today_wellbeing":          det.get("today_wellbeing", today.get("wellbeing", 0)),
            "delta":                    det.get("delta", 0),
            "days_flagged_consecutively": days,
            "trend_last_5_days":        get_trend(),
            "lowest_trait":             det.get("lowest_trait", "unknown"),
            "lowest_trait_value":       det.get("lowest_trait_value", 0),
            "recommended_action":       det.get("recommended_action", "Monitor"),
            "profile_image_b64":        info.get("profile_image_b64", ""),
        }

    # Run all detectors
    detectors_today = [
        detect_sudden_drop(today, baseline),
        detect_social_withdrawal(today, baseline),
        detect_hyperactivity_spike(today, baseline),
    ]
    detectors_history = [
        detect_sustained_low(history),
        detect_regression(history),
        detect_gaze_avoidance(history),
    ]

    for det in detectors_today + detectors_history:
        if det is not None:
            alerts.append(build_alert(det))

    # ---------------------------------------------------------------------------
    # BONUS: Peer-comparison anomaly — handled in main after all persons analysed
    # ---------------------------------------------------------------------------

    return alerts


# ---------------------------------------------------------------------------
# PEER-COMPARISON ANOMALY (BONUS)
# ---------------------------------------------------------------------------

def detect_peer_comparison(daily_data: dict, all_dates: list, person_days: dict,
                            person_info: dict) -> list:
    """
    Flag a person whose today's wellbeing is > 2 std below class average,
    even if their personal baseline is also low.
    Returns list of alert dicts.
    """
    alerts       = []
    today_str    = all_dates[-1] if all_dates else None
    if today_str is None or today_str not in daily_data:
        return alerts

    today_persons = daily_data[today_str]
    scores = {pid: pdata.get("wellbeing", 50)
              for pid, pdata in today_persons.items()}

    if len(scores) < 2:
        return alerts

    arr  = np.array(list(scores.values()), dtype=float)
    mean = float(np.mean(arr))
    std  = float(np.std(arr))

    if std < 1:
        return alerts

    for pid, wb in scores.items():
        if (mean - wb) > 2 * std:
            pdata = today_persons[pid]
            info  = person_info.get(pid, {"name": pid, "profile_image_b64": ""})
            name  = info.get("name", pid)
            traits = pdata.get("traits", {})
            lowest_trait       = min(traits, key=traits.get) if traits else "unknown"
            lowest_trait_value = traits.get(lowest_trait, 0)
            history = [person_days[pid][d]
                       for d in sorted(person_days[pid].keys())]
            alerts.append({
                "alert_id":                   f"ALT_PEER_{pid[-4:]}",
                "person_id":                  pid,
                "person_name":                name,
                "date":                       today_str,
                "severity":                   "monitor",
                "category":                   "PEER_COMPARISON",
                "title":                      "Significant peer-comparison anomaly",
                "description": (
                    f"{name}'s wellbeing ({wb}) is more than 2 standard deviations below "
                    f"the class average today (class avg: {mean:.0f}, std: {std:.0f})."
                ),
                "baseline_wellbeing":         round(mean),
                "today_wellbeing":            wb,
                "delta":                      round(wb - mean),
                "days_flagged_consecutively": 1,
                "trend_last_5_days":          [d.get("wellbeing", 0) for d in history[-5:]],
                "lowest_trait":               lowest_trait,
                "lowest_trait_value":         lowest_trait_value,
                "recommended_action":         "Check in — person is notably below class average today",
                "profile_image_b64":          info.get("profile_image_b64", ""),
            })

    return alerts


# ---------------------------------------------------------------------------
# HTML REPORT
# ---------------------------------------------------------------------------

def _sparkline_svg(scores: list, width: int = 80, height: int = 28) -> str:
    """Return an inline SVG sparkline for a list of scores (0-100)."""
    if not scores:
        return ""
    n = len(scores)
    pad = 3
    usable_w = width  - 2 * pad
    usable_h = height - 2 * pad
    min_s = min(scores)
    max_s = max(scores)
    rng   = max(max_s - min_s, 1)

    def px(i, s):
        x = pad + (i / max(n - 1, 1)) * usable_w
        y = pad + usable_h - ((s - min_s) / rng) * usable_h
        return f"{x:.1f},{y:.1f}"

    points = " ".join(px(i, s) for i, s in enumerate(scores))
    last   = scores[-1]
    dot_x, dot_y = [float(v) for v in px(n - 1, last).split(",")]

    colour = "#ef4444" if last < 45 else "#f59e0b" if last < 65 else "#22c55e"

    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'<polyline points="{points}" fill="none" stroke="{colour}" stroke-width="2" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
        f'<circle cx="{dot_x:.1f}" cy="{dot_y:.1f}" r="3" fill="{colour}"/>'
        f'</svg>'
    )


def _severity_badge(severity: str) -> str:
    colours = {
        "urgent":        ("bg:#fee2e2", "text:#991b1b", "🔴"),
        "monitor":       ("bg:#fef3c7", "text:#92400e", "🟡"),
        "informational": ("bg:#dbeafe", "text:#1e40af", "🔵"),
    }
    bg, fg, icon = colours.get(severity, ("bg:#f3f4f6", "text:#374151", "⚪"))
    bg_c  = bg.split(":")[1]
    fg_c  = fg.split(":")[1]
    label = severity.upper()
    return (
        f'<span style="display:inline-block;padding:2px 10px;border-radius:12px;'
        f'background:{bg_c};color:{fg_c};font-size:11px;font-weight:700;'
        f'letter-spacing:.5px;">{icon} {label}</span>'
    )


def generate_alert_digest(alerts: list, absence_flags: list,
                           school_summary: dict, output_path: Path):
    """
    Self-contained HTML counsellor report.
    Section 1: Today's alerts (sorted by severity).
    Section 2: School summary.
    Section 3: Persons flagged 3+ consecutive days.
    """
    today_str  = str(date.today())
    gen_time   = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Persistent alerts — flagged 3+ consecutive days
    persistent = [a for a in alerts if a.get("days_flagged_consecutively", 0) >= 3]

    # ── CSS ──────────────────────────────────────────────────────────────────
    css = """
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #f8fafc; color: #1e293b; line-height: 1.6;
    }
    .page { max-width: 900px; margin: 0 auto; padding: 32px 20px 64px; }
    header { margin-bottom: 32px; border-bottom: 2px solid #e2e8f0; padding-bottom: 20px; }
    header h1 { font-size: 26px; font-weight: 800; color: #0f172a; }
    header p  { color: #64748b; font-size: 13px; margin-top: 4px; }
    h2 { font-size: 18px; font-weight: 700; color: #0f172a; margin: 32px 0 14px; }
    h3 { font-size: 15px; font-weight: 600; color: #334155; }
    .card {
        background: #fff; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 18px 20px; margin-bottom: 14px;
        display: flex; gap: 18px; align-items: flex-start;
        box-shadow: 0 1px 3px rgba(0,0,0,.06);
    }
    .card-left  { flex: 1; min-width: 0; }
    .card-right { flex-shrink: 0; text-align: right; }
    .card-name  { font-weight: 700; font-size: 15px; margin-bottom: 4px; }
    .card-title { font-size: 14px; color: #475569; margin: 4px 0; }
    .card-desc  { font-size: 13px; color: #64748b; margin-top: 6px; }
    .card-action{ font-size: 12px; color: #0ea5e9; margin-top: 8px; font-style: italic; }
    .card-score { font-size: 28px; font-weight: 800; margin-bottom: 4px; }
    .score-urgent  { color: #ef4444; }
    .score-monitor { color: #f59e0b; }
    .score-ok      { color: #22c55e; }
    .sparkline-label { font-size: 10px; color: #94a3b8; margin-top: 2px; }
    .summary-grid {
        display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 14px; margin-bottom: 14px;
    }
    .summary-tile {
        background: #fff; border: 1px solid #e2e8f0; border-radius: 10px;
        padding: 16px; text-align: center;
    }
    .summary-tile .val { font-size: 32px; font-weight: 800; color: #0f172a; }
    .summary-tile .lbl { font-size: 12px; color: #64748b; margin-top: 4px; }
    .absence-card {
        background: #fff7ed; border: 1px solid #fed7aa; border-radius: 10px;
        padding: 14px 18px; margin-bottom: 10px;
        display: flex; justify-content: space-between; align-items: center;
    }
    .absence-name  { font-weight: 700; }
    .absence-days  { color: #ea580c; font-weight: 700; }
    .absence-action{ font-size: 12px; color: #64748b; margin-top: 2px; }
    .persistent-card {
        background: #fef2f2; border: 1px solid #fecaca; border-radius: 10px;
        padding: 14px 18px; margin-bottom: 10px;
    }
    .empty { color: #94a3b8; font-size: 14px; padding: 12px 0; }
    .category-tag {
        display: inline-block; font-size: 11px; font-weight: 600;
        background: #f1f5f9; color: #475569;
        border-radius: 6px; padding: 2px 8px; margin-right: 6px;
    }
    """

    # ── Alert cards ───────────────────────────────────────────────────────────
    def alert_card(a: dict) -> str:
        score  = a.get("today_wellbeing", 0)
        sc_cls = "score-urgent" if score < 45 else "score-monitor" if score < 65 else "score-ok"
        spark  = _sparkline_svg(a.get("trend_last_5_days", []))
        badge  = _severity_badge(a.get("severity", "monitor"))
        return f"""
        <div class="card">
          <div class="card-left">
            <div class="card-name">{a.get('person_name','Unknown')}</div>
            <span class="category-tag">{a.get('category','')}</span>{badge}
            <div class="card-title">{a.get('title','')}</div>
            <div class="card-desc">{a.get('description','')}</div>
            <div class="card-action">↳ {a.get('recommended_action','')}</div>
          </div>
          <div class="card-right">
            <div class="card-score {sc_cls}">{score}</div>
            {spark}
            <div class="sparkline-label">5-day trend</div>
          </div>
        </div>"""

    cards_html = "\n".join(alert_card(a) for a in alerts) if alerts else \
        '<p class="empty">No alerts raised today. ✅</p>'

    # ── Absence section ───────────────────────────────────────────────────────
    def absence_card(af: dict) -> str:
        return f"""
        <div class="absence-card">
          <div>
            <div class="absence-name">{af.get('person_name','Unknown')}</div>
            <div class="absence-action">Last seen: {af.get('last_seen_date','?')} · {af.get('recommended_action','')}</div>
          </div>
          <div class="absence-days">{af.get('days_absent',0)}d absent</div>
        </div>"""

    absence_html = "\n".join(absence_card(af) for af in absence_flags) if absence_flags else \
        '<p class="empty">No absence flags.</p>'

    # ── Persistent alerts section ─────────────────────────────────────────────
    def persistent_card(a: dict) -> str:
        return f"""
        <div class="persistent-card">
          <strong>{a.get('person_name','?')}</strong>
          <span class="category-tag" style="margin-left:8px">{a.get('category','')}</span>
          — flagged for <strong>{a.get('days_flagged_consecutively',0)} consecutive days</strong>.
          <span style="color:#64748b;font-size:13px"> {a.get('description','')}</span>
        </div>"""

    persistent_html = "\n".join(persistent_card(a) for a in persistent) if persistent else \
        '<p class="empty">No persons flagged 3+ consecutive days.</p>'

    # ── Summary tiles ─────────────────────────────────────────────────────────
    s   = school_summary
    avg_wb  = s.get("school_avg_wellbeing_today", 0)
    avg_cls = "score-urgent" if avg_wb < 45 else "score-monitor" if avg_wb < 65 else "score-ok"

    summary_html = f"""
    <div class="summary-grid">
      <div class="summary-tile">
        <div class="val">{s.get('total_persons_tracked',0)}</div>
        <div class="lbl">Persons tracked</div>
      </div>
      <div class="summary-tile">
        <div class="val" style="color:#ef4444">{s.get('persons_flagged_today',0)}</div>
        <div class="lbl">Flagged today</div>
      </div>
      <div class="summary-tile">
        <div class="val" style="color:#f59e0b">{s.get('persons_flagged_yesterday',0)}</div>
        <div class="lbl">Flagged yesterday</div>
      </div>
      <div class="summary-tile">
        <div class="val {avg_cls}">{avg_wb}</div>
        <div class="lbl">Avg wellbeing today</div>
      </div>
      <div class="summary-tile">
        <div class="val" style="font-size:16px;padding-top:6px">{s.get('most_common_anomaly_this_week','—')}</div>
        <div class="lbl">Top anomaly this week</div>
      </div>
    </div>"""

    # ── Assemble full HTML ────────────────────────────────────────────────────
    urgent_count  = sum(1 for a in alerts if a.get("severity") == "urgent")
    monitor_count = sum(1 for a in alerts if a.get("severity") == "monitor")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sentio Mind · Alert Digest · {today_str}</title>
  <style>{css}</style>
</head>
<body>
<div class="page">

  <header>
    <h1>🧠 Sentio Mind — Alert Digest</h1>
    <p>{SCHOOL} &nbsp;·&nbsp; Generated: {gen_time} &nbsp;·&nbsp;
       <span style="color:#ef4444;font-weight:700">{urgent_count} urgent</span> &nbsp;
       <span style="color:#f59e0b;font-weight:700">{monitor_count} monitor</span> &nbsp;
       {len(absence_flags)} absence flag(s)
    </p>
  </header>

  <!-- ── SECTION 1: Today's Alerts ── -->
  <h2>📋 Section 1 — Today's Alerts</h2>
  {cards_html}

  <!-- ── Absence Flags ── -->
  <h2>🚨 Absence Flags</h2>
  {absence_html}

  <!-- ── SECTION 2: School Summary ── -->
  <h2>🏫 Section 2 — School Summary</h2>
  {summary_html}

  <!-- ── SECTION 3: Persistent Alerts ── -->
  <h2>🔁 Section 3 — Persistent Alerts (3+ consecutive days)</h2>
  {persistent_html}

</div>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"  HTML report written → {output_path}")


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

    # Bonus: peer-comparison anomaly detection
    peer_alerts = detect_peer_comparison(daily_data, all_dates, person_days, person_info)
    all_alerts.extend(peer_alerts)

    # Sort by severity
    sev_order = {"urgent": 0, "monitor": 1, "informational": 2}
    all_alerts.sort(key=lambda a: sev_order.get(a.get("severity", "informational"), 3))

    today_str     = all_dates[-1] if all_dates else str(date.today())
    flagged_today = sum(1 for a in all_alerts if a.get("date") == today_str)
    cat_counter   = Counter(a.get("category") for a in all_alerts)
    top_category  = cat_counter.most_common(1)[0][0] if cat_counter else "none"

    # School avg wellbeing for today
    if today_str in daily_data:
        wb_vals = [p.get("wellbeing", 0) for p in daily_data[today_str].values()]
        school_avg_wb = round(float(np.mean(wb_vals))) if wb_vals else 0
    else:
        school_avg_wb = 0

    school_summary = {
        "total_persons_tracked":         len(person_days),
        "persons_flagged_today":         flagged_today,
        "persons_flagged_yesterday":     0,
        "most_common_anomaly_this_week": top_category,
        "school_avg_wellbeing_today":    school_avg_wb,
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
    print(f"  JSON feed written    → {FEED_OUT}")

    generate_alert_digest(all_alerts, absence_flags, school_summary, REPORT_OUT)

    print()
    print("=" * 50)
    print(f"  Alerts:  {feed['alert_summary']['total_alerts']} total  "
          f"({feed['alert_summary']['urgent']} urgent, {feed['alert_summary']['monitor']} monitor)")
    print(f"  Absence flags: {len(absence_flags)}")
    print(f"  Report → {REPORT_OUT}")
    print(f"  JSON   → {FEED_OUT}")
    print("=" * 50)
