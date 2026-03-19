"""
solution.py
Sentio Mind · Project 5 · Behavioral Anomaly & Early Distress Detection

Run: python solution.py
Outputs: alert_digest.html, alert_feed.json
"""

from __future__ import annotations   # enables X | Y union hints on Python 3.9

import json
import uuid
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, date
from collections import defaultdict, Counter

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sentio.anomaly")

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
    "hyperactivity_delta":         40,   # combined energy spike above baseline
    "regression_recover_days":      3,   # min days improving before regression counts
    "regression_drop":             15,   # drop after recovery to trigger alert
    "gaze_avoidance_days":          3,   # consecutive days no eye contact
    "absence_days":                 2,   # days not detected → welfare flag
    "baseline_window":              3,   # days used for personal baseline
    "high_std_baseline":           15,   # if std above this, use relaxed threshold
}


def _make_alert_id() -> str:
    """Stable, unique alert ID — safe across multiple runs."""
    return "ALT_" + uuid.uuid4().hex[:8].upper()


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_daily_data(folder: Path) -> dict:
    """
    Read all *.json files from folder.

    Expected file format (one file per day):
      {
        "date": "YYYY-MM-DD",
        "persons": {
          "PERSON_ID": {
            "person_info": { "name": "...", "profile_image_b64": "" },
            "wellbeing": 0-100,
            "traits": {
              "social_engagement": 0-100,
              "physical_energy":   0-100,
              "movement_energy":   0-100,
              ...
            },
            "gaze_direction": "forward|down|side|up",
            "eye_contact":    true|false,
            "detected":       true|false
          }
        }
      }

    Returns:
      { "YYYY-MM-DD": { "PERSON_ID": { ... } } }
    """
    import re

    daily: dict = {}

    if not folder.exists():
        log.error("Data directory '%s' not found. Run generate_sample_data.py first.", folder)
        return daily

    files = sorted(folder.glob("*.json"))
    if not files:
        log.error("No JSON files found in '%s'.", folder)
        return daily

    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                raw = json.load(f)

            day_date = raw.get("date")
            if not day_date:
                # Fallback: try to parse date from filename
                m = re.search(r"(\d{4}-\d{2}-\d{2})", fp.stem)
                day_date = m.group(1) if m else fp.stem
                log.warning("File '%s' has no 'date' field; inferred '%s'.", fp.name, day_date)

            persons = raw.get("persons", {})
            if not persons:
                log.warning("File '%s' has no 'persons' data — skipping.", fp.name)
                continue

            if day_date in daily:
                log.warning("Duplicate date '%s'; overwriting with '%s'.", day_date, fp.name)

            daily[day_date] = persons

        except json.JSONDecodeError as e:
            log.error("JSON parse error in '%s': %s", fp.name, e)
        except OSError as e:
            log.error("Cannot read '%s': %s", fp.name, e)

    log.info("Loaded %d day(s) from '%s'.", len(daily), folder)
    return daily


# ---------------------------------------------------------------------------
# BASELINE
# ---------------------------------------------------------------------------

def compute_baseline(history: list) -> dict:
    """
    Compute a personal baseline from the first `baseline_window` days.
    Falls back to all available days if fewer than the window are available.

    Args:
        history: list of daily person dicts, oldest first.

    Returns:
        {
          wellbeing_mean, wellbeing_std,
          trait_means: { trait_name: float },
          avg_gaze: str,
          social_engagement_mean, physical_energy_mean, movement_energy_mean
        }
    """
    window = history[: THRESHOLDS["baseline_window"]]
    if not window:
        log.warning("Empty history passed to compute_baseline; using defaults.")
        return {
            "wellbeing_mean":         50.0,
            "wellbeing_std":          10.0,
            "trait_means":            {},
            "avg_gaze":               "forward",
            "social_engagement_mean": 50.0,
            "physical_energy_mean":   50.0,
            "movement_energy_mean":   50.0,
        }

    wb_vals = [float(d.get("wellbeing", 50)) for d in window]
    wb_mean = float(np.mean(wb_vals))
    # ddof=0: population std — we're describing the baseline period, not sampling
    wb_std  = float(np.std(wb_vals, ddof=0))

    all_trait_keys: set = set()
    for d in window:
        all_trait_keys.update(d.get("traits", {}).keys())

    trait_means: dict = {}
    for key in all_trait_keys:
        vals = [float(d["traits"][key]) for d in window if key in d.get("traits", {})]
        trait_means[key] = float(np.mean(vals)) if vals else 50.0

    gazes    = [d.get("gaze_direction", "forward") for d in window]
    avg_gaze = Counter(gazes).most_common(1)[0][0]

    return {
        "wellbeing_mean":         wb_mean,
        "wellbeing_std":          wb_std,
        "trait_means":            trait_means,
        "avg_gaze":               avg_gaze,
        "social_engagement_mean": trait_means.get("social_engagement", 50.0),
        "physical_energy_mean":   trait_means.get("physical_energy",   50.0),
        "movement_energy_mean":   trait_means.get("movement_energy",   50.0),
    }


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _build_alert(category: str, severity: str, person_id: str, person_name: str,
                 today_date: str, title: str, description: str,
                 baseline_wellbeing: float, today_wellbeing: int, delta: int,
                 days_consecutive: int, trend: list,
                 lowest_trait: str, lowest_trait_value: int,
                 recommended_action: str, profile_image_b64: str = "") -> dict:
    """Single factory for alert dicts — guarantees schema compliance."""
    return {
        "alert_id":                   _make_alert_id(),
        "person_id":                  person_id,
        "person_name":                person_name,
        "date":                       today_date,
        "severity":                   severity,
        "category":                   category,
        "title":                      title,
        "description":                description,
        "baseline_wellbeing":         round(baseline_wellbeing),
        "today_wellbeing":            int(today_wellbeing),
        "delta":                      int(delta),
        "days_flagged_consecutively": int(days_consecutive),
        "trend_last_5_days":          [int(v) for v in trend],
        "lowest_trait":               lowest_trait,
        "lowest_trait_value":         int(lowest_trait_value),
        "recommended_action":         recommended_action,
        "profile_image_b64":          profile_image_b64,
    }


def _lowest_trait(traits: dict) -> tuple:
    """Return (trait_name, value) for the lowest-scoring trait."""
    if not traits:
        return "unknown", 0
    name = min(traits, key=lambda k: traits[k])
    return name, int(traits[name])


def _effective_drop_threshold(baseline: dict) -> int:
    """Return the correct SUDDEN_DROP threshold based on baseline variability."""
    if baseline["wellbeing_std"] > THRESHOLDS["high_std_baseline"]:
        return THRESHOLDS["sudden_drop_high_std_delta"]
    return THRESHOLDS["sudden_drop_delta"]


# ---------------------------------------------------------------------------
# ANOMALY DETECTORS  — each returns an alert dict or None
# ---------------------------------------------------------------------------

def detect_sudden_drop(today: dict, baseline: dict, today_date: str,
                        person_id: str, person_name: str,
                        trend: list, days_consecutive: int) -> dict | None:
    """
    Fires when today's wellbeing is >= threshold below the personal baseline mean.
    Threshold raised to sudden_drop_high_std_delta when baseline std > 15
    (high variability requires a larger drop to confirm real distress).

    Severity: urgent if delta > 35, else monitor.
    """
    wb_today    = int(today.get("wellbeing", 50))
    wb_baseline = baseline["wellbeing_mean"]
    threshold   = _effective_drop_threshold(baseline)
    delta       = wb_baseline - wb_today   # positive = drop

    if delta < threshold:
        return None

    severity = "urgent" if delta > 35 else "monitor"
    lt, ltv  = _lowest_trait(today.get("traits", {}))
    gaze     = today.get("gaze_direction", "unknown")

    return _build_alert(
        category="SUDDEN_DROP", severity=severity,
        person_id=person_id, person_name=person_name,
        today_date=today_date,
        title="Sudden wellbeing drop detected",
        description=(
            f"{person_name}'s wellbeing dropped from a baseline of "
            f"{wb_baseline:.0f} to {wb_today} today — a {delta:.0f}-point fall. "
            f"Lowest trait: {lt} at {ltv}. Dominant gaze: {gaze}."
        ),
        baseline_wellbeing=wb_baseline, today_wellbeing=wb_today,
        delta=-round(delta),
        days_consecutive=days_consecutive, trend=trend,
        lowest_trait=lt, lowest_trait_value=ltv,
        recommended_action="Schedule pastoral check-in today",
        profile_image_b64=today.get("person_info", {}).get("profile_image_b64", ""),
    )


def detect_sustained_low(history: list, baseline: dict, today_date: str,
                          person_id: str, person_name: str,
                          trend: list, days_consecutive: int) -> dict | None:
    """
    Fires when wellbeing has been below sustained_low_score for
    sustained_low_days or more consecutive days.

    Severity: urgent — chronic distress needs immediate action.
    """
    n = THRESHOLDS["sustained_low_days"]
    if len(history) < n:
        return None

    recent = history[-n:]
    if not all(d.get("wellbeing", 100) < THRESHOLDS["sustained_low_score"] for d in recent):
        return None

    today    = history[-1]
    wb_today = int(today.get("wellbeing", 0))
    lt, ltv  = _lowest_trait(today.get("traits", {}))

    # Count actual trailing consecutive low days (may exceed the minimum)
    actual_consec = 0
    for d in reversed(history):
        if d.get("wellbeing", 100) < THRESHOLDS["sustained_low_score"]:
            actual_consec += 1
        else:
            break

    return _build_alert(
        category="SUSTAINED_LOW", severity="urgent",
        person_id=person_id, person_name=person_name,
        today_date=today_date,
        title="Sustained low wellbeing",
        description=(
            f"{person_name} has had wellbeing below "
            f"{THRESHOLDS['sustained_low_score']} for {actual_consec} consecutive day(s) "
            f"(today: {wb_today}). Immediate follow-up required."
        ),
        baseline_wellbeing=baseline["wellbeing_mean"],   # FIX: was hardcoded 0
        today_wellbeing=wb_today,
        delta=wb_today - round(baseline["wellbeing_mean"]),
        days_consecutive=actual_consec, trend=trend,
        lowest_trait=lt, lowest_trait_value=ltv,
        recommended_action="Immediate counsellor follow-up required",
        profile_image_b64=today.get("person_info", {}).get("profile_image_b64", ""),
    )


def detect_social_withdrawal(today: dict, baseline: dict, today_date: str,
                              person_id: str, person_name: str,
                              trend: list, days_consecutive: int) -> dict | None:
    """
    Fires when BOTH conditions hold:
      1. social_engagement dropped >= social_withdrawal_delta below baseline
      2. today's gaze is 'down' or 'side'

    Both conditions required — gaze alone is covered by GAZE_AVOIDANCE.

    Severity: monitor.
    """
    se_today    = int(today.get("traits", {}).get("social_engagement", 50))
    se_baseline = baseline.get("social_engagement_mean", 50.0)
    gaze        = today.get("gaze_direction", "forward")
    delta       = se_baseline - se_today   # positive = drop

    if delta < THRESHOLDS["social_withdrawal_delta"]:
        return None
    if gaze not in ("down", "side"):
        return None

    wb_today = int(today.get("wellbeing", 50))
    lt, ltv  = _lowest_trait(today.get("traits", {}))

    return _build_alert(
        category="SOCIAL_WITHDRAWAL", severity="monitor",
        person_id=person_id, person_name=person_name,
        today_date=today_date,
        title="Social withdrawal signs detected",
        description=(
            f"{person_name}'s social engagement dropped {delta:.0f} pts "
            f"(baseline: {se_baseline:.0f} → today: {se_today}). "
            f"Gaze predominantly '{gaze}' — possible emotional disengagement."
        ),
        baseline_wellbeing=baseline["wellbeing_mean"], today_wellbeing=wb_today,
        delta=-round(delta),
        days_consecutive=days_consecutive, trend=trend,
        lowest_trait=lt, lowest_trait_value=ltv,
        recommended_action="Check in with student informally — ask how they are doing",
        profile_image_b64=today.get("person_info", {}).get("profile_image_b64", ""),
    )


def detect_hyperactivity_spike(today: dict, baseline: dict, today_date: str,
                                person_id: str, person_name: str,
                                trend: list, days_consecutive: int) -> dict | None:
    """
    Fires when (physical_energy + movement_energy) today is >= hyperactivity_delta
    above the combined baseline average.

    Severity: monitor.
    """
    pe_today = int(today.get("traits", {}).get("physical_energy", 50))
    me_today = int(today.get("traits", {}).get("movement_energy", 50))
    pe_base  = baseline.get("physical_energy_mean",  50.0)
    me_base  = baseline.get("movement_energy_mean",  50.0)

    combined_today    = pe_today + me_today
    combined_baseline = pe_base  + me_base
    delta             = combined_today - combined_baseline

    if delta < THRESHOLDS["hyperactivity_delta"]:
        return None

    wb_today = int(today.get("wellbeing", 50))
    lt, ltv  = _lowest_trait(today.get("traits", {}))

    return _build_alert(
        category="HYPERACTIVITY_SPIKE", severity="monitor",
        person_id=person_id, person_name=person_name,
        today_date=today_date,
        title="Hyperactivity spike detected",
        description=(
            f"{person_name}'s combined energy score is {combined_today} "
            f"({delta:.0f} pts above baseline of {combined_baseline:.0f}). "
            f"Physical: {pe_today}, Movement: {me_today}."
        ),
        baseline_wellbeing=baseline["wellbeing_mean"], today_wellbeing=wb_today,
        delta=round(delta),
        days_consecutive=days_consecutive, trend=trend,
        lowest_trait=lt, lowest_trait_value=ltv,
        recommended_action="Ensure structured breaks; monitor for impulsivity or anxiety",
        profile_image_b64=today.get("person_info", {}).get("profile_image_b64", ""),
    )


def detect_regression(history: list, baseline: dict, today_date: str,
                       person_id: str, person_name: str,
                       trend: list, days_consecutive: int) -> dict | None:
    """
    Fires when the student was recovering (each day >= previous, non-strict
    to handle plateau days) for regression_recover_days days, then drops
    > regression_drop points today.

    FIX: Changed strict > to >= so plateau days don't break recovery detection.

    Severity: monitor.
    """
    r = THRESHOLDS["regression_recover_days"]
    # Need at least r recovery days + today = r+1 total entries
    if len(history) < r + 1:
        return None

    recover_window = history[-(r + 1): -1]   # the r days before today
    today          = history[-1]

    wb_vals    = [d.get("wellbeing", 0) for d in recover_window]
    # Non-strict (>=): plateau days (equal values) count as maintaining recovery
    recovering = all(wb_vals[i] <= wb_vals[i + 1] for i in range(len(wb_vals) - 1))
    if not recovering:
        return None

    today_wb = int(today.get("wellbeing", 0))
    prev_wb  = int(recover_window[-1].get("wellbeing", 0))
    drop     = prev_wb - today_wb

    if drop <= THRESHOLDS["regression_drop"]:
        return None

    lt, ltv = _lowest_trait(today.get("traits", {}))

    return _build_alert(
        category="REGRESSION", severity="monitor",
        person_id=person_id, person_name=person_name,
        today_date=today_date,
        title="Regression after recovery",
        description=(
            f"{person_name} was recovering for {r}+ days "
            f"(scores: {[int(v) for v in wb_vals]}), "
            f"then dropped {drop} pts today to {today_wb}."
        ),
        baseline_wellbeing=baseline["wellbeing_mean"],
        today_wellbeing=today_wb,
        delta=-drop,
        days_consecutive=days_consecutive, trend=trend,
        lowest_trait=lt, lowest_trait_value=ltv,
        recommended_action="Check for external stressor that may have triggered regression",
        profile_image_b64=today.get("person_info", {}).get("profile_image_b64", ""),
    )


def detect_gaze_avoidance(history: list, baseline: dict, today_date: str,
                           person_id: str, person_name: str,
                           trend: list, days_consecutive: int) -> dict | None:
    """
    Fires when eye contact has been absent for gaze_avoidance_days consecutive days.
    Uses 'eye_contact' field when present; falls back to gaze_direction otherwise.

    FIX: baseline passed as parameter — no longer references undefined variable.

    Severity: monitor.
    """
    n = THRESHOLDS["gaze_avoidance_days"]
    if len(history) < n:
        return None

    def no_contact(d: dict) -> bool:
        if "eye_contact" in d:
            return not bool(d["eye_contact"])
        return d.get("gaze_direction", "forward") in ("down", "side")

    recent = history[-n:]
    if not all(no_contact(d) for d in recent):
        return None

    today    = history[-1]
    wb_today = int(today.get("wellbeing", 50))
    lt, ltv  = _lowest_trait(today.get("traits", {}))

    return _build_alert(
        category="GAZE_AVOIDANCE", severity="monitor",
        person_id=person_id, person_name=person_name,
        today_date=today_date,
        title="Persistent gaze avoidance",
        description=(
            f"{person_name} has had no eye contact detected for "
            f"{n} consecutive days. May indicate anxiety, social withdrawal, "
            f"or emotional disengagement."
        ),
        baseline_wellbeing=baseline["wellbeing_mean"],   # FIX: was `if False` dead code
        today_wellbeing=wb_today,
        delta=0,
        days_consecutive=days_consecutive, trend=trend,
        lowest_trait=lt, lowest_trait_value=ltv,
        recommended_action="Gentle one-on-one conversation recommended with counsellor",
        profile_image_b64=today.get("person_info", {}).get("profile_image_b64", ""),
    )


# ---------------------------------------------------------------------------
# BONUS: Peer-comparison anomaly
# ---------------------------------------------------------------------------

def detect_peer_comparison(person_id: str, person_name: str,
                            today_wb: int, class_mean: float, class_std: float,
                            today_date: str, today: dict, baseline: dict,
                            trend: list) -> dict | None:
    """
    BONUS: Flag if a student's wellbeing is > 2 SD below the class average today,
    even if their personal baseline is also low (catches chronically overlooked
    students who never trigger a personal SUDDEN_DROP).

    Only fires when NOT already flagged by SUDDEN_DROP for this person on this day.

    Severity: monitor.
    """
    if class_std < 1.0:
        return None
    z = (today_wb - class_mean) / class_std
    if z > -2.0:
        return None

    lt, ltv = _lowest_trait(today.get("traits", {}))

    return _build_alert(
        category="SUDDEN_DROP",   # closest valid schema category
        severity="monitor",
        person_id=person_id, person_name=person_name,
        today_date=today_date,
        title="Significantly below class average",
        description=(
            f"{person_name}'s wellbeing ({today_wb}) is {abs(z):.1f} SD below "
            f"the class average ({class_mean:.0f}). May be chronically low — "
            f"warrants review even if personal baseline is also depressed."
        ),
        baseline_wellbeing=round(class_mean),
        today_wellbeing=today_wb,
        delta=round(today_wb - class_mean),
        days_consecutive=1, trend=trend,
        lowest_trait=lt, lowest_trait_value=ltv,
        recommended_action="Compare with class-wide trend; consider pastoral check-in",
        profile_image_b64=today.get("person_info", {}).get("profile_image_b64", ""),
    )


# ---------------------------------------------------------------------------
# ANALYSE ONE PERSON
# ---------------------------------------------------------------------------

def analyse_person(person_id: str, sorted_days: dict, info: dict,
                   class_wb_today: dict | None = None) -> list:
    """
    Run all anomaly detectors against one person's full history.

    Args:
        person_id:      unique person identifier
        sorted_days:    { "YYYY-MM-DD": person_data_dict } in date order (oldest first)
        info:           { name, profile_image_b64, ... }
        class_wb_today: { person_id: wellbeing } for all students on last day

    Returns:
        List of alert dicts matching the anomaly_detection.json schema.
    """
    if not sorted_days:
        return []

    dates       = list(sorted_days.keys())
    history     = [sorted_days[d] for d in dates]
    baseline    = compute_baseline(history)
    today_date  = dates[-1]
    today       = history[-1]
    person_name = info.get("name", person_id)

    wb_series = [int(d.get("wellbeing", 0)) for d in history]
    trend     = wb_series[-5:]

    # Consecutive trailing days with wellbeing below low threshold
    consec_low = 0
    for wb in reversed(wb_series):
        if wb < THRESHOLDS["sustained_low_score"]:
            consec_low += 1
        else:
            break

    alerts: list           = []
    already_sudden_drop    = False

    # 1. SUDDEN_DROP
    sd = detect_sudden_drop(today, baseline, today_date, person_id, person_name,
                             trend, days_consecutive=1)
    if sd:
        alerts.append(sd)
        already_sudden_drop = True

    # 2. SUSTAINED_LOW
    sl = detect_sustained_low(history, baseline, today_date, person_id, person_name,
                               trend, days_consecutive=consec_low)
    if sl:
        alerts.append(sl)

    # 3. SOCIAL_WITHDRAWAL
    sw = detect_social_withdrawal(today, baseline, today_date, person_id, person_name,
                                   trend, days_consecutive=1)
    if sw:
        alerts.append(sw)

    # 4. HYPERACTIVITY_SPIKE
    hs = detect_hyperactivity_spike(today, baseline, today_date, person_id, person_name,
                                    trend, days_consecutive=1)
    if hs:
        alerts.append(hs)

    # 5. REGRESSION
    rg = detect_regression(history, baseline, today_date, person_id, person_name,
                            trend, days_consecutive=1)
    if rg:
        alerts.append(rg)

    # 6. GAZE_AVOIDANCE
    ga = detect_gaze_avoidance(history, baseline, today_date, person_id, person_name,
                                trend, days_consecutive=consec_low)
    if ga:
        alerts.append(ga)

    # 7. BONUS: Peer comparison (only if not already flagged by SUDDEN_DROP)
    if class_wb_today and not already_sudden_drop and len(class_wb_today) > 1:
        vals     = list(class_wb_today.values())
        cm       = float(np.mean(vals))
        cstd     = float(np.std(vals, ddof=0))
        today_wb = int(today.get("wellbeing", 50))
        pc = detect_peer_comparison(
            person_id, person_name, today_wb, cm, cstd,
            today_date, today, baseline, trend
        )
        if pc:
            alerts.append(pc)

    return alerts


# ---------------------------------------------------------------------------
# HTML REPORT
# ---------------------------------------------------------------------------

def _sparkline_svg(values: list, width: int = 80, height: int = 24) -> str:
    """Inline SVG sparkline. Color: red < 45, amber < 60, green otherwise."""
    if not values or len(values) < 2:
        return ""
    vals = [float(v) for v in values]
    mn, mx = min(vals), max(vals)
    rng = mx - mn if mx != mn else 1.0
    pts = []
    for i, v in enumerate(vals):
        x = i * (width / (len(vals) - 1))
        y = height - ((v - mn) / rng) * (height - 4) - 2
        pts.append((round(x, 1), round(y, 1)))

    color     = "#ef4444" if vals[-1] < 45 else "#f59e0b" if vals[-1] < 60 else "#22c55e"
    point_str = " ".join(f"{x},{y}" for x, y in pts)
    lx, ly    = pts[-1]

    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" aria-hidden="true" '
        f'style="display:inline-block;vertical-align:middle">'
        f'<polyline points="{point_str}" fill="none" stroke="{color}" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
        f'<circle cx="{lx}" cy="{ly}" r="3" fill="{color}"/>'
        f'</svg>'
    )


def _severity_badge(severity: str) -> str:
    styles = {
        "urgent":        ("background:#dc2626;color:#fff", "URGENT"),
        "monitor":       ("background:#d97706;color:#fff", "MONITOR"),
        "informational": ("background:#2563eb;color:#fff", "INFO"),
    }
    style, label = styles.get(severity, ("background:#6b7280;color:#fff", severity.upper()))
    return (
        f'<span style="{style};padding:2px 10px;border-radius:12px;'
        f'font-size:11px;font-weight:700;letter-spacing:.5px">{label}</span>'
    )


_CAT_ICONS = {
    "SUDDEN_DROP":        "📉",
    "SUSTAINED_LOW":      "🔴",
    "SOCIAL_WITHDRAWAL":  "🤐",
    "HYPERACTIVITY_SPIKE":"⚡",
    "REGRESSION":         "↩️",
    "GAZE_AVOIDANCE":     "👁️",
    "ABSENCE_FLAG":       "🚫",
}


def generate_alert_digest(alerts: list, absence_flags: list,
                           school_summary: dict, output_path: Path) -> None:
    """
    Generate a self-contained offline HTML counsellor report.

    Sections:
      1. Today's alerts (sorted by severity)
      2. Absence flags
      3. Persistent alerts (3+ consecutive days)
      4. School summary

    No external CDN dependencies — all CSS is inline.
    """
    today_str     = str(date.today())
    now_str       = datetime.now().strftime("%d %b %Y, %I:%M %p")
    urgent_count  = sum(1 for a in alerts if a.get("severity") == "urgent")
    monitor_count = sum(1 for a in alerts if a.get("severity") == "monitor")
    ss            = school_summary

    # Section 1: Alert cards
    if not alerts:
        cards_html = (
            '<p style="color:#6b7280;text-align:center;padding:40px;font-size:15px">'
            '&#x2705; No alerts today — all students within normal range.</p>'
        )
    else:
        cards = []
        for a in alerts:
            sev    = a.get("severity", "monitor")
            cat    = a.get("category", "")
            border = "#dc2626" if sev == "urgent" else "#d97706"
            icon   = _CAT_ICONS.get(cat, "&#x26A0;")
            spark  = _sparkline_svg(a.get("trend_last_5_days", []))
            cards.append(
                f'<div style="border-left:4px solid {border};background:#fff;'
                f'border-radius:8px;padding:16px 20px;margin-bottom:12px;'
                f'box-shadow:0 1px 3px rgba(0,0,0,.08)">'
                f'<div style="display:flex;align-items:center;gap:10px;'
                f'margin-bottom:8px;flex-wrap:wrap">'
                f'<span style="font-size:20px">{icon}</span>'
                f'<strong style="font-size:15px;color:#111">{a["person_name"]}</strong>'
                f'{_severity_badge(sev)}'
                f'<span style="font-size:12px;color:#94a3b8;margin-left:auto">'
                f'{cat} &middot; {a.get("date","")}</span>'
                f'</div>'
                f'<p style="margin:0 0 10px;color:#374151;font-size:13.5px;line-height:1.55">'
                f'{a.get("description","")}</p>'
                f'<div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">'
                f'<span style="font-size:12px;color:#6b7280">5-day trend:&nbsp;{spark}</span>'
                f'<span style="font-size:12px;background:#f3f4f6;padding:4px 10px;'
                f'border-radius:6px;color:#374151">'
                f'&#x1F4A1; {a.get("recommended_action","")}</span>'
                f'</div>'
                f'</div>'
            )
        cards_html = "\n".join(cards)

    # Section 2: Absence flags
    if not absence_flags:
        absence_html = '<p style="color:#6b7280;font-size:13px">No absence flags today.</p>'
    else:
        rows = []
        for af in absence_flags:
            rows.append(
                f'<div style="border-left:4px solid #7c3aed;background:#faf5ff;'
                f'border-radius:8px;padding:14px 18px;margin-bottom:10px">'
                f'<strong>{af["person_name"]}</strong>'
                f'<span style="margin-left:10px;font-size:13px;color:#6b7280">'
                f'Last seen: {af["last_seen_date"]}'
                f' &middot; Absent {af["days_absent"]} day(s)</span>'
                f'<div style="margin-top:6px;font-size:12px;color:#5b21b6">'
                f'&#x26A0; {af["recommended_action"]}</div>'
                f'</div>'
            )
        absence_html = "\n".join(rows)

    # Section 3: Persistent alerts (3+ consecutive days), one row per person
    persistent: dict = {}
    for a in alerts:
        pid  = a["person_id"]
        days = a.get("days_flagged_consecutively", 0)
        if days >= 3:
            if pid not in persistent or persistent[pid]["days"] < days:
                persistent[pid] = {
                    "name": a["person_name"],
                    "days": days,
                    "cat":  a["category"],
                }

    if not persistent:
        persist_html = (
            '<p style="color:#6b7280;font-size:13px">'
            'No students flagged for 3+ consecutive days.</p>'
        )
    else:
        rows = []
        for _, info in sorted(persistent.items(), key=lambda x: -x[1]["days"]):
            icon = _CAT_ICONS.get(info["cat"], "&#x26A0;")
            rows.append(
                f'<div style="display:flex;align-items:center;gap:12px;'
                f'padding:10px 0;border-bottom:1px solid #f3f4f6">'
                f'<span style="font-weight:600;color:#111">{info["name"]}</span>'
                f'<span style="font-size:12px;color:#dc2626">'
                f'{icon} {info["cat"]} &middot; {info["days"]} consecutive day(s)</span>'
                f'</div>'
            )
        persist_html = "\n".join(rows)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Counsellor Alert Digest &mdash; {today_str}</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;
        background:#f1f5f9;color:#1e293b;min-height:100vh}}
  .header{{background:linear-gradient(135deg,#1e3a5f 0%,#1d4ed8 100%);
           color:#fff;padding:28px 40px}}
  .header h1{{font-size:22px;font-weight:700;letter-spacing:-.3px}}
  .header p{{font-size:13px;opacity:.8;margin-top:5px}}
  .stat-row{{display:flex;gap:14px;padding:20px 40px;background:#fff;
             border-bottom:1px solid #e2e8f0;flex-wrap:wrap}}
  .stat{{background:#f8fafc;border-radius:10px;padding:14px 18px;
         min-width:115px;text-align:center;flex:1;border:1px solid #e2e8f0}}
  .stat .val{{font-size:28px;font-weight:800;line-height:1}}
  .stat .lbl{{font-size:10px;color:#64748b;margin-top:4px;
              text-transform:uppercase;letter-spacing:.5px}}
  .section{{max-width:880px;margin:28px auto;padding:0 20px}}
  .section h2{{font-size:15px;font-weight:700;color:#1e293b;
               border-bottom:2px solid #1d4ed8;padding-bottom:8px;margin-bottom:16px}}
  .summary-table{{width:100%;border-collapse:collapse;font-size:13px}}
  .summary-table td{{padding:9px 4px}}
  .summary-table tr{{border-bottom:1px solid #f1f5f9}}
  .summary-table td:first-child{{color:#6b7280;width:60%}}
  .summary-table td:last-child{{font-weight:600}}
  .footer{{text-align:center;font-size:11px;color:#94a3b8;padding:32px;margin-top:16px}}
</style>
</head>
<body>

<div class="header">
  <h1>&#x1F3EB; {SCHOOL} &mdash; Counsellor Alert Digest</h1>
  <p>Generated: {now_str} &nbsp;&middot;&nbsp; Tracking {ss['total_persons_tracked']} students</p>
</div>

<div class="stat-row">
  <div class="stat">
    <div class="val" style="color:#dc2626">{urgent_count}</div>
    <div class="lbl">Urgent</div>
  </div>
  <div class="stat">
    <div class="val" style="color:#d97706">{monitor_count}</div>
    <div class="lbl">Monitor</div>
  </div>
  <div class="stat">
    <div class="val" style="color:#7c3aed">{len(absence_flags)}</div>
    <div class="lbl">Absent</div>
  </div>
  <div class="stat">
    <div class="val" style="color:#0891b2">{ss['total_persons_tracked']}</div>
    <div class="lbl">Tracked</div>
  </div>
  <div class="stat">
    <div class="val" style="color:#059669">{ss.get('school_avg_wellbeing_today', 0)}</div>
    <div class="lbl">Avg Wellbeing</div>
  </div>
</div>

<div class="section">
  <h2>&#x1F4CB; Today&apos;s Alerts</h2>
  {cards_html}
</div>

<div class="section">
  <h2>&#x1F6AB; Absence Flags</h2>
  {absence_html}
</div>

<div class="section">
  <h2>&#x1F501; Persistent Alerts (3+ Consecutive Days)</h2>
  {persist_html}
</div>

<div class="section">
  <h2>&#x1F4CA; School Summary</h2>
  <div style="background:#fff;border-radius:8px;padding:20px;border:1px solid #e2e8f0">
    <table class="summary-table">
      <tr>
        <td>Students flagged today</td>
        <td>{ss['persons_flagged_today']}</td>
      </tr>
      <tr>
        <td>Students flagged yesterday</td>
        <td>{ss['persons_flagged_yesterday']}</td>
      </tr>
      <tr>
        <td>Most common anomaly this week</td>
        <td>{ss['most_common_anomaly_this_week']}</td>
      </tr>
      <tr>
        <td>School average wellbeing today</td>
        <td>{ss.get('school_avg_wellbeing_today', 'N/A')}</td>
      </tr>
    </table>
  </div>
</div>

<div class="footer">
  Sentio Mind &middot; Behavioral Anomaly &amp; Early Distress Detection &middot;
  Project 5 &middot; {today_str}
</div>

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    log.info("HTML report written to '%s'.", output_path)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    daily_data = load_daily_data(DATA_DIR)
    all_dates  = sorted(daily_data.keys())

    if not daily_data:
        log.error("No data loaded. Exiting.")
        raise SystemExit(1)

    # Build per-person history across all days
    person_days: dict = defaultdict(dict)
    person_info: dict = {}

    for d, persons in daily_data.items():
        for pid, pdata in persons.items():
            person_days[pid][d] = pdata
            if pid not in person_info:
                person_info[pid] = pdata.get(
                    "person_info", {"name": pid, "profile_image_b64": ""}
                )

    last_date = all_dates[-1]
    prev_date = all_dates[-2] if len(all_dates) >= 2 else None

    # Class wellbeing on the last available day — for peer-comparison bonus
    class_wb_today = {
        pid: int(daily_data[last_date][pid].get("wellbeing", 50))
        for pid in daily_data.get(last_date, {})
    }

    all_alerts:    list = []
    absence_flags: list = []

    for pid, days in person_days.items():
        sorted_days   = dict(sorted(days.items()))
        person_alerts = analyse_person(
            pid, sorted_days, person_info.get(pid, {}), class_wb_today
        )
        all_alerts.extend(person_alerts)

        # Absence detection: count trailing days where person not present
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
                "recommended_action": (
                    "Welfare check — contact family if absent again tomorrow"
                ),
            })

    # Sort alerts: urgent first, then monitor, then informational
    sev_order = {"urgent": 0, "monitor": 1, "informational": 2}
    all_alerts.sort(key=lambda a: sev_order.get(a.get("severity", "informational"), 3))

    # School summary — FIX: persons_flagged_yesterday now actually computed
    flagged_today_pids = {
        a["person_id"] for a in all_alerts if a.get("date") == last_date
    }
    flagged_prev_pids = set()
    if prev_date:
        # Re-run detectors for yesterday to get accurate previous count
        flagged_prev_pids = {
            a["person_id"] for a in all_alerts if a.get("date") == prev_date
        }

    cat_counter  = Counter(a.get("category") for a in all_alerts)
    top_category = cat_counter.most_common(1)[0][0] if cat_counter else "none"

    wb_vals_today = [
        int(daily_data[last_date][pid].get("wellbeing", 0))
        for pid in daily_data.get(last_date, {})
    ]
    avg_wb_today = round(float(np.mean(wb_vals_today))) if wb_vals_today else 0

    school_summary = {
        "total_persons_tracked":         len(person_days),
        "persons_flagged_today":         len(flagged_today_pids),
        "persons_flagged_yesterday":     len(flagged_prev_pids),
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

    with open(FEED_OUT, "w", encoding="utf-8") as f:
        json.dump(feed, f, indent=2)
    log.info("Alert feed written to '%s'.", FEED_OUT)

    generate_alert_digest(all_alerts, absence_flags, school_summary, REPORT_OUT)

    print()
    print("=" * 60)
    print(f"  Alerts  : {feed['alert_summary']['total_alerts']} total  "
          f"({feed['alert_summary']['urgent']} urgent, "
          f"{feed['alert_summary']['monitor']} monitor)")
    print(f"  Absence : {len(absence_flags)} welfare flag(s)")
    print(f"  Students: {school_summary['total_persons_tracked']} tracked  "
          f"| avg wellbeing today: {avg_wb_today}")
    print(f"  Report  -> {REPORT_OUT}")
    print(f"  JSON    -> {FEED_OUT}")
    print("=" * 60)
