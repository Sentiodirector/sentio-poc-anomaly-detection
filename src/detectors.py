"""
detectors.py
------------
Contains one detection function for each of the 7 anomaly categories.

Each function:
    - Takes a student's full list of records + their baseline
    - Returns a list of alert dictionaries (empty list = no anomaly found)

Helper function `_make_alert()` creates a standardised alert dict.
`run_all_detectors()` ties everything together — call this from solution.py.
"""

import uuid
from .baseline import compute_baseline


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _make_alert(person_id, alert_type, severity, detected_on, description, details):
    """
    Creates one standardised alert dictionary.

    severity: "HIGH" | "MEDIUM" | "LOW"
    """
    return {
        "alert_id":    str(uuid.uuid4()),   # unique ID for each alert
        "person_id":   person_id,
        "alert_type":  alert_type,
        "severity":    severity,
        "detected_on": detected_on,
        "description": description,
        "details":     details,
    }


# ---------------------------------------------------------------------------
# Detector 1 — SUDDEN_DROP
# ---------------------------------------------------------------------------

def detect_sudden_drop(records: list, baseline: dict) -> list:
    """
    SUDDEN_DROP: wellbeing drops >= 20 points compared to the personal baseline
    in a single observation day.

    Only checks days AFTER the baseline period (day index 3 onwards).
    Only fires once per drop episode (won't re-alert while already low).
    """
    alerts = []
    threshold        = baseline["thresholds"]["sudden_drop"]
    baseline_wellbeing = baseline["wellbeing_mean"]
    person_id        = records[0]["person_id"]

    already_in_drop = False   # track if we already alerted for this episode

    for i, record in enumerate(records):
        # Skip baseline days (first 3)
        if i < 3:
            already_in_drop = False
            continue

        if not record.get("person_detected", True):
            already_in_drop = False
            continue

        current = record["wellbeing_score"]
        drop    = baseline_wellbeing - current

        if drop >= threshold:
            if not already_in_drop:
                # First day of this drop episode — fire the alert
                alerts.append(_make_alert(
                    person_id    = person_id,
                    alert_type   = "SUDDEN_DROP",
                    severity     = "HIGH",
                    detected_on  = record["date"],
                    description  = (
                        f"Wellbeing dropped {drop:.1f} points below personal baseline "
                        f"({baseline_wellbeing:.1f} → {current})."
                    ),
                    details = {
                        "metric":          "wellbeing_score",
                        "current_value":   current,
                        "baseline_value":  round(baseline_wellbeing, 1),
                        "drop":            round(drop, 1),
                        "threshold":       threshold,
                    }
                ))
            already_in_drop = True
        else:
            # Wellbeing recovered — reset the flag
            already_in_drop = False

    return alerts


# ---------------------------------------------------------------------------
# Detector 2 — SUSTAINED_LOW
# ---------------------------------------------------------------------------

def detect_sustained_low(records: list, baseline: dict) -> list:
    """
    SUSTAINED_LOW: wellbeing score stays below 45 for 3 or more consecutive days.
    Alert fires on the 3rd consecutive low day.
    """
    alerts         = []
    person_id      = records[0]["person_id"]
    LOW_THRESHOLD  = 45          # fixed threshold from the assignment spec
    consecutive    = []          # running list of consecutive low-wellbeing records

    for record in records:
        # Absence resets the streak
        if not record.get("person_detected", True):
            consecutive = []
            continue

        if record["wellbeing_score"] < LOW_THRESHOLD:
            consecutive.append(record)
        else:
            consecutive = []    # wellbeing is back — reset streak

        # Fire exactly when the streak hits 3
        if len(consecutive) == 3:
            alerts.append(_make_alert(
                person_id   = person_id,
                alert_type  = "SUSTAINED_LOW",
                severity    = "HIGH",
                detected_on = record["date"],
                description = (
                    f"Wellbeing has been below {LOW_THRESHOLD} for 3 consecutive days "
                    f"(since {consecutive[0]['date']})."
                ),
                details = {
                    "metric":           "wellbeing_score",
                    "threshold":        LOW_THRESHOLD,
                    "consecutive_days": 3,
                    "start_date":       consecutive[0]["date"],
                    "scores":           [r["wellbeing_score"] for r in consecutive],
                }
            ))

    return alerts


# ---------------------------------------------------------------------------
# Detector 3 — SOCIAL_WITHDRAWAL
# ---------------------------------------------------------------------------

def detect_social_withdrawal(records: list, baseline: dict) -> list:
    """
    SOCIAL_WITHDRAWAL: social_engagement_score is >= 25 points below baseline
                       AND gaze_direction is "downward" on the same day.

    Only checked after the baseline period.
    Fires once per withdrawal episode.
    """
    alerts          = []
    threshold       = baseline["thresholds"]["social_drop"]
    baseline_social = baseline["social_mean"]
    person_id       = records[0]["person_id"]

    already_flagged = False

    for i, record in enumerate(records):
        if i < 3:
            already_flagged = False
            continue

        if not record.get("person_detected", True):
            already_flagged = False
            continue

        current_social = record["social_engagement_score"]
        drop           = baseline_social - current_social
        gaze_down      = record.get("gaze_direction", "forward") == "downward"

        if drop >= threshold and gaze_down:
            if not already_flagged:
                alerts.append(_make_alert(
                    person_id   = person_id,
                    alert_type  = "SOCIAL_WITHDRAWAL",
                    severity    = "MEDIUM",
                    detected_on = record["date"],
                    description = (
                        f"Social engagement dropped {drop:.1f} points "
                        f"(baseline: {baseline_social:.1f} → {current_social}) "
                        f"and gaze is consistently downward."
                    ),
                    details = {
                        "metric":          "social_engagement_score",
                        "current_value":   current_social,
                        "baseline_value":  round(baseline_social, 1),
                        "drop":            round(drop, 1),
                        "threshold":       threshold,
                        "gaze_direction":  record.get("gaze_direction"),
                    }
                ))
            already_flagged = True
        else:
            already_flagged = False

    return alerts


# ---------------------------------------------------------------------------
# Detector 4 — HYPERACTIVITY_SPIKE
# ---------------------------------------------------------------------------

def detect_hyperactivity_spike(records: list, baseline: dict) -> list:
    """
    HYPERACTIVITY_SPIKE: combined energy (energy_score + activity_level)
    is >= 40 points above the personal energy baseline in a single day.

    Only checked after the baseline period.
    """
    alerts          = []
    threshold       = baseline["thresholds"]["hyperactivity"]
    baseline_energy = baseline["energy_combined_mean"]
    person_id       = records[0]["person_id"]

    for i, record in enumerate(records):
        if i < 3:
            continue

        if not record.get("person_detected", True):
            continue

        combined = record["energy_score"] + record["activity_level"]
        spike    = combined - baseline_energy

        if spike >= threshold:
            alerts.append(_make_alert(
                person_id   = person_id,
                alert_type  = "HYPERACTIVITY_SPIKE",
                severity    = "MEDIUM",
                detected_on = record["date"],
                description = (
                    f"Combined energy ({combined}) is {spike:.1f} points above "
                    f"personal baseline ({baseline_energy:.1f})."
                ),
                details = {
                    "metric":          "combined_energy",
                    "current_value":   combined,
                    "baseline_value":  round(baseline_energy, 1),
                    "spike":           round(spike, 1),
                    "threshold":       threshold,
                    "energy_score":    record["energy_score"],
                    "activity_level":  record["activity_level"],
                }
            ))

    return alerts


# ---------------------------------------------------------------------------
# Detector 5 — REGRESSION
# ---------------------------------------------------------------------------

def detect_regression(records: list, baseline: dict) -> list:
    """
    REGRESSION: student was recovering (wellbeing increasing day-over-day)
    for 3+ consecutive days, then suddenly drops more than 15 points in one day.

    This catches students who seemed to be getting better but then relapsed.
    Requires at least 4 records (3 recovery + 1 drop day).
    """
    alerts    = []
    person_id = records[0]["person_id"]
    threshold = baseline["thresholds"]["regression_drop"]

    # We need index i (the drop day) and records[i-3 .. i-1] (recovery window)
    for i in range(3, len(records)):
        current_record = records[i]

        if not current_record.get("person_detected", True):
            continue

        # The 3 days before the current day must all be detected
        prev_3 = records[i - 3: i]
        all_detected = all(r.get("person_detected", True) for r in prev_3)
        if not all_detected:
            continue

        # Were those 3 days a recovery? Each day must be higher than the one before.
        day_a, day_b, day_c = prev_3[0], prev_3[1], prev_3[2]
        was_recovering = (
            day_b["wellbeing_score"] > day_a["wellbeing_score"] and
            day_c["wellbeing_score"] > day_b["wellbeing_score"]
        )

        if not was_recovering:
            continue

        # Did today drop sharply vs yesterday?
        prev_wellbeing    = records[i - 1]["wellbeing_score"]
        current_wellbeing = current_record["wellbeing_score"]
        drop              = prev_wellbeing - current_wellbeing

        if drop > threshold:
            alerts.append(_make_alert(
                person_id   = person_id,
                alert_type  = "REGRESSION",
                severity    = "HIGH",
                detected_on = current_record["date"],
                description = (
                    f"After 3 days of recovery ({day_a['wellbeing_score']} → "
                    f"{day_b['wellbeing_score']} → {day_c['wellbeing_score']}), "
                    f"wellbeing dropped {drop} points today ({prev_wellbeing} → {current_wellbeing})."
                ),
                details = {
                    "metric":           "wellbeing_score",
                    "current_value":    current_wellbeing,
                    "previous_value":   prev_wellbeing,
                    "drop":             drop,
                    "threshold":        threshold,
                    "recovery_start":   day_a["date"],
                    "recovery_scores":  [r["wellbeing_score"] for r in prev_3],
                }
            ))

    return alerts


# ---------------------------------------------------------------------------
# Detector 6 — GAZE_AVOIDANCE
# ---------------------------------------------------------------------------

def detect_gaze_avoidance(records: list, baseline: dict) -> list:
    """
    GAZE_AVOIDANCE: eye_contact_detected is False for 3 or more consecutive days.
    Alert fires on the 3rd consecutive day without eye contact.
    """
    alerts      = []
    person_id   = records[0]["person_id"]
    consecutive = []   # running list of no-eye-contact days

    for record in records:
        if not record.get("person_detected", True):
            consecutive = []
            continue

        if not record.get("eye_contact_detected", True):
            consecutive.append(record)
        else:
            consecutive = []    # eye contact restored — reset streak

        if len(consecutive) == 3:
            alerts.append(_make_alert(
                person_id   = person_id,
                alert_type  = "GAZE_AVOIDANCE",
                severity    = "MEDIUM",
                detected_on = record["date"],
                description = (
                    f"No eye contact detected for 3 consecutive days "
                    f"(since {consecutive[0]['date']})."
                ),
                details = {
                    "metric":           "eye_contact_detected",
                    "consecutive_days": 3,
                    "start_date":       consecutive[0]["date"],
                    "dates":            [r["date"] for r in consecutive],
                }
            ))

    return alerts


# ---------------------------------------------------------------------------
# Detector 7 — ABSENCE_FLAG
# ---------------------------------------------------------------------------

def detect_absence_flag(records: list, baseline: dict) -> list:
    """
    ABSENCE_FLAG: person not detected (person_detected = false) for 2 or more
    consecutive days. A welfare check is recommended.

    Alert fires on the 2nd consecutive absent day.
    """
    alerts      = []
    person_id   = records[0]["person_id"]
    consecutive = []   # running list of absent days

    for record in records:
        if not record.get("person_detected", True):
            consecutive.append(record)
        else:
            consecutive = []    # person is back — reset streak

        if len(consecutive) == 2:
            alerts.append(_make_alert(
                person_id   = person_id,
                alert_type  = "ABSENCE_FLAG",
                severity    = "HIGH",
                detected_on = record["date"],
                description = (
                    f"Person not detected for 2 consecutive days "
                    f"(since {consecutive[0]['date']}). Welfare check recommended."
                ),
                details = {
                    "metric":                  "person_detected",
                    "consecutive_absent_days": 2,
                    "start_date":              consecutive[0]["date"],
                    "dates":                   [r["date"] for r in consecutive],
                }
            ))

    return alerts


# ---------------------------------------------------------------------------
# Master runner — calls all 7 detectors for every student
# ---------------------------------------------------------------------------

def run_all_detectors(records_by_person: dict) -> list:
    """
    Runs all 7 anomaly detectors for every student in the dataset.

    Parameters:
        records_by_person (dict): Output of loader.load_daily_files()

    Returns:
        list: All alerts from all students, flat list sorted by detected_on date.
    """
    # List of all detector functions — easy to add more later
    detectors = [
        detect_sudden_drop,
        detect_sustained_low,
        detect_social_withdrawal,
        detect_hyperactivity_spike,
        detect_regression,
        detect_gaze_avoidance,
        detect_absence_flag,
    ]

    all_alerts = []

    for person_id, records in records_by_person.items():
        if not records:
            continue

        # Build this student's personal baseline
        baseline = compute_baseline(records)

        # Run every detector and collect results
        for detector_fn in detectors:
            try:
                alerts = detector_fn(records, baseline)
                all_alerts.extend(alerts)
            except Exception as e:
                print(f"  Warning: {detector_fn.__name__} failed for {person_id}: {e}")

    # Sort all alerts by date so the report is chronological
    all_alerts.sort(key=lambda a: a["detected_on"])

    return all_alerts
