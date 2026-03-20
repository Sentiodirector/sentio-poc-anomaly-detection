"""
baseline.py
-----------
Computes a personal baseline for each student from their first 3 days of data.

Why a baseline?
    Everyone has a different "normal". A wellbeing score of 55 might be great
    for one student but bad for another. The baseline captures each person's
    individual normal so we compare them to themselves, not to a global average.

Rule (from assignment):
    - Use first 3 days as baseline (or all days if fewer than 3 available).
    - If the wellbeing standard deviation > 15, increase drop thresholds by 50%
      to avoid false positives for naturally volatile students.
"""

import statistics


def compute_baseline(records: list) -> dict:
    """
    Calculates the personal baseline metrics for one student.

    Parameters:
        records (list): All daily records for ONE student, sorted by date.

    Returns:
        dict with:
            - wellbeing_mean       : average wellbeing in baseline period
            - social_mean          : average social engagement in baseline period
            - energy_combined_mean : average (energy_score + activity_level) in baseline
            - wellbeing_std        : standard deviation of wellbeing in baseline
            - high_variance        : True if std > 15 (thresholds will be higher)
            - thresholds           : dict of adjusted detection thresholds
    """
    # Take only the first 3 days as the baseline window
    baseline_records = records[:3] if len(records) >= 3 else records

    # Only include days where the person was actually detected
    detected = [r for r in baseline_records if r.get("person_detected", True)]

    # Extract the metric values we care about
    wellbeing_values = [r["wellbeing_score"] for r in detected]
    social_values    = [r["social_engagement_score"] for r in detected]
    energy_values    = [r["energy_score"] + r["activity_level"] for r in detected]

    # Compute averages (fall back to sensible defaults if no detected days)
    wellbeing_mean      = statistics.mean(wellbeing_values) if wellbeing_values else 50
    social_mean         = statistics.mean(social_values)    if social_values    else 50
    energy_combined_mean = statistics.mean(energy_values)  if energy_values    else 100

    # Standard deviation tells us how "jumpy" this student's wellbeing is
    wellbeing_std = statistics.stdev(wellbeing_values) if len(wellbeing_values) > 1 else 0

    # If the student is naturally volatile, raise the bar to avoid false alarms
    high_variance = wellbeing_std > 15

    # --- Base thresholds (from assignment spec) ---
    sudden_drop_threshold   = 20   # SUDDEN_DROP: drop >= 20 vs baseline
    social_drop_threshold   = 25   # SOCIAL_WITHDRAWAL: social drop >= 25
    hyperactivity_threshold = 40   # HYPERACTIVITY_SPIKE: combined energy >= 40 above baseline
    regression_drop_threshold = 15 # REGRESSION: drop > 15 after recovery

    # Increase thresholds by 50% for high-variance students
    if high_variance:
        sudden_drop_threshold    = int(sudden_drop_threshold    * 1.5)  # 30
        social_drop_threshold    = int(social_drop_threshold    * 1.5)  # 37
        hyperactivity_threshold  = int(hyperactivity_threshold  * 1.5)  # 60
        regression_drop_threshold = int(regression_drop_threshold * 1.5) # 22

    return {
        "wellbeing_mean":        round(wellbeing_mean, 2),
        "social_mean":           round(social_mean, 2),
        "energy_combined_mean":  round(energy_combined_mean, 2),
        "wellbeing_std":         round(wellbeing_std, 2),
        "high_variance":         high_variance,
        "thresholds": {
            "sudden_drop":      sudden_drop_threshold,
            "social_drop":      social_drop_threshold,
            "hyperactivity":    hyperactivity_threshold,
            "regression_drop":  regression_drop_threshold,
        }
    }
