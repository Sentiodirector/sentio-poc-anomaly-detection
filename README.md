# Behavioral Anomaly and Early Distress Detection

Sentio Mind POC Assignment - Project 5

## Why This Exists
Sentio Mind generates wellbeing scores and trait charts, but it never raises proactive alerts. This project adds that alert layer using only existing JSON outputs (no video or camera processing).

## Deliverables
- solution.py
- alert_digest.html
- alert_feed.json
- demo.mp4 (screen recording under 2 minutes)

## Workflow Overview
1. Ingest daily JSON files from sample_data/.
2. Normalize each person record (names, traits, gaze, eye contact, presence).
3. Build per-student history and compute baseline from the first 3 days.
4. Run all detectors and create alerts with severity and descriptions.
5. Assign alert IDs and summarize school-wide metrics.
6. Write alert_feed.json and generate alert_digest.html.

## Setup
- Python 3.9+ and numpy 1.26.4.
- Place daily JSON files in sample_data/ (one file per day).

## How To Run
1. Activate your virtual environment (if you created one).
2. Run:
   python solution.py
3. Outputs:
   - alert_digest.html (offline counselor report)
   - alert_feed.json (machine-readable alerts)

## How To Check Results
- Open alert_digest.html in a browser and verify alerts are grouped as expected.
- Open alert_feed.json and confirm top-level keys match the required schema.
- Confirm severity values are only: urgent, monitor, informational.
- Check that all 7 anomaly categories appear when sample data is present.

## Input Expectations
Minimum fields per person per day:
- person_id
- wellbeing (0-100)
- social_engagement (0-100)
Optional but supported:
- traits (physical_energy, movement_energy, etc.)
- gaze_direction (down, side, forward, up)
- eye_contact or eye_contact_duration
- person_detected
- person_name, profile_image_b64

## Output Schema
alert_feed.json matches the provided anomaly_detection.json schema exactly.
Top-level keys must be:
- source
- generated_at
- school
- alert_summary
- alerts
- absence_flags
- school_summary

## Anomaly Categories
- SUDDEN_DROP: wellbeing drops >= 20 vs baseline (>= 30 if baseline std > 15)
- SUSTAINED_LOW: wellbeing below 45 for 3+ consecutive days
- SOCIAL_WITHDRAWAL: social_engagement drop >= 25 and gaze is down or side
- HYPERACTIVITY_SPIKE: physical_energy + movement_energy >= 40 above baseline
- REGRESSION: 3+ days improving, then drop > 15 in one day
- GAZE_AVOIDANCE: no eye contact for 3+ consecutive days
- ABSENCE_FLAG: person not detected for 2+ consecutive days

## Severity Rules
- SUDDEN_DROP with delta > 35 -> urgent
- SUDDEN_DROP with delta 20-35 -> monitor
- SUSTAINED_LOW -> urgent
- ABSENCE_FLAG -> urgent
- All others -> monitor

## Hard Rules
- Do not rename functions in the template.
- Do not change key names in the JSON schema.
- All thresholds are defined in THRESHOLDS at the top of solution.py.
- No OpenCV or video processing.
- Python 3.9+ only. No Jupyter notebooks.
- HTML report must work offline (no external CDN dependencies).

## Notes
If your dataset has a different JSON structure, adjust parsing in load_daily_data().
