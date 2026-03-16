# Behavioral Anomaly & Early Distress Detection
**Sentio Mind · POC Assignment · Project 5**

GitHub: https://github.com/Sentiodirector/sentio-poc-anomaly-detection.git
Branch: FirstName_LastName_RollNumber

---

## Why This Exists

Sentio Mind generates wellbeing scores and trait charts. But it never says "something is wrong." A student can be in distress for three consecutive days and unless someone manually checks every profile every morning, it goes unnoticed. This project adds that proactive alert layer. No cameras, no video — pure data analysis on existing JSON files.

---

## What You Receive

```
p5_anomaly_detection/
├── sample_data/
│   ├── analysis_Day1.json      ← Sentio Mind output format, one per day
│   ├── analysis_Day2.json
│   ├── analysis_Day3.json
│   ├── analysis_Day4.json
│   └── analysis_Day5.json
├── anomaly_detection.py        ← your template — copy to solution.py
├── anomaly_detection.json      ← schema for alert_feed.json
└── README.md
```

If you don't have real Sentio Mind data, generate synthetic data or use a public student wellbeing dataset. Minimum fields needed: person_id, daily wellbeing score (0–100), social_engagement (0–100).

---

## What You Must Build

Run `python solution.py` → it must produce:

1. `alert_digest.html` — readable counsellor report, works offline
2. `alert_feed.json` — follows `anomaly_detection.json` schema exactly

### The 7 Anomaly Patterns

| Category | Trigger |
|---|---|
| SUDDEN_DROP | Wellbeing drops ≥ 20 pts vs baseline. If baseline std > 15, threshold raises to 30. |
| SUSTAINED_LOW | Wellbeing below 45 for 3+ consecutive days |
| SOCIAL_WITHDRAWAL | social_engagement drops ≥ 25 pts + dominant gaze is "down" or "side" |
| HYPERACTIVITY_SPIKE | physical_energy + movement_energy combined ≥ 40 pts above baseline |
| REGRESSION | 3+ days of improving scores, then drops > 15 pts in one day |
| GAZE_AVOIDANCE | No eye contact detected for 3+ consecutive days |
| ABSENCE_FLAG | Person not seen in any video for 2+ consecutive days |

### Baseline Rule

Use the first 3 days of data per person. If fewer than 3 days available, use all available days.
Compute: mean wellbeing, std wellbeing, mean per trait, most common gaze direction.

### Severity

- SUDDEN_DROP with delta > 35 → **urgent**
- SUDDEN_DROP with delta 20–35 → **monitor**
- SUSTAINED_LOW → **urgent**
- ABSENCE_FLAG → **urgent**
- All others → **monitor**

### Report Sections

1. Today's alerts sorted by severity — name, alert type badge, one-sentence description, 5-day sparkline
2. School summary — persons flagged today vs yesterday, most common anomaly this week
3. Persistent alerts — persons flagged for 3+ consecutive days

---

## Hard Rules

- Do not rename functions in `anomaly_detection.py`
- Do not change key names in `anomaly_detection.json`
- All thresholds in the THRESHOLDS dict at top of your script
- No OpenCV needed — pure Python + numpy
- Python 3.9+, no Jupyter notebooks

## Libraries

```
numpy==1.26.4   (everything else is Python stdlib)
```

---

## Submit

| # | File | What |
|---|------|------|
| 1 | `solution.py` | Working script |
| 2 | `alert_digest.html` | Counsellor report |
| 3 | `alert_feed.json` | Alert feed matching schema |
| 4 | `demo.mp4` | Screen recording under 2 min |

Push to your branch only. Do not touch main.

---

## Bonus

Peer-comparison anomaly: flag a person whose wellbeing is more than 2 standard deviations below the class average on the same day, even if their personal baseline is also low.

*Sentio Mind · 2026*
