# Behavioral Anomaly & Early Distress Detection
**Sentio Mind · POC Assignment · Project 5**

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [How It Works](#how-it-works)
3. [Project Structure](#project-structure)
4. [Setup & Installation](#setup--installation)
5. [Running the Solution](#running-the-solution)
6. [The 7 Anomaly Categories](#the-7-anomaly-categories)
7. [Baseline Computation](#baseline-computation)
8. [Output Files](#output-files)
9. [Dataset](#dataset)
10. [Bonus Feature](#bonus-feature)

---

## Problem Statement

Sentio Mind already generates daily wellbeing scores and behavioral trait charts for each student from video analysis. However, it never proactively says *"something is wrong."*

A student could show signs of distress — declining wellbeing, social withdrawal, persistent low scores — for multiple days in a row, and no one would be alerted unless someone manually checked every profile every morning.

**This project adds the alert layer.** No video processing, no cameras, no OpenCV — purely data analysis on top of the existing JSON outputs that Sentio Mind already produces.

---

## How It Works

```
sample_data/
  analysis_Day1.json
  analysis_Day2.json
  ...
  analysis_Day5.json
         │
         ▼
    solution.py
         │
    ┌────┴────┐
    ▼         ▼
alert_feed.json    alert_digest.html
```

1. `solution.py` reads all `analysis_*.json` files from the `sample_data/` folder
2. For each student, it builds a personal history and computes a **baseline** from the first 3 days
3. It then runs **7 anomaly detectors** on each day's data against that baseline
4. All triggered alerts are written to `alert_feed.json`
5.  `alert_digest.html` is generated for the school counsellor

---

## Project Structure

```
sentio-anomaly-detection/
├── sample_data/
│   ├── analysis_Day1.json      ← Day 1 (2026-03-16)
│   ├── analysis_Day2.json      ← Day 2 (2026-03-17)
│   ├── analysis_Day3.json      ← Day 3 (2026-03-18)
│   ├── analysis_Day4.json      ← Day 4 (2026-03-19)
│   └── analysis_Day5.json      ← Day 5 (2026-03-20) 
├── solution.py                 ← main detection script
├── alert_digest.html           ← counsellor report
├── alert_feed.json             
└── README.md
└── demo.mp4
```

---

## Setup & Installation

**Requirements:** Python 3.9+

Install the only external dependency:

```bash
pip install numpy==1.26.4
```

---

## Running the Solution

```bash
python solution.py
```

**Expected output:**

```
Loaded 5 days: ['2026-03-17', '2026-03-18', '2026-03-19', '2026-03-20', '2026-03-21']
  Report → alert_digest.html

====================================================
  Total alerts : 15
  Urgent       : 2
  Monitor      : 13
  Absence flags: 1
  Report → alert_digest.html
  JSON   → alert_feed.json
====================================================
```

This produces two files:
- `alert_digest.html`
- `alert_feed.json`

---

## The 7 Anomaly Categories

### 1. `SUDDEN_DROP`
**Trigger:** Wellbeing drops ≥ 20 points vs personal baseline in a single day.

- If baseline std > 15, threshold raises to 30 to avoid false positives on high-variability students
- **Severity:** `urgent` if delta > 35 pts · `monitor` if delta 20–35 pts
- **Example:** Baseline ~73 → drops to 31 today (delta = 42) → **urgent**

---

### 2. `SUSTAINED_LOW`
**Trigger:** Wellbeing stays below 45 for 3+ consecutive days.

- **Severity:** always `urgent`
- Distinguishes chronic distress from a one-off bad day
- **Example:** Scores of 40, 38, 42 across 3 days → alert triggered

---

### 3. `SOCIAL_WITHDRAWAL`
**Trigger:** `social_engagement` drops ≥ 25 pts below baseline **AND** `gaze_direction` is `"down"` or `"side"` on the same day.

- Both conditions must be true — a social drop alone is not enough
- **Severity:** `monitor`
- **Example:** Baseline social 70 → today 42 (delta 28) + gaze `"down"` → alert

---

### 4. `HYPERACTIVITY_SPIKE`
**Trigger:** `physical_energy + movement_energy` combined is ≥ 40 pts above combined baseline.

- **Severity:** `monitor`
- May indicate anxiety, mania, or extreme stress
- **Example:** Baseline combined ~95 → today 185 (delta 90) → alert

---

### 5. `REGRESSION`
**Trigger:** Student improved for 3+ consecutive days, then drops > 15 pts in one day.

- **Severity:** `monitor`
- Catches students who appeared to be recovering but relapsed
- **Example:** 50 → 55 → 62 → 65, then drops to 39 (drop = 26 > 15) → alert

---

### 6. `GAZE_AVOIDANCE`
**Trigger:** `eye_contact` is `False` for 3+ consecutive days.

- **Severity:** `monitor`
- May indicate social anxiety or avoidant behaviour
- **Example:** No eye contact on Days 3, 4, 5 → alert triggered on Day 5

---

### 7. `ABSENCE_FLAG`
**Trigger:** Student not detected in any session for 2+ consecutive days.

- **Severity:** `urgent`
- Handled at the data level — absence means no JSON entry at all
- Includes `last_seen_date` and a welfare check recommendation

---

## Baseline Computation

A personal baseline is computed per student from the **first 3 days** of available data (or all days if fewer than 3 exist).

| Field | How Computed |
|---|---|
| `wellbeing_mean` | Mean wellbeing across baseline window |
| `wellbeing_std` | Std dev of wellbeing — used to adjust SUDDEN_DROP threshold |
| `social_mean` | Mean social engagement |
| `physical_energy_mean` | Mean physical energy |
| `movement_energy_mean` | Mean movement energy |
| `avg_gaze` | Most common gaze direction in baseline window |

**High-variability rule:** If `wellbeing_std > 15`, SUDDEN_DROP threshold increases from 20 → 30 points to prevent false positives.

---

## Output Files

### `alert_feed.json`

Sample alert entry:

```json
{
  "alert_id": "ALT_001",
  "person_id": "SCHOOL_P0001",
  "person_name": "Aarav Sharma",
  "date": "2026-03-21",
  "severity": "urgent",
  "category": "SUDDEN_DROP",
  "title": "Sudden wellbeing drop detected",
  "description": "Aarav Sharma's wellbeing dropped from a baseline of 73 to 31 today — a 42-point fall. Lowest trait: social_engagement at 18. Dominant gaze: down.",
  "baseline_wellbeing": 73,
  "today_wellbeing": 31,
  "delta": -42,
  "days_flagged_consecutively": 1,
  "trend_last_5_days": [72, 74, 71, 75, 31],
  "lowest_trait": "social_engagement",
  "lowest_trait_value": 18,
  "recommended_action": "Schedule pastoral check-in today"
}
```

Top-level keys also include `alert_summary`, `absence_flags`, and `school_summary`.

### `alert_digest.html`

Self-contained counsellor report with four sections:

1. **School Summary** — students tracked, flagged today/yesterday, avg wellbeing, top anomaly
2. **Today's Alerts** — sorted urgent first, each card has name, severity badge, description, 5-day sparkline, recommended action
3. **Absence Flags** — students missing 2+ days with last seen date
4. **Persistent Alerts** — students flagged 3+ consecutive days

---

## Dataset

No real Sentio Mind data was available in the repository. Synthetic data was manually crafted and included in `sample_data/` covering 5 students over 5 days (2026-03-16 to 2026-03-20), with all 7 anomaly types deliberately seeded:

| Student | Anomaly |
|---|---|
| Aarav Sharma | `SUDDEN_DROP` — wellbeing crashes from ~73 to 31 on Day 5 (urgent) |
| Priya Rajan | `SUSTAINED_LOW` + `GAZE_AVOIDANCE` — wellbeing < 45 and no eye contact Days 3–5 |
| Arjun Mehta | `SOCIAL_WITHDRAWAL` Day 4 + `REGRESSION` Day 5 |
| Sneha Kulkarni | `HYPERACTIVITY_SPIKE` — combined energy spikes ~90 pts above baseline on Day 4 |
| Rohan Desai | `ABSENCE_FLAG` — not detected on Days 4 and 5 |

Each daily JSON file follows this structure:

```json
{
  "date": "2026-03-21",
  "school": "JAGRAN PUBLIC SCHOOL",
  "persons": {
    "SCHOOL_P0001": {
      "person_id": "SCHOOL_P0001",
      "person_info": { "name": "Aarav Sharma", "profile_image_b64": "" },
      "wellbeing": 31,
      "social_engagement": 18,
      "gaze_direction": "down",
      "eye_contact": false,
      "physical_energy": 20,
      "movement_energy": 15
    }
  }
}
```

---


## Bonus Feature

**Peer Comparison Anomaly:** Flags a student whose wellbeing is more than **2 standard deviations below the class average** on the same day — even if their personal baseline is also low.

This catches an important edge case: a student whose personal baseline is already depressed won't trigger `SUDDEN_DROP` (no personal drop detected), but they may still be significantly worse off than everyone else in the class. This bonus detector catches exactly that.

- Computes class mean and std from all other students present that day
- Flags if `(student_wellbeing − class_mean) / class_std < −2`
- Stored under category `SUDDEN_DROP` in the JSON (closest valid schema category — integration contract cannot be modified)
- Severity: `monitor`

---


---

*Sentio Mind · 2026*
