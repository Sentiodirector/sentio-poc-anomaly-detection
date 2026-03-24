# Behavioral Anomaly & Early Distress Detection

> A lightweight alert engine that watches the numbers so counsellors can watch the people.

---

## Why This Exists

Sentio Mind already does the hard part — it processes video feeds, computes engagement scores, tracks gaze patterns, and produces daily snapshots for every student. What it didn't do was *act* on any of that. The scores existed. Charts existed. But if a student's wellbeing quietly collapsed over three days, nobody got a notification. A counsellor would have to manually dig through data to notice, which means they often didn't.

This project adds the alert layer. It sits on top of the existing pipeline — no changes to the video processing, no new cameras, no new infrastructure — and does one thing: the moment a student's numbers deviate in a way that matters clinically, it raises a flag and tells a human being.

---

## What It Actually Does

Every day, the engine reads the latest JSON snapshot from `sample_data/`. For each student, it builds a personal rolling baseline from their own history (not some global average — someone who's naturally quiet shouldn't trigger the same alert as an extrovert going quiet). Then it checks whether today's metrics cross any of seven anomaly thresholds.

If something looks wrong, it generates two outputs:

- **`alert_digest.html`** — A browser-based counsellor report. No internet required, no login, no dashboard to navigate. Open it, scan it in under 30 seconds, decide who needs a check-in.
- **`alert_feed.json`** — A machine-readable version of the same data, returned via a new `/get_alerts` Flask endpoint for downstream systems that want to automate triage or integrate with school management software.

---

## The 7 Anomaly Categories

These aren't arbitrary thresholds. Each one was chosen to reflect something a trained counsellor would actually care about:

| # | Category | What It Catches | Trigger Threshold |
|---|----------|-----------------|-------------------|
| 1 | **SUDDEN_DROP** | Overnight wellbeing crash | ≥ 20 pts below personal baseline in one day |
| 2 | **SUSTAINED_LOW** | Prolonged low mood | Wellbeing < 45 for ≥ 3 consecutive days |
| 3 | **SOCIAL_WITHDRAWAL** | Disengagement + avoidance | Social engagement down ≥ 25 pts + gaze mostly downward |
| 4 | **HYPERACTIVITY_SPIKE** | Restlessness or agitation | Energy traits ≥ 40 pts above baseline |
| 5 | **REGRESSION** | Recovery that collapses | Improving 3+ days, then drops > 15 pts in one day |
| 6 | **GAZE_AVOIDANCE** | Persistent eye-contact loss | Zero eye contact for 3+ consecutive days |
| 7 | **ABSENCE_FLAG** | Welfare check trigger | Not detected for 2+ consecutive days |

All seven are actively coded, tested against purpose-built mock profiles, and surfaced visually in the HTML digest.

---

## How to Run

```bash
python solution.py
```

That's it. One command does everything:

1. Reads daily JSON snapshots from `sample_data/`
2. Computes rolling personal baselines per student
3. Evaluates all 7 anomaly rules against today's metrics
4. Exports `alert_feed.json` (machine-readable, schema-compliant)
5. Generates `alert_digest.html` (offline, zero CDN dependencies)
6. Starts the Flask server at `http://127.0.0.1:5000`

**Requirements:** Python 3.9+ with standard library only. No notebooks, no external servers, no pip installs.

---

## The Sample Data

The `sample_data/` folder contains five days of synthetic student snapshots, each named `day_1.json` through `day_5.json`. The mock profiles are deliberately crafted so that every anomaly path fires at least once — you can verify all 7 detection rules are working without ambiguity.

The students are named after Indian cricketers, because making test data boring is a choice and we didn't make it.

Here's what fires and why:

| Student | Category | Severity | What Happens in the Data |
|---------|----------|----------|--------------------------|
| Arjun Mehta | `SUDDEN_DROP` | urgent | Wellbeing crashes from a ~74 baseline down to 28 on day 4 — a 46-point drop |
| Rohan Sharma | `SUSTAINED_LOW` | urgent | Wellbeing stays below 45 on days 3, 4, and 5 |
| Ananya Singh | `SOCIAL_WITHDRAWAL` | monitor | Social engagement drops 33 pts below baseline, gaze turns downward on day 4 |
| Vikram Nair | `HYPERACTIVITY_SPIKE` | informational | Energy jumps from a ~55 baseline to 100 on day 5 (+45 pts) |
| Meera Pillai | `REGRESSION` | monitor | Recovers steadily for three days (52 → 56 → 62 → 67), then drops 21 pts on day 5 |
| Kabir Hassan | `GAZE_AVOIDANCE` | informational | Gaze is avoidant on days 3, 4, and 5 — three consecutive |
| Siddharth Bose | `SUSTAINED_LOW` | urgent | Similar sustained drop below 45 from day 3 onward |
| Priya Rajan | `ABSENCE_FLAG` | — | Not detected on days 4 or 5 — welfare check triggered |

---

## Design Choices Worth Knowing

**Weighted composite wellbeing score.** Wellbeing isn't a single number — it's a blend of social engagement (40%), energy traits (40%), and self-reported mood (20%). This mirrors how real behavioral assessment works: no single metric tells the full story, and this weighting reflects that social and physical signals tend to be more reliable than self-report alone.

**Personal rolling baselines, not global averages.** The engine uses each student's own first three days of data as their baseline (or all available data if fewer than three days exist). An introvert's baseline looks different from an extrovert's, and the alert logic respects that. If the standard deviation of someone's baseline is high (> 15), the drop threshold increases by 50% to reduce false positives for naturally variable students.

**Prioritized alert chain.** The order in which anomaly rules are checked matters. `ABSENCE_FLAG` fires first — you can't assess someone who isn't there. `SUDDEN_DROP` comes next (most urgent clinical signal), and `GAZE_AVOIDANCE` comes last (subtler, longer-developing signal). This ensures critical alerts surface without being masked by softer ones.

**Offline-first HTML output.** The digest uses inline SVG for sparklines and system fonts only. No CDN calls, no JavaScript frameworks, no internet dependency. A counsellor in a school with unreliable connectivity can still open it and get the full picture.

**Severity vocabulary aligned to the PoC schema.** Alert severities are expressed as `urgent`, `monitor`, and `informational` — not `HIGH/MEDIUM/LOW` — to match the integration contract expected by downstream systems.

---

## The `/get_alerts` Endpoint

Once `solution.py` is running, hit:

```
GET http://127.0.0.1:5000/get_alerts
```

It returns the full feed object — the same data as `alert_feed.json` — wrapped in the PoC schema envelope:

```json
{
  "_readme": "...",
  "source": "sentio-anomaly-engine",
  "generated_at": "...",
  "school": { ... },
  "alert_summary": { "total": 9, "urgent": 3, "monitor": 2, "informational": 4 },
  "alerts": [ ... ],
  "absence_flags": [ ... ],
  "school_summary": { ... }
}
```

Absence flags are in their own list — separate from `alerts[]` — exactly as the schema expects.

---

## Project Structure

```
sentio/
├── solution.py                  # Core engine: analysis + HTML/JSON generation + Flask
├── alert_digest.html            # Generated counsellor dashboard (offline, browser-ready)
├── alert_feed.json              # Generated machine-readable alerts (schema-compliant)
├── README.md                    # This file
├── sample_data/                 # Daily JSON snapshots (5 days of synthetic input)
│   ├── day_1.json
│   ├── day_2.json
│   ├── day_3.json
│   ├── day_4.json
│   └── day_5.json
└── assignment_assets/           # Mock data generator, debug tools, assignment PDF
```

---

## Deliverables

| File | Status | Notes |
|------|--------|-------|
| `solution.py` | ✅ | Core implementation; original function signatures preserved, helpers added |
| `alert_feed.json` | ✅ | Full PoC schema compliance, field-for-field verified |
| `alert_digest.html` | ✅ | Fully offline, inline SVG sparklines, system fonts only |
| `README.md` | ✅ | This document |

---

**Developed by:** Yashwanth Sai Kasarabada (220103012)  
**Institute:** Indian Institute of Information Technology Senapati, Manipur  
**Branch:** CSE AI & DS, 2026  
**LinkedIn:** [linkedin.com/in/yashwanth-sai-kasarabada](https://www.linkedin.com/in/yashwanth-sai-kasarabada-ba4265258/)
