# Behavioral Anomaly & Early Distress Detection

> A lightweight alert engine that watches the numbers so counsellors can watch the people.

---

## 🎥 Demo Video

Watch the full working demo here: https://drive.google.com/file/d/1_DY3qSPy7-SYoh-jQTijniMO_q-L8D2M/view?usp=sharing

---

## Why This Exists

Sentio Mind already does the hard part — it processes video feeds, computes engagement scores, tracks gaze patterns, and produces daily snapshots for every student. What it didn't do was *act* on any of that. The scores existed. Charts existed. But if a student's wellbeing quietly collapsed over three days, nobody got a notification. A counsellor would have to manually dig through data to notice, which means they often didn't.

I added the alert layer on top of that. No changes to the video processing, no new cameras, no new infrastructure — just one extra step: the moment a student's numbers deviate in a way that matters clinically, it raises a flag and tells a human being.

---

## What It Actually Does

Every day, the engine reads the latest JSON snapshot from `sample_data/`. For each student, I compute a personal rolling baseline from their own history (not some global average — someone who's naturally quiet shouldn't trigger the same alert as an extrovert going quiet). Then it checks whether today's metrics cross any of seven anomaly thresholds.

If something looks wrong, it generates two outputs:

- **`alert_digest.html`** — A browser-based counsellor report. No internet required, no login, no dashboard to navigate. Open it, scan it in under 30 seconds, decide who needs a check-in.
- **`alert_feed.json`** — A machine-readable version of the same data, returned via a new `/get_alerts` Flask endpoint for downstream systems that want to automate triage or integrate with school management software.

---

## The 7 Anomaly Categories

I didn't pick arbitrary numbers here. Each threshold was chosen to reflect something a trained counsellor would actually flag:

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

The `sample_data/` folder contains five days of synthetic snapshots (`day_1.json` through `day_5.json`). I crafted each mock profile to trigger a specific anomaly category — so all 7 detection rules can be verified without ambiguity. Each student hits exactly one case: a sudden crash, a sustained slump, a social withdrawal, an energy spike, a recovery that collapses, persistent gaze avoidance, and an unexplained absence.

The students are named after Indian cricketers, because making test data boring is a choice and I didn't make it.

---

## Design Choices Worth Knowing

**Weighted composite wellbeing score.** Wellbeing isn't a single number — I compute it as a weighted blend of social engagement (40%), energy traits (40%), and self-reported mood (20%). Social and physical signals tend to be more reliable than self-report alone, and the weighting reflects that.

**Personal rolling baselines, not global averages.** I use each student's own first three days of data as their baseline (or all available data if fewer than three days exist). An introvert's baseline looks different from an extrovert's, and the alert logic respects that. If the standard deviation of a baseline is high (> 15), I increase the drop threshold by 50% to reduce false positives for naturally variable students.

**Prioritized alert chain.** The order in which I check anomaly rules matters. `ABSENCE_FLAG` fires first — you can't assess someone who isn't there. `SUDDEN_DROP` comes next (most urgent clinical signal), and `GAZE_AVOIDANCE` comes last (subtler, longer-developing signal). This ensures critical alerts surface without being masked by softer ones.

**Offline-first HTML output.** The digest uses inline SVG for sparklines and system fonts only. No CDN calls, no JavaScript frameworks, no internet dependency. A counsellor in a school with unreliable connectivity can still open it and get the full picture.

**Severity vocabulary aligned to the PoC schema.** I express alert severities as `urgent`, `monitor`, and `informational` — not `HIGH/MEDIUM/LOW` — to match the integration contract expected by downstream systems.

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
├── solution.py                                    # Core engine: analysis + HTML/JSON generation + Flask
├── alert_digest.html                              # Generated counsellor dashboard (offline, browser-ready)
├── alert_feed.json                                # Generated machine-readable alerts (schema-compliant)
├── README.md                                      # This file
├── Yashwanth_Sai_Kasarabada_220103012.mp4         # Demo video (required submission artifact)
└── sample_data/                                   # Daily JSON snapshots (8 days of synthetic input)
    ├── day_1_2023-10-03.json
    ├── day_2_2023-10-04.json
    ├── day_3_2023-10-05.json
    ├── day_4_2023-10-06.json
    ├── day_5_2023-10-07.json
    ├── day_6_2023-10-08.json
    ├── day_7_2023-10-09.json
    └── day_8_2023-10-10.json
```

---

## Deliverables

| File | Status | Notes |
|------|--------|-------|
| `solution.py` | ✅ | Core implementation; original function signatures preserved |
| `alert_feed.json` | ✅ | Full PoC schema compliance |
| `alert_digest.html` | ✅ | Offline counsellor dashboard |
| `README.md` | ✅ | Documentation |
| `Yashwanth_Sai_Kasarabada_220103012.mp4` | ✅ | Demo video as per submission requirement |

---

**Developed by:** Yashwanth Sai Kasarabada (220103012)  
**Institute:** Indian Institute of Information Technology Senapati, Manipur  
**Branch:** CSE AI & DS, 2026  
**LinkedIn:** [linkedin.com/in/yashwanth-sai-kasarabada](https://www.linkedin.com/in/yashwanth-sai-kasarabada-ba4265258/)
