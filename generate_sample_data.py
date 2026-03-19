"""
generate_sample_data.py
Creates synthetic Sentio Mind daily JSON files for testing solution.py.

Scenarios designed to trigger all 7 anomaly categories + bonus peer comparison:
  SCHOOL_P0001 — SUDDEN_DROP (urgent: delta > 35)
  SCHOOL_P0002 — SOCIAL_WITHDRAWAL
  SCHOOL_P0003 — SUSTAINED_LOW (urgent)
  SCHOOL_P0004 — REGRESSION (fixed: plateau day included)
  SCHOOL_P0005 — GAZE_AVOIDANCE
  SCHOOL_P0006 — HYPERACTIVITY_SPIKE (fixed: energy spike large enough)
  SCHOOL_P0007 — Absent last 2 days → ABSENCE_FLAG

Run: python generate_sample_data.py
Creates: sample_data/analysis_Day1.json ... analysis_Day7.json
"""

import json
import random
from pathlib import Path

random.seed(42)

DATA_DIR = Path("sample_data")
DATA_DIR.mkdir(exist_ok=True)

NUM_DAYS   = 7
START_DATE = "2026-03-13"   # Day 1 date (YYYY-MM-DD)

# Compute date string for each day index
from datetime import date, timedelta
_start = date.fromisoformat(START_DATE)
DAY_DATES = [str(_start + timedelta(days=i)) for i in range(NUM_DAYS)]


def clamp(val: float, lo: int = 0, hi: int = 100) -> int:
    return max(lo, min(hi, int(round(val))))


def base_person(name: str, base_wb: int = 65) -> dict:
    """Generate a stable baseline day for a person."""
    wb = clamp(base_wb + random.gauss(0, 3))
    se = clamp(wb + random.gauss(0, 5))
    pe = clamp(50 + random.gauss(0, 6))
    me = clamp(50 + random.gauss(0, 6))
    return {
        "person_info": {"name": name, "profile_image_b64": ""},
        "wellbeing":   wb,
        "traits": {
            "social_engagement":   se,
            "physical_energy":     pe,
            "movement_energy":     me,
            "focus":               clamp(60 + random.gauss(0, 8)),
            "emotional_stability": clamp(wb + random.gauss(0, 5)),
        },
        "gaze_direction": "forward",
        "eye_contact":    True,
        "detected":       True,
    }


# Pre-build per-person data for all 7 days
persons_data: dict = {}

# ── P0001: SUDDEN_DROP (urgent, delta > 35) ───────────────────────────────
# Stable ~72 for days 1–6, then crashes to ~28 on day 7
p1 = [base_person("Aarav Sharma", base_wb=72) for _ in range(6)]
p1_drop = base_person("Aarav Sharma", base_wb=72)
p1_drop["wellbeing"] = 28
p1_drop["traits"]["social_engagement"] = 20
p1_drop["gaze_direction"] = "down"
p1_drop["eye_contact"]    = False
p1.append(p1_drop)
persons_data["SCHOOL_P0001"] = p1

# ── P0002: SOCIAL_WITHDRAWAL ──────────────────────────────────────────────
# Stable baseline SE ~65; days 6–7 drop SE by 30+ pts AND gaze goes down
p2 = [base_person("Priya Nair", base_wb=65) for _ in range(5)]
for _ in range(2):   # days 6 & 7
    d = base_person("Priya Nair", base_wb=60)
    d["traits"]["social_engagement"] = clamp(65 - 32 + random.gauss(0, 3))  # -32 pts drop
    d["gaze_direction"] = "down"
    d["eye_contact"]    = False
    p2.append(d)
persons_data["SCHOOL_P0002"] = p2

# ── P0003: SUSTAINED_LOW (urgent) ────────────────────────────────────────
# Wellbeing below 45 every day from day 4 onwards (4 consecutive low days)
p3 = [base_person("Arjun Mehta", base_wb=62) for _ in range(3)]
for _ in range(4):
    d = base_person("Arjun Mehta", base_wb=38)
    d["wellbeing"] = clamp(38 + random.gauss(0, 3))
    p3.append(d)
persons_data["SCHOOL_P0003"] = p3

# ── P0004: REGRESSION ────────────────────────────────────────────────────
# Days 1–3: baseline ~55; days 4–6: recovering (non-strict: 58, 64, 64);
# day 7: drops 28 pts → triggers regression (drop > 15, recovery window non-strict)
p4_wb = [55, 54, 56,   58, 64, 64,   36]   # non-strict recovery (plateau on day 6)
p4 = []
for wb in p4_wb:
    d = base_person("Sneha Reddy", base_wb=wb)
    d["wellbeing"] = clamp(wb + random.gauss(0, 1))
    p4.append(d)
persons_data["SCHOOL_P0004"] = p4

# ── P0005: GAZE_AVOIDANCE ────────────────────────────────────────────────
# Days 1–4: normal; days 5–7: eye_contact=False every day (3 consecutive)
p5 = [base_person("Rahul Verma", base_wb=62) for _ in range(4)]
for _ in range(3):
    d = base_person("Rahul Verma", base_wb=58)
    d["gaze_direction"] = "down"
    d["eye_contact"]    = False
    p5.append(d)
persons_data["SCHOOL_P0005"] = p5

# ── P0006: HYPERACTIVITY_SPIKE ───────────────────────────────────────────
# Baseline PE+ME ~100 (each ~50); day 7: PE=95, ME=90 → combined=185, delta=85 > 40
p6 = []
for i in range(6):
    d = base_person("Kavya Iyer", base_wb=68)
    d["traits"]["physical_energy"] = clamp(50 + random.gauss(0, 4))
    d["traits"]["movement_energy"] = clamp(50 + random.gauss(0, 4))
    p6.append(d)
d_spike = base_person("Kavya Iyer", base_wb=68)
d_spike["traits"]["physical_energy"] = 95
d_spike["traits"]["movement_energy"] = 90
p6.append(d_spike)
persons_data["SCHOOL_P0006"] = p6

# ── P0007: ABSENCE_FLAG (not present days 6 & 7) ────────────────────────
p7 = [base_person("Vikram Pillai", base_wb=70) for _ in range(5)]
# Days 6 & 7: absent — no entry in those day files
persons_data["SCHOOL_P0007"] = p7  # only 5 entries


# ── Write daily files ─────────────────────────────────────────────────────
for day_idx in range(NUM_DAYS):
    day_persons: dict = {}

    for pid, entries in persons_data.items():
        if day_idx < len(entries):
            day_persons[pid] = entries[day_idx]
        # Persons with fewer entries than day_idx are simply absent (ABSENCE_FLAG)

    payload = {
        "date":    DAY_DATES[day_idx],
        "persons": day_persons,
    }
    out = DATA_DIR / f"analysis_Day{day_idx + 1}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Created {out}  ({len(day_persons)} persons)")

print(f"\nDone — {NUM_DAYS} day files written to '{DATA_DIR}/'")
print("Scenarios covered:")
print("  P0001 SUDDEN_DROP (urgent)     P0002 SOCIAL_WITHDRAWAL")
print("  P0003 SUSTAINED_LOW (urgent)   P0004 REGRESSION")
print("  P0005 GAZE_AVOIDANCE           P0006 HYPERACTIVITY_SPIKE")
print("  P0007 ABSENCE_FLAG")
