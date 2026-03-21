"""
generate_mock_data.py
Generates sample_data/analysis_Day1.json through analysis_Day5.json.
Each file is a JSON array of student records with injected anomalies.
"""

import json
import os
from pathlib import Path

DATES = [
    "2026-03-17",  # Day 1
    "2026-03-18",  # Day 2
    "2026-03-19",  # Day 3
    "2026-03-20",  # Day 4
    "2026-03-21",  # Day 5  (today)
]


def rec(pid, name, date, wb, soc, phys, move, gaze, seen=True):
    return {
        "person_id": pid,
        "person_name": name,
        "date": date,
        "wellbeing": wb,
        "social_engagement": soc,
        "physical_energy": phys,
        "movement_energy": move,
        "gaze": gaze,
        "seen_in_video": seen,
    }


# ── per-student day values ──────────────────────────────────────────────────
# Student A — normal baseline ~80, no anomaly
A = [
    rec("STU_A", "Alice Sharma",  DATES[0], 80, 78, 75, 72, "forward"),
    rec("STU_A", "Alice Sharma",  DATES[1], 82, 79, 76, 73, "forward"),
    rec("STU_A", "Alice Sharma",  DATES[2], 79, 77, 74, 71, "forward"),
    rec("STU_A", "Alice Sharma",  DATES[3], 81, 78, 75, 72, "forward"),
    rec("STU_A", "Alice Sharma",  DATES[4], 80, 78, 75, 72, "forward"),
]

# Student B — SUDDEN_DROP: baseline ~80, Day 5 drops 45 pts → urgent
B = [
    rec("STU_B", "Bob Nair",      DATES[0], 82, 80, 75, 70, "forward"),
    rec("STU_B", "Bob Nair",      DATES[1], 80, 78, 73, 68, "forward"),
    rec("STU_B", "Bob Nair",      DATES[2], 79, 77, 72, 67, "forward"),
    rec("STU_B", "Bob Nair",      DATES[3], 81, 79, 74, 69, "forward"),
    rec("STU_B", "Bob Nair",      DATES[4], 35, 30, 40, 38, "down"),   # drop
]

# Student C — SUSTAINED_LOW: wellbeing = 35 from Day 3 onward (3 consec < 45)
C = [
    rec("STU_C", "Chitra Patel",  DATES[0], 65, 62, 58, 55, "forward"),
    rec("STU_C", "Chitra Patel",  DATES[1], 62, 60, 56, 53, "forward"),
    rec("STU_C", "Chitra Patel",  DATES[2], 35, 32, 40, 38, "down"),
    rec("STU_C", "Chitra Patel",  DATES[3], 35, 30, 38, 36, "down"),
    rec("STU_C", "Chitra Patel",  DATES[4], 35, 28, 36, 34, "down"),
]

# Student D — ABSENCE_FLAG: not present on Day 4 and Day 5 (2 consecutive absent)
D = [
    rec("STU_D", "David Kumar",   DATES[0], 75, 72, 68, 65, "forward"),
    rec("STU_D", "David Kumar",   DATES[1], 73, 70, 66, 63, "forward"),
    rec("STU_D", "David Kumar",   DATES[2], 70, 68, 64, 61, "forward"),
    # Days 4 & 5 intentionally absent (seen_in_video = False represented by missing records)
]

# Student E — GAZE_AVOIDANCE: gaze = down/away for Days 3, 4, 5 (3 consecutive)
E = [
    rec("STU_E", "Esha Reddy",    DATES[0], 72, 70, 65, 62, "forward"),
    rec("STU_E", "Esha Reddy",    DATES[1], 71, 69, 64, 61, "forward"),
    rec("STU_E", "Esha Reddy",    DATES[2], 68, 65, 62, 59, "down"),
    rec("STU_E", "Esha Reddy",    DATES[3], 66, 63, 60, 57, "away"),
    rec("STU_E", "Esha Reddy",    DATES[4], 65, 62, 59, 56, "down"),
]

# Student F — SOCIAL_WITHDRAWAL: social drops 36 pts + gaze=down on Day 4
# Baseline social ≈ 78; Day 4 social = 42 → delta = 36 ≥ 25, gaze = down
F = [
    rec("STU_F", "Farhan Malik",  DATES[0], 78, 80, 72, 68, "forward"),
    rec("STU_F", "Farhan Malik",  DATES[1], 76, 78, 70, 66, "forward"),
    rec("STU_F", "Farhan Malik",  DATES[2], 75, 76, 69, 65, "forward"),
    rec("STU_F", "Farhan Malik",  DATES[3], 70, 42, 65, 62, "down"),   # withdrawal
    rec("STU_F", "Farhan Malik",  DATES[4], 72, 44, 67, 64, "forward"),
]

# Student G — HYPERACTIVITY_SPIKE: combined energy spikes 102 pts on Day 4
# Baseline phys ≈ 30.3, move ≈ 25.3  →  combined baseline ≈ 55.7
# Day 4: 80 + 78 = 158  →  spike = 102 ≥ 40
G = [
    rec("STU_G", "Gauri Singh",   DATES[0], 65, 63, 30, 25, "forward"),
    rec("STU_G", "Gauri Singh",   DATES[1], 67, 65, 32, 27, "forward"),
    rec("STU_G", "Gauri Singh",   DATES[2], 64, 62, 29, 24, "forward"),
    rec("STU_G", "Gauri Singh",   DATES[3], 68, 66, 80, 78, "forward"),  # spike
    rec("STU_G", "Gauri Singh",   DATES[4], 66, 64, 31, 26, "forward"),
]

# Student H — REGRESSION: Days 2-4 improving (+5, +7, +8), Day 5 drops 20 pts
H = [
    rec("STU_H", "Harish Iyer",   DATES[0], 50, 55, 52, 48, "forward"),
    rec("STU_H", "Harish Iyer",   DATES[1], 55, 58, 55, 51, "forward"),
    rec("STU_H", "Harish Iyer",   DATES[2], 62, 62, 58, 54, "forward"),
    rec("STU_H", "Harish Iyer",   DATES[3], 70, 67, 62, 58, "forward"),
    rec("STU_H", "Harish Iyer",   DATES[4], 50, 52, 48, 44, "forward"),  # regression
]

# ── assemble daily buckets ──────────────────────────────────────────────────
daily: dict[str, list] = {d: [] for d in DATES}

for student_records in [A, B, C, D, E, F, G, H]:
    for r in student_records:
        daily[r["date"]].append(r)

# ── write files ─────────────────────────────────────────────────────────────
out_dir = Path("sample_data")
out_dir.mkdir(exist_ok=True)

for i, d in enumerate(DATES, start=1):
    fp = out_dir / f"analysis_Day{i}.json"
    with open(fp, "w") as f:
        json.dump(daily[d], f, indent=2)
    print(f"  {fp}  ({len(daily[d])} records)")

print("\nDone — 5 files written to sample_data/")
print("\nAnomalies injected:")
print("  STU_A - no anomaly (baseline ~80)")
print("  STU_B - SUDDEN_DROP   : Day 5 wellbeing 35 (baseline 80, delta 45, urgent)")
print("  STU_C - SUSTAINED_LOW : Days 3-5 wellbeing 35 (< 45 for 3 days, urgent)")
print("  STU_D - ABSENCE_FLAG  : missing from Day 4 & 5 (2 consecutive absences)")
print("  STU_E - GAZE_AVOIDANCE: gaze down/away Days 3-5 (3 consecutive days)")
print("  STU_F - SOCIAL_WITHDRAW: social -36 pts + gaze=down on Day 4")
print("  STU_G - HYPERACTIVITY : phys+move spike 102 pts above baseline on Day 4")
print("  STU_H - REGRESSION    : 3-day recovery (50->55->62->70) then drops 20 on Day 5")
