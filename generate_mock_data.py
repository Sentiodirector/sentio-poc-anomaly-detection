"""
generate_mock_data.py
Sentio Mind · Project 5
Generates sample_data/analysis_Day1.json through analysis_Day5.json.
"""

import json
import numpy as np
from pathlib import Path

rng = np.random.default_rng(42)   # reproducible

DATES = ["2026-03-17", "2026-03-18", "2026-03-19", "2026-03-20", "2026-03-21"]


def n(base, std=4):
    """Noisy integer, clamped 0–100."""
    return int(np.clip(int(rng.normal(base, std)), 0, 100))


def rec(pid, name, date, wb, soc, phys, move, gaze="forward", seen=True):
    return {
        "person_id":         pid,
        "person_name":       name,
        "date":              date,
        "wellbeing":         int(np.clip(wb,   0, 100)),
        "social_engagement": int(np.clip(soc,  0, 100)),
        "physical_energy":   int(np.clip(phys, 0, 100)),
        "movement_energy":   int(np.clip(move, 0, 100)),
        "gaze":              gaze,
        "seen_in_video":     bool(seen),
    }


daily = {d: [] for d in DATES}


# ── Student A  NORMAL  baseline ~80 ─────────────────────────────────────────
for i, d in enumerate(DATES):
    daily[d].append(rec("SCHOOL_P0001", "Alice Sharma", d,
                        n(80), n(78), n(72), n(68)))

# ── Student B  SUDDEN_DROP  Day 5 wellbeing = 35 ────────────────────────────
# Baseline (Days 1-3) ≈ 80; delta ≈ 45 > 20 → urgent
for i, d in enumerate(DATES):
    wb  = 35    if i == 4 else n(80)
    soc = n(72) if i == 4 else n(78)
    daily[d].append(rec("SCHOOL_P0002", "Bob Nair", d,
                        wb, soc, n(72), n(68)))

# ── Student C  SUSTAINED_LOW (+ SUDDEN_DROP)  chronic ───────────────────────
# Fixed Days 1-2 so baseline is deterministic:
#   baseline = (80+78+30)/3 = 62.7, std = 23.1 > 15 → threshold = 30
#   delta = 62.7 – 30 = 32.7 > 30 → SUDDEN_DROP fires Days 3-5 (monitor)
#   sustained_low: [30,30,30] < 45 → fires Day 5 (urgent)
#   days_flagged_consecutively = 3 → CHRONIC
C_wb  = [80, 78, 30, 30, 30]
C_soc = [78, 76, 35, 33, 31]
for i, d in enumerate(DATES):
    gaze = "down" if i >= 2 else "forward"
    daily[d].append(rec("SCHOOL_P0003", "Chitra Patel", d,
                        C_wb[i], C_soc[i], n(58), n(54), gaze))

# ── Student D  ABSENCE_FLAG  seen_in_video=False Days 4-5 ───────────────────
for i, d in enumerate(DATES):
    seen = i < 3
    daily[d].append(rec("SCHOOL_P0004", "David Kumar", d,
                        n(72), n(70), n(65), n(61), seen=seen))

# ── Student E  GAZE_AVOIDANCE  gaze=down Days 3-5 ───────────────────────────
for i, d in enumerate(DATES):
    gaze = "down" if i >= 2 else "forward"
    daily[d].append(rec("SCHOOL_P0005", "Esha Reddy", d,
                        n(72), n(70), n(65), n(61), gaze))

# ── Student F  SOCIAL_WITHDRAWAL  social drop Days 3-5 + gaze=down ──────────
# Fixed social: [80, 78, 40, 40, 40]
#   baseline_social = (80+78+40)/3 = 66; delta = 26 ≥ 25 AND gaze=down → fires
#   Also GAZE_AVOIDANCE fires on Day 5 (3 consecutive down-gaze days)
#   days_flagged_consecutively = 3 → CHRONIC
F_soc = [80, 78, 40, 40, 40]
for i, d in enumerate(DATES):
    gaze = "down" if i >= 2 else "forward"
    daily[d].append(rec("SCHOOL_P0006", "Farhan Malik", d,
                        n(78), F_soc[i], n(70), n(66), gaze))

# ── Student G  HYPERACTIVITY_SPIKE  energy spike Days 4-5 ───────────────────
# Baseline energy: (30+25+32+27+29+24)/3 combined = 55.7
# Days 4-5: 82+80=162 > 55.7+40=95.7 → fires
# days_flagged_consecutively = 2
G_phys = [30, 32, 29, 82, 80]
G_move = [25, 27, 24, 80, 79]
for i, d in enumerate(DATES):
    daily[d].append(rec("SCHOOL_P0007", "Gauri Singh", d,
                        n(65), n(63), G_phys[i], G_move[i]))

# ── Student H  REGRESSION  wellbeing rises 3 days then drops ────────────────
# Fixed: [50, 56, 63, 71, 50]  56>50, 63>56, 71>63 → 3 improvements
# Drop: 71-50=21 > 15 → REGRESSION fires Day 5
H_wb = [50, 56, 63, 71, 50]
for i, d in enumerate(DATES):
    daily[d].append(rec("SCHOOL_P0008", "Harish Iyer", d,
                        H_wb[i], n(58 + i * 2), n(52), n(48)))


# ── Write files ───────────────────────────────────────────────────────────────
out = Path("sample_data")
out.mkdir(exist_ok=True)

for i, d in enumerate(DATES, 1):
    fp = out / f"analysis_Day{i}.json"
    with open(fp, "w") as f:
        json.dump(daily[d], f, indent=2)
    print(f"  {fp}  ({len(daily[d])} records)")

print("\nDone - 5 files written to sample_data/")
print("\nAnomalies injected:")
print("  SCHOOL_P0001 Alice Sharma  - Normal baseline ~80")
print("  SCHOOL_P0002 Bob Nair      - SUDDEN_DROP     : Day 5 wb=35 (delta~45, urgent)")
print("  SCHOOL_P0003 Chitra Patel  - SUSTAINED_LOW   : Days 3-5 wb=30 + SUDDEN_DROP (chronic)")
print("  SCHOOL_P0004 David Kumar   - ABSENCE_FLAG    : seen_in_video=False Days 4-5")
print("  SCHOOL_P0005 Esha Reddy    - GAZE_AVOIDANCE  : gaze=down Days 3-5")
print("  SCHOOL_P0006 Farhan Malik  - SOCIAL_WITHDRAWAL: social drop+gaze=down Days 3-5 (chronic)")
print("  SCHOOL_P0007 Gauri Singh   - HYPERACTIVITY   : phys+move spike Days 4-5")
print("  SCHOOL_P0008 Harish Iyer   - REGRESSION      : recovery then 21pt drop Day 5")
