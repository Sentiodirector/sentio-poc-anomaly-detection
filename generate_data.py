"""
generate_data.py
Run this ONCE to create the sample_data/ folder with 5 days of synthetic student data.
Run: python generate_data.py
"""

import json
import random
import os
from pathlib import Path

random.seed(42)

STUDENTS = [
    {"id": "SCHOOL_P0001", "name": "Aarav Sharma"},
    {"id": "SCHOOL_P0002", "name": "Sparshita Reddy"},
    {"id": "SCHOOL_P0003", "name": "Ranjhana Patel"},
    {"id": "SCHOOL_P0004", "name": "Kabir Nair"},
    {"id": "SCHOOL_P0005", "name": "Diya Menon"},
    {"id": "SCHOOL_P0006", "name": "Arjun Mehta"},
]

DAYS = [
    "2026-01-01",
    "2026-01-02",
    "2026-01-03",
    "2026-01-04",
    "2026-01-05",
]

def clamp(v, lo=0, hi=100):
    return max(lo, min(hi, int(v)))

def make_student_day(student, day_index, scenario=None):
    """Generate one student's data for one day."""
    base_wb     = random.randint(60, 85)
    base_social = random.randint(55, 80)
    base_phys   = random.randint(40, 65)
    base_move   = random.randint(40, 65)
    gaze        = random.choice(["forward", "forward", "forward", "side", "down"])
    eye_contact = gaze == "forward"

    if scenario == "sudden_drop" and day_index == 4:
        # Day 5: big drop
        wb     = clamp(base_wb - random.randint(30, 45))
        social = clamp(base_social - random.randint(28, 40))
        gaze   = "down"
        eye_contact = False
    elif scenario == "sustained_low":
        # Days 3-5: consistently low
        if day_index >= 2:
            wb     = clamp(random.randint(25, 44))
            social = clamp(random.randint(20, 40))
            gaze   = "down"
            eye_contact = False
        else:
            wb     = clamp(base_wb + random.randint(-5, 5))
            social = clamp(base_social + random.randint(-5, 5))
    elif scenario == "hyperactivity":
        # Days 4-5: energy spike
        if day_index >= 3:
            base_phys = clamp(base_phys + random.randint(40, 55))
            base_move = clamp(base_move + random.randint(40, 55))
            wb        = clamp(base_wb + random.randint(-10, 5))
            social    = clamp(base_social + random.randint(-5, 5))
        else:
            wb     = clamp(base_wb + random.randint(-5, 5))
            social = clamp(base_social + random.randint(-5, 5))
    elif scenario == "regression":
        # Days 1-4 improving, day 5 drops hard
        improving_wb = [40, 50, 60, 70]
        if day_index < 4:
            wb     = clamp(improving_wb[day_index] + random.randint(-3, 3))
            social = clamp(improving_wb[day_index] - 5 + random.randint(-3, 3))
        else:
            wb     = clamp(improving_wb[3] - random.randint(20, 30))
            social = clamp(social if 'social' in dir() else 40 - 20)
            gaze   = "side"
            eye_contact = False
    elif scenario == "gaze_avoidance":
        # Last 3 days: no eye contact
        if day_index >= 2:
            gaze        = random.choice(["down", "side"])
            eye_contact = False
            wb          = clamp(base_wb + random.randint(-10, 0))
            social      = clamp(base_social - random.randint(10, 20))
        else:
            wb     = clamp(base_wb + random.randint(-5, 5))
            social = clamp(base_social + random.randint(-5, 5))
    else:
        # Normal variation
        wb     = clamp(base_wb + random.randint(-8, 8))
        social = clamp(base_social + random.randint(-8, 8))

    physical_energy  = clamp(base_phys + random.randint(-5, 5))
    movement_energy  = clamp(base_move + random.randint(-5, 5))

    return {
        "person_info": {
            "name":              student["name"],
            "profile_image_b64": ""
        },
        "wellbeing":         wb,
        "traits": {
            "social_engagement":  social,
            "physical_energy":    physical_energy,
            "movement_energy":    movement_energy,
            "focus":              clamp(random.randint(45, 75) + random.randint(-5, 5)),
            "calmness":           clamp(random.randint(50, 80) + random.randint(-5, 5)),
        },
        "gaze_direction":    gaze,
        "eye_contact":       eye_contact,
        "detection_count":   random.randint(8, 20),
    }


# Assign scenarios to specific students
SCENARIOS = {
    "SCHOOL_P0002": "sudden_drop",
    "SCHOOL_P0003": "sustained_low",
    "SCHOOL_P0004": "hyperactivity",
    "SCHOOL_P0005": "regression",
    "SCHOOL_P0006": "gaze_avoidance",
    # P0001 = normal
    # P0007 will be absent (not included in days 4-5)
}

# Extra absent student
ABSENT_STUDENT = {"id": "SCHOOL_P0007", "name": "Priya Rajan"}

DATA_DIR = Path("sample_data")
DATA_DIR.mkdir(exist_ok=True)

for day_idx, day_str in enumerate(DAYS):
    day_data = {"date": day_str, "persons": {}}

    for student in STUDENTS:
        scenario = SCENARIOS.get(student["id"])
        day_data["persons"][student["id"]] = make_student_day(student, day_idx, scenario)

    # Absent student only appears in first 2 days
    if day_idx < 2:
        day_data["persons"][ABSENT_STUDENT["id"]] = make_student_day(ABSENT_STUDENT, day_idx, None)

    fname = DATA_DIR / f"analysis_{day_str}.json"
    with open(fname, "w") as f:
        json.dump(day_data, f, indent=2)

    print(f"Created {fname}")

print(f"\nDone! Created {len(DAYS)} files in sample_data/")
print("Students with scenarios:")
for sid, sc in SCENARIOS.items():
    name = next(s["name"] for s in STUDENTS if s["id"] == sid)
    print(f"  {name} ({sid}): {sc}")
print(f"  {ABSENT_STUDENT['name']} ({ABSENT_STUDENT['id']}): absent last 3 days")