import json
import random
from pathlib import Path

DATA_DIR = Path("sample_data")
DATA_DIR.mkdir(exist_ok=True)

NUM_DAYS = 5
PERSONS = ["P1", "P2", "P3", "P4", "P5"]

def random_traits():
    return {
        "physical_energy": random.randint(20, 80),
        "movement_energy": random.randint(20, 80),
        "social_engagement": random.randint(20, 80),
        "eye_contact": random.choice([True, False])
    }

def random_gaze():
    return random.choice(["forward", "down", "side"])

for day in range(1, NUM_DAYS + 1):
    data = {}

    for pid in PERSONS:
        # simulate realistic behaviour
        wellbeing = random.randint(30, 90)

        data[pid] = {
            "person_info": {
                "name": f"Student_{pid}"
            },
            "wellbeing": wellbeing,
            "social_engagement": random.randint(20, 80),
            "gaze_direction": random_gaze(),
            "traits": random_traits()
        }

    file_path = DATA_DIR / f"analysis_Day{day}.json"
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

print(" Synthetic data generated in sample_data/")
