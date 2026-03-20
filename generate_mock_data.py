import json
import os
import numpy as np
from datetime import datetime, timedelta

def generate_data():
    os.makedirs("sample_data", exist_ok=True)
    start_date = datetime(2026, 1, 1)
    
    students = [
        {"id": "SCHOOL_P0001", "name": "Aisha Khan", "type": "Normal Student", "baseline": 80},
        {"id": "SCHOOL_P0002", "name": "Liam Smith", "type": "Sudden Drop Case", "baseline": 75},
        {"id": "SCHOOL_P0003", "name": "Arjun Mehta", "type": "Sustained Low Case", "baseline": 60},
        {"id": "SCHOOL_P0004", "name": "Priya Rajan", "type": "Absence Case", "baseline": 85}
    ]

    for day in range(1, 6):
        current_date = (start_date + timedelta(days=day-1)).strftime("%Y-%m-%d")
        daily_records = []

        for student in students:
            # Default normal behavior
            wellbeing = int(np.random.normal(student["baseline"], 5))
            social = int(np.random.normal(student["baseline"], 5))
            energy = int(np.random.normal(student["baseline"], 5))
            gaze = "forward"
            seen = True
            
            # Inject Anomalies based on the day 
            if student["type"] == "Sudden Drop Case" and day == 5:
                wellbeing -= 45 # Massive drop on day 5
                social -= 30
                gaze = "down"
            
            elif student["type"] == "Sustained Low Case" and day >= 3:
                wellbeing = 35 # Drops below 45 for 3+ days
            
            elif student["type"] == "Absence Case" and day >= 4:
                seen = False # Absent for day 4 and 5
            
            # Clamp values between 0 and 100
            wellbeing = max(0, min(100, wellbeing))
            
            record = {
                "person_id": student["id"],
                "person_name": student["name"],
                "date": current_date,
                "wellbeing": wellbeing,
                "social_engagement": social,
                "physical_energy": energy,
                "movement_energy": energy,
                "gaze": gaze,
                "seen_in_video": seen,
            }
            
            
            if seen:
                daily_records.append(record)
            
        with open(f"sample_data/analysis_Day{day}.json", "w") as f:
            json.dump(daily_records, f, indent=4)
            
    print("Successfully generated 5 days of realistic mock data in sample_data/")

if __name__ == "__main__":
    generate_data()