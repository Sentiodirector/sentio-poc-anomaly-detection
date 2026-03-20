"""
loader.py
---------
Responsible for reading all daily JSON files from the sample_data/ folder.

Each JSON file = one day of observations.
Each file contains a list of student records for that day.

Returns a dictionary:
    { "student_001": [day1_record, day2_record, ...], ... }
"""

import json
from pathlib import Path


def load_daily_files(data_folder: str = "sample_data") -> dict:
    """
    Reads every .json file in data_folder, groups records by person_id,
    and returns them sorted by date.

    Parameters:
        data_folder (str): Path to the folder containing daily JSON files.

    Returns:
        dict: { person_id (str): list of daily records sorted by date }
    """
    data_path = Path(data_folder)

    # Safety check: does the folder exist?
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data folder '{data_folder}' not found. "
            "Please create it and add daily JSON files."
        )

    # Collect all .json files, sort them by filename (alphabetical = chronological)
    json_files = sorted(data_path.glob("*.json"))

    if len(json_files) == 0:
        raise ValueError(f"No JSON files found in '{data_folder}'.")

    if len(json_files) < 5:
        print(f"  Warning: Only {len(json_files)} day(s) of data found. 5+ recommended.")

    # This dict will hold each student's records
    records_by_person = {}

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            # Each file is a list of student records for one day
            daily_records = json.load(f)

        # Each record belongs to one person — group them
        for record in daily_records:
            person_id = record["person_id"]

            if person_id not in records_by_person:
                records_by_person[person_id] = []

            records_by_person[person_id].append(record)

    # Sort each person's records by date (just in case files were out of order)
    for person_id in records_by_person:
        records_by_person[person_id].sort(key=lambda r: r["date"])

    return records_by_person
