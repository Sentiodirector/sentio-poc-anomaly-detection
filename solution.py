"""
solution.py
-----------
Main entry point for the Sentio Mind Anomaly Detection system.

What this script does (in order):
    1. Loads daily JSON files from sample_data/
    2. Computes a personal baseline for each student (first 3 days)
    3. Runs all 7 anomaly detectors against every student's data
    4. Saves results to:
           alert_feed.json    — machine-readable, served by /get_alerts endpoint
           alert_digest.html  — human-readable counsellor report with sparklines

Usage:
    python solution.py
    python solution.py --data sample_data --output .
"""

import argparse
import sys
from pathlib import Path

# Make sure Python can find the src/ package
sys.path.insert(0, str(Path(__file__).parent))

from src.loader    import load_daily_files
from src.detectors import run_all_detectors
from src.reporter  import generate_alert_feed, generate_alert_digest


def main(data_folder: str = "sample_data", output_folder: str = ".") -> list:
    """
    Runs the full anomaly detection pipeline.

    Parameters:
        data_folder  (str): Folder containing daily JSON files.
        output_folder (str): Where to save alert_feed.json and alert_digest.html.

    Returns:
        list: All detected alerts (also written to disk).
    """
    print()
    print("=" * 55)
    print("   Sentio Mind — Behavioral Anomaly Detection")
    print("=" * 55)

    # ------------------------------------------------------------------ #
    # Step 1 — Load data                                                   #
    # ------------------------------------------------------------------ #
    print(f"\n[1/3] Loading data from '{data_folder}/' ...")
    records_by_person = load_daily_files(data_folder)
    print(f"      Found {len(records_by_person)} student(s): "
          f"{', '.join(records_by_person.keys())}")

    # ------------------------------------------------------------------ #
    # Step 2 — Detect anomalies                                            #
    # ------------------------------------------------------------------ #
    print("\n[2/3] Running anomaly detection ...")
    all_alerts = run_all_detectors(records_by_person)

    if all_alerts:
        print(f"      Detected {len(all_alerts)} anomal(ies):")
        for alert in all_alerts:
            print(f"        • [{alert['severity']:6s}] {alert['alert_type']:22s} "
                  f"{alert['person_id']}  ({alert['detected_on']})")
    else:
        print("      No anomalies detected. All students within normal range.")

    # ------------------------------------------------------------------ #
    # Step 3 — Generate output files                                       #
    # ------------------------------------------------------------------ #
    print("\n[3/3] Generating output files ...")

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    feed_path   = output_path / "alert_feed.json"
    digest_path = output_path / "alert_digest.html"

    generate_alert_feed(all_alerts, str(feed_path))
    generate_alert_digest(all_alerts, records_by_person, str(digest_path))

    print()
    print("=" * 55)
    print("  Done! Output files created:")
    print(f"    alert_feed.json   -> {feed_path.resolve()}")
    print(f"    alert_digest.html -> {digest_path.resolve()}")
    print()
    print("  Next steps:")
    print("    1. Open alert_digest.html in your browser to view the report.")
    print("    2. Run 'python src/app.py' to start the Flask server.")
    print("    3. Visit http://localhost:5001/get_alerts for the JSON feed.")
    print("=" * 55)
    print()

    return all_alerts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sentio Mind — Behavioral Anomaly Detection"
    )
    parser.add_argument(
        "--data",
        default="sample_data",
        help="Path to the folder containing daily JSON files (default: sample_data)"
    )
    parser.add_argument(
        "--output",
        default=".",
        help="Output folder for generated files (default: current directory)"
    )
    args = parser.parse_args()

    main(data_folder=args.data, output_folder=args.output)
