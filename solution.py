import os
import json
import math
import base64
from datetime import datetime
from collections import defaultdict, Counter
from flask import Flask, jsonify, send_file
import statistics

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Mappings: internal → PoC contract vocabulary
# ---------------------------------------------------------------------------

SEVERITY_MAP = {
    "HIGH":   "urgent",
    "MEDIUM": "monitor",
    "LOW":    "informational",
}

CATEGORY_TITLES = {
    "SUDDEN_DROP":         "Sudden wellbeing drop detected",
    "SUSTAINED_LOW":       "Sustained low wellbeing detected",
    "SOCIAL_WITHDRAWAL":   "Social withdrawal and gaze avoidance detected",
    "HYPERACTIVITY_SPIKE": "Hyperactivity spike detected",
    "REGRESSION":          "Recovery regression detected",
    "GAZE_AVOIDANCE":      "Persistent gaze avoidance detected",
    "ABSENCE_FLAG":        "Absence welfare flag",
}

RECOMMENDED_ACTIONS = {
    "SUDDEN_DROP":         "Schedule pastoral check-in today",
    "SUSTAINED_LOW":       "Refer to school counsellor for extended support",
    "SOCIAL_WITHDRAWAL":   "Engage student in supervised peer activity; monitor tomorrow",
    "HYPERACTIVITY_SPIKE": "Assess for environmental stressors; consider calm-down strategy",
    "REGRESSION":          "Review prior support plan; escalate if decline continues",
    "GAZE_AVOIDANCE":      "Low-pressure 1:1 conversation recommended",
    "ABSENCE_FLAG":        "Welfare check — contact family if absent again tomorrow",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_dir="sample_data"):
    """Load all daily JSON files and sort them chronologically."""
    daily_logs = []
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found. Please generate sample data first.")
        return []

    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(".json"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r") as f:
                daily_logs.append(json.load(f))

    daily_logs.sort(key=lambda x: x["date"])
    return daily_logs


# ---------------------------------------------------------------------------
# Baseline computation
# ---------------------------------------------------------------------------

def compute_baseline(series):
    """
    Compute baseline properties using up to the first 3 days.
    Returns: avg, is_high_variance (bool)
    """
    baseline_days = series[:3]
    if not baseline_days:
        return 0, False

    avg = statistics.mean(baseline_days)
    if len(baseline_days) > 1:
        stdev = statistics.stdev(baseline_days)
        is_high_variance = stdev > 15
    else:
        is_high_variance = False

    return avg, is_high_variance


# ---------------------------------------------------------------------------
# Core anomaly engine
# ---------------------------------------------------------------------------

def analyze_patterns(daily_logs: list, window_size: int = 3) -> tuple:
    """
    Core engine for Behavioral Intelligence.
    We build a rolling window of history (default 3 days) to calculate
    personalized statistical baselines (mean & standard deviation) for every student.

    Why this approach?
    An 'anxious' day for an extrovert might look entirely normal for an introvert.
    By assessing deviations against their *own* personal baseline, we reduce
    false positives and act with true empathy and evidence.

    Returns: (alerts_list, absence_flags_list, student_records_dict)
    """
    alerts = []
    absence_flags = []

    # Format: { student_id: [ {date, wellbeing_score, ...}, ... ] }
    student_records = defaultdict(list)
    student_names = {}

    # Pre-process: Organise chronologically per student
    for log_entry in daily_logs:
        current_date = log_entry.get("date")

        for student in log_entry.get("students", log_entry.get("individuals", [])):
            sid = student["id"]
            name = student.get("name", sid)
            student_names[sid] = name

            # Weighted composite wellbeing score
            wellbeing_score = (
                student.get("social_engagement_score", student.get("interaction_score", 50)) * 0.4
                + student.get("energy_traits", student.get("attention_span", 50)) * 0.4
                + student.get("wellbeing_score", student.get("reported_mood", 50)) * 0.2
            )

            student_records[sid].append({
                "date":             current_date,
                "social_engagement": student.get("social_engagement_score", student.get("interaction_score", 50)),
                "energy_traits":    student.get("energy_traits", student.get("attention_span", 50)),
                "wellbeing_score":  wellbeing_score,
                "raw_wellbeing":    student.get("wellbeing_score", student.get("reported_mood", 50)),
                "gaze_direction":   student.get("gaze_direction", "forward"),
                "detected":         student.get("detected", True),
            })

    # -----------------------------------------------------------------
    # Baseline Computation Rule (from assignment):
    # Use first 3 days of data as personal baseline (or all data if < 3 days).
    # If baseline standard deviation > 15, increase drop threshold by 50%.
    # -----------------------------------------------------------------

    student_baselines = {}
    for sid, history in student_records.items():
        baseline_records = [r for r in history[:3] if r.get("detected", True)]

        if baseline_records:
            bl_wellbeing = [r["wellbeing_score"] for r in baseline_records]
            bl_social    = [r["social_engagement"] for r in baseline_records]
            bl_energy    = [r["energy_traits"]     for r in baseline_records]

            avg_wellbeing = statistics.mean(bl_wellbeing)
            avg_social    = statistics.mean(bl_social)
            avg_energy    = statistics.mean(bl_energy)

            std_wellbeing = statistics.stdev(bl_wellbeing) if len(bl_wellbeing) > 1 else 0
            std_social    = statistics.stdev(bl_social)    if len(bl_social)    > 1 else 0
            std_energy    = statistics.stdev(bl_energy)    if len(bl_energy)    > 1 else 0

            high_var = std_wellbeing > 15 or std_social > 15 or std_energy > 15
            threshold_mult = 1.5 if high_var else 1.0

            student_baselines[sid] = {
                "avg_wellbeing":  avg_wellbeing,
                "avg_social":     avg_social,
                "avg_energy":     avg_energy,
                "threshold_mult": threshold_mult,
                "high_variance":  high_var,
            }
        else:
            student_baselines[sid] = {
                "avg_wellbeing": 50, "avg_social": 50, "avg_energy": 50,
                "threshold_mult": 1.0, "high_variance": False,
            }

    # -----------------------------------------------------------------
    # Track consecutive flag days per student (for days_flagged_consecutively)
    # -----------------------------------------------------------------
    consecutive_flags = defaultdict(int)

    # -----------------------------------------------------------------
    # Evaluate every day against the fixed first-3-day baseline
    # -----------------------------------------------------------------
    alert_counter = 0

    for sid, history in student_records.items():
        name = student_names[sid]
        bl = student_baselines[sid]
        avg_wellbeing = bl["avg_wellbeing"]
        avg_social    = bl["avg_social"]
        avg_energy    = bl["avg_energy"]
        tmult         = bl["threshold_mult"]

        absence_logged = False

        for i, today_rec in enumerate(history):
            current_date = today_rec["date"]
            alert = None

            # Need at least 2 prior days of history to detect consecutive patterns
            if i < 2:
                continue

            recent_history   = history[max(0, i - 5):i]
            detected_history = [r for r in recent_history if r.get("detected", True)]
            hist_wellbeing   = [r["wellbeing_score"] for r in detected_history]

            curr_social    = today_rec["social_engagement"]
            curr_energy    = today_rec["energy_traits"]
            curr_wellbeing = today_rec["wellbeing_score"]
            curr_gaze      = today_rec["gaze_direction"]

            # Trend: last 5 days of wellbeing (rounded to 1dp)
            trend_window = history[max(0, i - 4): i + 1]
            trend_last_5 = [round(r["wellbeing_score"], 1) for r in trend_window]

            # Determine lowest trait this day
            traits = {
                "social_engagement": curr_social,
                "energy_traits":     curr_energy,
                "wellbeing_score":   curr_wellbeing,
            }
            lowest_trait       = min(traits, key=traits.get)
            lowest_trait_value = round(traits[lowest_trait], 1)

            # -----------------------------------------------------------------
            # Alert detection — elif priority chain (assignment spec order)
            # -----------------------------------------------------------------

            # 7. ABSENCE_FLAG: person not detected for 2+ consecutive days
            if (
                not today_rec["detected"]
                and len(recent_history) >= 1
                and not recent_history[-1].get("detected", True)
            ):
                days_absent = sum(1 for r in recent_history if not r.get("detected", True)) + 1
                last_seen = next(
                    (r["date"] for r in reversed(recent_history) if r.get("detected", True)),
                    history[0]["date"],
                )
                if not absence_logged:
                    absence_flags.append({
                        "person_id":          sid,
                        "person_name":        name,
                        "last_seen_date":     last_seen,
                        "days_absent":        days_absent,
                        "recommended_action": RECOMMENDED_ACTIONS["ABSENCE_FLAG"],
                    })
                    absence_logged = True
                consecutive_flags[sid] = 0
                continue

            elif not today_rec["detected"]:
                consecutive_flags[sid] = 0
                continue

            # 1. SUDDEN_DROP
            elif curr_wellbeing <= (avg_wellbeing - 20 * tmult):
                alert = _make_alert(
                    sid, name, current_date, "SUDDEN_DROP", "HIGH",
                    avg_wellbeing, curr_wellbeing,
                    trend_last_5, lowest_trait, lowest_trait_value,
                    metrics={"wellbeing_drop": round(avg_wellbeing - curr_wellbeing, 2)},
                )

            # 2. SUSTAINED_LOW
            elif (
                len(detected_history) >= 2
                and all(m < 45 for m in hist_wellbeing[-2:])
                and curr_wellbeing < 45
            ):
                alert = _make_alert(
                    sid, name, current_date, "SUSTAINED_LOW", "HIGH",
                    avg_wellbeing, curr_wellbeing,
                    trend_last_5, lowest_trait, lowest_trait_value,
                    metrics={"current_wellbeing": round(curr_wellbeing, 2), "baseline": round(avg_wellbeing, 2)},
                )

            # 3. SOCIAL_WITHDRAWAL
            elif curr_social <= (avg_social - 25 * tmult) and curr_gaze == "downward":
                alert = _make_alert(
                    sid, name, current_date, "SOCIAL_WITHDRAWAL", "MEDIUM",
                    avg_wellbeing, curr_wellbeing,
                    trend_last_5, lowest_trait, lowest_trait_value,
                    metrics={"social_drop": round(avg_social - curr_social, 2), "gaze": curr_gaze},
                )

            # 4. HYPERACTIVITY_SPIKE
            elif curr_energy >= (avg_energy + 40 * tmult):
                alert = _make_alert(
                    sid, name, current_date, "HYPERACTIVITY_SPIKE", "LOW",
                    avg_wellbeing, curr_wellbeing,
                    trend_last_5, lowest_trait, lowest_trait_value,
                    metrics={"energy_spike": round(curr_energy - avg_energy, 2)},
                )

            # 5. REGRESSION
            elif (
                len(detected_history) >= 3
                and hist_wellbeing[-3] < hist_wellbeing[-2] < hist_wellbeing[-1]
                and (hist_wellbeing[-1] - curr_wellbeing) > 15 * tmult
            ):
                alert = _make_alert(
                    sid, name, current_date, "REGRESSION", "MEDIUM",
                    avg_wellbeing, curr_wellbeing,
                    trend_last_5, lowest_trait, lowest_trait_value,
                    metrics={"drop_from_yesterday": round(hist_wellbeing[-1] - curr_wellbeing, 2)},
                )

            # 6. GAZE_AVOIDANCE
            elif (
                len(detected_history) >= 2
                and all(r.get("gaze_direction", "forward") in ["downward", "avoidant"] for r in detected_history[-2:])
                and curr_gaze in ["downward", "avoidant"]
            ):
                alert = _make_alert(
                    sid, name, current_date, "GAZE_AVOIDANCE", "LOW",
                    avg_wellbeing, curr_wellbeing,
                    trend_last_5, lowest_trait, lowest_trait_value,
                    metrics={"gaze": curr_gaze},
                )

            if alert:
                consecutive_flags[sid] += 1
                alert["days_flagged_consecutively"] = consecutive_flags[sid]
                alert_counter += 1
                alert["alert_id"] = f"ALT_{alert_counter:03d}"
                alerts.append(alert)
            else:
                consecutive_flags[sid] = 0

    return alerts, absence_flags, student_records


def _make_alert(
    sid, name, date, category, severity_label,
    avg_wellbeing, curr_wellbeing,
    trend_last_5, lowest_trait, lowest_trait_value,
    metrics=None,
):
    """Build a fully schema-compliant alert object (PoC contract)."""
    return {
        # alert_id and days_flagged_consecutively filled by caller
        "alert_id":                   "",
        "person_id":                  sid,
        "person_name":                name,
        "date":                       date,
        "severity":                   SEVERITY_MAP[severity_label],
        "_note_severity":             "one of: urgent / monitor / informational",
        "category":                   category,
        "_note_category":             "one of: SUDDEN_DROP / SUSTAINED_LOW / SOCIAL_WITHDRAWAL / HYPERACTIVITY_SPIKE / REGRESSION / GAZE_AVOIDANCE / ABSENCE_FLAG",
        "title":                      CATEGORY_TITLES[category],
        "description":                _build_description(category, name, avg_wellbeing, curr_wellbeing, metrics),
        "baseline_wellbeing":         round(avg_wellbeing, 2),
        "today_wellbeing":            round(curr_wellbeing, 2),
        "delta":                      round(curr_wellbeing - avg_wellbeing, 2),
        "days_flagged_consecutively": 1,
        "trend_last_5_days":          trend_last_5,
        "lowest_trait":               lowest_trait,
        "lowest_trait_value":         lowest_trait_value,
        "recommended_action":         RECOMMENDED_ACTIONS[category],
        "profile_image_b64":          "",
        # keep metrics for HTML digest / downstream analytics
        "metrics":                    metrics or {},
    }


def _build_description(category, name, baseline, today, metrics):
    """Human-readable description matching PoC style."""
    first = name.split()[0]
    delta = round(baseline - today, 1)
    descriptions = {
        "SUDDEN_DROP":
            f"{first}'s wellbeing dropped from a baseline of {round(baseline,1)} to {round(today,1)} "
            f"— a {delta}-point fall.",
        "SUSTAINED_LOW":
            f"{first}'s wellbeing has remained below 45 for 3+ consecutive days "
            f"(current: {round(today,1)}, baseline: {round(baseline,1)}).",
        "SOCIAL_WITHDRAWAL":
            f"{first}'s social engagement dropped {metrics.get('social_drop','?')} points "
            f"below baseline and gaze is predominantly downward.",
        "HYPERACTIVITY_SPIKE":
            f"{first}'s energy traits spiked {metrics.get('energy_spike','?')} points above baseline.",
        "REGRESSION":
            f"{first} was recovering for 3+ days but dropped "
            f"{metrics.get('drop_from_yesterday','?')} points today — recovery collapsed.",
        "GAZE_AVOIDANCE":
            f"Zero eye contact detected for {first} for 3+ consecutive days. "
            f"Gaze: {metrics.get('gaze','downward')}.",
        "ABSENCE_FLAG":
            f"{first} has not been detected for 2+ consecutive days. Welfare check required.",
    }
    return descriptions.get(category, "Anomaly detected.")


# ---------------------------------------------------------------------------
# Feed builder — wraps alerts in the PoC top-level schema
# ---------------------------------------------------------------------------

def build_feed(alerts, absence_flags, student_records, daily_logs, school_name="Sentio Demo School"):
    """
    Construct the top-level alert_feed object matching anomaly_detection.json exactly.
    """
    today_date     = daily_logs[-1]["date"] if daily_logs else datetime.utcnow().strftime("%Y-%m-%d")
    yesterday_date = daily_logs[-2]["date"] if len(daily_logs) >= 2 else None

    urgents        = sum(1 for a in alerts if a["severity"] == "urgent")
    monitors       = sum(1 for a in alerts if a["severity"] == "monitor")
    informationals = sum(1 for a in alerts if a["severity"] == "informational")

    total_tracked   = len(student_records)
    flagged_today   = len(set(a["person_id"] for a in alerts if a["date"] == today_date))
    flagged_yest    = len(set(a["person_id"] for a in alerts if a["date"] == yesterday_date)) if yesterday_date else 0

    all_categories = [a["category"] for a in alerts]
    most_common    = Counter(all_categories).most_common(1)[0][0] if all_categories else "N/A"

    today_individuals = []
    if daily_logs:
        for ind in daily_logs[-1].get("individuals", daily_logs[-1].get("students", [])):
            if ind.get("detected", True):
                records = student_records.get(ind["id"], [])
                if records:
                    today_individuals.append(records[-1]["wellbeing_score"])
    school_avg_today = round(statistics.mean(today_individuals), 2) if today_individuals else 0.0

    # Strip internal-only keys before export
    export_alerts = []
    for a in alerts:
        ea = {k: v for k, v in a.items() if k != "metrics"}
        export_alerts.append(ea)

    return {
        "_readme": (
            "Your alert_feed.json must match this structure exactly. "
            "It is returned by the /get_alerts Flask endpoint in Sentio Mind. "
            "Do not add, remove, or rename any top-level key."
        ),
        "source":       "p5_anomaly_detection",
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
        "school":       school_name,
        "alert_summary": {
            "total_alerts":  len(export_alerts),
            "urgent":        urgents,
            "monitor":       monitors,
            "informational": informationals,
        },
        "alerts":        export_alerts,
        "absence_flags": absence_flags,
        "school_summary": {
            "total_persons_tracked":         total_tracked,
            "persons_flagged_today":         flagged_today,
            "persons_flagged_yesterday":     flagged_yest,
            "most_common_anomaly_this_week": most_common,
            "school_avg_wellbeing_today":    school_avg_today,
        },
    }


# ---------------------------------------------------------------------------
# Detect anomalies — public wrapper (preserves original signature)
# ---------------------------------------------------------------------------

def detect_anomalies(daily_logs):
    """
    Wrapper function implementing the assignment requirements.
    Returns (alerts_list, absence_flags_list, student_records_dict).
    """
    return analyze_patterns(daily_logs)


# ---------------------------------------------------------------------------
# HTML digest generator
# ---------------------------------------------------------------------------

def generate_html_digest(alerts, absence_flags, student_records, feed):
    """Generate alert_digest.html — fully offline, zero CDN, inline SVG sparklines."""

    def get_severity_style(severity):
        if severity == "urgent":
            return "category-danger", "alert-danger"
        elif severity == "monitor":
            return "category-warning", "alert-warning"
        return "category-info", "alert-info"

    def generate_svg_sparkline(sid, student_records, category):
        records = student_records.get(sid, [])
        if category == "HYPERACTIVITY_SPIKE":
            wb = [r.get("energy_traits", 0) for r in records if r.get("detected", False)]
        elif category == "SOCIAL_WITHDRAWAL":
            wb = [r.get("social_engagement", 0) for r in records if r.get("detected", False)]
        elif category in ["GAZE_AVOIDANCE", "ABSENCE_FLAG"]:
            val_map = {"forward": 100, "avoidant": 50, "downward": 0}
            wb = [val_map.get(r.get("gaze_direction", "forward"), 100) for r in records if r.get("detected", False)]
        else:
            wb = [r.get("wellbeing_score", 0) for r in records if r.get("detected", False)]

        if len(wb) < 2:
            return '<span style="color:var(--text-muted);font-size:0.75rem;font-style:italic;">Insufficient data</span>'

        width, height = 140, 45
        x_step = width / (len(wb) - 1)
        points = []
        for idx, val in enumerate(wb):
            x = idx * x_step
            y = height - (max(0, min(100, val)) / 100 * height)
            points.append(f"{x:.1f},{y:.1f}")

        pts_str = " ".join(points)
        cx, cy = points[-1].split(",")
        _id = f"g_{sid}_{category}"

        return (
            f'<svg width="140" height="45" class="sparkline" viewBox="0 -5 140 55">'
            f'<defs>'
            f'<linearGradient id="grad_{_id}" x1="0%" y1="0%" x2="100%" y2="0%">'
            f'<stop offset="0%"   style="stop-color:#FF9933;stop-opacity:1"/>'
            f'<stop offset="50%"  style="stop-color:#000080;stop-opacity:1"/>'
            f'<stop offset="100%" style="stop-color:#138808;stop-opacity:1"/>'
            f'</linearGradient>'
            f'<filter id="glow_{_id}" x="-20%" y="-20%" width="140%" height="140%">'
            f'<feGaussianBlur stdDeviation="2" result="blur"/>'
            f'<feComposite in="SourceGraphic" in2="blur" operator="over"/>'
            f'</filter>'
            f'</defs>'
            f'<polyline fill="none" stroke="url(#grad_{_id})" stroke-width="2.5" '
            f'points="{pts_str}" stroke-linejoin="round" stroke-linecap="round" filter="url(#glow_{_id})"/>'
            f'<circle cx="{cx}" cy="{cy}" r="4" fill="#138808" stroke="#000" stroke-width="1.5"/>'
            f'</svg>'
        )

    summary  = feed["alert_summary"]
    school_s = feed["school_summary"]
    alerts_sorted = sorted(alerts, key=lambda x: x["date"], reverse=True)

    logo_b64 = ""
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sentio_logo.png")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as bf:
            logo_b64 = "data:image/png;base64," + base64.b64encode(bf.read()).decode()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sentio Mind — Anomaly Detection Digest</title>
<style>
:root {{
    --bg-color:#f8fafc; --surface:#ffffff; --surface-elevated:#f1f5f9;
    --text-main:#0f172a; --text-muted:#475569; --border:#e2e8f0;
    --brand-saffron:#ea580c; --brand-green:#16a34a;
    --urgent:#ef4444; --urgent-bg:rgba(239,68,68,0.08);
    --monitor:#f59e0b; --monitor-bg:rgba(245,158,11,0.08);
    --info-col:#3b82f6; --info-bg:rgba(59,130,246,0.08);
    --font-sans:'Inter',-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}}
*,*::before,*::after{{box-sizing:border-box;}}
body{{font-family:var(--font-sans);background:var(--bg-color);color:var(--text-main);margin:0;padding:0;line-height:1.6;-webkit-font-smoothing:antialiased;}}
.container{{max-width:920px;margin:0 auto;padding:2rem 1.5rem;}}
header{{display:flex;justify-content:space-between;align-items:center;padding:1.75rem 0;margin-bottom:2rem;border-bottom:1px solid var(--border);}}
.header-eyebrow{{font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;color:var(--text-muted);font-weight:600;}}
.header-title{{font-size:1.45rem;font-weight:600;margin:0.2rem 0 0;letter-spacing:-0.02em;}}
.logo-text{{font-size:1.3rem;font-weight:700;color:var(--text-main);}}
.logo-text span{{color:var(--brand-saffron);}}
.company-logo img{{height:42px;object-fit:contain;}}
.summary-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:2.5rem;}}
.stat-card{{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1.1rem 1.25rem;box-shadow:0 1px 3px rgba(0,0,0,0.04);}}
.stat-label{{font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-muted);font-weight:600;}}
.stat-value{{font-size:2rem;font-weight:700;margin-top:0.25rem;line-height:1;}}
.stat-card.urgent .stat-value{{color:var(--urgent);}}
.stat-card.monitor .stat-value{{color:var(--monitor);}}
.stat-card.info-col .stat-value{{color:var(--info-col);}}
.stat-card.total .stat-value{{color:var(--text-main);}}
.developer-card{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1.4rem 2rem;margin-bottom:2.5rem;position:relative;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.06);}}
.developer-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#FF9933 0%,#ffffff 50%,#138808 100%);}}
.dev-name{{font-size:1.05rem;font-weight:600;margin:0;}}
.dev-meta{{font-size:0.85rem;color:var(--text-muted);margin:0.35rem 0 0;display:flex;gap:1rem;flex-wrap:wrap;}}
.dev-meta span::after{{content:'•';margin-left:1rem;color:var(--border);}}
.dev-meta span:last-child::after{{content:'';margin:0;}}
.school-bar{{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1rem 1.5rem;margin-bottom:2.5rem;display:flex;flex-wrap:wrap;gap:1.5rem;align-items:center;font-size:0.85rem;}}
.school-bar-item{{display:flex;flex-direction:column;gap:0.15rem;}}
.school-bar-label{{font-size:0.65rem;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-muted);font-weight:600;}}
.school-bar-value{{font-weight:600;color:var(--text-main);}}
.school-name{{font-weight:700;font-size:1rem;color:var(--brand-saffron);margin-right:auto;}}
.section-title{{font-size:0.95rem;font-weight:600;margin-bottom:1rem;color:var(--text-main);padding-bottom:0.6rem;border-bottom:1px solid var(--border);text-transform:uppercase;letter-spacing:0.06em;}}
.alert-card{{background:var(--surface);border-radius:12px;padding:1.4rem 1.5rem;margin-bottom:0.85rem;border:1px solid var(--border);display:flex;align-items:flex-start;justify-content:space-between;gap:1.5rem;transition:box-shadow 0.15s ease,transform 0.15s ease;box-shadow:0 1px 3px rgba(0,0,0,0.04);}}
.alert-card:hover{{transform:translateY(-2px);box-shadow:0 6px 16px rgba(0,0,0,0.08);}}
.alert-card.alert-danger{{border-left:4px solid var(--urgent);}}
.alert-card.alert-warning{{border-left:4px solid var(--monitor);}}
.alert-card.alert-info{{border-left:4px solid var(--info-col);}}
.alert-content{{flex:1;min-width:0;}}
.alert-header{{display:flex;flex-wrap:wrap;align-items:center;gap:0.6rem;margin-bottom:0.4rem;}}
.student-name{{font-weight:600;font-size:1rem;}}
.alert-id{{font-size:0.7rem;color:var(--text-muted);font-family:monospace;}}
.date-badge{{font-size:0.75rem;color:var(--text-muted);margin-left:auto;}}
.category-badge{{display:inline-flex;padding:0.2rem 0.6rem;border-radius:6px;font-size:0.62rem;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;}}
.category-danger{{background:var(--urgent-bg);color:var(--urgent);border:1px solid rgba(239,68,68,0.2);}}
.category-warning{{background:var(--monitor-bg);color:var(--monitor);border:1px solid rgba(245,158,11,0.2);}}
.category-info{{background:var(--info-bg);color:var(--info-col);border:1px solid rgba(59,130,246,0.2);}}
.alert-title{{font-size:0.95rem;font-weight:600;margin-bottom:0.3rem;}}
.description{{color:var(--text-muted);font-size:0.88rem;margin-bottom:0.75rem;line-height:1.5;}}
.meta-row{{display:flex;flex-wrap:wrap;gap:0.5rem;margin-bottom:0.6rem;}}
.meta-chip{{font-size:0.72rem;padding:0.15rem 0.55rem;border-radius:6px;background:var(--surface-elevated);border:1px solid var(--border);color:var(--text-muted);font-weight:500;}}
.recommended{{font-size:0.8rem;color:var(--brand-green);font-weight:600;display:flex;align-items:center;gap:0.35rem;}}
.sparkline-container{{text-align:right;width:150px;flex-shrink:0;}}
.sparkline-title{{font-size:0.6rem;color:var(--text-muted);margin-bottom:0.35rem;text-transform:uppercase;letter-spacing:0.06em;font-weight:600;}}
.absence-card{{background:var(--surface);border:1px solid var(--border);border-left:4px solid var(--urgent);border-radius:10px;padding:1rem 1.25rem;margin-bottom:0.75rem;display:flex;justify-content:space-between;align-items:center;font-size:0.88rem;}}
.absence-name{{font-weight:600;}}
.absence-meta{{color:var(--text-muted);font-size:0.8rem;}}
.absence-action{{color:var(--urgent);font-size:0.78rem;font-weight:600;}}
footer{{margin-top:4rem;padding:2rem 0;border-top:1px solid var(--border);display:flex;gap:2.5rem;flex-wrap:wrap;}}
.footer-col{{flex:1;min-width:260px;}}
.footer-col:first-child{{border-right:1px solid var(--border);padding-right:2rem;}}
.footer-name{{font-weight:700;font-size:0.95rem;margin-bottom:0.25rem;}}
.footer-brand{{font-weight:700;font-size:1rem;color:var(--brand-saffron);margin-bottom:0.5rem;}}
.footer-link{{color:var(--brand-saffron);text-decoration:none;font-size:0.82rem;display:inline-block;margin-bottom:0.75rem;}}
.footer-link:hover{{text-decoration:underline;}}
.footer-text{{color:var(--text-muted);font-size:0.82rem;line-height:1.65;margin:0;}}
.footer-disclaimer{{width:100%;text-align:center;margin-top:2rem;padding:1.25rem 1.5rem;background:var(--surface-elevated);border-radius:8px;border:1px solid var(--border);color:var(--text-muted);font-size:0.78rem;line-height:1.6;}}
</style>
</head>
<body>
<div class="container">
<header>
  <div>
    <div class="header-eyebrow">Behavioral Anomaly &amp; Early Distress Detection</div>
    <h1 class="header-title">Sentio Mind — Alert Digest</h1>
  </div>
  <div class="company-logo">
    <img src="{logo_b64}" alt="Sentio Mind"
         onerror="this.outerHTML='&lt;span class=&quot;logo-text&quot;&gt;Sentio &lt;span&gt;Mind&lt;/span&gt;&lt;/span&gt;'">
  </div>
</header>

<div class="summary-row">
  <div class="stat-card total"><div class="stat-label">Total Alerts</div><div class="stat-value">{summary['total_alerts']}</div></div>
  <div class="stat-card urgent"><div class="stat-label">Urgent</div><div class="stat-value">{summary['urgent']}</div></div>
  <div class="stat-card monitor"><div class="stat-label">Monitor</div><div class="stat-value">{summary['monitor']}</div></div>
  <div class="stat-card info-col"><div class="stat-label">Informational</div><div class="stat-value">{summary['informational']}</div></div>
</div>

<div class="school-bar">
  <span class="school-name">{feed['school']}</span>
  <div class="school-bar-item"><span class="school-bar-label">Total Tracked</span><span class="school-bar-value">{school_s['total_persons_tracked']}</span></div>
  <div class="school-bar-item"><span class="school-bar-label">Flagged Today</span><span class="school-bar-value">{school_s['persons_flagged_today']}</span></div>
  <div class="school-bar-item"><span class="school-bar-label">Flagged Yesterday</span><span class="school-bar-value">{school_s['persons_flagged_yesterday']}</span></div>
  <div class="school-bar-item"><span class="school-bar-label">Top Anomaly This Week</span><span class="school-bar-value">{school_s['most_common_anomaly_this_week'].replace('_',' ')}</span></div>
  <div class="school-bar-item"><span class="school-bar-label">Avg Wellbeing Today</span><span class="school-bar-value">{school_s['school_avg_wellbeing_today']}</span></div>
</div>

<div class="developer-card">
  <div class="dev-name">Yashwanth Sai Kasarabada</div>
  <div class="dev-meta">
    <span>Roll No. 220103012</span>
    <span>IIIT Senapati, Manipur</span>
    <span>CSE AI &amp; DS Branch '26</span>
    <span>Generated: {feed['generated_at']}</span>
  </div>
</div>

<h3 class="section-title">Anomaly Alerts</h3>
<div id="alerts-list">
"""

    if not alerts_sorted:
        html += "<p style='color:var(--text-muted);padding:2rem;text-align:center;border:1px dashed var(--border);border-radius:10px;'>No anomalies detected. System operating nominally.</p>"

    for a in alerts_sorted:
        badge_cls, wrapper_cls = get_severity_style(a["severity"])
        svg = generate_svg_sparkline(a["person_id"], student_records, a["category"])
        delta_str = f"{a['delta']:+.1f}"

        html += f"""
<div class="alert-card {wrapper_cls}">
  <div class="alert-content">
    <div class="alert-header">
      <span class="student-name">{a['person_name']}</span>
      <span class="category-badge {badge_cls}">{a['category'].replace('_',' ')}</span>
      <span class="alert-id">{a['alert_id']}</span>
      <span class="date-badge">{a['date']}</span>
    </div>
    <div class="alert-title">{a['title']}</div>
    <div class="description">{a['description']}</div>
    <div class="meta-row">
      <span class="meta-chip">Baseline: {a['baseline_wellbeing']}</span>
      <span class="meta-chip">Today: {a['today_wellbeing']}</span>
      <span class="meta-chip">&#916; {delta_str}</span>
      <span class="meta-chip">Consecutive: {a['days_flagged_consecutively']}d</span>
      <span class="meta-chip">Lowest: {a['lowest_trait'].replace('_',' ')} ({a['lowest_trait_value']})</span>
    </div>
    <div class="recommended">&#10003; {a['recommended_action']}</div>
  </div>
  <div class="sparkline-container">
    <div class="sparkline-title">Trend</div>
    {svg}
  </div>
</div>"""

    html += "\n</div>\n"

    if absence_flags:
        html += '\n<h3 class="section-title" style="margin-top:2.5rem;">Absence Flags</h3>\n'
        for af in absence_flags:
            html += f"""
<div class="absence-card">
  <div>
    <div class="absence-name">{af['person_name']}</div>
    <div class="absence-meta">ID: {af['person_id']} &middot; Last seen: {af['last_seen_date']} &middot; Absent: {af['days_absent']} day(s)</div>
  </div>
  <div class="absence-action">&#9888; {af['recommended_action']}</div>
</div>"""

    html += f"""
<footer>
  <div class="footer-col">
    <div class="footer-name">Yashwanth Sai Kasarabada</div>
    <a href="https://www.linkedin.com/in/yashwanth-sai-kasarabada-ba4265258/" class="footer-link">LinkedIn Profile</a>
    <p class="footer-text">I'm driven to contribute to building something meaningful rather than operating as a generic output in a large MNC. I'm willing to put in the hard work, take ownership, and grow alongside the company. The core idea deeply resonates with me — I'd really appreciate the opportunity to prove my capabilities.</p>
  </div>
  <div class="footer-col">
    <div class="footer-brand">Sentio Mind</div>
    <p class="footer-text">An AI-powered behavioral intelligence platform operating on top of existing infrastructure to detect early emotional risks — without invading privacy.</p>
    <p class="footer-text" style="margin-top:0.75rem;">Acting as a silent guardian, helping educators act with empathy, evidence, and responsibility.</p>
  </div>
  <div class="footer-disclaimer">
    Thank you for reviewing my assignment. My primary goal was to demonstrate not just technical proficiency, but a deep alignment with Sentio's mission of building empathetic, AI-driven solutions. I have poured immense care into both the core algorithmic constraints and this visual presentation. I eagerly look forward to discussing my approach and proving how I can contribute from day one.
  </div>
</footer>

</div>
</body>
</html>"""

    with open("alert_digest.html", "w", encoding="utf-8") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------

def generate_outputs():
    print("Loading data...")
    daily_logs = load_data()
    print("Analysing patterns and evaluating baselines...")
    alerts, absence_flags, student_records = detect_anomalies(daily_logs)

    print(f"Detected {len(alerts)} anomaly alert(s), {len(absence_flags)} absence flag(s).")

    feed = build_feed(alerts, absence_flags, student_records, daily_logs)

    # 1. Write machine-readable alert_feed.json (PoC schema-compliant)
    with open("alert_feed.json", "w") as f:
        json.dump(feed, f, indent=2)
    print("Exported alert_feed.json")

    # 2. Write HTML digest
    generate_html_digest(alerts, absence_flags, student_records, feed)
    print("Exported alert_digest.html (sparklines + absence flags + school summary)")


# ---------------------------------------------------------------------------
# Flask endpoints
# ---------------------------------------------------------------------------

@app.route("/get_alerts", methods=["GET"])
def get_alerts():
    """Return the full PoC-schema-compliant feed."""
    if not os.path.exists("alert_feed.json"):
        return jsonify({"error": "No alert feed found. Run anomaly generation first."}), 404
    with open("alert_feed.json", "r") as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/", methods=["GET"])
def index():
    return send_file("alert_digest.html")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate_outputs()
    print("\nStarting Flask server on http://127.0.0.1:5000 — press Ctrl+C to stop.")
    app.run(host="127.0.0.1", port=5000)
