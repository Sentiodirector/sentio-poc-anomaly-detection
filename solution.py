"""
solution.py
Sentio Mind · Project 5 · Behavioral Anomaly & Early Distress Detection
Run: python solution.py
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime, date
from collections import defaultdict, Counter

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR   = Path("sample_data")
REPORT_OUT = Path("alert_digest.html")
FEED_OUT   = Path("alert_feed.json")
SCHOOL     = "Demo School"

THRESHOLDS = {
    "sudden_drop_delta":           20,
    "sudden_drop_high_std_delta":  30,
    "sustained_low_score":         45,
    "sustained_low_days":           3,
    "social_withdrawal_delta":     25,
    "hyperactivity_delta":         40,
    "regression_recover_days":      3,
    "regression_drop":             15,
    "gaze_avoidance_days":          3,
    "absence_days":                 2,
    "baseline_window":              3,
    "high_std_baseline":           15,
}

_alert_counter = [0]

def _next_alert_id():
    _alert_counter[0] += 1
    return f"ALT_{_alert_counter[0]:03d}"


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_daily_data(folder: Path) -> dict:
    """
    Read all analysis_*.json files from folder.
    Return: { "YYYY-MM-DD": { "PERSON_ID": { wellbeing, traits, gaze, name, ... } } }
    """
    daily = {}
    for fp in sorted(folder.glob("*.json")):
        try:
            with open(fp) as f:
                raw = json.load(f)

            # Support two formats:
            # Format A: { "date": "...", "persons": { pid: {...} } }
            # Format B: { pid: { "date": "...", ... } }  (flat)
            if "persons" in raw and "date" in raw:
                day_str = raw["date"]
                persons = raw["persons"]
            elif "date" in raw:
                day_str = raw["date"]
                persons = {k: v for k, v in raw.items() if k != "date"}
            else:
                # Infer date from filename e.g. analysis_2026-01-01.json
                stem = fp.stem  # analysis_2026-01-01
                parts = stem.split("_")
                day_str = parts[-1] if len(parts) > 1 else stem
                persons = raw

            daily[day_str] = persons
        except Exception as e:
            print(f"  Warning: could not load {fp.name}: {e}")

    return daily


# ---------------------------------------------------------------------------
# BASELINE
# ---------------------------------------------------------------------------

def compute_baseline(history: list) -> dict:
    """
    history: list of daily dicts (oldest first).
    Use first baseline_window days.
    Returns: { wellbeing_mean, wellbeing_std, trait_means, avg_gaze,
               social_engagement_mean, physical_energy_mean, movement_energy_mean }
    """
    window = history[: THRESHOLDS["baseline_window"]]
    if not window:
        return {
            "wellbeing_mean": 50.0, "wellbeing_std": 10.0,
            "trait_means": {}, "avg_gaze": "forward",
            "social_engagement_mean": 50.0,
            "physical_energy_mean":   50.0,
            "movement_energy_mean":   50.0,
        }

    wb_vals = [d.get("wellbeing", 50) for d in window]
    wb_mean = float(np.mean(wb_vals))
    wb_std  = float(np.std(wb_vals)) if len(wb_vals) > 1 else 0.0

    # Per-trait means
    trait_keys = set()
    for d in window:
        trait_keys.update(d.get("traits", {}).keys())

    trait_means = {}
    for tk in trait_keys:
        vals = [d.get("traits", {}).get(tk, 50) for d in window if tk in d.get("traits", {})]
        trait_means[tk] = float(np.mean(vals)) if vals else 50.0

    # Most common gaze
    gazes   = [d.get("gaze_direction", "forward") for d in window]
    avg_gaze = Counter(gazes).most_common(1)[0][0]

    return {
        "wellbeing_mean":          wb_mean,
        "wellbeing_std":           wb_std,
        "trait_means":             trait_means,
        "avg_gaze":                avg_gaze,
        "social_engagement_mean":  trait_means.get("social_engagement", 50.0),
        "physical_energy_mean":    trait_means.get("physical_energy",   50.0),
        "movement_energy_mean":    trait_means.get("movement_energy",   50.0),
    }


# ---------------------------------------------------------------------------
# ANOMALY DETECTORS
# ---------------------------------------------------------------------------

def detect_sudden_drop(today: dict, baseline: dict, today_date: str, person_name: str, trend: list) -> dict | None:
    today_wb   = today.get("wellbeing", 50)
    base_mean  = baseline["wellbeing_mean"]
    base_std   = baseline["wellbeing_std"]
    delta      = base_mean - today_wb

    threshold = (THRESHOLDS["sudden_drop_high_std_delta"]
                 if base_std > THRESHOLDS["high_std_baseline"]
                 else THRESHOLDS["sudden_drop_delta"])

    if delta < threshold:
        return None

    severity = "urgent" if delta > 35 else "monitor"

    traits = today.get("traits", {})
    lowest_trait = min(traits, key=lambda k: traits[k]) if traits else "unknown"
    lowest_val   = traits.get(lowest_trait, 0)

    return {
        "alert_id":                  _next_alert_id(),
        "person_id":                 today.get("_person_id", ""),
        "person_name":               person_name,
        "date":                      today_date,
        "severity":                  severity,
        "category":                  "SUDDEN_DROP",
        "title":                     "Sudden wellbeing drop detected",
        "description":               (f"{person_name}'s wellbeing dropped from a baseline of "
                                      f"{base_mean:.0f} to {today_wb} today — a {delta:.0f}-point fall. "
                                      f"Lowest trait: {lowest_trait} at {lowest_val}. "
                                      f"Dominant gaze: {today.get('gaze_direction', 'unknown')}."),
        "baseline_wellbeing":        round(base_mean),
        "today_wellbeing":           today_wb,
        "delta":                     -round(delta),
        "days_flagged_consecutively": 1,
        "trend_last_5_days":         trend,
        "lowest_trait":              lowest_trait,
        "lowest_trait_value":        lowest_val,
        "recommended_action":        "Schedule pastoral check-in today",
        "profile_image_b64":         today.get("person_info", {}).get("profile_image_b64", ""),
    }


def detect_sustained_low(history: list, today_date: str, person_name: str, pid: str, trend: list) -> dict | None:
    n = THRESHOLDS["sustained_low_days"]
    if len(history) < n:
        return None
    last_n = history[-n:]
    if all(d.get("wellbeing", 100) < THRESHOLDS["sustained_low_score"] for d in last_n):
        today_wb = last_n[-1].get("wellbeing", 0)
        traits   = last_n[-1].get("traits", {})
        lowest   = min(traits, key=lambda k: traits[k]) if traits else "unknown"

        return {
            "alert_id":                  _next_alert_id(),
            "person_id":                 pid,
            "person_name":               person_name,
            "date":                      today_date,
            "severity":                  "urgent",
            "category":                  "SUSTAINED_LOW",
            "title":                     f"Wellbeing below 45 for {n}+ consecutive days",
            "description":               (f"{person_name} has scored below {THRESHOLDS['sustained_low_score']} "
                                          f"for {n} consecutive days. Current wellbeing: {today_wb}."),
            "baseline_wellbeing":        0,
            "today_wellbeing":           today_wb,
            "delta":                     0,
            "days_flagged_consecutively": n,
            "trend_last_5_days":         trend,
            "lowest_trait":              lowest,
            "lowest_trait_value":        traits.get(lowest, 0),
            "recommended_action":        "Immediate pastoral support recommended",
            "profile_image_b64":         last_n[-1].get("person_info", {}).get("profile_image_b64", ""),
        }
    return None


def detect_social_withdrawal(today: dict, baseline: dict, today_date: str, person_name: str, pid: str, trend: list) -> dict | None:
    social_today    = today.get("traits", {}).get("social_engagement", 50)
    social_baseline = baseline.get("social_engagement_mean", 50)
    delta           = social_baseline - social_today
    gaze            = today.get("gaze_direction", "forward")

    if delta >= THRESHOLDS["social_withdrawal_delta"] and gaze in ("down", "side"):
        return {
            "alert_id":                  _next_alert_id(),
            "person_id":                 pid,
            "person_name":               person_name,
            "date":                      today_date,
            "severity":                  "monitor",
            "category":                  "SOCIAL_WITHDRAWAL",
            "title":                     "Social withdrawal detected",
            "description":               (f"{person_name}'s social engagement dropped {delta:.0f} points "
                                          f"below baseline (now {social_today:.0f}). Gaze: {gaze}."),
            "baseline_wellbeing":        round(baseline["wellbeing_mean"]),
            "today_wellbeing":           today.get("wellbeing", 0),
            "delta":                     -round(delta),
            "days_flagged_consecutively": 1,
            "trend_last_5_days":         trend,
            "lowest_trait":              "social_engagement",
            "lowest_trait_value":        round(social_today),
            "recommended_action":        "Monitor social interactions; consider peer support",
            "profile_image_b64":         today.get("person_info", {}).get("profile_image_b64", ""),
        }
    return None


def detect_hyperactivity_spike(today: dict, baseline: dict, today_date: str, person_name: str, pid: str, trend: list) -> dict | None:
    phys_today  = today.get("traits", {}).get("physical_energy", 50)
    move_today  = today.get("traits", {}).get("movement_energy", 50)
    phys_base   = baseline.get("physical_energy_mean", 50)
    move_base   = baseline.get("movement_energy_mean", 50)
    combined_delta = (phys_today + move_today) - (phys_base + move_base)

    if combined_delta >= THRESHOLDS["hyperactivity_delta"]:
        return {
            "alert_id":                  _next_alert_id(),
            "person_id":                 pid,
            "person_name":               person_name,
            "date":                      today_date,
            "severity":                  "monitor",
            "category":                  "HYPERACTIVITY_SPIKE",
            "title":                     "Hyperactivity spike detected",
            "description":               (f"{person_name}'s combined energy (physical + movement) is "
                                          f"{combined_delta:.0f} points above baseline today."),
            "baseline_wellbeing":        round(baseline["wellbeing_mean"]),
            "today_wellbeing":           today.get("wellbeing", 0),
            "delta":                     round(combined_delta),
            "days_flagged_consecutively": 1,
            "trend_last_5_days":         trend,
            "lowest_trait":              "movement_energy",
            "lowest_trait_value":        round(move_today),
            "recommended_action":        "Check for anxiety, overstimulation, or disrupted routine",
            "profile_image_b64":         today.get("person_info", {}).get("profile_image_b64", ""),
        }
    return None


def detect_regression(history: list, today_date: str, person_name: str, pid: str, trend: list) -> dict | None:
    n = THRESHOLDS["regression_recover_days"]
    # Need at least n+1 entries (n improving days + today's drop)
    if len(history) < n + 1:
        return None

    recover_window = history[-(n + 1):-1]   # the n days before today
    today_wb       = history[-1].get("wellbeing", 50)

    # Check each day was improving (each > previous)
    was_improving = all(
        recover_window[i].get("wellbeing", 0) > recover_window[i - 1].get("wellbeing", 0)
        for i in range(1, len(recover_window))
    )

    if not was_improving:
        return None

    peak_wb = recover_window[-1].get("wellbeing", 50)
    drop    = peak_wb - today_wb

    if drop > THRESHOLDS["regression_drop"]:
        return {
            "alert_id":                  _next_alert_id(),
            "person_id":                 pid,
            "person_name":               person_name,
            "date":                      today_date,
            "severity":                  "monitor",
            "category":                  "REGRESSION",
            "title":                     "Regression after recovery period",
            "description":               (f"{person_name} had been improving for {n} days "
                                          f"(peaked at {peak_wb}) but dropped {drop:.0f} points today to {today_wb}."),
            "baseline_wellbeing":        round(peak_wb),
            "today_wellbeing":           today_wb,
            "delta":                     -round(drop),
            "days_flagged_consecutively": 1,
            "trend_last_5_days":         trend,
            "lowest_trait":              "wellbeing",
            "lowest_trait_value":        today_wb,
            "recommended_action":        "Check for relapse triggers; schedule follow-up",
            "profile_image_b64":         history[-1].get("person_info", {}).get("profile_image_b64", ""),
        }
    return None


def detect_gaze_avoidance(history: list, today_date: str, person_name: str, pid: str, trend: list) -> dict | None:
    n = THRESHOLDS["gaze_avoidance_days"]
    if len(history) < n:
        return None

    last_n = history[-n:]
    no_contact = all(
        d.get("eye_contact") is False or d.get("eye_contact") == 0
        for d in last_n
    )

    if no_contact:
        today_wb = last_n[-1].get("wellbeing", 0)
        return {
            "alert_id":                  _next_alert_id(),
            "person_id":                 pid,
            "person_name":               person_name,
            "date":                      today_date,
            "severity":                  "monitor",
            "category":                  "GAZE_AVOIDANCE",
            "title":                     f"No eye contact for {n} consecutive days",
            "description":               (f"{person_name} has shown no eye contact for "
                                          f"{n} consecutive days. Possible withdrawal or anxiety."),
            "baseline_wellbeing":        0,
            "today_wellbeing":           today_wb,
            "delta":                     0,
            "days_flagged_consecutively": n,
            "trend_last_5_days":         trend,
            "lowest_trait":              "eye_contact",
            "lowest_trait_value":        0,
            "recommended_action":        "Gentle 1-on-1 check-in recommended",
            "profile_image_b64":         last_n[-1].get("person_info", {}).get("profile_image_b64", ""),
        }
    return None


# ---------------------------------------------------------------------------
# BONUS: Peer-comparison anomaly
# ---------------------------------------------------------------------------

def detect_peer_comparison(today: dict, class_mean: float, class_std: float,
                            today_date: str, person_name: str, pid: str, trend: list) -> dict | None:
    """Flag if wellbeing is more than 2 std devs below the class average."""
    today_wb = today.get("wellbeing", 50)
    if class_std > 0 and (class_mean - today_wb) > 2 * class_std:
        return {
            "alert_id":                  _next_alert_id(),
            "person_id":                 pid,
            "person_name":               person_name,
            "date":                      today_date,
            "severity":                  "monitor",
            "category":                  "SUDDEN_DROP",   # closest schema category
            "title":                     "Significantly below class average",
            "description":               (f"{person_name} is scoring {today_wb}, which is more than "
                                          f"2 standard deviations below today's class average of {class_mean:.0f}."),
            "baseline_wellbeing":        round(class_mean),
            "today_wellbeing":           today_wb,
            "delta":                     -round(class_mean - today_wb),
            "days_flagged_consecutively": 1,
            "trend_last_5_days":         trend,
            "lowest_trait":              "wellbeing",
            "lowest_trait_value":        today_wb,
            "recommended_action":        "Peer-comparison flag — consider counsellor referral",
            "profile_image_b64":         today.get("person_info", {}).get("profile_image_b64", ""),
        }
    return None


# ---------------------------------------------------------------------------
# ANALYSE ONE PERSON
# ---------------------------------------------------------------------------

def analyse_person(person_id: str, sorted_days: dict, info: dict,
                   class_wb_today: dict = None) -> list:
    """
    sorted_days: { "YYYY-MM-DD": person_data_dict }
    Returns list of alert dicts.
    """
    alerts  = []
    dates   = list(sorted_days.keys())
    history = []

    for d in dates:
        entry = dict(sorted_days[d])
        entry["_person_id"] = person_id
        history.append(entry)

    if not history:
        return alerts

    baseline     = compute_baseline(history)
    today_date   = dates[-1]
    today        = history[-1]
    today["_person_id"] = person_id

    # Build 5-day wellbeing trend
    trend = [d.get("wellbeing", 0) for d in history[-5:]]
    while len(trend) < 5:
        trend.insert(0, 0)

    person_name = info.get("name", person_id)

    # Run all detectors
    a = detect_sudden_drop(today, baseline, today_date, person_name, trend)
    if a: alerts.append(a)

    a = detect_sustained_low(history, today_date, person_name, person_id, trend)
    if a: alerts.append(a)

    a = detect_social_withdrawal(today, baseline, today_date, person_name, person_id, trend)
    if a: alerts.append(a)

    a = detect_hyperactivity_spike(today, baseline, today_date, person_name, person_id, trend)
    if a: alerts.append(a)

    a = detect_regression(history, today_date, person_name, person_id, trend)
    if a: alerts.append(a)

    a = detect_gaze_avoidance(history, today_date, person_name, person_id, trend)
    if a: alerts.append(a)

    # Bonus: peer comparison
    if class_wb_today:
        wb_vals   = list(class_wb_today.values())
        c_mean    = float(np.mean(wb_vals)) if wb_vals else 50.0
        c_std     = float(np.std(wb_vals))  if wb_vals else 10.0
        a = detect_peer_comparison(today, c_mean, c_std, today_date, person_name, person_id, trend)
        if a: alerts.append(a)

    return alerts


# ---------------------------------------------------------------------------
# HTML REPORT  (offline, no CDN)
# ---------------------------------------------------------------------------

def _sparkline_svg(trend: list) -> str:
    """Generate a tiny inline SVG sparkline from a 5-value list."""
    W, H   = 80, 30
    pad    = 3
    vals   = [max(0, min(100, v)) for v in trend]
    min_v  = min(vals) if vals else 0
    max_v  = max(vals) if vals else 100
    rng    = max(max_v - min_v, 1)

    def px(i):
        x = pad + i * (W - 2 * pad) / max(len(vals) - 1, 1)
        y = H - pad - (vals[i] - min_v) / rng * (H - 2 * pad)
        return f"{x:.1f},{y:.1f}"

    points = " ".join(px(i) for i in range(len(vals)))
    last_x, last_y = px(len(vals) - 1).split(",")

    return (
        f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg">'
        f'<polyline points="{points}" fill="none" stroke="#6366f1" stroke-width="2" stroke-linejoin="round"/>'
        f'<circle cx="{last_x}" cy="{last_y}" r="3" fill="#ef4444"/>'
        f'</svg>'
    )


def _severity_badge(severity: str) -> str:
    colors = {
        "urgent":        ("bg-urgent",   "🔴 URGENT"),
        "monitor":       ("bg-monitor",  "🟡 MONITOR"),
        "informational": ("bg-info",     "🔵 INFO"),
    }
    cls, label = colors.get(severity, ("bg-info", severity.upper()))
    return f'<span class="badge {cls}">{label}</span>'


def _category_icon(cat: str) -> str:
    icons = {
        "SUDDEN_DROP":        "📉",
        "SUSTAINED_LOW":      "🌧️",
        "SOCIAL_WITHDRAWAL":  "🚪",
        "HYPERACTIVITY_SPIKE":"⚡",
        "REGRESSION":         "↩️",
        "GAZE_AVOIDANCE":     "👁️",
        "ABSENCE_FLAG":       "❓",
    }
    return icons.get(cat, "⚠️")


def generate_alert_digest(alerts: list, absence_flags: list,
                           school_summary: dict, output_path: Path):
    today_str   = str(date.today())
    today_alerts = [a for a in alerts if a.get("date") == today_str]
    # If no alerts for today, show all (synthetic data has fixed dates)
    if not today_alerts:
        today_alerts = alerts

    urgent_alerts  = [a for a in today_alerts if a.get("severity") == "urgent"]
    monitor_alerts = [a for a in today_alerts if a.get("severity") == "monitor"]
    other_alerts   = [a for a in today_alerts if a.get("severity") not in ("urgent", "monitor")]
    sorted_alerts  = urgent_alerts + monitor_alerts + other_alerts

    # Persistent: flagged 3+ consecutive days
    consec_counter = Counter(a.get("person_name") for a in alerts)
    persistent     = {name for name, cnt in consec_counter.items() if cnt >= 3}

    def alert_card(a):
        spark = _sparkline_svg(a.get("trend_last_5_days", [0]*5))
        badge = _severity_badge(a.get("severity", "informational"))
        icon  = _category_icon(a.get("category", ""))
        return f"""
        <div class="card severity-{a.get('severity','informational')}">
          <div class="card-header">
            <div>
              <span class="person-name">{a.get('person_name','')}</span>
              {badge}
            </div>
            <div class="sparkline">{spark}</div>
          </div>
          <div class="card-body">
            <div class="category">{icon} {a.get('category','')}</div>
            <div class="title">{a.get('title','')}</div>
            <p class="description">{a.get('description','')}</p>
            <div class="action">💡 {a.get('recommended_action','')}</div>
            <div class="meta">
              Baseline: <strong>{a.get('baseline_wellbeing','—')}</strong> &nbsp;|&nbsp;
              Today: <strong>{a.get('today_wellbeing','—')}</strong> &nbsp;|&nbsp;
              Delta: <strong>{a.get('delta','—')}</strong>
            </div>
          </div>
        </div>"""

    cards_html = "\n".join(alert_card(a) for a in sorted_alerts) if sorted_alerts else "<p class='no-alerts'>No alerts today ✅</p>"

    absence_rows = ""
    for af in absence_flags:
        absence_rows += f"""
        <tr>
          <td>{af['person_name']}</td>
          <td>{af['last_seen_date']}</td>
          <td><strong>{af['days_absent']}</strong></td>
          <td>{af['recommended_action']}</td>
        </tr>"""

    persistent_list = "".join(f"<li>⚠️ <strong>{n}</strong></li>" for n in sorted(persistent)) \
                      if persistent else "<li>None at this time ✅</li>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sentio Mind · Alert Digest · {today_str}</title>
<style>
  :root {{
    --urgent:  #ef4444;
    --monitor: #f59e0b;
    --info:    #6366f1;
    --bg:      #0f172a;
    --surface: #1e293b;
    --border:  #334155;
    --text:    #e2e8f0;
    --muted:   #94a3b8;
    --green:   #22c55e;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 24px;
    line-height: 1.6;
  }}
  header {{
    border-bottom: 1px solid var(--border);
    padding-bottom: 16px;
    margin-bottom: 28px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
  }}
  header h1 {{ font-size: 1.6rem; font-weight: 700; color: #fff; }}
  header .sub {{ color: var(--muted); font-size: 0.9rem; }}
  .section {{ margin-bottom: 36px; }}
  .section h2 {{
    font-size: 1.1rem; font-weight: 600; color: #fff;
    border-left: 3px solid var(--info);
    padding-left: 10px; margin-bottom: 16px;
  }}
  .summary-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px;
    margin-bottom: 8px;
  }}
  .stat-box {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
  }}
  .stat-box .num {{ font-size: 2rem; font-weight: 800; }}
  .stat-box .lbl {{ font-size: 0.78rem; color: var(--muted); margin-top: 4px; }}
  .stat-box.urgent-box .num {{ color: var(--urgent); }}
  .stat-box.monitor-box .num {{ color: var(--monitor); }}
  .stat-box.green-box .num {{ color: var(--green); }}
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-bottom: 14px;
    overflow: hidden;
    transition: box-shadow 0.2s;
  }}
  .card.severity-urgent  {{ border-left: 4px solid var(--urgent);  }}
  .card.severity-monitor {{ border-left: 4px solid var(--monitor); }}
  .card.severity-informational {{ border-left: 4px solid var(--info); }}
  .card-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    background: rgba(255,255,255,0.03);
    border-bottom: 1px solid var(--border);
  }}
  .person-name {{ font-weight: 700; font-size: 1rem; margin-right: 10px; }}
  .badge {{
    display: inline-block;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
  }}
  .bg-urgent  {{ background: rgba(239,68,68,0.2);  color: var(--urgent);  }}
  .bg-monitor {{ background: rgba(245,158,11,0.2); color: var(--monitor); }}
  .bg-info    {{ background: rgba(99,102,241,0.2); color: var(--info);    }}
  .card-body {{ padding: 14px 16px; }}
  .category {{ font-size: 0.75rem; color: var(--muted); margin-bottom: 4px; }}
  .title {{ font-weight: 600; margin-bottom: 6px; }}
  .description {{ font-size: 0.88rem; color: #cbd5e1; margin-bottom: 8px; }}
  .action {{
    font-size: 0.82rem;
    background: rgba(99,102,241,0.1);
    border-radius: 6px;
    padding: 6px 10px;
    margin-bottom: 8px;
    color: #a5b4fc;
  }}
  .meta {{ font-size: 0.78rem; color: var(--muted); }}
  .sparkline svg {{ display: block; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
    background: var(--surface);
    border-radius: 10px;
    overflow: hidden;
  }}
  th {{
    background: rgba(255,255,255,0.05);
    text-align: left;
    padding: 10px 14px;
    color: var(--muted);
    font-weight: 600;
    font-size: 0.78rem;
    text-transform: uppercase;
  }}
  td {{ padding: 10px 14px; border-top: 1px solid var(--border); }}
  ul {{ list-style: none; padding: 0; }}
  li {{ padding: 8px 12px; background: var(--surface); border-radius: 8px;
        margin-bottom: 6px; border: 1px solid var(--border); }}
  .no-alerts {{ color: var(--green); padding: 20px; text-align: center; font-size: 1.1rem; }}
  .generated {{ color: var(--muted); font-size: 0.78rem; margin-top: 40px; text-align: center; }}
</style>
</head>
<body>

<header>
  <div>
    <h1>🧠 Sentio Mind · Alert Digest</h1>
    <div class="sub">{SCHOOL} &nbsp;·&nbsp; Generated {datetime.now().strftime('%d %b %Y, %H:%M')}</div>
  </div>
  <div class="sub">Project 5 · Behavioral Anomaly Detection</div>
</header>

<!-- SECTION 1: SCHOOL SUMMARY -->
<div class="section">
  <h2>📊 School Summary</h2>
  <div class="summary-grid">
    <div class="stat-box">
      <div class="num">{school_summary['total_persons_tracked']}</div>
      <div class="lbl">Students Tracked</div>
    </div>
    <div class="stat-box urgent-box">
      <div class="num">{sum(1 for a in alerts if a.get('severity')=='urgent')}</div>
      <div class="lbl">Urgent Alerts</div>
    </div>
    <div class="stat-box monitor-box">
      <div class="num">{sum(1 for a in alerts if a.get('severity')=='monitor')}</div>
      <div class="lbl">Monitor Alerts</div>
    </div>
    <div class="stat-box">
      <div class="num">{len(absence_flags)}</div>
      <div class="lbl">Absent Students</div>
    </div>
    <div class="stat-box green-box">
      <div class="num">{school_summary.get('school_avg_wellbeing_today', 0)}</div>
      <div class="lbl">Avg Wellbeing Today</div>
    </div>
    <div class="stat-box">
      <div class="num" style="font-size:1rem">{school_summary.get('most_common_anomaly_this_week','—')}</div>
      <div class="lbl">Top Anomaly This Week</div>
    </div>
  </div>
</div>

<!-- SECTION 2: TODAY'S ALERTS -->
<div class="section">
  <h2>🚨 Today's Alerts ({len(sorted_alerts)})</h2>
  {cards_html}
</div>

<!-- SECTION 3: ABSENCE FLAGS -->
<div class="section">
  <h2>❓ Absence Flags ({len(absence_flags)})</h2>
  {'<table><thead><tr><th>Student</th><th>Last Seen</th><th>Days Absent</th><th>Action</th></tr></thead><tbody>' + absence_rows + '</tbody></table>' if absence_flags else '<p class="no-alerts">No absence flags ✅</p>'}
</div>

<!-- SECTION 4: PERSISTENT ALERTS -->
<div class="section">
  <h2>🔁 Flagged 3+ Times</h2>
  <ul>{persistent_list}</ul>
</div>

<div class="generated">Generated by solution.py · Sentio Mind POC · Project 5 · {datetime.now().isoformat()}</div>

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    daily_data = load_daily_data(DATA_DIR)
    all_dates  = sorted(daily_data.keys())
    print(f"Loaded {len(daily_data)} days: {all_dates}")

    if not daily_data:
        print("\n⚠️  No data found in sample_data/. Run generate_data.py first.")
        exit(1)

    # Build per-person history
    person_days = defaultdict(dict)
    person_info = {}
    for d, persons in daily_data.items():
        for pid, pdata in persons.items():
            person_days[pid][d] = pdata
            if pid not in person_info:
                person_info[pid] = pdata.get("person_info", {"name": pid, "profile_image_b64": ""})

    # Compute class wellbeing for today (for bonus peer-comparison)
    last_date = all_dates[-1] if all_dates else None
    class_wb_today = {}
    if last_date and last_date in daily_data:
        for pid, pdata in daily_data[last_date].items():
            class_wb_today[pid] = pdata.get("wellbeing", 50)

    all_alerts    = []
    absence_flags = []

    for pid, days in person_days.items():
        sorted_days   = dict(sorted(days.items()))
        person_alerts = analyse_person(pid, sorted_days, person_info.get(pid, {}), class_wb_today)
        all_alerts.extend(person_alerts)

        # Check absence
        present = set(days.keys())
        absent  = 0
        for d in reversed(all_dates):
            if d not in present:
                absent += 1
            else:
                break
        if absent >= THRESHOLDS["absence_days"]:
            last_seen = sorted(present)[-1] if present else "unknown"
            absence_flags.append({
                "person_id":          pid,
                "person_name":        person_info.get(pid, {}).get("name", pid),
                "last_seen_date":     last_seen,
                "days_absent":        absent,
                "recommended_action": "Welfare check — contact family if absent again tomorrow",
            })

    sev_order = {"urgent": 0, "monitor": 1, "informational": 2}
    all_alerts.sort(key=lambda a: sev_order.get(a.get("severity", "informational"), 3))

    today_str    = str(date.today())
    flagged_today = sum(1 for a in all_alerts if a.get("date") == today_str)
    cat_counter   = Counter(a.get("category") for a in all_alerts)
    top_category  = cat_counter.most_common(1)[0][0] if cat_counter else "none"

    # Compute avg wellbeing today from last day's data
    avg_wb_today = 0
    if last_date and last_date in daily_data:
        wb_vals = [v.get("wellbeing", 0) for v in daily_data[last_date].values()]
        avg_wb_today = round(float(np.mean(wb_vals))) if wb_vals else 0

    school_summary = {
        "total_persons_tracked":         len(person_days),
        "persons_flagged_today":         flagged_today,
        "persons_flagged_yesterday":     0,
        "most_common_anomaly_this_week": top_category,
        "school_avg_wellbeing_today":    avg_wb_today,
    }

    feed = {
        "source":        "p5_anomaly_detection",
        "generated_at":  datetime.now().isoformat(),
        "school":        SCHOOL,
        "alert_summary": {
            "total_alerts":  len(all_alerts),
            "urgent":        sum(1 for a in all_alerts if a.get("severity") == "urgent"),
            "monitor":       sum(1 for a in all_alerts if a.get("severity") == "monitor"),
            "informational": sum(1 for a in all_alerts if a.get("severity") == "informational"),
        },
        "alerts":         all_alerts,
        "absence_flags":  absence_flags,
        "school_summary": school_summary,
    }

    with open(FEED_OUT, "w") as f:
        json.dump(feed, f, indent=2)

    generate_alert_digest(all_alerts, absence_flags, school_summary, REPORT_OUT)

    print()
    print("=" * 55)
    print(f"  Students tracked : {len(person_days)}")
    print(f"  Alerts total     : {feed['alert_summary']['total_alerts']}")
    print(f"    Urgent         : {feed['alert_summary']['urgent']}")
    print(f"    Monitor        : {feed['alert_summary']['monitor']}")
    print(f"  Absence flags    : {len(absence_flags)}")
    print(f"  Avg wellbeing    : {avg_wb_today}")
    print(f"  Report → {REPORT_OUT}")
    print(f"  JSON   → {FEED_OUT}")
    print("=" * 55)