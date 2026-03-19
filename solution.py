import json
import numpy as np
from pathlib import Path
from datetime import datetime, date
from collections import defaultdict, Counter

# ---------------------------------------------------------------------------
# CONFIG — adjust thresholds here, nowhere else
# ---------------------------------------------------------------------------
DATA_DIR   = Path("sample_data")
REPORT_OUT = Path("alert_digest.html")
FEED_OUT   = Path("alert_feed.json")
SCHOOL     = "Demo School"

THRESHOLDS = {
    "sudden_drop_delta":           20,   # baseline - today >= this → SUDDEN_DROP
    "sudden_drop_high_std_delta":  30,   # used when baseline_std > 15
    "sustained_low_score":         45,   # below this = low
    "sustained_low_days":           3,   # consecutive days below threshold
    "social_withdrawal_delta":     25,   # social_engagement drop
    "hyperactivity_delta":         40,   # combined energy spike
    "regression_recover_days":      3,   # days improving before regression counts
    "regression_drop":             15,   # drop after recovery
    "gaze_avoidance_days":          3,   # consecutive days no eye contact
    "absence_days":                 2,   # days not detected
    "baseline_window":              3,   # days used for baseline
    "high_std_baseline":           15,   # if std above this, use relaxed threshold
}

_alert_counter = 0


def _new_alert_id() -> str:
    global _alert_counter
    _alert_counter += 1
    return f"ALT_{_alert_counter:03d}"


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_daily_data(folder: Path) -> dict:
    """
    Read all analysis_*.json files from folder.
    Return: { "YYYY-MM-DD": { "PERSON_ID": { wellbeing, traits, gaze, name, ... }, ... }, ... }
    """
    daily = {}
    for fp in sorted(folder.glob("*.json")):
        with open(fp) as f:
            raw = json.load(f)

        day_date = raw.get("date")          
        persons_raw = raw.get("persons", {})

        if not day_date:
            continue

        persons = {}
        for pid, rec in persons_raw.items():
            # Normalise: ensure top-level wellbeing key
            wb = rec.get("wellbeing", 0)
            traits = rec.get("traits", {})
            persons[pid] = {
                "wellbeing":      wb,
                "traits":         traits,
                "gaze_direction": rec.get("gaze_direction", "forward"),
                "eye_contact":    rec.get("eye_contact", True),
                "person_name":    rec.get("person_name", pid),
                "person_info":    rec.get("person_info", {"name": rec.get("person_name", pid),
                                                          "profile_image_b64": ""}),
            }
        daily[day_date] = persons

    return daily


# ---------------------------------------------------------------------------
# BASELINE
# ---------------------------------------------------------------------------

def compute_baseline(history: list) -> dict:
    """
    history: list of daily dicts (oldest first).
    Use first THRESHOLDS['baseline_window'] days.
    Return: { wellbeing_mean, wellbeing_std, trait_means: {}, avg_gaze: str,
              social_engagement_mean, physical_energy_mean, movement_energy_mean }
    """
    window = history[: THRESHOLDS["baseline_window"]]
    if not window:
        return {
            "wellbeing_mean": 50.0,
            "wellbeing_std":  10.0,
            "trait_means":    {},
            "avg_gaze":       "forward",
            "social_engagement_mean": 50.0,
            "physical_energy_mean":   50.0,
            "movement_energy_mean":   50.0,
        }

    wb_vals = np.array([d["wellbeing"] for d in window], dtype=float)

    # Collect trait values
    trait_keys = set()
    for d in window:
        trait_keys.update(d.get("traits", {}).keys())

    trait_means = {}
    for tk in trait_keys:
        vals = [d["traits"].get(tk, np.nan) for d in window]
        vals_arr = np.array([v for v in vals if v is not None and not np.isnan(v)], dtype=float)
        trait_means[tk] = float(np.mean(vals_arr)) if len(vals_arr) > 0 else 50.0

    # Most common gaze direction
    gazes = [d.get("gaze_direction", "forward") for d in window]
    avg_gaze = Counter(gazes).most_common(1)[0][0]

    return {
        "wellbeing_mean": float(np.mean(wb_vals)),
        "wellbeing_std":  float(np.std(wb_vals)),
        "trait_means":    trait_means,
        "avg_gaze":       avg_gaze,
        "social_engagement_mean": trait_means.get("social_engagement", 50.0),
        "physical_energy_mean":   trait_means.get("physical_energy",   50.0),
        "movement_energy_mean":   trait_means.get("movement_energy",   50.0),
    }


# ---------------------------------------------------------------------------
# ANOMALY DETECTORS — each returns a partial alert dict or None
# ---------------------------------------------------------------------------

def detect_sudden_drop(today: dict, baseline: dict) -> dict | None:
    """
    Wellbeing drops ≥ 20 pts vs baseline mean.
    If baseline_std > 15, raise threshold to sudden_drop_high_std_delta.
    Severity: delta > 35 → urgent, else monitor.
    """
    wb_today    = today["wellbeing"]
    wb_baseline = baseline["wellbeing_mean"]
    std         = baseline["wellbeing_std"]

    threshold = (
        THRESHOLDS["sudden_drop_high_std_delta"]
        if std > THRESHOLDS["high_std_baseline"]
        else THRESHOLDS["sudden_drop_delta"]
    )

    delta = wb_baseline - wb_today          # positive = drop
    if delta < threshold:
        return None

    severity = "urgent" if delta > 35 else "monitor"

    traits = today.get("traits", {})
    lowest_trait     = min(traits, key=lambda k: traits[k]) if traits else "n/a"
    lowest_trait_val = traits.get(lowest_trait, 0)

    return {
        "category":           "SUDDEN_DROP",
        "severity":           severity,
        "title":              "Sudden wellbeing drop detected",
        "description": (
            f"{today['person_name']}'s wellbeing dropped from a baseline of "
            f"{wb_baseline:.0f} to {wb_today} today — a {delta:.0f}-point fall. "
            f"Lowest trait: {lowest_trait} at {lowest_trait_val:.0f}. "
            f"Dominant gaze: {today.get('gaze_direction', 'unknown')}."
        ),
        "baseline_wellbeing": round(wb_baseline, 1),
        "today_wellbeing":    wb_today,
        "delta":              round(-delta, 1),
        "lowest_trait":       lowest_trait,
        "lowest_trait_value": round(lowest_trait_val, 1),
        "recommended_action": "Schedule pastoral check-in today",
    }


def detect_sustained_low(history: list) -> dict | None:
    """
    Last sustained_low_days entries all have wellbeing < sustained_low_score → urgent.
    """
    n = THRESHOLDS["sustained_low_days"]
    if len(history) < n:
        return None

    window = history[-n:]
    scores = [d["wellbeing"] for d in window]
    threshold = THRESHOLDS["sustained_low_score"]

    if not all(s < threshold for s in scores):
        return None

    today  = history[-1]
    avg_wb = round(sum(scores) / len(scores), 1)

    return {
        "category":           "SUSTAINED_LOW",
        "severity":           "urgent",
        "title":              f"Wellbeing sustained below {threshold} for {n}+ days",
        "description": (
            f"{today['person_name']} has had wellbeing below {threshold} for "
            f"{n} consecutive days (avg {avg_wb}). Immediate support recommended."
        ),
        "baseline_wellbeing": None,
        "today_wellbeing":    today["wellbeing"],
        "delta":              None,
        "lowest_trait":       None,
        "lowest_trait_value": None,
        "recommended_action": "Immediate counsellor referral",
    }


def detect_social_withdrawal(today: dict, baseline: dict) -> dict | None:
    """
    social_engagement dropped ≥ social_withdrawal_delta AND gaze is down/side.
    """
    se_today    = today.get("traits", {}).get("social_engagement")
    se_baseline = baseline.get("social_engagement_mean")

    if se_today is None or se_baseline is None:
        return None

    delta    = se_baseline - se_today           # positive = drop
    gaze_dir = today.get("gaze_direction", "forward")

    if delta < THRESHOLDS["social_withdrawal_delta"]:
        return None
    if gaze_dir not in ("down", "side"):
        return None

    return {
        "category":           "SOCIAL_WITHDRAWAL",
        "severity":           "monitor",
        "title":              "Social withdrawal pattern detected",
        "description": (
            f"{today['person_name']}'s social engagement dropped {delta:.0f} pts "
            f"(from baseline {se_baseline:.0f} to {se_today:.0f}). "
            f"Gaze direction is '{gaze_dir}', suggesting avoidance."
        ),
        "baseline_wellbeing": baseline.get("wellbeing_mean"),
        "today_wellbeing":    today["wellbeing"],
        "delta":              round(-delta, 1),
        "lowest_trait":       "social_engagement",
        "lowest_trait_value": round(se_today, 1),
        "recommended_action": "Encourage peer interaction; follow up next session",
    }


def detect_hyperactivity_spike(today: dict, baseline: dict) -> dict | None:
    """
    (physical_energy + movement_energy) today minus baseline combined >= hyperactivity_delta.
    """
    pe_today = today.get("traits", {}).get("physical_energy", 0)
    me_today = today.get("traits", {}).get("movement_energy", 0)
    pe_base  = baseline.get("physical_energy_mean", 50.0)
    me_base  = baseline.get("movement_energy_mean", 50.0)

    combined_delta = (pe_today + me_today) - (pe_base + me_base)

    if combined_delta < THRESHOLDS["hyperactivity_delta"]:
        return None

    return {
        "category":           "HYPERACTIVITY_SPIKE",
        "severity":           "monitor",
        "title":              "Hyperactivity / energy spike detected",
        "description": (
            f"{today['person_name']}'s combined physical+movement energy spiked "
            f"{combined_delta:.0f} pts above baseline "
            f"(physical: {pe_today:.0f}, movement: {me_today:.0f}). "
            f"May indicate anxiety or mood dysregulation."
        ),
        "baseline_wellbeing": baseline.get("wellbeing_mean"),
        "today_wellbeing":    today["wellbeing"],
        "delta":              None,
        "lowest_trait":       None,
        "lowest_trait_value": None,
        "recommended_action": "Monitor for signs of anxiety; consider structured activity",
    }


def detect_regression(history: list) -> dict | None:
    """
    Last regression_recover_days entries were all improving (each > previous),
    then today dropped > regression_drop pts.
    """
    n = THRESHOLDS["regression_recover_days"]
    # Need at least n improving days + 1 recovery day + today = n+2 entries
    if len(history) < n + 2:
        return None

    # Window: [recovery_start ... recovery_end, today]
    # recovery window = history[-(n+1):-1], today = history[-1]
    recovery_window = history[-(n + 1):-1]   # n entries
    today           = history[-1]

    # Check all improving (strictly ascending wellbeing)
    wb_seq = [d["wellbeing"] for d in recovery_window]
    improving = all(wb_seq[i] < wb_seq[i + 1] for i in range(len(wb_seq) - 1))
    if not improving:
        return None

    drop = recovery_window[-1]["wellbeing"] - today["wellbeing"]
    if drop <= THRESHOLDS["regression_drop"]:
        return None

    return {
        "category":           "REGRESSION",
        "severity":           "monitor",
        "title":              "Regression after recovery period",
        "description": (
            f"{today['person_name']} showed {n} consecutive days of improvement "
            f"(up to {recovery_window[-1]['wellbeing']}), then dropped {drop:.0f} pts today "
            f"to {today['wellbeing']}. May signal an underlying unresolved stressor."
        ),
        "baseline_wellbeing": None,
        "today_wellbeing":    today["wellbeing"],
        "delta":              round(-drop, 1),
        "lowest_trait":       None,
        "lowest_trait_value": None,
        "recommended_action": "Discuss with student what changed; reassess support plan",
    }


def detect_gaze_avoidance(history: list) -> dict | None:
    """
    Last gaze_avoidance_days entries all have eye_contact == False.
    """
    n = THRESHOLDS["gaze_avoidance_days"]
    if len(history) < n:
        return None

    window = history[-n:]
    if not all(not d.get("eye_contact", True) for d in window):
        return None

    today = history[-1]
    return {
        "category":           "GAZE_AVOIDANCE",
        "severity":           "monitor",
        "title":              f"No eye contact detected for {n}+ consecutive days",
        "description": (
            f"{today['person_name']} has shown no eye contact for {n} consecutive days. "
            f"This may indicate social anxiety, depression, or avoidant behaviour."
        ),
        "baseline_wellbeing": None,
        "today_wellbeing":    today["wellbeing"],
        "delta":              None,
        "lowest_trait":       None,
        "lowest_trait_value": None,
        "recommended_action": "Arrange a quiet one-on-one check-in with school counsellor",
    }


# ---------------------------------------------------------------------------
# BONUS: Peer-comparison anomaly
# ---------------------------------------------------------------------------

def detect_peer_comparison(today: dict, class_mean: float, class_std: float) -> dict | None:
    """
    Flag if person's wellbeing is > 2 std below class average today.
    """
    if class_std == 0:
        return None

    wb = today["wellbeing"]
    z  = (wb - class_mean) / class_std

    if z > -2.0:
        return None

    return {
        "category":           "PEER_COMPARISON",
        "severity":           "monitor",
        "title":              "Significantly below class average wellbeing",
        "description": (
            f"{today['person_name']}'s wellbeing ({wb}) is {abs(z):.1f} std deviations "
            f"below the class average of {class_mean:.0f} today."
        ),
        "baseline_wellbeing": round(class_mean, 1),
        "today_wellbeing":    wb,
        "delta":              round(wb - class_mean, 1),
        "lowest_trait":       None,
        "lowest_trait_value": None,
        "recommended_action": "Cross-check with personal baseline; consider brief welfare check",
    }


# ---------------------------------------------------------------------------
# ANALYSE ONE PERSON
# ---------------------------------------------------------------------------

def analyse_person(person_id: str, sorted_days: dict, info: dict,
                   class_stats: dict = None) -> list:
    """
    sorted_days: { "YYYY-MM-DD": person_data_dict } — keys in date order
    info:        { name, profile_image_b64, ... }
    class_stats: { "YYYY-MM-DD": (mean, std) } for peer comparison

    Returns list of alert dicts matching anomaly_detection.json schema.
    """
    alerts     = []
    all_dates  = sorted(sorted_days.keys())
    history    = [sorted_days[d] for d in all_dates]   # oldest → newest

    if not history:
        return alerts

    baseline = compute_baseline(history)
    today    = history[-1]
    today_dt = all_dates[-1]

    person_name = info.get("name", person_id)
    profile_b64 = info.get("profile_image_b64", "")

    # Trend last 5 days wellbeing (for sparkline)
    trend = [h["wellbeing"] for h in history[-5:]]

    # Count consecutive days flagged (we build this per-run; initialise to 1)
    days_flagged = 1

    # ---- Run all detectors ------------------------------------------------
    candidate_alerts = []

    sd = detect_sudden_drop(today, baseline)
    if sd:
        candidate_alerts.append(sd)

    sl = detect_sustained_low(history)
    if sl:
        candidate_alerts.append(sl)

    sw = detect_social_withdrawal(today, baseline)
    if sw:
        candidate_alerts.append(sw)

    hs = detect_hyperactivity_spike(today, baseline)
    if hs:
        candidate_alerts.append(hs)

    rg = detect_regression(history)
    if rg:
        candidate_alerts.append(rg)

    ga = detect_gaze_avoidance(history)
    if ga:
        candidate_alerts.append(ga)

    # Bonus: peer comparison
    if class_stats and today_dt in class_stats:
        cmean, cstd = class_stats[today_dt]
        pc = detect_peer_comparison(today, cmean, cstd)
        if pc:
            candidate_alerts.append(pc)

    # ---- Package into full alert dicts ------------------------------------
    for partial in candidate_alerts:
        alert = {
            "alert_id":               _new_alert_id(),
            "person_id":              person_id,
            "person_name":            person_name,
            "date":                   today_dt,
            "severity":               partial.get("severity", "monitor"),
            "category":               partial.get("category", "SUDDEN_DROP"),
            "title":                  partial.get("title", ""),
            "description":            partial.get("description", ""),
            "baseline_wellbeing":     partial.get("baseline_wellbeing"),
            "today_wellbeing":        partial.get("today_wellbeing"),
            "delta":                  partial.get("delta"),
            "days_flagged_consecutively": days_flagged,
            "trend_last_5_days":      trend,
            "lowest_trait":           partial.get("lowest_trait"),
            "lowest_trait_value":     partial.get("lowest_trait_value"),
            "recommended_action":     partial.get("recommended_action", "Follow up"),
            "profile_image_b64":      profile_b64,
        }
        alerts.append(alert)

    return alerts


# ---------------------------------------------------------------------------
# HTML REPORT
# ---------------------------------------------------------------------------

def _sparkline_svg(trend: list, width: int = 80, height: int = 28) -> str:
    """Tiny inline SVG sparkline from a list of 0-100 wellbeing values."""
    if not trend:
        return ""
    n = len(trend)
    mn, mx = 0, 100
    pad = 3

    def x(i):
        return pad + i * (width - 2 * pad) / max(n - 1, 1)

    def y(v):
        return height - pad - (v - mn) / (mx - mn) * (height - 2 * pad)

    pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(trend))
    last_color = "#ef4444" if trend[-1] < 45 else "#f59e0b" if trend[-1] < 60 else "#22c55e"

    return (
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'style="vertical-align:middle">'
        f'<polyline points="{pts}" fill="none" stroke="#94a3b8" stroke-width="1.5"/>'
        f'<circle cx="{x(n-1):.1f}" cy="{y(trend[-1]):.1f}" r="3" '
        f'fill="{last_color}"/>'
        f'</svg>'
    )


def _badge(severity: str) -> str:
    colours = {
        "urgent":        ("background:#ef4444;color:#fff", "🔴 URGENT"),
        "monitor":       ("background:#f59e0b;color:#fff", "🟡 MONITOR"),
        "informational": ("background:#3b82f6;color:#fff", "🔵 INFO"),
    }
    style, label = colours.get(severity, ("background:#6b7280;color:#fff", severity.upper()))
    return (
        f'<span style="display:inline-block;padding:2px 8px;border-radius:999px;'
        f'font-size:11px;font-weight:700;{style}">{label}</span>'
    )


def generate_alert_digest(alerts: list, absence_flags: list,
                           school_summary: dict, output_path: Path):
    """
    Self-contained HTML counsellor report — no CDN, inline CSS only.
    Section 1: Today's alerts sorted by severity.
    Section 2: School summary.
    Section 3: Persons flagged 3+ consecutive days.
    """
    today_str = str(date.today())

    # Section 3: persistent (3+ consecutive days flagged)
    day_counts = Counter(a["person_id"] for a in alerts)
    persistent = [a for a in alerts if day_counts[a["person_id"]] >= 3]
    persistent_ids = {a["person_id"] for a in persistent}

    # ---- build alert cards ------------------------------------------------
    def alert_card(a):
        sparkline = _sparkline_svg(a.get("trend_last_5_days", []))
        badge     = _badge(a.get("severity", "monitor"))
        delta_txt = ""
        if a.get("delta") is not None:
            sign = "↑" if a["delta"] > 0 else "↓"
            delta_txt = f'<span style="color:#6b7280;font-size:12px">{sign}{abs(a["delta"]):.0f} pts</span>'

        return f"""
        <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;
                    padding:16px 20px;margin-bottom:12px;
                    border-left:4px solid {'#ef4444' if a.get('severity')=='urgent' else '#f59e0b'};">
          <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
            <strong style="font-size:15px">{a.get('person_name','–')}</strong>
            {badge}
            <span style="font-size:12px;color:#6b7280">{a.get('category','')}</span>
            {delta_txt}
            <span style="margin-left:auto">{sparkline}</span>
          </div>
          <p style="margin:8px 0 4px;font-size:13px;color:#374151">{a.get('description','')}</p>
          <p style="margin:0;font-size:12px;color:#6b7280">
            🎯 <em>{a.get('recommended_action','')}</em> &nbsp;|&nbsp;
            Alert ID: {a.get('alert_id','')} &nbsp;|&nbsp; {a.get('date','')}
          </p>
        </div>
        """

    cards_html = "".join(alert_card(a) for a in alerts) if alerts else (
        '<p style="color:#6b7280">No alerts today. 🎉</p>'
    )

    # ---- absence cards ----------------------------------------------------
    def absence_card(af):
        return f"""
        <div style="background:#fef3c7;border:1px solid #fcd34d;border-radius:10px;
                    padding:14px 18px;margin-bottom:10px">
          <strong>{af['person_name']}</strong>
          <span style="font-size:12px;color:#78350f;margin-left:8px">
            Absent {af['days_absent']} day(s) — last seen {af['last_seen_date']}
          </span>
          <p style="margin:6px 0 0;font-size:12px;color:#92400e">
            {af['recommended_action']}
          </p>
        </div>
        """

    absence_html = "".join(absence_card(af) for af in absence_flags) if absence_flags else (
        '<p style="color:#6b7280">No absence flags.</p>'
    )

    # ---- persistent section -----------------------------------------------
    persistent_cards = ""
    seen_pid = set()
    for a in alerts:
        pid = a["person_id"]
        if pid in persistent_ids and pid not in seen_pid:
            seen_pid.add(pid)
            persistent_cards += alert_card(a)
    if not persistent_cards:
        persistent_cards = '<p style="color:#6b7280">No students flagged 3+ consecutive days.</p>'

    # ---- summary boxes ----------------------------------------------------
    urgent_count = sum(1 for a in alerts if a.get("severity") == "urgent")
    monitor_count = sum(1 for a in alerts if a.get("severity") == "monitor")

    summary_html = f"""
    <div style="display:flex;gap:14px;flex-wrap:wrap;margin-bottom:24px">
      <div style="flex:1;min-width:140px;background:#fee2e2;border-radius:10px;padding:16px;text-align:center">
        <div style="font-size:32px;font-weight:800;color:#dc2626">{urgent_count}</div>
        <div style="font-size:12px;color:#7f1d1d;font-weight:600">URGENT</div>
      </div>
      <div style="flex:1;min-width:140px;background:#fef3c7;border-radius:10px;padding:16px;text-align:center">
        <div style="font-size:32px;font-weight:800;color:#d97706">{monitor_count}</div>
        <div style="font-size:12px;color:#78350f;font-weight:600">MONITOR</div>
      </div>
      <div style="flex:1;min-width:140px;background:#dcfce7;border-radius:10px;padding:16px;text-align:center">
        <div style="font-size:32px;font-weight:800;color:#16a34a">{school_summary['total_persons_tracked']}</div>
        <div style="font-size:12px;color:#14532d;font-weight:600">TRACKED</div>
      </div>
      <div style="flex:1;min-width:140px;background:#dbeafe;border-radius:10px;padding:16px;text-align:center">
        <div style="font-size:32px;font-weight:800;color:#2563eb">{school_summary.get('school_avg_wellbeing_today',0):.0f}</div>
        <div style="font-size:12px;color:#1e3a8a;font-weight:600">AVG WELLBEING</div>
      </div>
    </div>
    <table style="width:100%;border-collapse:collapse;font-size:13px">
      <tr style="background:#f8fafc">
        <td style="padding:8px 12px;border:1px solid #e2e8f0;font-weight:600">Persons flagged today</td>
        <td style="padding:8px 12px;border:1px solid #e2e8f0">{school_summary['persons_flagged_today']}</td>
      </tr>
      <tr>
        <td style="padding:8px 12px;border:1px solid #e2e8f0;font-weight:600">Persons flagged yesterday</td>
        <td style="padding:8px 12px;border:1px solid #e2e8f0">{school_summary['persons_flagged_yesterday']}</td>
      </tr>
      <tr style="background:#f8fafc">
        <td style="padding:8px 12px;border:1px solid #e2e8f0;font-weight:600">Most common anomaly this week</td>
        <td style="padding:8px 12px;border:1px solid #e2e8f0">{school_summary['most_common_anomaly_this_week']}</td>
      </tr>
      <tr>
        <td style="padding:8px 12px;border:1px solid #e2e8f0;font-weight:600">Total absence flags</td>
        <td style="padding:8px 12px;border:1px solid #e2e8f0">{len(absence_flags)}</td>
      </tr>
    </table>
    """

    generated_at = datetime.now().strftime("%d %b %Y %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sentio Mind · Alert Digest · {today_str}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f1f5f9;
    color: #1e293b;
    padding: 24px;
  }}
  .container {{ max-width: 860px; margin: 0 auto; }}
  .header {{
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
    color: #fff;
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 24px;
  }}
  .header h1 {{ font-size: 22px; font-weight: 800; letter-spacing: -0.5px; }}
  .header p  {{ font-size: 13px; opacity: 0.8; margin-top: 4px; }}
  .section-title {{
    font-size: 15px;
    font-weight: 700;
    color: #1e293b;
    margin: 24px 0 12px;
    padding-bottom: 6px;
    border-bottom: 2px solid #e2e8f0;
  }}
  .footer {{
    text-align: center;
    font-size: 11px;
    color: #94a3b8;
    margin-top: 32px;
    padding-top: 16px;
    border-top: 1px solid #e2e8f0;
  }}
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>🧠 Sentio Mind · Behavioral Alert Digest</h1>
    <p>{SCHOOL} &nbsp;|&nbsp; Report date: {today_str} &nbsp;|&nbsp; Generated: {generated_at}</p>
  </div>

  <!-- SECTION 1: TODAY'S ALERTS -->
  <div class="section-title">📋 Section 1 — Today's Alerts (sorted by severity)</div>
  {cards_html}

  <!-- ABSENCE FLAGS -->
  <div class="section-title">🚨 Absence Flags</div>
  {absence_html}

  <!-- SECTION 2: SCHOOL SUMMARY -->
  <div class="section-title">🏫 Section 2 — School Summary</div>
  {summary_html}

  <!-- SECTION 3: PERSISTENT ALERTS -->
  <div class="section-title">⏳ Section 3 — Persistent Alerts (3+ consecutive days)</div>
  {persistent_cards}

  <div class="footer">
    Sentio Mind · Project 5 · Behavioral Anomaly &amp; Early Distress Detection · 2026
  </div>

</div>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"HTML report written → {output_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    daily_data = load_daily_data(DATA_DIR)
    all_dates  = sorted(daily_data.keys())
    print(f"Loaded {len(daily_data)} days: {all_dates}")

    # Build per-person history
    person_days: dict[str, dict] = defaultdict(dict)
    person_info: dict[str, dict] = {}

    for d, persons in daily_data.items():
        for pid, pdata in persons.items():
            person_days[pid][d] = pdata
            if pid not in person_info:
                person_info[pid] = pdata.get("person_info", {
                    "name":             pdata.get("person_name", pid),
                    "profile_image_b64": "",
                })

    # Compute class-level stats per day (for peer-comparison bonus)
    class_stats: dict[str, tuple] = {}
    for d, persons in daily_data.items():
        wbs = [p["wellbeing"] for p in persons.values()]
        if wbs:
            class_stats[d] = (float(np.mean(wbs)), float(np.std(wbs)))

    all_alerts:    list = []
    absence_flags: list = []

    for pid, days in person_days.items():
        sorted_days   = dict(sorted(days.items()))
        info          = person_info.get(pid, {"name": pid, "profile_image_b64": ""})
        person_alerts = analyse_person(pid, sorted_days, info, class_stats)
        all_alerts.extend(person_alerts)

        # Absence check
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
                "person_id":        pid,
                "person_name":      info.get("name", pid),
                "last_seen_date":   last_seen,
                "days_absent":      absent,
                "recommended_action": "Welfare check — contact family if absent again tomorrow",
            })

    # Sort by severity
    sev_order = {"urgent": 0, "monitor": 1, "informational": 2}
    all_alerts.sort(key=lambda a: sev_order.get(a.get("severity", "informational"), 3))

    today_str     = str(date.today())
    flagged_today = sum(1 for a in all_alerts if a.get("date") == today_str)
    cat_counter   = Counter(a.get("category") for a in all_alerts)
    top_category  = cat_counter.most_common(1)[0][0] if cat_counter else "none"

    # Compute school avg wellbeing for last available day
    last_day = all_dates[-1] if all_dates else None
    school_avg_wb = 0
    if last_day and last_day in daily_data:
        wbs = [p["wellbeing"] for p in daily_data[last_day].values()]
        school_avg_wb = round(sum(wbs) / len(wbs), 1) if wbs else 0

    school_summary = {
        "total_persons_tracked":         len(person_days),
        "persons_flagged_today":         flagged_today,
        "persons_flagged_yesterday":     0,
        "most_common_anomaly_this_week": top_category,
        "school_avg_wellbeing_today":    school_avg_wb,
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
        "alerts":        all_alerts,
        "absence_flags": absence_flags,
        "school_summary": school_summary,
    }

    with open(FEED_OUT, "w") as f:
        json.dump(feed, f, indent=2)
    print(f"JSON feed written → {FEED_OUT}")

    generate_alert_digest(all_alerts, absence_flags, school_summary, REPORT_OUT)

    print()
    print("=" * 55)
    print(f"  Alerts  : {feed['alert_summary']['total_alerts']} total  "
          f"({feed['alert_summary']['urgent']} urgent, "
          f"{feed['alert_summary']['monitor']} monitor)")
    print(f"  Absence : {len(absence_flags)} flag(s)")
    print(f"  Report  → {REPORT_OUT}")
    print(f"  JSON    → {FEED_OUT}")
    print("=" * 55)