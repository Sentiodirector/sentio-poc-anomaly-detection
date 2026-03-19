"""
solution.py
Sentio Mind · Project 5 · Behavioral Anomaly & Early Distress Detection

Run: python solution.py
Outputs: alert_digest.html, alert_feed.json
"""

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
    "peer_comparison_std":        2.0,
}


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
        with open(fp, "r", encoding="utf-8") as f:
            raw = json.load(f)
        day_date = raw.get("date", fp.stem)
        persons_dict = {}
        for p in raw.get("persons", []):
            pid = p["person_id"]
            persons_dict[pid] = {
                "person_id":         pid,
                "name":              p.get("name", pid),
                "wellbeing":         p.get("wellbeing"),
                "traits": {
                    "social_engagement": p.get("traits", {}).get("social_engagement"),
                    "physical_energy":   p.get("traits", {}).get("physical_energy"),
                    "movement_energy":   p.get("traits", {}).get("movement_energy"),
                },
                "gaze_direction":    p.get("gaze_direction", "forward"),
                "eye_contact":       p.get("eye_contact", True),
                "emotion":           p.get("emotion", "neutral"),
                "detected":          p.get("detected", True),
                "profile_image_b64": p.get("profile_image_b64", ""),
                "person_info":       p.get("person_info", {"name": p.get("name", pid), "profile_image_b64": ""}),
            }
        daily[day_date] = persons_dict
    return daily


# ---------------------------------------------------------------------------
# BASELINE
# ---------------------------------------------------------------------------

def compute_baseline(history: list) -> dict:
    """
    history: list of daily dicts (oldest first).
    Use first THRESHOLDS['baseline_window'] detected days.
    Return: { wellbeing_mean, wellbeing_std, trait_means: {}, avg_gaze: str }
    """
    window = THRESHOLDS["baseline_window"]
    base   = [h for h in history if h.get("detected", True) and h.get("wellbeing") is not None]
    base   = base[:window] if len(base) >= window else base

    if not base:
        return {
            "wellbeing_mean": 50.0,
            "wellbeing_std":  10.0,
            "trait_means":    {"social_engagement": 50.0, "physical_energy": 50.0, "movement_energy": 50.0},
            "avg_gaze":       "forward",
        }

    wb_vals = [b["wellbeing"] for b in base]
    trait_keys = ["social_engagement", "physical_energy", "movement_energy"]
    trait_means = {}
    for k in trait_keys:
        vals = [b["traits"].get(k) for b in base if b.get("traits", {}).get(k) is not None]
        trait_means[k] = float(np.mean(vals)) if vals else 50.0

    gazes = [b["gaze_direction"] for b in base if b.get("gaze_direction")]
    avg_gaze = Counter(gazes).most_common(1)[0][0] if gazes else "forward"

    return {
        "wellbeing_mean": float(np.mean(wb_vals)),
        "wellbeing_std":  float(np.std(wb_vals, ddof=0)) if len(wb_vals) > 1 else 0.0,
        "trait_means":    trait_means,
        "avg_gaze":       avg_gaze,
    }


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

_alert_counter = 0

def _new_alert_id() -> str:
    global _alert_counter
    _alert_counter += 1
    return f"ALT_{_alert_counter:03d}"

def _lowest_trait(traits: dict) -> tuple:
    valid = {k: v for k, v in traits.items() if v is not None}
    if not valid:
        return ("unknown", 0)
    k = min(valid, key=valid.get)
    return (k, valid[k])

def _trend_last_5(history: list) -> list:
    vals = [h.get("wellbeing") if h.get("detected", True) else 0 for h in history]
    vals = vals[-5:]
    while len(vals) < 5:
        vals.insert(0, 0)
    return vals


# ---------------------------------------------------------------------------
# ANOMALY DETECTORS
# ---------------------------------------------------------------------------

def detect_sudden_drop(today: dict, baseline: dict) -> dict | None:
    """
    today['wellbeing'] vs baseline['wellbeing_mean'].
    If baseline_std > 15, raise threshold to sudden_drop_high_std_delta.
    Severity: delta > 35 = urgent, else monitor.
    """
    wb_today = today.get("wellbeing")
    if wb_today is None:
        return None
    std = baseline.get("wellbeing_std", 0) or 0
    threshold = (
        THRESHOLDS["sudden_drop_high_std_delta"]
        if std > THRESHOLDS["high_std_baseline"]
        else THRESHOLDS["sudden_drop_delta"]
    )
    delta = baseline["wellbeing_mean"] - wb_today
    if delta < threshold:
        return None
    severity = "urgent" if delta > 35 else "monitor"
    lt_name, lt_val = _lowest_trait(today.get("traits", {}))
    return {
        "category": "SUDDEN_DROP",
        "severity": severity,
        "delta":    round(delta, 1),
        "title":    "Sudden wellbeing drop detected",
        "description": (
            f"{today['name']}'s wellbeing dropped from a baseline of "
            f"{baseline['wellbeing_mean']:.0f} to {wb_today} today — "
            f"a {delta:.0f}-point fall. Lowest trait: {lt_name} at {lt_val:.0f}. "
            f"Dominant gaze: {today.get('gaze_direction','?')}."
        ),
        "baseline_wellbeing": round(baseline["wellbeing_mean"], 1),
        "today_wellbeing":    wb_today,
        "lowest_trait":       lt_name,
        "lowest_trait_value": round(lt_val, 1),
        "recommended_action": "Schedule pastoral check-in today",
    }


def detect_sustained_low(history: list) -> dict | None:
    """
    Check the last sustained_low_days entries in history.
    If all have wellbeing < sustained_low_score → alert. Severity: urgent.
    """
    days_req  = THRESHOLDS["sustained_low_days"]
    threshold = THRESHOLDS["sustained_low_score"]
    detected  = [h for h in history if h.get("detected", True) and h.get("wellbeing") is not None]
    if len(detected) < days_req:
        return None
    last_n = detected[-days_req:]
    if not all(h["wellbeing"] < threshold for h in last_n):
        return None
    today = last_n[-1]
    return {
        "category": "SUSTAINED_LOW",
        "severity": "urgent",
        "delta":    None,
        "title":    f"Sustained low wellbeing ({days_req}+ days)",
        "description": (
            f"{today['name']} has had wellbeing below {threshold} for "
            f"{days_req}+ consecutive days (scores: {[h['wellbeing'] for h in last_n]})."
        ),
        "baseline_wellbeing": None,
        "today_wellbeing":    today["wellbeing"],
        "lowest_trait":       "",
        "lowest_trait_value": 0,
        "recommended_action": "Urgent counsellor meeting — arrange today",
    }


def detect_social_withdrawal(today: dict, baseline: dict) -> dict | None:
    """
    social_engagement dropped >= social_withdrawal_delta AND
    today's gaze_direction is "down" or "side". Severity: monitor.
    """
    se_today    = today.get("traits", {}).get("social_engagement")
    se_baseline = baseline.get("trait_means", {}).get("social_engagement")
    if se_today is None or se_baseline is None:
        return None
    drop = se_baseline - se_today
    gaze = today.get("gaze_direction", "forward")
    if drop < THRESHOLDS["social_withdrawal_delta"] or gaze not in ("down", "side"):
        return None
    return {
        "category": "SOCIAL_WITHDRAWAL",
        "severity": "monitor",
        "delta":    round(drop, 1),
        "title":    "Social withdrawal detected",
        "description": (
            f"{today['name']}'s social engagement dropped {drop:.0f} pts vs baseline "
            f"and gaze is '{gaze}'. May be withdrawing from peers."
        ),
        "baseline_wellbeing": round(baseline["wellbeing_mean"], 1),
        "today_wellbeing":    today.get("wellbeing"),
        "lowest_trait":       "social_engagement",
        "lowest_trait_value": round(se_today, 1),
        "recommended_action": "Observe at lunch — consider peer support referral",
    }


def detect_hyperactivity_spike(today: dict, baseline: dict) -> dict | None:
    """
    (today.physical_energy + today.movement_energy) minus
    (baseline.physical_energy_mean + baseline.movement_energy_mean) >= hyperactivity_delta.
    Severity: monitor.
    """
    pe_today = today.get("traits", {}).get("physical_energy")
    me_today = today.get("traits", {}).get("movement_energy")
    pe_base  = baseline.get("trait_means", {}).get("physical_energy")
    me_base  = baseline.get("trait_means", {}).get("movement_energy")
    if any(v is None for v in [pe_today, me_today, pe_base, me_base]):
        return None
    spike = (pe_today + me_today) - (pe_base + me_base)
    if spike < THRESHOLDS["hyperactivity_delta"]:
        return None
    return {
        "category": "HYPERACTIVITY_SPIKE",
        "severity": "monitor",
        "delta":    round(spike, 1),
        "title":    "Hyperactivity spike detected",
        "description": (
            f"{today['name']}'s combined energy is {spike:.0f} pts above baseline. "
            "May indicate anxiety or hyper-arousal."
        ),
        "baseline_wellbeing": round(baseline["wellbeing_mean"], 1),
        "today_wellbeing":    today.get("wellbeing"),
        "lowest_trait":       "movement_energy",
        "lowest_trait_value": round(me_today, 1),
        "recommended_action": "Monitor for rest of day — check for sleep issues",
    }


def detect_regression(history: list) -> dict | None:
    """
    Find if the last regression_recover_days entries were all improving (each > previous),
    then today dropped > regression_drop. Severity: monitor.
    """
    recover_days = THRESHOLDS["regression_recover_days"]
    drop_thresh  = THRESHOLDS["regression_drop"]
    detected = [h for h in history if h.get("detected", True) and h.get("wellbeing") is not None]
    if len(detected) < recover_days + 1:
        return None
    window = detected[-(recover_days + 1):]
    scores = [h["wellbeing"] for h in window]
    improving = all(scores[i] < scores[i + 1] for i in range(len(scores) - 2))
    drop = scores[-2] - scores[-1]
    if not improving or drop <= drop_thresh:
        return None
    today = detected[-1]
    return {
        "category": "REGRESSION",
        "severity": "monitor",
        "delta":    round(drop, 1),
        "title":    "Regression after recovery",
        "description": (
            f"{today['name']} was recovering for {recover_days}+ days, "
            f"then dropped {drop:.0f} pts today. May have relapsed."
        ),
        "baseline_wellbeing": None,
        "today_wellbeing":    today["wellbeing"],
        "lowest_trait":       "",
        "lowest_trait_value": 0,
        "recommended_action": "Follow-up with counsellor — check for recent stressors",
    }


def detect_gaze_avoidance(history: list) -> dict | None:
    """
    Last gaze_avoidance_days entries all have eye_contact == False.
    Severity: monitor.
    """
    days_req = THRESHOLDS["gaze_avoidance_days"]
    detected = [h for h in history if h.get("detected", True)]
    if len(detected) < days_req:
        return None
    last_n = detected[-days_req:]
    if not all(not h.get("eye_contact", True) for h in last_n):
        return None
    today = last_n[-1]
    return {
        "category": "GAZE_AVOIDANCE",
        "severity": "monitor",
        "delta":    None,
        "title":    f"Gaze avoidance — {days_req}+ consecutive days",
        "description": (
            f"{today['name']} has had no eye contact detected for "
            f"{days_req}+ consecutive days. May indicate anxiety or low confidence."
        ),
        "baseline_wellbeing": None,
        "today_wellbeing":    today.get("wellbeing"),
        "lowest_trait":       "",
        "lowest_trait_value": 0,
        "recommended_action": "Gentle 1-on-1 check-in from teacher — avoid group pressure",
    }


# ---------------------------------------------------------------------------
# ANALYSE ONE PERSON
# ---------------------------------------------------------------------------

def analyse_person(person_id: str, sorted_days: dict, info: dict,
                   class_mean: float = 0.0, class_std: float = 0.0) -> list:
    """
    sorted_days: { "YYYY-MM-DD": person_data_dict } — keys in date order
    info: { name, profile_image_b64, ... }
    Build history list, compute baseline, run all detectors.
    Return list of alert dicts matching anomaly_detection.json schema.
    """
    history    = list(sorted_days.values())
    baseline   = compute_baseline(history)
    today      = history[-1] if history else {}
    today_date = list(sorted_days.keys())[-1] if sorted_days else str(date.today())

    if not today.get("detected", True) or today.get("wellbeing") is None:
        return []

    trend = _trend_last_5(history)

    # BONUS: peer comparison
    wb_today   = today.get("wellbeing")
    peer_alert = None
    if wb_today is not None and class_std > 0:
        z = (wb_today - class_mean) / class_std
        if z < -THRESHOLDS["peer_comparison_std"]:
            peer_alert = {
                "category": "PEER_COMPARISON",
                "severity": "monitor",
                "delta":    round(class_mean - wb_today, 1),
                "title":    "Significantly below class average",
                "description": (
                    f"{today['name']}'s wellbeing ({wb_today}) is {abs(z):.1f} std deviations "
                    f"below today's class average ({class_mean:.0f})."
                ),
                "baseline_wellbeing": round(baseline["wellbeing_mean"], 1),
                "today_wellbeing":    wb_today,
                "lowest_trait":       "",
                "lowest_trait_value": 0,
                "recommended_action": "Peer support or counsellor check-in recommended",
            }

    detector_results = [
        detect_sudden_drop(today, baseline),
        detect_sustained_low(history),
        detect_social_withdrawal(today, baseline),
        detect_hyperactivity_spike(today, baseline),
        detect_regression(history),
        detect_gaze_avoidance(history),
        peer_alert,
    ]

    alerts = []
    for result in detector_results:
        if result is None:
            continue
        alerts.append({
            "alert_id":                  _new_alert_id(),
            "person_id":                 person_id,
            "person_name":               today.get("name", person_id),
            "date":                      today_date,
            "severity":                  result["severity"],
            "category":                  result["category"],
            "title":                     result["title"],
            "description":               result["description"],
            "baseline_wellbeing":        result.get("baseline_wellbeing"),
            "today_wellbeing":           result.get("today_wellbeing"),
            "delta":                     result.get("delta"),
            "days_flagged_consecutively": 1,
            "trend_last_5_days":         trend,
            "lowest_trait":              result.get("lowest_trait", ""),
            "lowest_trait_value":        result.get("lowest_trait_value", 0),
            "recommended_action":        result.get("recommended_action", ""),
            "profile_image_b64":         info.get("profile_image_b64", ""),
        })
    return alerts


# ---------------------------------------------------------------------------
# HTML REPORT
# ---------------------------------------------------------------------------

def _sparkline_svg(trend: list, width: int = 90, height: int = 28) -> str:
    vals = [v for v in trend if v and v > 0]
    if len(vals) < 2:
        return "<span style='color:#bbb;font-size:11px'>—</span>"
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1
    n = len(vals)
    step = width / max(n - 1, 1)
    pad = 3

    def py(v):
        return height - pad - (v - lo) / span * (height - 2 * pad)

    pts   = " ".join(f"{i*step:.1f},{py(v):.1f}" for i, v in enumerate(vals))
    color = "#e74c3c" if vals[-1] < 45 else "#27ae60" if vals[-1] >= 65 else "#e67e22"
    lx, ly = (n - 1) * step, py(vals[-1])

    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" style="display:block">'
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.2" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
        f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="3.5" fill="{color}"/>'
        f'</svg>'
    )


def _sev_badge(sev: str) -> str:
    bg = {"urgent": "#e74c3c", "monitor": "#e67e22", "informational": "#3498db"}.get(sev, "#888")
    return (f'<span style="background:{bg};color:#fff;padding:3px 10px;border-radius:20px;'
            f'font-size:11px;font-weight:700;letter-spacing:.6px">{sev.upper()}</span>')


def _cat_badge(cat: str) -> str:
    c = {"SUDDEN_DROP":"#c0392b","SUSTAINED_LOW":"#8e44ad","SOCIAL_WITHDRAWAL":"#2980b9",
         "HYPERACTIVITY_SPIKE":"#d35400","REGRESSION":"#16a085","GAZE_AVOIDANCE":"#7f8c8d",
         "ABSENCE_FLAG":"#2c3e50","PEER_COMPARISON":"#1abc9c"}.get(cat, "#555")
    return (f'<span style="background:{c};color:#fff;padding:2px 8px;border-radius:4px;'
            f'font-size:11px;font-weight:600">{cat}</span>')


def _profile_img(b64: str, name: str, size: int = 44) -> str:
    if b64:
        fmt = "jpeg" if b64.startswith("/9j") else "png"
        return (f'<img src="data:image/{fmt};base64,{b64}" '
                f'style="width:{size}px;height:{size}px;border-radius:50%;'
                f'object-fit:cover;border:2px solid #e0e0e0" alt="{name}"/>')
    initials = "".join(w[0].upper() for w in name.split()[:2]) or "?"
    return (f'<div style="width:{size}px;height:{size}px;border-radius:50%;'
            f'background:#0f3460;color:#fff;display:flex;align-items:center;'
            f'justify-content:center;font-weight:700;font-size:{size//3}px">{initials}</div>')


def generate_alert_digest(alerts: list, absence_flags: list,
                           school_summary: dict, output_path: Path):
    """
    Self-contained HTML — no CDN, inline CSS only.
    Section 1: Today's alerts sorted by severity with sparklines.
    Section 2: School summary numbers.
    Section 3: Persons flagged 3+ consecutive days.
    """
    today_str   = str(date.today())
    gen_time    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    urgent_cnt  = sum(1 for a in alerts if a.get("severity") == "urgent")
    monitor_cnt = sum(1 for a in alerts if a.get("severity") == "monitor")
    flagged     = len({a["person_id"] for a in alerts})
    top_cat     = school_summary.get("most_common_anomaly_this_week", "—")

    # Alert cards
    alert_cards = ""
    if not alerts:
        alert_cards = ('<div style="text-align:center;padding:40px;color:#aaa;font-size:15px">'
                       '✅ No alerts today — all students within normal range.</div>')
    else:
        for a in alerts:
            sev_col = "#e74c3c" if a["severity"] == "urgent" else "#e67e22"
            delta   = a.get("delta")
            delta_html = (f'<div><div style="font-size:10px;color:#aaa;text-transform:uppercase;'
                          f'letter-spacing:.5px">Drop</div>'
                          f'<div style="font-size:18px;font-weight:700;color:#e74c3c">Δ{delta:.0f}</div></div>'
                          if delta else "")
            wb_t = a.get("today_wellbeing", "N/A")
            wb_b = a.get("baseline_wellbeing", "N/A")

            alert_cards += f"""
<div style="border:1px solid #eee;border-left:5px solid {sev_col};border-radius:8px;
            padding:16px 20px;margin-bottom:14px;background:#fff;
            box-shadow:0 2px 6px rgba(0,0,0,.05);display:flex;gap:16px;align-items:flex-start">
  <div style="flex-shrink:0">{_profile_img(a.get('profile_image_b64',''), a['person_name'])}</div>
  <div style="flex:1;min-width:0">
    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px">
      <span style="font-weight:700;font-size:15px">{a['person_name']}</span>
      {_sev_badge(a['severity'])} {_cat_badge(a['category'])}
      <span style="font-size:12px;color:#bbb;margin-left:auto">{a['date']}</span>
    </div>
    <div style="font-size:13px;color:#444;margin-bottom:10px;line-height:1.5">{a['description']}</div>
    <div style="display:flex;align-items:center;gap:20px;flex-wrap:wrap">
      <div><div style="font-size:10px;color:#aaa;text-transform:uppercase;letter-spacing:.5px">5-day trend</div>
           {_sparkline_svg(a.get('trend_last_5_days',[]))}</div>
      <div><div style="font-size:10px;color:#aaa;text-transform:uppercase;letter-spacing:.5px">Today</div>
           <div style="font-size:22px;font-weight:700;color:{sev_col}">{wb_t}</div></div>
      <div><div style="font-size:10px;color:#aaa;text-transform:uppercase;letter-spacing:.5px">Baseline</div>
           <div style="font-size:22px;font-weight:700;color:#888">{wb_b}</div></div>
      {delta_html}
    </div>
    <div style="margin-top:10px;padding:8px 12px;background:#f8f9fb;border-radius:6px;font-size:12px;color:#555">
      <strong>Action:</strong> {a.get('recommended_action','—')}
    </div>
  </div>
</div>"""

    # Absence section
    absence_html = ""
    for ab in absence_flags:
        absence_html += f"""
<div style="border:1px solid #2c3e50;border-left:5px solid #2c3e50;border-radius:8px;
            padding:14px 20px;margin-bottom:12px;background:#fff">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
    <span style="font-weight:700">{ab['person_name']}</span>
    {_sev_badge('urgent')} {_cat_badge('ABSENCE_FLAG')}
  </div>
  <div style="font-size:13px;color:#444">
    Not detected for <strong>{ab['days_absent']} consecutive days</strong>. Last seen: {ab['last_seen_date']}.
  </div>
  <div style="margin-top:8px;padding:7px 12px;background:#f8f9fb;border-radius:6px;font-size:12px;color:#555">
    <strong>Action:</strong> {ab['recommended_action']}
  </div>
</div>"""
    if not absence_html:
        absence_html = '<p style="color:#aaa;font-size:13px;padding:8px 0">No absences flagged.</p>'

    # Persistent
    pid_cnt = Counter(a["person_id"] for a in alerts)
    persist_rows = ""
    for pid, cnt in pid_cnt.most_common():
        if cnt < 3:
            continue
        pa   = next(a for a in alerts if a["person_id"] == pid)
        cats = ", ".join(sorted({a["category"] for a in alerts if a["person_id"] == pid}))
        persist_rows += f"""
<tr>
  <td style="padding:12px 10px">
    <div style="display:flex;align-items:center;gap:10px">
      {_profile_img(pa.get('profile_image_b64',''), pa['person_name'], 36)}
      <strong>{pa['person_name']}</strong>
    </div>
  </td>
  <td style="padding:12px 10px;font-size:12px;color:#555">{cats}</td>
  <td style="padding:12px 10px;text-align:center;font-weight:700;color:#e74c3c">{cnt}</td>
  <td style="padding:12px 10px">{_sparkline_svg(pa.get('trend_last_5_days',[]))}</td>
</tr>"""
    if not persist_rows:
        persist_rows = '<tr><td colspan="4" style="text-align:center;color:#aaa;padding:24px">No persistent alerts this week.</td></tr>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Sentio Mind · Alert Digest · {today_str}</title>
<style>
  *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       background:#f0f2f5;color:#1a1a2e;font-size:14px;line-height:1.5}}
  header{{background:linear-gradient(135deg,#1a1a2e 0%,#16213e 60%,#0f3460 100%);
          color:#fff;padding:24px 40px;display:flex;align-items:center;justify-content:space-between}}
  header h1{{font-size:20px;font-weight:700}}
  header p{{font-size:12px;opacity:.65;margin-top:3px}}
  .pill{{background:rgba(255,255,255,.12);border-radius:20px;padding:5px 16px;font-size:12px;font-weight:600;letter-spacing:.8px}}
  main{{max-width:980px;margin:28px auto;padding:0 20px 60px}}
  .kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:26px}}
  .kpi{{background:#fff;border-radius:10px;padding:18px 20px;box-shadow:0 2px 8px rgba(0,0,0,.07);border-top:4px solid #0f3460}}
  .kpi.u{{border-top-color:#e74c3c}}.kpi.m{{border-top-color:#e67e22}}.kpi.c{{border-top-color:#8e44ad}}
  .kv{{font-size:30px;font-weight:700;color:#0f3460}}
  .kpi.u .kv{{color:#e74c3c}}.kpi.m .kv{{color:#e67e22}}
  .kl{{font-size:11px;color:#999;margin-top:3px;text-transform:uppercase;letter-spacing:.5px}}
  section{{background:#fff;border-radius:10px;padding:22px 24px;margin-bottom:22px;box-shadow:0 2px 8px rgba(0,0,0,.07)}}
  section h2{{font-size:15px;font-weight:700;color:#0f3460;margin-bottom:16px;border-bottom:2px solid #f0f2f5;padding-bottom:10px}}
  table{{width:100%;border-collapse:collapse}}
  thead th{{background:#f8f9fb;font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:#999;padding:9px 10px;text-align:left}}
  tbody tr{{border-bottom:1px solid #f4f4f4}}
  tbody tr:last-child{{border-bottom:none}}
  tbody tr:hover{{background:#fafafa}}
  footer{{text-align:center;color:#bbb;font-size:11px;padding:20px 0 40px}}
  @media(max-width:640px){{.kpis{{grid-template-columns:repeat(2,1fr)}}}}
</style>
</head>
<body>
<header>
  <div>
    <h1>&#127914; Sentio Mind &nbsp;·&nbsp; Alert Digest</h1>
    <p>Behavioral Anomaly Report &nbsp;|&nbsp; {today_str} &nbsp;|&nbsp; {SCHOOL}</p>
  </div>
  <div class="pill">COUNSELLOR REPORT</div>
</header>
<main>
<div class="kpis">
  <div class="kpi"><div class="kv">{flagged}</div><div class="kl">Persons Flagged</div></div>
  <div class="kpi u"><div class="kv">{urgent_cnt}</div><div class="kl">Urgent Alerts</div></div>
  <div class="kpi m"><div class="kv">{monitor_cnt}</div><div class="kl">Monitor Alerts</div></div>
  <div class="kpi c"><div class="kv" style="font-size:15px;padding-top:8px">{top_cat}</div><div class="kl">Top Anomaly This Week</div></div>
</div>
<section>
  <h2>&#9888;&#65039; Today's Alerts <span style="font-weight:400;color:#aaa;font-size:13px">sorted by severity</span></h2>
  {alert_cards}
</section>
<section>
  <h2>&#128680; Absence Flags</h2>
  {absence_html}
</section>
<section>
  <h2>&#127981; School Summary</h2>
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>
      <tr><td style="padding:11px 10px">Total persons tracked</td><td style="padding:11px 10px"><strong>{school_summary.get('total_persons_tracked','—')}</strong></td></tr>
      <tr><td style="padding:11px 10px">Persons flagged today</td><td style="padding:11px 10px"><strong>{school_summary.get('persons_flagged_today','—')}</strong></td></tr>
      <tr><td style="padding:11px 10px">Persons flagged yesterday</td><td style="padding:11px 10px"><strong>{school_summary.get('persons_flagged_yesterday','—')}</strong></td></tr>
      <tr><td style="padding:11px 10px">Most common anomaly this week</td><td style="padding:11px 10px"><strong>{top_cat}</strong></td></tr>
      <tr><td style="padding:11px 10px">School avg wellbeing today</td><td style="padding:11px 10px"><strong>{school_summary.get('school_avg_wellbeing_today','—')}</strong></td></tr>
    </tbody>
  </table>
</section>
<section>
  <h2>&#128204; Persistent Alerts <span style="font-weight:400;color:#aaa;font-size:13px">flagged 3+ times this week</span></h2>
  <table>
    <thead><tr><th>Student</th><th>Anomaly Types</th><th style="text-align:center">Count</th><th>Trend</th></tr></thead>
    <tbody>{persist_rows}</tbody>
  </table>
</section>
</main>
<footer>Sentio Mind &copy; 2026 &nbsp;&middot;&nbsp; Generated {gen_time} &nbsp;&middot;&nbsp; Confidential Counsellor Report</footer>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[✓] {output_path} written")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    daily_data = load_daily_data(DATA_DIR)
    all_dates  = sorted(daily_data.keys())
    print(f"Loaded {len(daily_data)} days: {all_dates}")

    person_days: dict = defaultdict(dict)
    person_info: dict = {}
    for d, persons in daily_data.items():
        for pid, pdata in persons.items():
            person_days[pid][d] = pdata
            if pid not in person_info:
                person_info[pid] = pdata.get("person_info", {"name": pdata.get("name", pid), "profile_image_b64": ""})

    # Class-wide stats for peer comparison
    today_key     = all_dates[-1] if all_dates else ""
    today_wb_vals = [
        v["wellbeing"] for v in daily_data.get(today_key, {}).values()
        if v.get("detected", True) and v.get("wellbeing") is not None
    ]
    class_mean  = float(np.mean(today_wb_vals)) if today_wb_vals else 0.0
    class_std   = float(np.std(today_wb_vals, ddof=0)) if len(today_wb_vals) > 1 else 0.0
    school_avg  = round(class_mean, 1) if today_wb_vals else 0

    all_alerts:    list = []
    absence_flags: list = []

    for pid, days in person_days.items():
        sorted_days   = dict(sorted(days.items()))
        person_alerts = analyse_person(pid, sorted_days, person_info.get(pid, {}),
                                       class_mean=class_mean, class_std=class_std)
        all_alerts.extend(person_alerts)

        # Absence detection
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

    sev_order  = {"urgent": 0, "monitor": 1, "informational": 2}
    all_alerts.sort(key=lambda a: sev_order.get(a.get("severity", "informational"), 3))

    flagged_today = len({a["person_id"] for a in all_alerts})
    cat_counter   = Counter(a.get("category") for a in all_alerts)
    top_category  = cat_counter.most_common(1)[0][0] if cat_counter else "none"

    school_summary = {
        "total_persons_tracked":         len(person_days),
        "persons_flagged_today":         flagged_today,
        "persons_flagged_yesterday":     0,
        "most_common_anomaly_this_week": top_category,
        "school_avg_wellbeing_today":    school_avg,
    }

    # Build feed — strip profile_image_b64 (too large for JSON feed)
    feed_alerts = [{k: v for k, v in a.items() if k != "profile_image_b64"} for a in all_alerts]

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
        "alerts":         feed_alerts,
        "absence_flags":  absence_flags,
        "school_summary": school_summary,
    }

    with open(FEED_OUT, "w") as f:
        json.dump(feed, f, indent=2)
    print(f"[✓] {FEED_OUT} written")

    generate_alert_digest(all_alerts, absence_flags, school_summary, REPORT_OUT)

    print()
    print("=" * 50)
    print(f"  Alerts:  {feed['alert_summary']['total_alerts']} total  "
          f"({feed['alert_summary']['urgent']} urgent, {feed['alert_summary']['monitor']} monitor)")
    print(f"  Absence flags: {len(absence_flags)}")
    print(f"  Report → {REPORT_OUT}")
    print(f"  JSON   → {FEED_OUT}")
    print("=" * 50)
