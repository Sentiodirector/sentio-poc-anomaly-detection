"""
reporter.py
-----------
Generates the two output files:

1. alert_feed.json   — machine-readable JSON for the /get_alerts Flask endpoint
2. alert_digest.html — human-readable counsellor report with sparkline charts

Important: The HTML report uses ONLY inline CSS and inline SVG.
           No external CDN links — works fully offline.
"""

import json
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. JSON output — alert_feed.json
# ---------------------------------------------------------------------------

def generate_alert_feed(alerts: list, output_path: str = "alert_feed.json") -> dict:
    """
    Writes all alerts to alert_feed.json.

    Also returns the dict so the Flask /get_alerts endpoint can serve it directly.
    """
    feed = {
        "schema_version": "1.0",
        "generated_at":   datetime.now().isoformat(),
        "total_alerts":   len(alerts),
        "alerts":         alerts,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(feed, f, indent=2)

    print(f"      Saved {len(alerts)} alert(s) -> {output_path}")
    return feed


# ---------------------------------------------------------------------------
# 2. HTML helpers
# ---------------------------------------------------------------------------

def _sparkline_svg(values: list, width: int = 120, height: int = 32) -> str:
    """
    Draws a tiny inline SVG line chart (sparkline) for a list of numbers.

    No external libraries — pure SVG polyline, works offline.
    High values appear at the top; low values at the bottom.
    """
    if not values or len(values) < 2:
        return f'<svg width="{width}" height="{height}"></svg>'

    min_v = min(values)
    max_v = max(values)
    value_range = (max_v - min_v) if (max_v != min_v) else 1

    step = width / (len(values) - 1)
    padding = 3   # small padding so lines aren't clipped at edges

    points = []
    for i, v in enumerate(values):
        x = i * step
        # SVG y-axis is inverted: 0 = top. We want high values at top.
        y = padding + (height - 2 * padding) * (1 - (v - min_v) / value_range)
        points.append(f"{x:.1f},{y:.1f}")

    polyline = " ".join(points)

    # Color the line red if last value is lower than first (downward trend)
    color = "#e74c3c" if values[-1] < values[0] else "#27ae60"

    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
        f'</svg>'
    )


# ---------------------------------------------------------------------------
# 3. HTML output — alert_digest.html
# ---------------------------------------------------------------------------

def generate_alert_digest(
    alerts: list,
    records_by_person: dict,
    output_path: str = "alert_digest.html"
) -> None:
    """
    Generates a standalone HTML counsellor report.

    Features:
        - Summary dashboard (counts by severity)
        - Per-student cards showing their alerts
        - Wellbeing sparkline chart for each student
        - Colour-coded severity badges
        - Fully offline (no CDN, no JavaScript libraries)
    """

    # ---- Group alerts by student ----------------------------------------
    alerts_by_person = {}
    for alert in alerts:
        pid = alert["person_id"]
        alerts_by_person.setdefault(pid, []).append(alert)

    # ---- Colour maps -------------------------------------------------------
    severity_color = {"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#27ae60"}
    alert_icon = {
        "SUDDEN_DROP":        "↓",
        "SUSTAINED_LOW":      "⚠",
        "SOCIAL_WITHDRAWAL":  "↩",
        "HYPERACTIVITY_SPIKE": "↑",
        "REGRESSION":         "↘",
        "GAZE_AVOIDANCE":     "👁",
        "ABSENCE_FLAG":       "✗",
    }

    # ---- Build per-student HTML cards ------------------------------------
    cards_html = ""

    for person_id, person_records in records_by_person.items():
        person_alerts = alerts_by_person.get(person_id, [])

        # Sparkline: wellbeing values for detected days only
        wellbeing_values = [
            r["wellbeing_score"]
            for r in person_records
            if r.get("person_detected", True)
        ]
        sparkline = _sparkline_svg(wellbeing_values)

        # Build the alert rows for this student's table
        rows_html = ""
        for alert in person_alerts:
            color = severity_color.get(alert["severity"], "#999")
            icon  = alert_icon.get(alert["alert_type"], "!")
            rows_html += f"""
                <tr>
                  <td><span style="font-weight:600;">{icon} {alert["alert_type"]}</span></td>
                  <td>
                    <span style="background:{color};color:white;padding:2px 10px;
                                 border-radius:4px;font-size:12px;font-weight:600;">
                      {alert["severity"]}
                    </span>
                  </td>
                  <td style="white-space:nowrap;">{alert["detected_on"]}</td>
                  <td style="color:#555;">{alert["description"]}</td>
                </tr>"""

        # If no alerts, show a green "all clear" row
        if not rows_html:
            rows_html = """
                <tr>
                  <td colspan="4" style="text-align:center;color:#27ae60;padding:16px;">
                    ✓ No anomalies detected for this student.
                  </td>
                </tr>"""

        alert_count_badge = (
            f'<span style="margin-left:auto;background:#e74c3c;color:white;'
            f'padding:3px 12px;border-radius:20px;font-size:13px;font-weight:600;">'
            f'{len(person_alerts)} alert(s)</span>'
            if person_alerts else
            f'<span style="margin-left:auto;background:#27ae60;color:white;'
            f'padding:3px 12px;border-radius:20px;font-size:13px;">All clear</span>'
        )

        cards_html += f"""
      <div style="background:white;border-radius:10px;margin-bottom:20px;
                  box-shadow:0 2px 8px rgba(0,0,0,0.08);overflow:hidden;">
        <!-- Student header -->
        <div style="display:flex;align-items:center;padding:16px 22px;
                    background:#f8f9fa;border-bottom:1px solid #e9ecef;gap:18px;flex-wrap:wrap;">
          <div style="font-size:17px;font-weight:700;color:#1a237e;">{person_id}</div>
          <div style="display:flex;align-items:center;gap:8px;">
            <span style="font-size:12px;color:#888;">Wellbeing trend</span>
            {sparkline}
          </div>
          {alert_count_badge}
        </div>
        <!-- Alerts table -->
        <table style="width:100%;border-collapse:collapse;">
          <thead>
            <tr style="background:#f8f9fa;">
              <th style="padding:9px 16px;text-align:left;font-size:11px;color:#888;
                         text-transform:uppercase;letter-spacing:.5px;border-bottom:1px solid #eee;">
                Alert Type
              </th>
              <th style="padding:9px 16px;text-align:left;font-size:11px;color:#888;
                         text-transform:uppercase;letter-spacing:.5px;border-bottom:1px solid #eee;">
                Severity
              </th>
              <th style="padding:9px 16px;text-align:left;font-size:11px;color:#888;
                         text-transform:uppercase;letter-spacing:.5px;border-bottom:1px solid #eee;">
                Date
              </th>
              <th style="padding:9px 16px;text-align:left;font-size:11px;color:#888;
                         text-transform:uppercase;letter-spacing:.5px;border-bottom:1px solid #eee;">
                Description
              </th>
            </tr>
          </thead>
          <tbody>{rows_html}
          </tbody>
        </table>
      </div>"""

    # ---- Summary numbers ------------------------------------------------
    high_count   = sum(1 for a in alerts if a["severity"] == "HIGH")
    medium_count = sum(1 for a in alerts if a["severity"] == "MEDIUM")
    students_at_risk = len(alerts_by_person)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---- Assemble full HTML page ----------------------------------------
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sentio Mind — Behavioral Alert Digest</title>
  <style>
    /* Reset */
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      background: #f0f2f5;
      color: #333;
      padding: 24px;
      min-height: 100vh;
    }}

    /* Responsive table on small screens */
    @media (max-width: 600px) {{
      table, thead, tbody, th, td, tr {{ display: block; }}
      thead tr {{ display: none; }}
    }}
  </style>
</head>
<body>

  <!-- Page header -->
  <div style="background:linear-gradient(135deg,#1a237e,#3949ab);color:white;
              padding:28px 32px;border-radius:12px;margin-bottom:22px;">
    <div style="font-size:10px;letter-spacing:2px;opacity:.7;margin-bottom:6px;">
      SENTIO MIND  •  CONFIDENTIAL
    </div>
    <h1 style="font-size:24px;font-weight:700;margin-bottom:4px;">
      Behavioral Alert Digest
    </h1>
    <p style="opacity:.8;font-size:14px;">Generated: {generated_at} &nbsp;|&nbsp; Counsellor Report</p>
  </div>

  <!-- Summary cards -->
  <div style="display:flex;gap:14px;margin-bottom:22px;flex-wrap:wrap;">
    <div style="background:white;padding:18px 24px;border-radius:10px;
                box-shadow:0 2px 8px rgba(0,0,0,.07);flex:1;min-width:130px;text-align:center;">
      <div style="font-size:34px;font-weight:700;color:#1a237e;">{len(records_by_person)}</div>
      <div style="font-size:12px;color:#888;margin-top:4px;">Students Monitored</div>
    </div>
    <div style="background:white;padding:18px 24px;border-radius:10px;
                box-shadow:0 2px 8px rgba(0,0,0,.07);flex:1;min-width:130px;text-align:center;">
      <div style="font-size:34px;font-weight:700;color:#e74c3c;">{len(alerts)}</div>
      <div style="font-size:12px;color:#888;margin-top:4px;">Total Alerts</div>
    </div>
    <div style="background:white;padding:18px 24px;border-radius:10px;
                box-shadow:0 2px 8px rgba(0,0,0,.07);flex:1;min-width:130px;text-align:center;">
      <div style="font-size:34px;font-weight:700;color:#e74c3c;">{high_count}</div>
      <div style="font-size:12px;color:#888;margin-top:4px;">High Severity</div>
    </div>
    <div style="background:white;padding:18px 24px;border-radius:10px;
                box-shadow:0 2px 8px rgba(0,0,0,.07);flex:1;min-width:130px;text-align:center;">
      <div style="font-size:34px;font-weight:700;color:#f39c12;">{medium_count}</div>
      <div style="font-size:12px;color:#888;margin-top:4px;">Medium Severity</div>
    </div>
    <div style="background:white;padding:18px 24px;border-radius:10px;
                box-shadow:0 2px 8px rgba(0,0,0,.07);flex:1;min-width:130px;text-align:center;">
      <div style="font-size:34px;font-weight:700;color:#333;">{students_at_risk}</div>
      <div style="font-size:12px;color:#888;margin-top:4px;">Students with Alerts</div>
    </div>
  </div>

  <!-- Per-student cards -->
  {cards_html}

  <!-- Footer -->
  <div style="text-align:center;color:#bbb;font-size:12px;margin-top:28px;padding-bottom:12px;">
    Sentio Mind Anomaly Detection &nbsp;|&nbsp;
    This report is auto-generated and must be reviewed by a qualified counsellor.
  </div>

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"      Saved alert digest  -> {output_path}")
