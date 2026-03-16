"""
app.py
------
Flask web application that exposes a /get_alerts endpoint.

This is the integration point with Sentio Mind — it returns alert_feed.json
as a JSON response. Zero changes needed to the existing analysis pipeline.

Usage:
    python src/app.py
    # Then open: http://localhost:5001/get_alerts
"""

import json
from pathlib import Path
from flask import Flask, jsonify

app = Flask(__name__)


@app.route("/get_alerts", methods=["GET"])
def get_alerts():
    """
    Returns the contents of alert_feed.json as a JSON response.

    Run solution.py first to generate alert_feed.json, then start this server.
    """
    feed_path = Path("alert_feed.json")

    if not feed_path.exists():
        return jsonify({
            "error": "alert_feed.json not found. Please run solution.py first."
        }), 404

    with open(feed_path, "r", encoding="utf-8") as f:
        feed = json.load(f)

    return jsonify(feed)


@app.route("/", methods=["GET"])
def index():
    """Root endpoint — shows available routes."""
    return jsonify({
        "service":   "Sentio Mind — Anomaly Detection",
        "version":   "1.0",
        "endpoints": {
            "GET /get_alerts": "Returns machine-readable alert feed (alert_feed.json)"
        }
    })


if __name__ == "__main__":
    print("Starting Sentio Mind Alert Server on http://localhost:5001")
    print("Make sure you have run solution.py first to generate alert_feed.json")
    app.run(debug=True, port=5001)
