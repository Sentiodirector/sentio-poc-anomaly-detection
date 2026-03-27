"""
app.py
Sentio Mind · Project 5 · Minimal Flask API

Endpoints:
  GET /get_alerts  — returns alert_feed.json as JSON
  GET /health      — returns {"status": "ok"}

Run: python app.py
"""

import json
from pathlib import Path
from flask import Flask, jsonify

app = Flask(__name__)
FEED_PATH = Path("alert_feed.json")


@app.route("/get_alerts")
def get_alerts():
    if not FEED_PATH.exists():
        return jsonify({"error": "alert_feed.json not found — run solution.py first"}), 404
    with open(FEED_PATH) as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(port=5000, debug=False)
