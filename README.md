# Sentio Mind · Behavioral Anomaly & Early Distress Detection

This project implements a rule-based anomaly detection system to identify early signs of distress in students using daily wellbeing and behavioral data.

It processes structured JSON outputs and generates:
- `alert_feed.json` → machine-readable alert stream
- `alert_digest.html` → human-readable counsellor report

---

## 🚀 How to Run

Follow the steps below to execute the pipeline:

### 1. Generate Sample Data
```bash
python generate.py
```
2. Run Anomaly Detection
```bash
python solution.py
```
📂 Output Files

After execution, the following files will be generated:

alert_feed.json
Structured alert data conforming to the required schema.

alert_digest.html
Offline-ready report for counsellors with categorized alerts and summaries.


Human-readable alert explanations

No external dependencies beyond numpy
