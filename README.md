# Web Log Anomaly Detector

End‑to‑end machine learning project that analyzes web server access logs, builds user sessions, and detects anomalous sessions (potential bots, attacks, or error‑heavy behavior).  
Training is done in a Kaggle notebook on a real Apache `access.log` dataset, and inference is exposed via a Streamlit dashboard.

---

## 1. Project Overview

This project turns raw Apache access logs into actionable **business intelligence** and **anomaly alerts**:

- Parses unstructured `access.log` lines into structured fields (IP, timestamp, method, URL, status, bytes).
- Groups requests into user sessions using a 30‑minute inactivity rule.
- Engineers session‑level behavior features (requests, duration, error rate, speed, time of day).
- Trains an **IsolationForest** model to learn normal traffic patterns and flag unusual sessions.
- Serves an interactive **Streamlit app** that lets you upload a log file and visualize anomalies.

It is designed to be:

- **Realistic**: based on real‑world log formats and session behavior.
- **Reproducible**: training code is in a Kaggle notebook; inference is in Python modules.
- **Deployable**: can be run locally or on Streamlit Community Cloud with no paid services.

---

## 2. Project Structure

```text
web-log-anomaly-detector/
├── models/
│   ├── session_anomaly_iforest.pkl        # trained IsolationForest + scaler pipeline
│   └── session_feature_cols.pkl           # list of feature column names used by the model
│
├── notebooks/
│   └── 01_kaggle_log_anomaly_training.ipynb  # training + analysis notebook (Kaggle)
│
├── log_processing.py                      # parsing & sessionization utilities
├── streamlit_app.py                       # Streamlit dashboard for anomaly exploration
├── requirements.txt                       # Python dependencies
└── README.md
