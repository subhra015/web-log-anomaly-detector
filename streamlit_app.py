import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*InconsistentVersionWarning.*"
)


from log_processing import parse_log_file, build_sessions

plt.style.use("seaborn-v0_8")
sns.set_palette("viridis")

st.set_page_config(page_title="Log Anomaly Dashboard", layout="wide")
st.title("Web Log Anomaly Detection for Business Intelligence")

model = joblib.load("models/session_anomaly_iforest.pkl")
feature_cols = joblib.load("models/session_feature_cols.pkl")

st.sidebar.header("Upload & Settings")
uploaded = st.sidebar.file_uploader("Upload an Apache access.log file", type=["log", "txt"])
session_minutes = st.sidebar.slider("Session timeout (minutes)", 5, 60, 30)
max_lines = st.sidebar.number_input(
    "Max lines to parse (for speed)", min_value=1000, max_value=2_000_000,
    value=200_000, step=50_000
)

if uploaded is None:
    st.info("Upload a log file (.log or .txt) in the sidebar to start analysis.")
else:
    tmp_path = "uploaded_access.log"
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    df_raw = parse_log_file(tmp_path, max_lines=max_lines)
    df_raw, sessions = build_sessions(df_raw, session_minutes=session_minutes)

    if sessions.empty:
        st.warning("No sessions detected in this log. Check file content or parsing pattern.")
    else:
        X_new = sessions[feature_cols].fillna(0.0)

        scores = model.decision_function(X_new)
        flags = model.predict(X_new)

        sessions["anomaly_score"] = scores
        sessions["is_anomalous"] = (flags == -1).astype(int)
        sessions["date"] = sessions["start_time"].dt.date

        total_sessions = sessions.shape[0]
        anomalous_sessions = int(sessions["is_anomalous"].sum())
        anomaly_rate = anomalous_sessions / total_sessions * 100 if total_sessions > 0 else 0.0

        c1, c2, c3 = st.columns(3)
        c1.metric("Total sessions", f"{total_sessions:,}")
        c2.metric("Anomalous sessions", f"{anomalous_sessions:,}")
        c3.metric("Anomaly rate (%)", f"{anomaly_rate:.2f}")

        daily = (
            sessions.groupby("date")["is_anomalous"]
            .mean()
            .mul(100)
            .rename("anomaly_rate_percent")
            .reset_index()
        )

        if not daily.empty:
            st.subheader("Daily anomalous session rate (%)")
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.lineplot(data=daily, x="date", y="anomaly_rate_percent", marker="o", ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

        st.subheader("Top anomalous sessions")
        st.dataframe(
            sessions[sessions["is_anomalous"] == 1]
            .sort_values("anomaly_score")
            .head(50)
        )

        st.subheader("Recent raw requests (sample)")
        st.dataframe(
            df_raw.sort_values("time", ascending=False)
            .head(100)[["time", "ip", "path", "status", "bytes"]]
        )
