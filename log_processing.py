# log_processing.py

import re
import pandas as pd
import numpy as np

LOG_PATTERN = (
    r'(?P<ip>\S+)\s+'
    r'(?P<ident>\S+)\s+'
    r'(?P<user>\S+)\s+'
    r'\[(?P<time>[^\]]+)\]\s+'
    r'"(?P<method>\S+)\s+'
    r'(?P<url>\S+)\s+'
    r'(?P<protocol>[^"]+)"\s+'
    r'(?P<status>\d{3})\s+'
    r'(?P<bytes>\S+)'
)

def parse_log_file(path: str, max_lines: int | None = None) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            m = re.match(LOG_PATTERN, line)
            if m:
                rows.append(m.groupdict())

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["status"] = pd.to_numeric(df["status"], errors="coerce").fillna(0).astype(int)
    df["bytes"] = pd.to_numeric(df["bytes"].replace("-", 0), errors="coerce").fillna(0).astype(int)

    df["time"] = pd.to_datetime(
        df["time"],
        format="%d/%b/%Y:%H:%M:%S %z",
        errors="coerce",
    )
    df = df.dropna(subset=["time"])

    df["date"] = df["time"].dt.date
    df["hour"] = df["time"].dt.hour
    df["dayofweek"] = df["time"].dt.day_name()
    df["path"] = df["url"].str.split("?", n=1).str[0]
    df["user_id"] = df["ip"]
    df["is_error"] = (df["status"] >= 400).astype(int)

    return df

def build_sessions(df: pd.DataFrame, session_minutes: int = 30):
    if df.empty:
        return df, pd.DataFrame()

    df = df.sort_values(["user_id", "time"])
    session_timeout = pd.Timedelta(minutes=session_minutes)

    df["prev_time"] = df.groupby("user_id")["time"].shift()
    df["time_diff"] = df["time"] - df["prev_time"]
    df["new_session"] = (df["time_diff"] > session_timeout) | df["time_diff"].isna()
    df["session_id"] = df.groupby("user_id")["new_session"].cumsum()

    sessions = (
        df.groupby(["user_id", "session_id"])
        .agg(
            start_time=("time", "min"),
            end_time=("time", "max"),
            num_requests=("path", "size"),
            unique_paths=("path", "nunique"),
            num_errors=("is_error", "sum"),
        )
        .reset_index()
    )

    sessions["duration_sec"] = (sessions["end_time"] - sessions["start_time"]).dt.total_seconds()
    sessions["error_rate"] = sessions["num_errors"] / sessions["num_requests"].replace(0, 1)
    sessions["requests_per_sec"] = sessions["num_requests"] / sessions["duration_sec"].replace(0, 1)
    sessions["start_hour"] = sessions["start_time"].dt.hour
    sessions["start_dayofweek"] = sessions["start_time"].dt.weekday

    for col in ["num_requests", "unique_paths", "duration_sec", "requests_per_sec"]:
        sessions[col] = sessions[col].clip(upper=sessions[col].quantile(0.99))

    return df, sessions
