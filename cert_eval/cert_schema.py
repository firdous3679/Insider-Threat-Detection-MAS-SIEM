"""Map CERT source tables into a normalized SIEM-like event schema."""
from __future__ import annotations

import pandas as pd


def _std(df: pd.DataFrame, event_type: str, source_file: str, resource_col: str = "resource", action_col: str = "activity") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["timestamp","user","event_type","source_file","pc","resource","action","metadata","label","day"])
    out = pd.DataFrame()
    out["timestamp"] = df.get("timestamp")
    out["user"] = df.get("user", df.get("employee_name", "unknown"))
    out["event_type"] = event_type
    out["source_file"] = source_file
    out["pc"] = df.get("pc", df.get("computer", ""))
    out["resource"] = df.get(resource_col, "")
    out["action"] = df.get(action_col, "")
    # Keep metadata lightweight: avoid row-wise dict expansion which is very memory-heavy
    # on CERT-scale data and can trigger OOM kills.
    out["metadata"] = ""
    out["metadata"] = df.apply(lambda r: r.to_dict(), axis=1)
    out["label"] = 0
    out["day"] = df.get("day")
    return out


def build_normalized_events(bundle, sort_events: bool = False) -> pd.DataFrame:
def build_normalized_events(bundle) -> pd.DataFrame:
    events = [
        _std(bundle.logon, "authentication", "logon.csv", action_col="activity"),
        _std(bundle.device, "device_activity", "device.csv", action_col="activity"),
        _std(bundle.file, "file_activity", "file.csv", resource_col="filename", action_col="activity"),
        _std(bundle.http, "web_activity", "http.csv", resource_col="url", action_col="activity"),
        _std(bundle.email, "email_activity", "email.csv", resource_col="to", action_col="activity"),
    ]
    all_events = pd.concat(events, ignore_index=True)
    # Sorting multi-million event rows can be expensive; make it optional.
    if sort_events and not all_events.empty:
        all_events = all_events.sort_values("timestamp", na_position="last").reset_index(drop=True)
    return all_events
    return all_events.sort_values("timestamp", na_position="last").reset_index(drop=True)
