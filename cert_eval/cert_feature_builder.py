"""Build reproducible user-day feature matrix for CERT evaluation."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

SENSITIVE_FILE_TERMS = ["confidential","secret","backup","password","config","source","payroll","finance","proprietary"]
SUSPICIOUS_HTTP_TERMS = ["dropbox","drive","pastebin","github","upload","filetransfer","career","job"]


def _safe_contains(series: pd.Series, terms):
    return series.fillna("").str.lower().str.contains("|".join(terms), regex=True)


def build_user_day_features(bundle, output_dir: str | Path) -> pd.DataFrame:
    users_days = []
    for name in ["logon", "device", "file", "http", "email"]:
        df = getattr(bundle, name)
        if not df.empty and "user" in df.columns and "day" in df.columns:
            users_days.append(df[["user", "day"]].dropna())
    base = pd.concat(users_days).drop_duplicates().reset_index(drop=True) if users_days else pd.DataFrame(columns=["user","day"])

    def add_count(df, key, cond=None):
        if df.empty or "user" not in df or "day" not in df:
            return pd.DataFrame(columns=["user","day",key])
        d = df[cond] if cond is not None else df
        return d.groupby(["user","day"]).size().reset_index(name=key)

    feats = base.copy()
    logon = bundle.logon
    feats = feats.merge(add_count(logon, "logon_count"), on=["user","day"], how="left")
    feats = feats.merge(add_count(logon, "after_hours_logon_count", logon.get("after_hours",0)==1), on=["user","day"], how="left")
    feats = feats.merge(add_count(logon, "weekend_logon_count", logon.get("weekend",0)==1), on=["user","day"], how="left")
    if not logon.empty and "pc" in logon.columns:
        upc = logon.groupby(["user","day"])["pc"].nunique().reset_index(name="unique_pc_count")
        feats = feats.merge(upc, on=["user","day"], how="left")

    device = bundle.device
    feats = feats.merge(add_count(device, "device_connect_count"), on=["user","day"], how="left")
    feats = feats.merge(add_count(device, "after_hours_device_count", device.get("after_hours",0)==1), on=["user","day"], how="left")

    file_df = bundle.file.copy()
    if not file_df.empty:
        file_df["is_sensitive"] = _safe_contains(file_df.get("filename", pd.Series(dtype=str)), SENSITIVE_FILE_TERMS)
        file_df["is_copy"] = file_df.get("activity", "").astype(str).str.lower().str.contains("copy|write|usb")
    feats = feats.merge(add_count(file_df, "file_access_count"), on=["user","day"], how="left")
    feats = feats.merge(add_count(file_df, "file_copy_count", file_df.get("is_copy", False)==True), on=["user","day"], how="left")
    feats = feats.merge(add_count(file_df, "sensitive_file_count", file_df.get("is_sensitive", False)==True), on=["user","day"], how="left")
    feats = feats.merge(add_count(file_df, "after_hours_file_count", file_df.get("after_hours",0)==1), on=["user","day"], how="left")

    http = bundle.http.copy()
    if not http.empty:
        http["is_suspicious"] = _safe_contains(http.get("url", pd.Series(dtype=str)), SUSPICIOUS_HTTP_TERMS)
    feats = feats.merge(add_count(http, "http_count"), on=["user","day"], how="left")
    feats = feats.merge(add_count(http, "suspicious_http_count", http.get("is_suspicious", False)==True), on=["user","day"], how="left")
    feats = feats.merge(add_count(http, "after_hours_http_count", http.get("after_hours",0)==1), on=["user","day"], how="left")

    email = bundle.email.copy()
    if not email.empty:
        to_series = email.get("to", pd.Series(dtype=str)).fillna("")
        email["external"] = to_series.str.contains("@") & ~to_series.str.contains("@dtaa\\.com", case=False, regex=True)
        email["external"] = to_series.str.contains("@") & ~to_series.str.contains("@dtaa\.com", case=False, regex=True)
        email["has_attachment"] = email.get("attachments", "").astype(str).str.lower().ne("")
        email["recipient_count"] = to_series.apply(lambda s: len([x for x in str(s).split(";") if x.strip()]))
    feats = feats.merge(add_count(email, "email_sent_count"), on=["user","day"], how="left")
    feats = feats.merge(add_count(email, "external_email_count", email.get("external", False)==True), on=["user","day"], how="left")
    feats = feats.merge(add_count(email, "attachment_email_count", email.get("has_attachment", False)==True), on=["user","day"], how="left")
    if not email.empty:
        uniq = email.groupby(["user","day"])["to"].nunique().reset_index(name="unique_recipient_count")
        tot = email.groupby(["user","day"])["recipient_count"].sum().reset_index(name="total_recipient_count")
        feats = feats.merge(uniq, on=["user","day"], how="left").merge(tot, on=["user","day"], how="left")

    if not bundle.ldap.empty and "user" in bundle.ldap.columns and "role" in bundle.ldap.columns:
        feats = feats.merge(bundle.ldap[["user","role"]].drop_duplicates(), on="user", how="left")
        role_mean = feats.groupby("role")["file_access_count"].transform("mean")
        role_std = feats.groupby("role")["file_access_count"].transform("std").replace(0, 1)
        feats["role_peer_deviation_score"] = ((feats["file_access_count"] - role_mean) / role_std).fillna(0)
    else:
        feats["role_peer_deviation_score"] = 0.0

    feats = feats.fillna(0)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    feats.to_csv(out / "cert_user_day_features.csv", index=False)
    return feats
