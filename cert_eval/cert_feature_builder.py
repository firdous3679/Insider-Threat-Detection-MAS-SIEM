"""Build a reproducible user-day feature matrix for CERT evaluation.

CERT releases differ in column names and even in which columns are
present. r4.2's ``file.csv``, for example, has no ``activity`` column,
which is why this module never relies on ``df.get(col, default_scalar)``
(that returns the default *as-is* if the column is missing — and a
scalar default has no ``.astype``). Instead we use ``_col(df, name)``,
which always returns a same-length Series even when the column is
absent.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

# Keyword lists for sensitive-file and suspicious-URL detection.
SENSITIVE_FILE_TERMS = [
    "confidential", "secret", "backup", "password", "config",
    "source", "payroll", "finance", "proprietary",
]
SUSPICIOUS_HTTP_TERMS = [
    "dropbox", "drive", "pastebin", "github",
    "upload", "filetransfer", "career", "job",
]


def _col(df: pd.DataFrame, name: str, fill: str = "") -> pd.Series:
    """Return ``df[name]`` as a same-length Series, or a fill Series if missing.

    This is the workhorse that lets the rest of the function handle
    any CERT release without crashing on missing columns.
    """
    if name in df.columns:
        return df[name]
    return pd.Series([fill] * len(df), index=df.index)


def _flag_col(df: pd.DataFrame, name: str) -> pd.Series:
    """Return a boolean Series for ``df[name]==1`` (False if column absent)."""
    if name in df.columns:
        return df[name] == 1
    return pd.Series(False, index=df.index)


def _safe_contains(series: pd.Series, terms) -> pd.Series:
    """Vectorized substring match against a list of keywords."""
    return series.fillna("").astype(str).str.lower().str.contains("|".join(terms), regex=True)


def build_user_day_features(bundle, output_dir: str | Path) -> pd.DataFrame:
    # ------------------------------------------------------------------
    # 1. Build the (user, day) index from every modality that has rows.
    # ------------------------------------------------------------------
    users_days = []
    for name in ["logon", "device", "file", "http", "email"]:
        df = getattr(bundle, name)
        if not df.empty and "user" in df.columns and "day" in df.columns:
            users_days.append(df[["user", "day"]].dropna())
    base = (
        pd.concat(users_days).drop_duplicates().reset_index(drop=True)
        if users_days
        else pd.DataFrame(columns=["user", "day"])
    )

    def add_count(df, key, cond=None):
        """Per-(user, day) row counter with optional boolean mask."""
        if df.empty or "user" not in df.columns or "day" not in df.columns:
            return pd.DataFrame(columns=["user", "day", key])
        d = df[cond] if cond is not None else df
        return d.groupby(["user", "day"]).size().reset_index(name=key)

    feats = base.copy()

    # ------------------------------------------------------------------
    # 2. Authentication features.
    # ------------------------------------------------------------------
    logon = bundle.logon
    feats = feats.merge(add_count(logon, "logon_count"), on=["user", "day"], how="left")
    feats = feats.merge(
        add_count(logon, "after_hours_logon_count", _flag_col(logon, "after_hours")),
        on=["user", "day"], how="left",
    )
    feats = feats.merge(
        add_count(logon, "weekend_logon_count", _flag_col(logon, "weekend")),
        on=["user", "day"], how="left",
    )
    if not logon.empty and "pc" in logon.columns:
        upc = logon.groupby(["user", "day"])["pc"].nunique().reset_index(name="unique_pc_count")
        feats = feats.merge(upc, on=["user", "day"], how="left")

    # ------------------------------------------------------------------
    # 3. Device (USB) features.
    # ------------------------------------------------------------------
    device = bundle.device
    feats = feats.merge(add_count(device, "device_connect_count"), on=["user", "day"], how="left")
    feats = feats.merge(
        add_count(device, "after_hours_device_count", _flag_col(device, "after_hours")),
        on=["user", "day"], how="left",
    )

    # ------------------------------------------------------------------
    # 4. File features. r4.2 has no `activity` column on file.csv, so
    #    we tolerate its absence: when missing, every file event counts
    #    as an access and `is_copy` stays False.
    # ------------------------------------------------------------------
    # All these blocks need to handle empty DataFrames cleanly. We
    # ensure the boolean condition columns exist BEFORE calling
    # ``add_count`` so the cond is always a proper Series, never a
    # bare bool / scalar default.
    file_df = bundle.file.copy()
    if not file_df.empty:
        file_df["is_sensitive"] = _safe_contains(_col(file_df, "filename"), SENSITIVE_FILE_TERMS)
        activity = _col(file_df, "activity").astype(str).str.lower()
        file_df["is_copy"] = activity.str.contains("copy|write|usb", regex=True, na=False)
    else:
        # Empty frames still need the columns so the masks below are valid.
        file_df = file_df.assign(is_sensitive=pd.Series(dtype=bool), is_copy=pd.Series(dtype=bool))
    feats = feats.merge(add_count(file_df, "file_access_count"), on=["user", "day"], how="left")
    feats = feats.merge(
        add_count(file_df, "file_copy_count", file_df["is_copy"].astype(bool) if "is_copy" in file_df else None),
        on=["user", "day"], how="left",
    )
    feats = feats.merge(
        add_count(file_df, "sensitive_file_count", file_df["is_sensitive"].astype(bool) if "is_sensitive" in file_df else None),
        on=["user", "day"], how="left",
    )
    feats = feats.merge(
        add_count(file_df, "after_hours_file_count", _flag_col(file_df, "after_hours")),
        on=["user", "day"], how="left",
    )

    # ------------------------------------------------------------------
    # 5. HTTP features.
    # ------------------------------------------------------------------
    http = bundle.http.copy()
    if not http.empty:
        http["is_suspicious"] = _safe_contains(_col(http, "url"), SUSPICIOUS_HTTP_TERMS)
    else:
        http = http.assign(is_suspicious=pd.Series(dtype=bool))
    feats = feats.merge(add_count(http, "http_count"), on=["user", "day"], how="left")
    feats = feats.merge(
        add_count(http, "suspicious_http_count", http["is_suspicious"].astype(bool) if "is_suspicious" in http else None),
        on=["user", "day"], how="left",
    )
    feats = feats.merge(
        add_count(http, "after_hours_http_count", _flag_col(http, "after_hours")),
        on=["user", "day"], how="left",
    )

    # ------------------------------------------------------------------
    # 6. Email features. We treat any address NOT on the synthetic
    #    CERT company domain (`dtaa.com`) as external.
    # ------------------------------------------------------------------
    email = bundle.email.copy()
    if not email.empty:
        to_series = _col(email, "to").fillna("").astype(str)
        email["external"] = to_series.str.contains("@") & ~to_series.str.contains(
            r"@dtaa\.com", case=False, regex=True
        )
        attachments = _col(email, "attachments").astype(str)
        email["has_attachment"] = attachments.str.lower().ne("")
        email["recipient_count"] = to_series.apply(
            lambda s: len([x for x in str(s).split(";") if x.strip()])
        )
        if "to" not in email.columns:
            email["to"] = to_series
    else:
        email = email.assign(
            external=pd.Series(dtype=bool),
            has_attachment=pd.Series(dtype=bool),
            recipient_count=pd.Series(dtype=int),
        )
    feats = feats.merge(add_count(email, "email_sent_count"), on=["user", "day"], how="left")
    feats = feats.merge(
        add_count(email, "external_email_count", email["external"].astype(bool) if "external" in email else None),
        on=["user", "day"], how="left",
    )
    feats = feats.merge(
        add_count(email, "attachment_email_count", email["has_attachment"].astype(bool) if "has_attachment" in email else None),
        on=["user", "day"], how="left",
    )
    if not email.empty and "to" in email.columns:
        uniq = email.groupby(["user", "day"])["to"].nunique().reset_index(name="unique_recipient_count")
        tot = email.groupby(["user", "day"])["recipient_count"].sum().reset_index(name="total_recipient_count")
        feats = feats.merge(uniq, on=["user", "day"], how="left").merge(tot, on=["user", "day"], how="left")

    # ------------------------------------------------------------------
    # 7. LDAP-driven peer deviation. Skip cleanly when LDAP is missing
    #    or doesn't have the columns we need.
    # ------------------------------------------------------------------
    ldap = bundle.ldap
    if (
        not ldap.empty
        and "user" in ldap.columns
        and "role" in ldap.columns
        and "file_access_count" in feats.columns
    ):
        feats = feats.merge(ldap[["user", "role"]].drop_duplicates(), on="user", how="left")
        role_mean = feats.groupby("role")["file_access_count"].transform("mean")
        role_std = feats.groupby("role")["file_access_count"].transform("std").replace(0, 1)
        feats["role_peer_deviation_score"] = (
            (feats["file_access_count"] - role_mean) / role_std
        ).fillna(0)
    else:
        feats["role_peer_deviation_score"] = 0.0

    # ------------------------------------------------------------------
    # 8. Persist features. Any column the merges didn't populate (because
    #    a modality was missing entirely) is filled with 0 so downstream
    #    code can treat the matrix as fully numeric.
    # ------------------------------------------------------------------
    expected_cols = [
        "logon_count", "after_hours_logon_count", "weekend_logon_count", "unique_pc_count",
        "device_connect_count", "after_hours_device_count",
        "file_access_count", "file_copy_count", "sensitive_file_count", "after_hours_file_count",
        "http_count", "suspicious_http_count", "after_hours_http_count",
        "email_sent_count", "external_email_count", "attachment_email_count",
        "unique_recipient_count", "total_recipient_count",
        "role_peer_deviation_score",
    ]
    for c in expected_cols:
        if c not in feats.columns:
            feats[c] = 0
    feats = feats.fillna(0)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    feats.to_csv(out / "cert_user_day_features.csv", index=False)
    return feats
