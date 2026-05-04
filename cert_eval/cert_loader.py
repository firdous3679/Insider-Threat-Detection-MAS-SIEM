"""CERT r5.2 data loading utilities.

This module is intentionally defensive because CERT releases can vary in column names.
It loads files if available, standardizes timestamps/columns, and emits warnings (not crashes)
for missing files to keep experiments reproducible across environments.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class CertDataBundle:
    """Container for loaded CERT tables."""

    logon: pd.DataFrame
    device: pd.DataFrame
    file: pd.DataFrame
    http: pd.DataFrame
    email: pd.DataFrame
    ldap: pd.DataFrame
    answers: pd.DataFrame


def _warn(msg: str) -> None:
    print(f"[CERT-LOADER WARNING] {msg}")


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None


def _add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    ts_col = _find_column(df, ["timestamp", "date", "time", "datetime"])
    if ts_col is None:
        _warn("No timestamp-like column found; adding empty time fields.")
        df["timestamp"] = pd.NaT
    else:
        df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)

    df["day"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour.fillna(-1).astype(int)
    df["after_hours"] = df["hour"].apply(lambda h: int(h < 8 or h > 18 if h >= 0 else 0))
    df["weekend"] = df["timestamp"].dt.dayofweek.fillna(-1).astype(int).isin([5, 6]).astype(int)
    return df


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        _warn(f"Missing file: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    return _add_time_columns(df)


def _load_ldap(data_dir: Path) -> pd.DataFrame:
    ldap_dir = data_dir / "LDAP"
    if not ldap_dir.exists():
        _warn(f"Missing LDAP directory: {ldap_dir}")
        return pd.DataFrame()
    files = sorted(ldap_dir.glob("*.csv"))
    if not files:
        _warn("No LDAP csv files found.")
        return pd.DataFrame()
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df.columns = [c.lower() for c in df.columns]
            df["source_file"] = f.name
            frames.append(df)
        except Exception as exc:
            _warn(f"Failed loading LDAP file {f}: {exc}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_answers(data_dir: Path) -> pd.DataFrame:
    answers_dir = data_dir / "answers"
    if not answers_dir.exists():
        _warn(f"Missing answers directory: {answers_dir}")
        return pd.DataFrame()
    frames = []
    for f in sorted(answers_dir.glob("*.csv")):
        try:
            df = pd.read_csv(f)
            df.columns = [c.lower() for c in df.columns]
            df["source_file"] = f.name
            df = _add_time_columns(df)
            frames.append(df)
        except Exception as exc:
            _warn(f"Failed loading answer file {f}: {exc}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_cert_data(data_dir: str | Path) -> CertDataBundle:
    """Load CERT r5.2 tables with resilient schema handling."""
    root = Path(data_dir)
    return CertDataBundle(
        logon=_read_csv_if_exists(root / "logon.csv"),
        device=_read_csv_if_exists(root / "device.csv"),
        file=_read_csv_if_exists(root / "file.csv"),
        http=_read_csv_if_exists(root / "http.csv"),
        email=_read_csv_if_exists(root / "email.csv"),
        ldap=_load_ldap(root),
        answers=_load_answers(root),
    )
