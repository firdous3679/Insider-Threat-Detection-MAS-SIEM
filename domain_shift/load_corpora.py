#!/usr/bin/env python3
"""Load Phase 1 email corpora into a shared schema."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ATTACK_CATEGORIES = {
    "BMS_ACCESS_REQUEST",
    "PHYSICAL_SECURITY_SYSTEM_ACCESS",
    "NETWORK_CONFIGURATION_REQUEST",
    "VENDOR_REMOTE_ACCESS_REQUEST",
    "IOT_DEVICE_ACCESS",
    "IAM_PRIVILEGE_ESCALATION",
    "DATA_EXTRACTION_REQUEST",
    "LOGS_AND_MONITORING_REQUEST",
    "SECURITY_POLICY_BYPASS",
    "EMERGENCY_SYSTEM_TAMPERING",
}


@dataclass(frozen=True)
class CorpusBundle:
    enron: pd.DataFrame
    municipal: pd.DataFrame
    kurdi: pd.DataFrame


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required Phase 1 corpus is missing: {path}")


def first_existing(repo_root: Path, candidates: list[str]) -> Path:
    for candidate in candidates:
        path = repo_root / candidate
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find any candidate corpus path: "
        + ", ".join(str(repo_root / c) for c in candidates)
    )


def _combine_text(subject: pd.Series, body: pd.Series) -> pd.Series:
    return (
        subject.fillna("").astype(str).str.strip()
        + "\n\n"
        + body.fillna("").astype(str).str.strip()
    ).str.strip()


def load_enron_spam(path: Path) -> pd.DataFrame:
    _require(path)
    raw = pd.read_csv(path)
    required = {"Subject", "Message", "Spam/Ham"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Enron spam file missing columns: {sorted(missing)}")

    labels = raw["Spam/Ham"].fillna("").astype(str).str.strip().str.lower()
    text = _combine_text(raw["Subject"], raw["Message"])
    df = pd.DataFrame(
        {
            "dataset": "enron_spam",
            "text": text,
            "label": labels.eq("spam").astype(int),
            "category": labels,
            "subcategory": labels,
            "group": text.str.lower().str.replace(r"\s+", " ", regex=True),
        }
    )
    # Keep duplicate rows because the public Enron spam/ham CSV contains many
    # repeated spam examples; grouped CV below prevents those repeats from
    # leaking across folds while preserving the published corpus size.
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    return df


def load_municipal(path: Path) -> pd.DataFrame:
    _require(path)
    raw = pd.read_csv(path)
    required = {"subject", "body", "is_phishing", "category", "subcategory"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Municipal corpus missing columns: {sorted(missing)}")

    text = _combine_text(raw["subject"], raw["body"])
    return pd.DataFrame(
        {
            "dataset": "municipal_synthetic",
            "text": text,
            "label": raw["is_phishing"].astype(int),
            "category": raw["category"].fillna("").astype(str),
            "subcategory": raw["subcategory"].fillna("").astype(str),
            "group": text.str.lower().str.replace(r"\s+", " ", regex=True),
        }
    ).reset_index(drop=True)


def load_municipal_v2(path: Path) -> pd.DataFrame:
    _require(path)
    raw = pd.read_csv(path)
    required = {"subject", "body", "is_phishing", "category", "subcategory"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Municipal V2 corpus missing columns: {sorted(missing)}")

    text = _combine_text(raw["subject"], raw["body"])
    df = pd.DataFrame(
        {
            "dataset": "municipal_synthetic_v2",
            "text": text,
            "label": raw["is_phishing"].astype(int),
            "category": raw["category"].fillna("").astype(str),
            "subcategory": raw["subcategory"].fillna("").astype(str),
            "group": text.str.lower().str.replace(r"\s+", " ", regex=True),
            "template_id": raw.get("template_id", pd.Series([""] * len(raw))).fillna("").astype(str),
            "template_family": raw.get("template_family", pd.Series([""] * len(raw))).fillna("").astype(str),
            "is_hard_negative": raw.get("is_hard_negative", pd.Series([0] * len(raw))).fillna(0).astype(int),
            "approval_context": raw.get("approval_context", pd.Series(["unknown"] * len(raw))).fillna("unknown").astype(str),
            "sender_domain_type": raw.get("sender_domain_type", pd.Series(["unknown"] * len(raw))).fillna("unknown").astype(str),
            "expected_detection_signal": raw.get("expected_detection_signal", pd.Series(["none"] * len(raw))).fillna("none").astype(str),
            "subject": raw["subject"].fillna("").astype(str),
            "body": raw["body"].fillna("").astype(str),
            "has_attachment": raw.get("has_attachment", pd.Series([pd.NA] * len(raw))),
            "has_external_link": raw.get("has_external_link", pd.Series([pd.NA] * len(raw))),
            "body_word_count": raw.get("body_word_count", pd.Series([pd.NA] * len(raw))),
        }
    )
    df.attrs["missing_optional_columns"] = sorted(
        {
            "template_id",
            "template_family",
            "is_hard_negative",
            "approval_context",
            "sender_domain_type",
            "expected_detection_signal",
            "has_attachment",
            "has_external_link",
            "body_word_count",
        }.difference(raw.columns)
    )
    return df.reset_index(drop=True)


def _extract_json_arrays(text: str) -> list[list[dict]]:
    arrays: list[list[dict]] = []
    decoder = json.JSONDecoder()
    pos = 0
    while True:
        start = text.find("[", pos)
        if start == -1:
            break
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            pos = start + 1
            continue
        if isinstance(value, list):
            arrays.append(value)
        pos = start + end
    return arrays


def load_kurdi(path: Path) -> pd.DataFrame:
    _require(path)
    arrays = _extract_json_arrays(path.read_text(encoding="utf-8"))
    if len(arrays) < 2:
        raise ValueError(
            "Kurdi corpus should contain two JSON arrays: suspicious emails and normal emails."
        )

    rows = []
    for idx, items in enumerate(arrays[:2]):
        for item in items:
            label_name = str(item.get("label", "")).strip()
            rows.append(
                {
                    "dataset": "kurdi_smart_building",
                    "text": str(item.get("email_text", "")).strip(),
                    "label": 1 if idx == 0 or label_name in ATTACK_CATEGORIES else 0,
                    "category": label_name,
                    "subcategory": label_name,
                }
            )
    df = pd.DataFrame(rows)
    df["group"] = df["text"].str.lower().str.replace(r"\s+", " ", regex=True)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    return df


def load_all(
    repo_root: Path,
    enron_path: str = "enron_spam_data.csv",
    municipal_path: str = "data/municipal_facilities_emails.csv",
    kurdi_path: str = "data/Kurdi_cyber_insider_smart_building_muncipality.json",
) -> CorpusBundle:
    return CorpusBundle(
        enron=load_enron_spam(repo_root / enron_path),
        municipal=load_municipal(repo_root / municipal_path),
        kurdi=load_kurdi(repo_root / kurdi_path),
    )


def describe(bundle: CorpusBundle) -> pd.DataFrame:
    rows = []
    for name, df in [
        ("enron_spam", bundle.enron),
        ("municipal_synthetic", bundle.municipal),
        ("kurdi_smart_building", bundle.kurdi),
    ]:
        rows.append(
            {
                "dataset": name,
                "n": len(df),
                "positive": int(df["label"].sum()),
                "negative": int((df["label"] == 0).sum()),
                "positive_rate": float(df["label"].mean()),
                "avg_words": float(df["text"].map(lambda s: len(re.findall(r"\b\w+\b", s))).mean()),
            }
        )
    return pd.DataFrame(rows)
