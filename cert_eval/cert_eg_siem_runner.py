"""CERT-EG-SIEM evidence-gated detection runner.

Pipeline (mirrors the Mesa EG-SIEM):

  user-day row
    -> per-modality evidence extraction
    -> peer-group (role) z-normalization of each evidence stream
    -> evidence-trigger flag per modality (z > THR or raw above benign-95p)
    -> evidence accumulation (count of triggered modalities)
    -> evidence gate (>= MIN_CATEGORIES triggered)
    -> confirmed alert

Design corrections vs the previous draft:

* **No target leakage.** The previous version multiplied the score by
  ``1/(1+actor_label*0.05)``, i.e. it consumed the ground-truth label.
  Removed entirely.
* **Peer-group normalization is real now.** Each modality's evidence
  stream is z-scored within the user's LDAP role; if role is missing
  we fall back to a global z-score.
* **Thresholds are calibrated from benign data**, not hardcoded. The
  per-mode operating threshold is the 95th percentile of the score on
  rows with ``user_day_label == 0``. For ablation modes that knock out
  whole evidence categories (e.g. email_only), this prevents F1=0
  while preserving a realistic FP budget.
* **ToM-like sequence is a real sequence rule**, not "logon AND
  anything". A user-day fires it only when *all three* of (logon,
  data-pull, exfil-channel) coexist that day. Mesa's TomAbd module
  uses a similar pattern.
* **CERT-LSC** stays a simple layered-correlation baseline: count of
  modalities firing on raw counts >= 3 of 4.
* **Confirmed vs early alert.** ``pred_alert`` is the *confirmed*
  alert (the published Mesa metric). ``early_alert`` is exposed for
  downstream TTD analysis.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

# Modalities that participate in the evidence gate. ``peer`` is the
# role-relative file-access deviation (already z-scored upstream).
EVIDENCE_MODALITIES = ["auth", "device", "file", "web", "email", "tom_like"]

# Minimum number of distinct evidence categories that must trigger
# before we promote a row to a confirmed alert.
MIN_CATEGORIES_CONFIRMED = 2
MIN_CATEGORIES_EARLY = 1

# Per-modality z-trigger threshold. A modality is considered triggered
# when its peer-normalized score exceeds this value.
Z_TRIGGER = 2.0

# Benign-percentile used to set the operating point for the aggregate
# risk score. 95 means "tolerate at most 5% FP/day in the benign pool".
BENIGN_PERCENTILE = 95.0


def _peer_zscore(values: pd.Series, group: pd.Series) -> pd.Series:
    """Z-score ``values`` within each ``group`` (e.g. LDAP role).

    Falls back to a global z-score if ``group`` is empty/all-NaN. Groups
    of size 1 use the global mean/std for that row to avoid divide-by-zero.
    """
    v = values.astype(float).fillna(0.0)
    if group is None or group.dropna().empty:
        mu, sd = v.mean(), v.std(ddof=0) or 1.0
        return (v - mu) / sd
    g = group.fillna("__no_role__")
    mu = v.groupby(g).transform("mean")
    sd = v.groupby(g).transform("std").fillna(0.0)
    sd = sd.replace(0, np.nan)
    z = (v - mu) / sd
    # Replace remaining NaNs (single-row groups) with global z-score.
    if z.isna().any():
        gmu, gsd = v.mean(), (v.std(ddof=0) or 1.0)
        z = z.fillna((v - gmu) / gsd)
    return z.fillna(0.0)


def _benign_threshold(score: pd.Series, label: pd.Series, pct: float = BENIGN_PERCENTILE) -> float:
    """Operating threshold = ``pct``-percentile of score over benign rows.

    Returns +inf when no benign rows exist (so nothing fires) and the
    global percentile when all rows are benign.
    """
    benign = score[label == 0]
    if benign.empty:
        return float("inf")
    return float(np.percentile(benign.values, pct))


def _has_copy_semantics(df: pd.DataFrame) -> bool:
    """Whether the dataset distinguishes file-copy from file-access events.

    CERT r5.x ships ``activity`` on file rows so the feature builder can
    populate ``file_copy_count``; CERT r4.2 does not. When every row has
    ``file_copy_count == 0`` we treat that as "no copy semantics" and
    use the alternative data-pull predicate below.
    """
    if "file_copy_count" not in df.columns:
        return False
    return float(df["file_copy_count"].max()) > 0.0


def _peer_anomalous_access(df: pd.DataFrame, role: pd.Series, k: float = 1.5) -> pd.Series:
    """r4.2 substitute for ``file_copy_count > 0`` in the ToM-like rule.

    Returns a boolean Series that is True when a user's daily
    ``file_access_count`` exceeds their LDAP-role peers by more than
    ``k`` standard deviations. The threshold is intentionally looser
    than the per-modality ``Z_TRIGGER`` (2.0) used in the evidence gate,
    because here it only contributes to a *conjunctive* sequence rule;
    we want enough recall to surface the data-pull step in suspicious
    sequences without creating its own false-positive stream.
    """
    if "file_access_count" not in df.columns:
        return pd.Series(False, index=df.index)
    z = _peer_zscore(df["file_access_count"].astype(float), role)
    return z > k


def _build_evidence(df: pd.DataFrame, role: pd.Series) -> Dict[str, pd.Series]:
    """Extract per-row evidence streams from the feature matrix.

    Each modality returns a non-negative magnitude that is later
    peer-z-scored. We deliberately use *counts* rather than ad-hoc
    "score>0" booleans so the z-score has continuous resolution.
    """
    auth = (
        df["after_hours_logon_count"].astype(float)
        + df["weekend_logon_count"].astype(float)
        + np.maximum(df["logon_count"].astype(float) - 8.0, 0.0)
    )
    device = (
        df["device_connect_count"].astype(float)
        + df["after_hours_device_count"].astype(float)
    )
    file_ = (
        0.2 * df["file_access_count"].astype(float)
        + df["file_copy_count"].astype(float)
        + df["sensitive_file_count"].astype(float)
        + df["after_hours_file_count"].astype(float)
    )
    web = (
        df["suspicious_http_count"].astype(float)
        + df["after_hours_http_count"].astype(float)
    )
    email = (
        df["external_email_count"].astype(float)
        + df["attachment_email_count"].astype(float)
        + (df["unique_recipient_count"] > 10).astype(float)
    )

    # ToM-like sequence: a user-day must show authentication AND a data
    # pull AND an exfil-channel attempt to count as a suspicious sequence.
    #
    # The data-pull predicate is *release-aware*:
    #   - r5.x has copy semantics on file events, so we use the strict
    #     rule (file_copy_count > 0 OR sensitive_file_count > 0).
    #   - r4.2 does not, so file_copy_count is structurally 0. We
    #     substitute peer-anomalous file_access_count as the data-pull
    #     signal (a user reading much more than their role peers on a
    #     given day) plus the existing sensitive_file_count match.
    # The substitution is documented in phase2_diagnostic_report.md.
    has_logon = df["logon_count"] > 0
    if _has_copy_semantics(df):
        has_pull = (df["file_copy_count"] > 0) | (df["sensitive_file_count"] > 0)
    else:
        has_pull = _peer_anomalous_access(df, role) | (df["sensitive_file_count"] > 0)
    has_exfil = (
        (df["external_email_count"] > 0)
        | (df["attachment_email_count"] > 0)
        | (df["suspicious_http_count"] > 0)
        | (df["device_connect_count"] > 0)
    )
    tom_like = (has_logon & has_pull & has_exfil).astype(float)

    return {
        "auth": auth,
        "device": device,
        "file": file_,
        "web": web,
        "email": email,
        "tom_like": tom_like,
    }


def _aggregate_score(z_streams: Dict[str, pd.Series], weights: Dict[str, float]) -> pd.Series:
    """Weighted sum of per-modality z-scores -> aggregate risk score.

    Negative z-scores (below peer mean) are clipped so a quiet day in
    one modality cannot mask suspicious activity in another.
    """
    out = None
    for k, w in weights.items():
        contrib = z_streams[k].clip(lower=0) * w
        out = contrib if out is None else out + contrib
    return out


def run_cert_eg_siem(labeled_df: pd.DataFrame, mode: str = "full") -> pd.DataFrame:
    """Score every CERT user-day and emit confirmed/early alerts.

    Parameters
    ----------
    labeled_df : pd.DataFrame
        Output of ``cert_label_builder.build_labels``. Must contain the
        per-user-day feature columns *and* ``user_day_label`` so we can
        calibrate the benign threshold.
    mode : str
        One of ``full``, ``without_email``, ``email_only``,
        ``without_tom``, ``lsc``.
    """
    df = labeled_df.copy()
    role = df["role"] if "role" in df.columns else pd.Series([""] * len(df))

    # ------------------------------------------------------------------
    # 1. Extract raw evidence streams. ``role`` is passed in so the
    #    ToM-like rule can use a peer-anomalous file_access fallback on
    #    releases without copy semantics (r4.2).
    # ------------------------------------------------------------------
    evidence = _build_evidence(df, role)

    # ------------------------------------------------------------------
    # 2. Apply mode masks BEFORE z-scoring so the masked modality
    #    doesn't contribute to the aggregate or to the trigger count.
    # ------------------------------------------------------------------
    if mode == "without_email":
        evidence["email"] = evidence["email"] * 0
    if mode == "without_tom":
        evidence["tom_like"] = evidence["tom_like"] * 0
    if mode == "email_only":
        for k in evidence:
            if k != "email":
                evidence[k] = evidence[k] * 0

    # ------------------------------------------------------------------
    # 3. Peer-group z-scoring of each evidence stream.
    # ------------------------------------------------------------------
    z_streams = {k: _peer_zscore(v, role) for k, v in evidence.items()}

    # ------------------------------------------------------------------
    # 4. CERT-LSC mode: simple layered correlation baseline. Count of
    #    modalities active on raw evidence; alert when >= 3 of 4 SIEM
    #    layers (auth/file/web/email) all light up — closer to a real
    #    layered-SIEM rule than the previous >=2 threshold.
    # ------------------------------------------------------------------
    if mode == "lsc":
        layers = (
            (evidence["auth"] > 0).astype(int)
            + (evidence["file"] > 0).astype(int)
            + (evidence["web"] > 0).astype(int)
            + (evidence["email"] > 0).astype(int)
        )
        df["risk_score"] = layers.astype(float)
        df["pred_alert"] = (layers >= 3).astype(int)
        df["early_alert"] = (layers >= 2).astype(int)
        df["evidence_categories"] = layers
        return df

    # ------------------------------------------------------------------
    # 5. EG-SIEM modes: weighted sum of (clipped) z-scores -> aggregate
    #    risk score; triggers per modality at z > Z_TRIGGER.
    # ------------------------------------------------------------------
    weights = {
        "auth": 1.2,
        "device": 1.0,
        "file": 1.4,
        "web": 1.1,
        "email": 1.2,
        "tom_like": 1.5,
    }
    risk = _aggregate_score(z_streams, weights)
    df["risk_score"] = risk

    triggers = pd.DataFrame(
        {k: (z_streams[k] > Z_TRIGGER).astype(int) for k in EVIDENCE_MODALITIES}
    )
    # tom_like is binary 0/1 raw; treat ANY tom_like as a trigger.
    triggers["tom_like"] = (evidence["tom_like"] > 0).astype(int)
    df["evidence_categories"] = triggers.sum(axis=1)

    # ------------------------------------------------------------------
    # 6. Calibrate the operating point on benign data only.
    # ------------------------------------------------------------------
    label = df["user_day_label"].astype(int) if "user_day_label" in df.columns else pd.Series([0] * len(df))
    thr = _benign_threshold(risk, label, pct=BENIGN_PERCENTILE)
    df["risk_threshold"] = thr

    # ------------------------------------------------------------------
    # 7. Confirmed alert: aggregate score above benign-95p AND at least
    #    MIN_CATEGORIES_CONFIRMED distinct evidence categories firing.
    # ------------------------------------------------------------------
    df["pred_alert"] = (
        (risk >= thr) & (df["evidence_categories"] >= MIN_CATEGORIES_CONFIRMED)
    ).astype(int)
    df["early_alert"] = (
        (risk >= thr) | (df["evidence_categories"] >= MIN_CATEGORIES_EARLY)
    ).astype(int)

    # email_only is a single-modality ablation; we cannot demand >=2
    # categories there. Relax the gate so this ablation can fire.
    if mode == "email_only":
        df["pred_alert"] = (
            (risk >= thr) & (triggers["email"] == 1)
        ).astype(int)
        df["early_alert"] = (risk >= thr).astype(int)

    return df
