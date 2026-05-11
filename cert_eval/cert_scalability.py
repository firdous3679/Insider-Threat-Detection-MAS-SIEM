"""Scalability sweep for the CERT pipeline.

The previous implementation timed only the EG-SIEM scoring step on an
already-aggregated user-day matrix, producing implausible 5.8M events/s
numbers. The real per-user cost is in feature engineering over the raw
modality CSV rows, so this version times the **end-to-end** path:

    bundle (per-modality raw rows)
      -> feature build  (cert_feature_builder)
      -> label build    (cert_label_builder)
      -> EG-SIEM full   (cert_eg_siem_runner)

User-size sweep defaults to 100 / 250 / 500 / 1000 (per the diagnostic
brief). ``events_processed`` reports the total raw events handled at
that scale, not the user-day row count, so events/sec is meaningful.
"""
from __future__ import annotations

import gc
import time
import tracemalloc
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import pandas as pd

from .cert_eg_siem_runner import run_cert_eg_siem
from .cert_feature_builder import build_user_day_features
from .cert_label_builder import build_labels


def _filter_bundle(bundle, keep_users: set):
    """Return a copy of ``bundle`` with each table filtered to ``keep_users``."""

    def _f(df):
        if df.empty or "user" not in df.columns:
            return df
        return df[df["user"].astype(str).isin(keep_users)].reset_index(drop=True)

    return replace(
        bundle,
        logon=_f(bundle.logon),
        device=_f(bundle.device),
        file=_f(bundle.file),
        http=_f(bundle.http),
        email=_f(bundle.email),
        ldap=_f(bundle.ldap),
        answers=bundle.answers,  # keep all answers so labels still map
    )


def _count_raw_events(bundle) -> int:
    """Sum of rows across every modality table — the true 'events processed'."""
    return sum(
        len(getattr(bundle, name))
        for name in ("logon", "device", "file", "http", "email")
    )


def run_scalability(
    bundle,
    output_dir: str | Path,
    user_sizes: Iterable[int] = (100, 250, 500, 1000),
) -> pd.DataFrame:
    """Sweep through ``user_sizes`` and time the end-to-end CERT pipeline.

    Parameters
    ----------
    bundle : CertDataBundle
        The already-loaded CERT tables (the loader bears the IO cost
        once; this sweep only varies how much of each table is fed
        into the pipeline).
    output_dir : str | Path
        Where to write per-scale intermediate CSVs (the labeled file is
        overwritten for each size in a temp subfolder).
    user_sizes : iterable of int
        User counts to evaluate. The largest size is capped at the
        actual user count present in the bundle.
    """
    # Stable user ordering: prefer LDAP order, fall back to logon order.
    if not bundle.ldap.empty and "user" in bundle.ldap.columns:
        users_sorted = (
            bundle.ldap["user"].dropna().astype(str).drop_duplicates().tolist()
        )
    else:
        users_sorted = (
            bundle.logon["user"].dropna().astype(str).drop_duplicates().tolist()
            if not bundle.logon.empty
            else []
        )
    total_users = len(users_sorted)

    rows = []
    out_root = Path(output_dir) / "_scalability_intermediate"
    out_root.mkdir(parents=True, exist_ok=True)

    # Dedupe sweep points after clamping: if total_users=252, requesting
    # both 500 and 1000 collapses to a single 252-user run.
    seen_sizes = set()

    for n in user_sizes:
        n_eff = min(int(n), total_users) if total_users else int(n)
        if n_eff in seen_sizes:
            continue
        seen_sizes.add(n_eff)
        keep = set(users_sorted[:n_eff])
        sub = _filter_bundle(bundle, keep)
        events_in = _count_raw_events(sub)

        gc.collect()
        tracemalloc.start()
        t0 = time.perf_counter()

        # Full pipeline at this scale.
        feats = build_user_day_features(sub, out_root / f"n{n_eff}")
        labeled = build_labels(feats, sub.answers, out_root / f"n{n_eff}")
        scored = run_cert_eg_siem(labeled, mode="full")

        runtime = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        rows.append(
            {
                "num_users": n_eff,
                "events_processed": int(events_in),
                "user_day_rows": int(len(scored)),
                "runtime_seconds": float(runtime),
                "peak_memory_mb": peak / (1024 * 1024),
                "events_per_second": (events_in / runtime) if runtime > 0 else 0.0,
            }
        )

        # Free per-scale frames before next iteration.
        del feats, labeled, scored, sub
        gc.collect()

    return pd.DataFrame(rows)
