"""Classification, alert, and time-to-detection metrics for CERT evaluation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)
from scipy import stats
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Classification + alert metrics
# ---------------------------------------------------------------------------


def compute_classification_metrics(y_true, y_pred, y_score=None):
    """Standard P/R/F1 + ROC-AUC + PR-AUC."""
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    out = {"precision": p, "recall": r, "f1": f}
    if y_score is not None and len(set(y_true)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
        out["pr_auc"] = average_precision_score(y_true, y_score)
    else:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan
    return out


def actor_level_metrics(df: pd.DataFrame, pred_col: str = "pred_alert") -> dict:
    """Actor-level precision/recall/F1.

    A user is "detected" if any of their user-day rows fired
    ``pred_col``. Ground truth is ``actor_label`` taken as max over
    rows for that user.
    """
    if "user" not in df.columns or "actor_label" not in df.columns:
        return {"actor_precision": np.nan, "actor_recall": np.nan, "actor_f1": np.nan}
    grp = df.groupby("user")
    actor_true = grp["actor_label"].max()
    actor_pred = grp[pred_col].max()
    tp = int(((actor_true == 1) & (actor_pred == 1)).sum())
    fp = int(((actor_true == 0) & (actor_pred == 1)).sum())
    fn = int(((actor_true == 1) & (actor_pred == 0)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"actor_precision": p, "actor_recall": r, "actor_f1": f1}


# ---------------------------------------------------------------------------
# Time to detection
# ---------------------------------------------------------------------------


def _to_dt(values: pd.Series) -> pd.Series:
    """Coerce a date-like column to ``pd.Timestamp`` (UTC)."""
    return pd.to_datetime(values, errors="coerce", utc=True)


def time_to_detection(
    pred_df: pd.DataFrame,
    pred_col: str = "pred_alert",
    day_col: str = "day",
    actor_col: str = "actor_label",
    user_day_label_col: str = "user_day_label",
) -> pd.DataFrame:
    """Per-actor TTD = first confirmed alert minus first malicious user-day.

    Returns one row per actor with ``ttd_hours`` (NaN when never
    detected). Aggregates can be computed on the result by the caller.

    The previous implementation hardcoded ``ttd_hours = 0`` for every
    method, which is what the diagnostic flagged. This function computes
    a real TTD from the per-row predictions and the answers-derived
    ``user_day_label``.
    """
    needed = {day_col, actor_col, user_day_label_col, pred_col, "user"}
    if not needed.issubset(pred_df.columns):
        return pd.DataFrame(columns=["user", "ttd_hours", "detected"])

    df = pred_df[pred_df[actor_col] == 1].copy()
    if df.empty:
        return pd.DataFrame(columns=["user", "ttd_hours", "detected"])
    df["_dt"] = _to_dt(df[day_col])

    rows = []
    for user, sub in df.groupby("user"):
        first_mal = sub.loc[sub[user_day_label_col] == 1, "_dt"].min()
        first_alert = sub.loc[sub[pred_col] == 1, "_dt"].min()
        if pd.isna(first_mal):
            # Actor with no labeled malicious day in this subset: skip.
            continue
        if pd.isna(first_alert) or first_alert < first_mal:
            # Never detected, OR alert fired before any labeled
            # malicious day (a false-positive that happens to be on the
            # actor; can't credit it as a detection).
            ttd = np.nan
            detected = 0
        else:
            ttd = (first_alert - first_mal).total_seconds() / 3600.0
            detected = 1
        rows.append({"user": user, "ttd_hours": ttd, "detected": detected})
    return pd.DataFrame(rows)


def summarize_ttd(ttd_df: pd.DataFrame) -> dict:
    """Summary stats over the per-actor TTD frame."""
    if ttd_df.empty:
        return {"ttd_mean_hours": np.nan, "ttd_median_hours": np.nan, "actors_detected": 0}
    detected = ttd_df.dropna(subset=["ttd_hours"])
    return {
        "ttd_mean_hours": float(detected["ttd_hours"].mean()) if not detected.empty else np.nan,
        "ttd_median_hours": float(detected["ttd_hours"].median()) if not detected.empty else np.nan,
        "actors_detected": int((ttd_df["detected"] == 1).sum()),
    }


# ---------------------------------------------------------------------------
# Statistical reporting
# ---------------------------------------------------------------------------


def summarize_statistics(values):
    a = np.array(values, dtype=float)
    if a.size == 0:
        return {"mean": np.nan, "std": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}
    mean = float(np.nanmean(a))
    std = float(np.nanstd(a, ddof=1)) if len(a) > 1 else 0.0
    ci = 1.96 * std / np.sqrt(len(a)) if len(a) > 1 else 0.0
    return {"mean": mean, "std": std, "ci95_low": mean - ci, "ci95_high": mean + ci}


def paired_significance(a, b, test: str = "wilcoxon"):
    """Paired comparison between two metric vectors (e.g. F1 across seeds)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if test == "t-test":
        stat, p = stats.ttest_rel(a, b, nan_policy="omit")
    else:
        # Wilcoxon errors when all differences are zero or n<10; degrade
        # gracefully so the runner doesn't crash.
        try:
            stat, p = stats.wilcoxon(a, b)
        except Exception:
            return float("nan"), float("nan")
    return float(stat), float(p)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def create_plots(
    results_df: pd.DataFrame,
    scalability_df: pd.DataFrame,
    output_dir: str | Path,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if "f1" in results_df.columns and "method" in results_df.columns:
        results_df.boxplot(column="f1", by="method")
        plt.suptitle("")
        plt.title("F1 distributions")
        plt.savefig(out / "fig_f1_boxplot.png", bbox_inches="tight")
        plt.close()

    if "ttd_hours" in results_df.columns and "method" in results_df.columns:
        results_df.boxplot(column="ttd_hours", by="method")
        plt.suptitle("")
        plt.title("TTD distributions")
        plt.savefig(out / "fig_ttd_boxplot.png", bbox_inches="tight")
        plt.close()

    if not scalability_df.empty:
        plt.plot(scalability_df["num_users"], scalability_df["runtime_seconds"], marker="o")
        plt.xlabel("Users")
        plt.ylabel("Runtime (s)")
        plt.title("Scalability runtime")
        plt.savefig(out / "fig_scalability_runtime.png", bbox_inches="tight")
        plt.close()

    if "fp_per_day" in results_df.columns and "method" in results_df.columns:
        fp = results_df.groupby("method")["fp_per_day"].mean().reset_index()
        plt.bar(fp["method"], fp["fp_per_day"])
        plt.xticks(rotation=45, ha="right")
        plt.title("False positive comparison")
        plt.savefig(out / "fig_false_positive_comparison.png", bbox_inches="tight")
        plt.close()
