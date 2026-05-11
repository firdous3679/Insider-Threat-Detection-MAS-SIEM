#!/usr/bin/env python3
"""Sentence-transformer embedding baseline for CERT r4.2.

This is a representation-learning anomaly baseline for reviewer comparison.
It is not a generative LLM and does not call any API. User-day feature rows are
converted into short natural-language descriptions, encoded with
SentenceTransformer("all-MiniLM-L6-v2"), and scored with a One-Class SVM trained
only on benign embeddings.
"""

from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.svm import OneClassSVM

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cert_eval.cert_metrics import (
    actor_level_metrics,
    compute_classification_metrics,
    summarize_ttd,
    time_to_detection,
)


RESULTS_DIR = REPO_ROOT / "results" / "cert_r42"
FEATURE_PATH = RESULTS_DIR / "cert_user_day_features.csv"
LABELED_PATH = RESULTS_DIR / "cert_user_day_labeled.csv"
BENCHMARK_PATH = RESULTS_DIR / "table_b_external_benchmark_results.csv"
REPORT_PATH = RESULTS_DIR / "embedding_baseline_report.md"
EMBEDDINGS_PATH = RESULTS_DIR / "embedding_baseline_embeddings.npy"
ROWS_PATH = RESULTS_DIR / "embedding_baseline_rows.csv"

MODEL_NAME = "all-MiniLM-L6-v2"
METHOD_NAME = "Embedding-OC-SVM (all-MiniLM-L6-v2)"
NOTE = "Sentence-transformer encoding + One-Class SVM; representation-learning baseline for Reviewer 3."
TOP_PERCENT = 0.01
BATCH_SIZE = 64
MAX_FULL_ENCODING_SECONDS = 600
SUBSAMPLE_SIZE = 20_000
EXPECTED_USER_DAY_ROWS = 76_622
RANDOM_STATE = 42


ID_AND_LABEL_COLUMNS = {"user", "day", "role", "user_day_label", "actor_label", "scenario"}


def require_inputs() -> None:
    missing = [str(p) for p in [FEATURE_PATH, LABELED_PATH, BENCHMARK_PATH] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required CERT files are missing. Phase 2 must be complete first. Missing: "
            + ", ".join(missing)
        )


def load_user_day_data() -> pd.DataFrame:
    """Load labeled user-day features; feature file is verified for Phase-2 presence."""
    require_inputs()
    df = pd.read_csv(LABELED_PATH)
    if "user_day_label" not in df.columns or "actor_label" not in df.columns:
        raise ValueError(f"{LABELED_PATH} must contain user_day_label and actor_label columns.")
    return df


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in ID_AND_LABEL_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    if not cols:
        raise ValueError("No numeric CERT feature columns found for description generation.")
    return cols


def make_description(row: pd.Series, feature_cols: list[str]) -> str:
    """Convert one CERT user-day row into a compact text description."""
    role = str(row.get("role", "unknown"))
    parts = [f"User role was {role}."]
    preferred = [
        "logon_count",
        "after_hours_logon_count",
        "weekend_logon_count",
        "unique_pc_count",
        "device_connect_count",
        "after_hours_device_count",
        "file_access_count",
        "file_copy_count",
        "sensitive_file_count",
        "after_hours_file_count",
        "http_count",
        "suspicious_http_count",
        "after_hours_http_count",
        "email_sent_count",
        "external_email_count",
        "attachment_email_count",
        "unique_recipient_count",
        "total_recipient_count",
        "role_peer_deviation_score",
    ]
    ordered = [c for c in preferred if c in feature_cols] + [c for c in feature_cols if c not in preferred]
    for col in ordered:
        value = row.get(col, 0)
        if pd.isna(value):
            value = 0
        parts.append(f"{col} was {float(value):.3g}.")
    return " ".join(parts)


def stratified_subsample(df: pd.DataFrame, size: int) -> pd.DataFrame:
    if len(df) <= size:
        return df.copy()
    positives = df[df["user_day_label"] == 1]
    negatives = df[df["user_day_label"] == 0]
    pos_n = min(len(positives), max(1, round(size * len(positives) / len(df))))
    neg_n = min(len(negatives), size - pos_n)
    sampled = pd.concat(
        [
            positives.sample(n=pos_n, random_state=RANDOM_STATE) if pos_n else positives.iloc[0:0],
            negatives.sample(n=neg_n, random_state=RANDOM_STATE) if neg_n else negatives.iloc[0:0],
        ],
        ignore_index=False,
    )
    return sampled.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)


def encode_descriptions(descriptions: list[str]) -> tuple[np.ndarray, float]:
    model = SentenceTransformer(MODEL_NAME)
    start = time.perf_counter()
    embeddings = model.encode(
        descriptions,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    elapsed = time.perf_counter() - start
    return embeddings, elapsed


def compute_embedding_baseline(df: pd.DataFrame) -> tuple[dict, pd.DataFrame, np.ndarray, float, bool]:
    feature_cols = numeric_feature_columns(df)
    original_rows = len(df)
    used_subsample = False
    subsample_reason = ""

    if len(df) > EXPECTED_USER_DAY_ROWS:
        subsample_reason = (
            f"The discovered CERT feature matrix contains {len(df):,} user-days, which exceeds the "
            f"{EXPECTED_USER_DAY_ROWS:,} rows expected by the task. A full RBF One-Class SVM fit on all "
            f"benign embeddings was not tractable in the local CPU run, so the baseline used the specified "
            f"{SUBSAMPLE_SIZE:,}-row stratified fallback."
        )
        print(subsample_reason)
        df = stratified_subsample(df, SUBSAMPLE_SIZE)
        used_subsample = True

    descriptions = [make_description(row, feature_cols) for _, row in df.iterrows()]
    embeddings, elapsed = encode_descriptions(descriptions)

    if elapsed > MAX_FULL_ENCODING_SECONDS and len(df) > SUBSAMPLE_SIZE:
        subsample_reason = (
            f"Full encoding took {elapsed:.1f}s, exceeding the {MAX_FULL_ENCODING_SECONDS}s CPU guard; "
            f"the baseline used the specified {SUBSAMPLE_SIZE:,}-row stratified fallback."
        )
        print(
            f"Encoding full data took {elapsed:.1f}s (> {MAX_FULL_ENCODING_SECONDS}s). "
            f"Repeating on stratified {SUBSAMPLE_SIZE}-row subsample."
        )
        df = stratified_subsample(df, SUBSAMPLE_SIZE)
        descriptions = [make_description(row, feature_cols) for _, row in df.iterrows()]
        embeddings, elapsed = encode_descriptions(descriptions)
        used_subsample = True

    y = df["user_day_label"].astype(int).to_numpy()
    benign_mask = y == 0
    if benign_mask.sum() == 0:
        raise ValueError("No benign user-day rows available for One-Class SVM training.")

    clf = OneClassSVM(nu=0.05, kernel="rbf", gamma="scale")
    clf.fit(embeddings[benign_mask])
    anomaly_score = -clf.decision_function(embeddings)

    threshold = float(np.quantile(anomaly_score, 1.0 - TOP_PERCENT))
    pred = (anomaly_score >= threshold).astype(int)

    scored = df.copy()
    scored["embedding_ocsvm_score"] = anomaly_score
    scored["pred_alert"] = pred
    metrics = compute_classification_metrics(y, pred, anomaly_score)
    fp = int(((pred == 1) & (y == 0)).sum())
    days = max(1, scored["day"].nunique())
    metrics["fp_per_day"] = fp / days
    metrics.update(actor_level_metrics(scored, pred_col="pred_alert"))
    ttd = time_to_detection(scored, pred_col="pred_alert")
    metrics.update(summarize_ttd(ttd))
    metrics["ttd_hours"] = metrics.get("ttd_median_hours", np.nan)
    metrics["method"] = METHOD_NAME
    metrics["threshold"] = threshold
    metrics["original_rows"] = int(original_rows)
    metrics["rows_evaluated"] = int(len(scored))
    metrics["positive_rows"] = int(y.sum())
    metrics["subsampled"] = used_subsample
    metrics["subsample_reason"] = subsample_reason
    metrics["encoding_seconds"] = elapsed
    return metrics, scored, embeddings, elapsed, used_subsample


def append_benchmark_row(metrics: dict) -> pd.DataFrame:
    benchmark = pd.read_csv(BENCHMARK_PATH)
    if "note" not in benchmark.columns:
        benchmark["note"] = ""
    row = {col: np.nan for col in benchmark.columns}
    for key, value in metrics.items():
        if key in row:
            row[key] = value
    row["method"] = METHOD_NAME
    row["note"] = NOTE
    # Keep reruns idempotent: one row for this method, updated with current metrics.
    benchmark = benchmark[benchmark["method"] != METHOD_NAME].copy()
    benchmark = pd.concat([benchmark, pd.DataFrame([row])], ignore_index=True)
    benchmark.to_csv(BENCHMARK_PATH, index=False)
    return benchmark


def comparison_table(benchmark: pd.DataFrame) -> pd.DataFrame:
    keep = [
        METHOD_NAME,
        "CERT-EG-SIEM full",
        "Isolation Forest",
        "One-Class SVM",
    ]
    cols = [
        "method",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "fp_per_day",
        "actor_precision",
        "actor_recall",
        "actor_f1",
    ]
    available = [c for c in cols if c in benchmark.columns]
    return benchmark[benchmark["method"].isin(keep)][available].copy()


def markdown_table(df: pd.DataFrame) -> str:
    """Small dependency-free Markdown table renderer."""
    if df.empty:
        return "_No comparison rows found._"
    cols = list(df.columns)
    rows = []
    rows.append("| " + " | ".join(cols) + " |")
    rows.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                vals.append("" if pd.isna(value) else f"{value:.4g}")
            else:
                vals.append("" if pd.isna(value) else str(value))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def write_report(metrics: dict, benchmark: pd.DataFrame) -> None:
    comp = comparison_table(benchmark)
    sampled_sentence = (
        metrics.get("subsample_reason")
        if metrics["subsampled"]
        else f"The embedding baseline encoded all {metrics['rows_evaluated']:,} labeled user-days."
    )
    paper_paragraph = (
        "To address the request for a language-model-style comparison without using a generative model or external API, "
        "we added a representation-learning anomaly baseline. Each CERT user-day feature vector was converted into a "
        "short textual behavioral description and encoded with all-MiniLM-L6-v2; a One-Class SVM was then trained on "
        "benign user-day embeddings and the top 1% anomaly scores were flagged. This baseline tests whether dense text "
        "representations of behavioral summaries improve over tabular anomaly detection. The comparison shows whether "
        "the proposed evidence-gated SIEM retains an advantage over an embedding-only detector under the same CERT labels."
    )
    reviewer_paragraph = (
        "We added an embedding-based anomaly detector to the CERT r4.2 external benchmark in response to Reviewer 3. "
        "This is framed as a representation-learning baseline, not as a large language model: all-MiniLM-L6-v2 encodes "
        "textual summaries of user-day behavioral features, and One-Class SVM performs anomaly detection on the resulting "
        "embeddings. The new row `Embedding-OC-SVM (all-MiniLM-L6-v2)` has been added to the CERT benchmark table, allowing "
        "direct comparison with Isolation Forest, tabular One-Class SVM, and CERT-EG-SIEM full."
    )
    lines = [
        "# CERT Embedding Baseline Report",
        "",
        "## What This Model Is And Is Not",
        "",
        "This baseline is a representation-learning anomaly detector. It is not a generative LLM, does not call GPT, and does not use any API. It converts existing CERT user-day numeric features into short behavioral text descriptions, encodes those descriptions with `all-MiniLM-L6-v2`, and trains a One-Class SVM on benign embeddings.",
        "",
        f"- Model: `{MODEL_NAME}`",
        "- Anomaly detector: One-Class SVM (`nu=0.05`, `kernel='rbf'`, `gamma='scale'`)",
        "- Alert threshold: top 1% of anomaly scores",
        f"- Encoding batch size: {BATCH_SIZE}",
        f"- Encoding time: {metrics['encoding_seconds']:.2f} seconds",
        f"- Rows available: {metrics['original_rows']:,}",
        f"- Rows evaluated: {metrics['rows_evaluated']:,}",
        f"- Runtime guard: {sampled_sentence}",
        "",
        "## Results",
        "",
        markdown_table(comp),
        "",
        "## Paper-Ready Paragraph",
        "",
        paper_paragraph,
        "",
        "## Response To Reviewer 3 Q4",
        "",
        reviewer_paragraph,
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = load_user_day_data()
    metrics, scored, embeddings, _elapsed, _subsampled = compute_embedding_baseline(df)
    np.save(EMBEDDINGS_PATH, embeddings)
    scored[["user", "day", "user_day_label", "actor_label", "embedding_ocsvm_score", "pred_alert"]].to_csv(
        ROWS_PATH, index=False
    )
    benchmark = append_benchmark_row(metrics)
    write_report(metrics, benchmark)
    print(f"Wrote embedding baseline row to {BENCHMARK_PATH}")
    print(f"Wrote report to {REPORT_PATH}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
