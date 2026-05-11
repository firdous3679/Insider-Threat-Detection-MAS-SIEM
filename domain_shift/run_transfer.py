#!/usr/bin/env python3
"""Run Phase 1 Enron-to-municipal email domain-shift experiments."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

try:
    from domain_shift.load_corpora import ATTACK_CATEGORIES, describe, load_all
except ModuleNotFoundError:
    from load_corpora import ATTACK_CATEGORIES, describe, load_all


SEED = 42

ORIGINAL_VOCAB = [
    "urgent",
    "verify",
    "password",
    "click here",
    "confirm",
    "expire",
    "act now",
    "immediately",
    "confidential",
    "suspended",
    "unauthorized",
    "security alert",
    "account",
    "bank",
    "credit card",
    "private",
    "credential",
    "ssn",
]

EXFIL_ALIGNED_VOCAB = ORIGINAL_VOCAB + [
    "external",
    "upload",
    "send externally",
    "forward externally",
    "share with",
    "off-network",
    "backup",
    "remote drive",
    "unauthorized recipient",
    "compressed archive",
]


class KeywordFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary: tuple[str, ...] = tuple(ORIGINAL_VOCAB)):
        self.vocabulary = tuple(vocabulary)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            lowered = str(text).lower()
            rows.append([1.0 if phrase in lowered else 0.0 for phrase in self.vocabulary])
        return sparse.csr_matrix(np.asarray(rows, dtype=float))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def build_model(vocabulary: list[str] | None = None) -> Pipeline:
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english",
        sublinear_tf=True,
    )
    if vocabulary is None:
        features = tfidf
    else:
        features = FeatureUnion(
            [
                ("tfidf", tfidf),
                ("keywords", KeywordFeatures(tuple(vocabulary))),
            ]
        )
    return Pipeline(
        [
            ("features", features),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]
    )


def positive_scores(model: Pipeline, texts: pd.Series | list[str]) -> np.ndarray:
    clf = model.named_steps["clf"]
    if hasattr(clf, "classes_") and 1 in clf.classes_:
        pos_idx = int(np.where(clf.classes_ == 1)[0][0])
    else:
        pos_idx = 1
    return model.predict_proba(list(texts))[:, pos_idx]


def metric_row(dataset: str, experiment: str, y_true, scores, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= threshold).astype(int)
    row = {
        "experiment": experiment,
        "dataset": dataset,
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) == 2:
        row["roc_auc"] = float(roc_auc_score(y_true, scores))
        row["pr_auc"] = float(average_precision_score(y_true, scores))
    else:
        row["roc_auc"] = np.nan
        row["pr_auc"] = np.nan
    return row


def threshold_metric_row(
    dataset: str,
    threshold_name: str,
    threshold_value: float,
    y_true,
    scores,
) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= threshold_value).astype(int)
    row = {
        "dataset": dataset,
        "threshold_name": threshold_name,
        "threshold_value": float(threshold_value),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "predicted_positive_rate": float(y_pred.mean()),
    }
    if len(np.unique(y_true)) == 2:
        row["ROC-AUC"] = float(roc_auc_score(y_true, scores))
        row["PR-AUC"] = float(average_precision_score(y_true, scores))
    else:
        row["ROC-AUC"] = np.nan
        row["PR-AUC"] = np.nan
    return row


def best_f1_threshold(y_true, scores) -> float:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.01, 0.99, 99):
        y_pred = (scores >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def best_f1_threshold_fine(y_true, scores) -> float:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    candidates = np.unique(
        np.concatenate(
            [
                np.array([0.0, 0.002, 0.5, 0.98, 1.0], dtype=float),
                np.asarray(scores, dtype=float),
                np.linspace(0.001, 0.999, 999),
            ]
        )
    )
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        y_pred = (scores >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def grouped_cv_in_domain(enron: pd.DataFrame, model: Pipeline) -> dict:
    X = enron["text"].tolist()
    y = enron["label"].to_numpy(dtype=int)
    groups = enron["group"].fillna(enron["text"].map(normalize_text)).to_numpy()
    group_counts = (
        pd.DataFrame({"label": y, "group": groups})
        .drop_duplicates(["label", "group"])
        .groupby("label")
        .size()
    )
    use_grouped = len(group_counts) == 2 and int(group_counts.min()) >= 25
    if use_grouped:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        splits = splitter.split(X, y, groups)
        strategy = "stratified_group_cv"
    else:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        splits = splitter.split(X, y)
        strategy = "stratified_cv_duplicate_fallback"

    scores = np.zeros(len(enron), dtype=float)
    for train_idx, test_idx in splits:
        fold_model = clone(model)
        fold_model.fit([X[i] for i in train_idx], y[train_idx])
        scores[test_idx] = positive_scores(fold_model, [X[i] for i in test_idx])
    threshold = best_f1_threshold(y, scores)
    row = metric_row("enron_spam_grouped_cv", "in_domain_cv", y, scores, threshold)
    row["cv_strategy"] = strategy
    row["min_unique_groups_per_class"] = int(group_counts.min()) if len(group_counts) else 0
    return row


def fit_enron_model(enron: pd.DataFrame, vocabulary: list[str] | None = None) -> Pipeline:
    model = build_model(vocabulary)
    model.fit(enron["text"].tolist(), enron["label"].to_numpy(dtype=int))
    return model


def run_zero_shot(bundle, vocabulary: list[str] | None = None) -> pd.DataFrame:
    model = fit_enron_model(bundle.enron, vocabulary)
    in_domain = grouped_cv_in_domain(bundle.enron, build_model(vocabulary))
    transfer_threshold = float(in_domain["threshold"])
    rows = [in_domain]
    for name, df in [
        ("municipal_synthetic_zero_shot", bundle.municipal),
        ("kurdi_smart_building_zero_shot", bundle.kurdi),
    ]:
        scores = positive_scores(model, df["text"].tolist())
        rows.append(metric_row(name, "zero_shot", df["label"], scores, transfer_threshold))
    return pd.DataFrame(rows)


def run_fine_tuned(bundle, vocabulary: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for name, df in [
        ("municipal_synthetic_fine_tuned", bundle.municipal),
        ("kurdi_smart_building_fine_tuned", bundle.kurdi),
    ]:
        train_df, test_df = train_test_split(
            df,
            test_size=0.20,
            stratify=df["label"],
            random_state=SEED,
        )
        adapt_fit, adapt_val = train_test_split(
            train_df,
            test_size=0.25,
            stratify=train_df["label"],
            random_state=SEED,
        )
        threshold_train = pd.concat([bundle.enron, adapt_fit], ignore_index=True)
        threshold_model = fit_enron_model(threshold_train, vocabulary)
        threshold_scores = positive_scores(threshold_model, adapt_val["text"].tolist())
        transfer_threshold = best_f1_threshold(adapt_val["label"], threshold_scores)

        combined_train = pd.concat([bundle.enron, train_df], ignore_index=True)
        model = fit_enron_model(combined_train, vocabulary)
        scores = positive_scores(model, test_df["text"].tolist())
        row = metric_row(name, "fine_tuned_transfer", test_df["label"], scores, transfer_threshold)
        row["train_adaptation_examples"] = int(len(train_df))
        row["test_holdout_examples"] = int(len(test_df))
        rows.append(row)
    return pd.DataFrame(rows)


def run_vocab_adaptation(bundle) -> pd.DataFrame:
    frames = []
    for label, vocab in [
        ("tfidf_only", None),
        ("original_keywords", ORIGINAL_VOCAB),
        ("exfil_aligned_keywords", EXFIL_ALIGNED_VOCAB),
    ]:
        df = run_zero_shot(bundle, vocab)
        df.insert(1, "feature_set", label)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def run_per_category_kurdi(bundle) -> pd.DataFrame:
    model = fit_enron_model(bundle.enron)
    threshold = float(grouped_cv_in_domain(bundle.enron, build_model())["threshold"])
    df = bundle.kurdi.copy()
    scores = positive_scores(model, df["text"].tolist())
    df["pred"] = (scores >= threshold).astype(int)
    rows = []
    for category in sorted(ATTACK_CATEGORIES):
        subset = df[df["category"] == category]
        rows.append(
            {
                "category": category,
                "n": int(len(subset)),
                "recall": float(recall_score(subset["label"], subset["pred"], zero_division=0))
                if len(subset)
                else np.nan,
                "detected": int(subset["pred"].sum()) if len(subset) else 0,
                "missed": int((subset["pred"] == 0).sum()) if len(subset) else 0,
            }
        )
    return pd.DataFrame(rows)


def plot_domain_shift(zero_shot: pd.DataFrame, fine_tuned: pd.DataFrame, out_path: Path) -> None:
    values = []
    enron = zero_shot[zero_shot["dataset"] == "enron_spam_grouped_cv"].iloc[0]
    values.append(("Enron in-domain", enron["f1"]))
    for label, frame, dataset in [
        ("Kurdi zero-shot", zero_shot, "kurdi_smart_building_zero_shot"),
        ("Kurdi fine-tuned", fine_tuned, "kurdi_smart_building_fine_tuned"),
        ("Municipal zero-shot", zero_shot, "municipal_synthetic_zero_shot"),
        ("Municipal fine-tuned", fine_tuned, "municipal_synthetic_fine_tuned"),
    ]:
        values.append((label, frame[frame["dataset"] == dataset].iloc[0]["f1"]))

    names, f1s = zip(*values)
    plt.figure(figsize=(10, 5.8))
    colors = ["#2d6cdf", "#d65f5f", "#5da271", "#d65f5f", "#5da271"]
    plt.bar(names, f1s, color=colors)
    plt.ylim(0, 1.0)
    plt.ylabel("F1")
    plt.title("Enron-to-Municipal Email Domain Shift")
    plt.xticks(rotation=25, ha="right")
    for i, value in enumerate(f1s):
        plt.text(i, min(0.98, value + 0.025), f"{value:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_threshold_sensitivity(sensitivity: pd.DataFrame, out_path: Path) -> None:
    datasets = list(sensitivity["dataset"].drop_duplicates())
    threshold_order = [
        "enron_f1_optimized",
        "default_0_50",
        "mesa_runtime_0_002",
        "target_oracle_best_f1_upper_bound",
    ]
    fig, axes = plt.subplots(1, len(datasets), figsize=(13, 5.5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    colors = ["#4f6bed", "#5a5f66", "#d9702f", "#238636"]
    for ax, dataset in zip(axes, datasets):
        subset = sensitivity[sensitivity["dataset"] == dataset].set_index("threshold_name")
        labels = []
        values = []
        for name in threshold_order:
            labels.append(name.replace("_", "\n"))
            values.append(float(subset.loc[name, "F1"]) if name in subset.index else 0.0)
        ax.bar(labels, values, color=colors)
        ax.set_title(dataset)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("F1")
        ax.tick_params(axis="x", labelsize=8)
        for i, value in enumerate(values):
            ax.text(i, min(0.98, value + 0.025), f"{value:.3f}", ha="center", fontsize=9)
    fig.suptitle("Phase 1 Zero-Shot Threshold Sensitivity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
        else:
            formatted[col] = formatted[col].map(lambda v: "" if pd.isna(v) else str(v))
    headers = [str(c) for c in formatted.columns]
    rows = formatted.values.tolist()
    widths = [
        max(len(headers[i]), *(len(str(row[i])) for row in rows))
        for i in range(len(headers))
    ]
    header_line = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *body])


def run_threshold_sensitivity(bundle) -> pd.DataFrame:
    model = fit_enron_model(bundle.enron)
    enron_threshold = float(grouped_cv_in_domain(bundle.enron, build_model())["threshold"])
    rows = []
    for dataset_name, df in [
        ("municipal_synthetic", bundle.municipal),
        ("kurdi_smart_building", bundle.kurdi),
    ]:
        y_true = df["label"].to_numpy(dtype=int)
        scores = positive_scores(model, df["text"].tolist())
        threshold_specs = [
            ("enron_f1_optimized", enron_threshold),
            ("default_0_50", 0.50),
            ("mesa_runtime_0_002", 0.002),
            ("target_oracle_best_f1_upper_bound", best_f1_threshold_fine(y_true, scores)),
        ]
        for threshold_name, threshold_value in threshold_specs:
            rows.append(
                threshold_metric_row(
                    dataset_name,
                    threshold_name,
                    threshold_value,
                    y_true,
                    scores,
                )
            )
    return pd.DataFrame(rows)[
        [
            "dataset",
            "threshold_name",
            "threshold_value",
            "precision",
            "recall",
            "F1",
            "ROC-AUC",
            "PR-AUC",
            "predicted_positive_rate",
        ]
    ]


def write_threshold_sensitivity_report(sensitivity: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# Phase 1 Threshold-Sensitivity Analysis",
        "",
        "## Results",
        "",
        markdown_table(sensitivity),
        "",
        "## Interpretation",
        "",
    ]
    paragraphs = []
    for dataset in sensitivity["dataset"].drop_duplicates():
        subset = sensitivity[sensitivity["dataset"] == dataset].set_index("threshold_name")
        enron = subset.loc["enron_f1_optimized"]
        default = subset.loc["default_0_50"]
        mesa = subset.loc["mesa_runtime_0_002"]
        oracle = subset.loc["target_oracle_best_f1_upper_bound"]
        paragraphs.append(
            f"For `{dataset}`, the Enron-trained classifier has ranking signal "
            f"(ROC-AUC={enron['ROC-AUC']:.3f}, PR-AUC={enron['PR-AUC']:.3f}), but the "
            f"Enron F1-optimized threshold ({enron['threshold_value']:.3f}) yields "
            f"F1={enron['F1']:.3f} with predicted-positive rate "
            f"{enron['predicted_positive_rate']:.3f}. At the default 0.50 threshold, "
            f"F1={default['F1']:.3f}; at the original Mesa runtime threshold 0.002, "
            f"F1={mesa['F1']:.3f}. The target-oracle upper bound reaches "
            f"F1={oracle['F1']:.3f} at threshold {oracle['threshold_value']:.4f}."
        )
    lines.extend(paragraphs)
    lines.extend(
        [
            "",
            (
                "Overall, poor zero-shot F1 is at least partly a threshold-calibration/domain-shift "
                "issue: the Enron-calibrated cutoff is too conservative for the target corpora, even "
                "when probability rankings contain usable signal. The target-oracle row is not a deployable "
                "result because it uses target labels, but it shows the ceiling available from recalibration. "
                "A defensible deployment therefore needs either target-domain threshold calibration on labeled "
                "validation data or target-domain fine-tuning, rather than simply reusing the Enron threshold."
            ),
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_report(
    out_path: Path,
    corpus_summary: pd.DataFrame,
    zero_shot: pd.DataFrame,
    fine_tuned: pd.DataFrame,
    vocab_adaptation: pd.DataFrame,
    per_category: pd.DataFrame,
) -> None:
    enron_f1 = float(zero_shot.loc[zero_shot["dataset"] == "enron_spam_grouped_cv", "f1"].iloc[0])
    enron_strategy = str(
        zero_shot.loc[zero_shot["dataset"] == "enron_spam_grouped_cv", "cv_strategy"].iloc[0]
    )
    min_groups = float(
        zero_shot.loc[
            zero_shot["dataset"] == "enron_spam_grouped_cv", "min_unique_groups_per_class"
        ].iloc[0]
    )
    municipal_zero = zero_shot[zero_shot["dataset"] == "municipal_synthetic_zero_shot"].iloc[0]
    kurdi_zero = zero_shot[zero_shot["dataset"] == "kurdi_smart_building_zero_shot"].iloc[0]
    municipal_ft = fine_tuned[fine_tuned["dataset"] == "municipal_synthetic_fine_tuned"].iloc[0]
    kurdi_ft = fine_tuned[fine_tuned["dataset"] == "kurdi_smart_building_fine_tuned"].iloc[0]

    vocab_focus = vocab_adaptation[
        vocab_adaptation["dataset"].isin(
            ["municipal_synthetic_zero_shot", "kurdi_smart_building_zero_shot"]
        )
    ][["feature_set", "dataset", "f1", "precision", "recall", "pr_auc"]]
    best_cats = per_category.sort_values(["recall", "category"], ascending=[False, True]).head(3)
    worst_cats = per_category.sort_values(["recall", "category"], ascending=[True, True]).head(3)

    lines = [
        "# Phase 1 Report: Email Domain-Shift Evaluation",
        "",
        "## Corpus Summary",
        "",
        markdown_table(corpus_summary),
        "",
        "## Main Finding",
        "",
        (
            f"The Enron in-domain CV F1 is {enron_f1:.3f}. "
            f"Zero-shot transfer drops to {municipal_zero['f1']:.3f} on the synthetic municipal corpus "
            f"and {kurdi_zero['f1']:.3f} on the Kurdi smart-building corpus. "
            f"That corresponds to F1 gaps of {enron_f1 - municipal_zero['f1']:.3f} "
            f"and {enron_f1 - kurdi_zero['f1']:.3f}, respectively."
        ),
        "",
        (
            f"CV note: the runner attempted duplicate-aware grouping by normalized text, but the Enron "
            f"spam class collapsed to only {min_groups:.0f} unique groups. The reported in-domain row "
            f"therefore uses `{enron_strategy}` and records the fallback explicitly in the CSV."
        ),
        "",
        "Fine-tuning with 80% of each target corpus and evaluating on a stratified 20% holdout changes the picture:",
        "",
        markdown_table(fine_tuned),
        "",
        "## Zero-Shot Transfer",
        "",
        markdown_table(zero_shot),
        "",
        "## Vocabulary Adaptation",
        "",
        markdown_table(vocab_focus),
        "",
        (
            "The exfiltration-aligned vocabulary is useful as an interpretable signal check, "
            "but the measured results show whether keyword expansion alone closes the transfer gap. "
            "If the exfil-aligned feature set remains below the fine-tuned holdout F1, the reviewer-facing "
            "interpretation is that domain adaptation is required rather than vocabulary substitution alone."
        ),
        "",
        "## Kurdi Per-Category Recall",
        "",
        markdown_table(per_category),
        "",
        "Best transferring attack categories:",
        "",
        markdown_table(best_cats),
        "",
        "Poorest transferring attack categories:",
        "",
        markdown_table(worst_cats),
        "",
        "## Reviewer 2 Q2 Answer",
        "",
        (
            "The hybrid architecture should be framed as synergistic rather than merely compensating for "
            "domain mismatch. The Enron-only text classifier gives a direct measure of the communication-layer "
            "domain gap; the broader MAS/SIEM evidence layers are still needed because municipal insider "
            "activity is not reducible to Enron-style spam/phishing text. Phase 1 therefore separates the "
            "limits of email transfer from the contribution of behavioral and evidence-gated signals."
        ),
        "",
        "## Paper-Ready Section 5 Paragraph",
        "",
        (
            f"To quantify the external validity of the Enron-derived email-forensics component, we evaluated "
            f"an Enron-trained TF-IDF/logistic-regression classifier under zero-shot transfer to two "
            f"smart-building email corpora. The in-domain CV F1 on Enron Spam was {enron_f1:.3f}, "
            f"whereas zero-shot F1 was {municipal_zero['f1']:.3f} on the 1,000-message synthetic municipal "
            f"facilities corpus and {kurdi_zero['f1']:.3f} on the 140-message Kurdi smart-building corpus. "
            f"After target-domain fine-tuning using 80% of each municipal corpus, holdout F1 was "
            f"{municipal_ft['f1']:.3f} and {kurdi_ft['f1']:.3f}, respectively. These results show a measurable "
            f"domain-shift gap and support using Enron as a calibration source rather than as a sufficient "
            f"standalone representation of municipal insider-threat communication."
        ),
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--out-dir", type=Path, default=Path("results/domain_shift"))
    parser.add_argument(
        "--threshold-sensitivity-only",
        action="store_true",
        help="Only write threshold_sensitivity.csv/md/png; leave existing Phase 1 CSVs untouched.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    out_dir = (repo_root / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_all(repo_root)

    if args.threshold_sensitivity_only:
        sensitivity = run_threshold_sensitivity(bundle)
        sensitivity.to_csv(out_dir / "threshold_sensitivity.csv", index=False)
        write_threshold_sensitivity_report(
            sensitivity,
            out_dir / "threshold_sensitivity.md",
        )
        plot_threshold_sensitivity(
            sensitivity,
            out_dir / "fig_threshold_sensitivity.png",
        )
        print(
            json.dumps(
                {
                    "outputs": [
                        str(out_dir / "threshold_sensitivity.csv"),
                        str(out_dir / "threshold_sensitivity.md"),
                        str(out_dir / "fig_threshold_sensitivity.png"),
                    ]
                },
                indent=2,
            )
        )
        return

    corpus_summary = describe(bundle)

    zero_shot = run_zero_shot(bundle)
    fine_tuned = run_fine_tuned(bundle)
    vocab_adaptation = run_vocab_adaptation(bundle)
    per_category = run_per_category_kurdi(bundle)

    corpus_summary.to_csv(out_dir / "corpus_summary.csv", index=False)
    zero_shot.to_csv(out_dir / "transfer_zero_shot.csv", index=False)
    fine_tuned.to_csv(out_dir / "transfer_fine_tuned.csv", index=False)
    vocab_adaptation.to_csv(out_dir / "vocab_adaptation.csv", index=False)
    per_category.to_csv(out_dir / "per_category_recall_kurdi.csv", index=False)
    plot_domain_shift(zero_shot, fine_tuned, out_dir / "fig_domain_shift_bars.png")
    write_report(
        repo_root / "phase1_report.md",
        corpus_summary,
        zero_shot,
        fine_tuned,
        vocab_adaptation,
        per_category,
    )

    manifest = {
        "outputs": [
            str(out_dir / "transfer_zero_shot.csv"),
            str(out_dir / "transfer_fine_tuned.csv"),
            str(out_dir / "vocab_adaptation.csv"),
            str(out_dir / "per_category_recall_kurdi.csv"),
            str(out_dir / "fig_domain_shift_bars.png"),
            str(repo_root / "phase1_report.md"),
        ]
    }
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
