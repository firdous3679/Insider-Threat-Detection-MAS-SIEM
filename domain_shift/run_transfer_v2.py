#!/usr/bin/env python3
"""Run Phase 1 V2 municipal corpus domain-shift evaluations."""

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
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

try:
    from domain_shift.load_corpora import (
        first_existing,
        load_enron_spam,
        load_kurdi,
        load_municipal,
        load_municipal_v2,
    )
    from domain_shift.run_transfer import (
        EXFIL_ALIGNED_VOCAB,
        ORIGINAL_VOCAB,
        best_f1_threshold_fine,
        build_model,
        fit_enron_model,
        grouped_cv_in_domain,
        markdown_table,
        positive_scores,
        threshold_metric_row,
    )
except ModuleNotFoundError:
    from load_corpora import first_existing, load_enron_spam, load_kurdi, load_municipal, load_municipal_v2
    from run_transfer import (
        EXFIL_ALIGNED_VOCAB,
        ORIGINAL_VOCAB,
        best_f1_threshold_fine,
        build_model,
        fit_enron_model,
        grouped_cv_in_domain,
        markdown_table,
        positive_scores,
        threshold_metric_row,
    )


SEED = 42
MESA_THRESHOLD = 0.002
MUNICIPAL_SECURITY_VOCAB = EXFIL_ALIGNED_VOCAB + [
    "vpn",
    "badge",
    "firmware",
    "audit log",
    "log export",
    "remote support",
    "security exception",
    "change advisory",
    "change ticket",
    "cab",
    "alarm testing",
    "procurement",
    "banking update",
    "approved",
    "expiration",
    "rollback",
    "vendor roster",
    "bms",
    "controller",
    "credential",
]


def word_count_series(df: pd.DataFrame) -> pd.Series:
    if "body_word_count" in df.columns and df["body_word_count"].notna().any():
        return pd.to_numeric(df["body_word_count"], errors="coerce")
    return df["text"].map(lambda s: len(re.findall(r"\b\w+\b", str(s))))


def binary_count(df: pd.DataFrame, col: str):
    if col not in df.columns or df[col].isna().all():
        return np.nan
    return int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())


def metric_row(dataset: str, experiment: str, y_true, scores, threshold: float, extra=None) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)
    row = {
        "dataset": dataset,
        "experiment": experiment,
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else np.nan,
        "PR-AUC": float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) == 2 else np.nan,
        "predicted_positive_rate": float(pred.mean()),
    }
    if extra:
        row.update(extra)
    return row


def corpus_summary(corpora: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in corpora.items():
        wc = word_count_series(df)
        rows.append(
            {
                "corpus_name": name,
                "total_emails": int(len(df)),
                "positive_count": int(df["label"].sum()),
                "negative_count": int((df["label"] == 0).sum()),
                "positive_rate": float(df["label"].mean()),
                "average_body_word_count": float(wc.mean()),
                "median_body_word_count": float(wc.median()),
                "number_of_categories": int(df["category"].nunique()) if "category" in df else np.nan,
                "number_of_subcategories": int(df["subcategory"].nunique()) if "subcategory" in df else np.nan,
                "number_of_templates": int(df["template_id"].replace("", np.nan).nunique()) if "template_id" in df else np.nan,
                "number_of_template_families": int(df["template_family"].replace("", np.nan).nunique()) if "template_family" in df else np.nan,
                "hard_negative_count": int(df["is_hard_negative"].sum()) if "is_hard_negative" in df else np.nan,
                "emails_with_attachment": binary_count(df, "has_attachment"),
                "emails_with_external_link": binary_count(df, "has_external_link"),
            }
        )
    return pd.DataFrame(rows)


def target_corpora(v1: pd.DataFrame, v2: pd.DataFrame, kurdi: pd.DataFrame) -> dict[str, pd.DataFrame]:
    v1 = v1.copy()
    v1["dataset"] = "municipal_v1"
    v2 = v2.copy()
    v2["dataset"] = "municipal_v2"
    kurdi = kurdi.copy()
    kurdi["dataset"] = "kurdi_smart_building"
    return {"municipal_v1": v1, "municipal_v2": v2, "kurdi_smart_building": kurdi}


def enron_cutoff(enron: pd.DataFrame) -> float:
    return float(grouped_cv_in_domain(enron, build_model())["threshold"])


def zero_shot(enron: pd.DataFrame, corpora: dict[str, pd.DataFrame], threshold: float) -> pd.DataFrame:
    model = fit_enron_model(enron)
    rows = []
    for name, df in corpora.items():
        scores = positive_scores(model, df["text"].tolist())
        rows.append(metric_row(name, "zero_shot_enron", df["label"], scores, threshold))
    return pd.DataFrame(rows)


def calibrated_threshold_eval(scores, labels, dataset: str) -> dict:
    df = pd.DataFrame({"score": scores, "label": np.asarray(labels, dtype=int)})
    cal, test = train_test_split(df, test_size=0.50, stratify=df["label"], random_state=SEED)
    threshold = best_f1_threshold_fine(cal["label"], cal["score"])
    return threshold_metric_row(dataset, "target_calibrated_heldout", threshold, test["label"], test["score"])


def threshold_sensitivity(enron: pd.DataFrame, corpora: dict[str, pd.DataFrame], threshold: float) -> pd.DataFrame:
    model = fit_enron_model(enron)
    rows = []
    for name, df in corpora.items():
        y = df["label"].to_numpy(dtype=int)
        scores = positive_scores(model, df["text"].tolist())
        specs = [
            ("enron_f1_optimized", threshold),
            ("default_0_50", 0.50),
            ("mesa_runtime_0_002", MESA_THRESHOLD),
            ("target_oracle_best_f1_upper_bound", best_f1_threshold_fine(y, scores)),
        ]
        for threshold_name, value in specs:
            rows.append(threshold_metric_row(name, threshold_name, value, y, scores))
        rows.append(calibrated_threshold_eval(scores, y, name))
    return pd.DataFrame(rows)


def tune_threshold_from_train(enron: pd.DataFrame, train_df: pd.DataFrame) -> float:
    fit_df, val_df = train_test_split(train_df, test_size=0.25, stratify=train_df["label"], random_state=SEED)
    model = fit_enron_model(pd.concat([enron, fit_df], ignore_index=True))
    scores = positive_scores(model, val_df["text"].tolist())
    return best_f1_threshold_fine(val_df["label"], scores)


def fine_tuned_random(enron: pd.DataFrame, corpora: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in corpora.items():
        train_df, test_df = train_test_split(df, test_size=0.20, stratify=df["label"], random_state=SEED)
        threshold = tune_threshold_from_train(enron, train_df)
        model = fit_enron_model(pd.concat([enron, train_df], ignore_index=True))
        scores = positive_scores(model, test_df["text"].tolist())
        rows.append(
            metric_row(
                name,
                "fine_tuned_random_split",
                test_df["label"],
                scores,
                threshold,
                {
                    "train_size": int(len(train_df)),
                    "test_size": int(len(test_df)),
                    "positive_rate_train": float(train_df["label"].mean()),
                    "positive_rate_test": float(test_df["label"].mean()),
                    "threshold_method": "target_train_validation_best_f1",
                },
            )
        )
    return pd.DataFrame(rows)


def grouped_split(df: pd.DataFrame, group_col: str):
    groups = df[group_col].astype(str)
    splitter = GroupShuffleSplit(n_splits=200, test_size=0.20, random_state=SEED)
    best = None
    best_score = 1e9
    for train_idx, test_idx in splitter.split(df, df["label"], groups):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        if train["label"].nunique() < 2 or test["label"].nunique() < 2:
            continue
        score = abs(float(test["label"].mean()) - float(df["label"].mean()))
        if score < best_score:
            best = (train_idx, test_idx, "strict_grouped_split")
            best_score = score
    if best:
        return best
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.20,
        stratify=df["label"],
        random_state=SEED,
    )
    return train_idx, test_idx, f"fallback_stratified_random_split_grouping_failed_for_{group_col}"


def fine_tuned_grouped(enron: pd.DataFrame, v2: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in v2.columns or v2[group_col].replace("", np.nan).dropna().nunique() < 2:
        group_col = "subcategory"
        split_note = "requested_group_missing_or_unusable_fallback_to_subcategory"
    else:
        split_note = ""
    train_idx, test_idx, split_strategy = grouped_split(v2, group_col)
    train_df = v2.iloc[train_idx].copy()
    test_df = v2.iloc[test_idx].copy()
    shared = set(train_df[group_col].astype(str)).intersection(set(test_df[group_col].astype(str)))
    threshold = tune_threshold_from_train(enron, train_df)
    model = fit_enron_model(pd.concat([enron, train_df], ignore_index=True))
    scores = positive_scores(model, test_df["text"].tolist())
    return pd.DataFrame(
        [
            metric_row(
                "municipal_v2",
                f"fine_tuned_grouped_by_{group_col}",
                test_df["label"],
                scores,
                threshold,
                {
                    "grouping_column": group_col,
                    "number_of_train_groups": int(train_df[group_col].nunique()),
                    "number_of_test_groups": int(test_df[group_col].nunique()),
                    "shared_group_count": int(len(shared)),
                    "split_strategy": split_strategy,
                    "split_note": split_note,
                    "train_size": int(len(train_df)),
                    "test_size": int(len(test_df)),
                },
            )
        ]
    )


def hard_negative_outputs(enron, v2, enron_threshold, random_result, grouped_result):
    hard = v2[v2.get("is_hard_negative", 0) == 1].copy()
    rows = []
    top_rows = []

    def add_setting(name, model, threshold, df_for_eval):
        if df_for_eval.empty:
            rows.append(
                {
                    "setting": name,
                    "hard_negative_count": 0,
                    "false_positive_count": 0,
                    "false_positive_rate": np.nan,
                    "specificity_on_hard_negatives": np.nan,
                    "threshold": threshold,
                }
            )
            return
        scores = positive_scores(model, df_for_eval["text"].tolist())
        pred = (scores >= threshold).astype(int)
        rows.append(
            {
                "setting": name,
                "hard_negative_count": int(len(df_for_eval)),
                "false_positive_count": int(pred.sum()),
                "false_positive_rate": float(pred.mean()),
                "specificity_on_hard_negatives": float(1.0 - pred.mean()),
                "threshold": float(threshold),
            }
        )
        tmp = df_for_eval.copy()
        tmp["setting"] = name
        tmp["score"] = scores
        tmp["predicted_positive"] = pred
        cols = ["setting", "score", "predicted_positive", "subject", "template_id", "template_family", "approval_context", "expected_detection_signal", "text"]
        top_rows.extend(tmp.sort_values("score", ascending=False).head(10)[[c for c in cols if c in tmp.columns]].to_dict("records"))

    zero_model = fit_enron_model(enron)
    add_setting("enron_zero_shot_enron_threshold", zero_model, enron_threshold, hard)
    zero_scores_all = positive_scores(zero_model, v2["text"].tolist())
    cal, _test = train_test_split(v2, test_size=0.50, stratify=v2["label"], random_state=SEED)
    cal_scores = pd.Series(zero_scores_all, index=v2.index).loc[cal.index]
    target_threshold = best_f1_threshold_fine(cal["label"], cal_scores)
    add_setting("enron_zero_shot_target_calibrated_threshold", zero_model, target_threshold, hard)

    train_df, test_df = train_test_split(v2, test_size=0.20, stratify=v2["label"], random_state=SEED)
    rand_threshold = float(random_result.iloc[0]["threshold"])
    rand_model = fit_enron_model(pd.concat([enron, train_df], ignore_index=True))
    add_setting("fine_tuned_random_split_hard_negative_holdout", rand_model, rand_threshold, test_df[test_df["is_hard_negative"] == 1])

    group_col = str(grouped_result.iloc[0]["grouping_column"])
    train_idx, test_idx, _ = grouped_split(v2, group_col)
    gtrain = v2.iloc[train_idx].copy()
    gtest = v2.iloc[test_idx].copy()
    gthreshold = float(grouped_result.iloc[0]["threshold"])
    gmodel = fit_enron_model(pd.concat([enron, gtrain], ignore_index=True))
    add_setting(f"fine_tuned_grouped_{group_col}_hard_negative_holdout", gmodel, gthreshold, gtest[gtest["is_hard_negative"] == 1])

    return pd.DataFrame(rows), pd.DataFrame(top_rows)


def vocab_adaptation(enron, v2) -> pd.DataFrame:
    rows = []
    for feature_set, vocab in [
        ("tfidf_only", None),
        ("original_keywords", ORIGINAL_VOCAB),
        ("exfil_aligned_keywords", EXFIL_ALIGNED_VOCAB),
        ("municipal_security_ops_keywords", MUNICIPAL_SECURITY_VOCAB),
    ]:
        model = fit_enron_model(enron, vocab)
        threshold = float(grouped_cv_in_domain(enron, build_model(vocab))["threshold"])
        scores = positive_scores(model, v2["text"].tolist())
        row = metric_row("municipal_v2", f"zero_shot_{feature_set}", v2["label"], scores, threshold)
        row["feature_set"] = feature_set
        rows.append(row)
    return pd.DataFrame(rows)


def per_subcategory(df: pd.DataFrame, model, threshold: float, dataset: str) -> pd.DataFrame:
    scores = positive_scores(model, df["text"].tolist())
    pred = (scores >= threshold).astype(int)
    tmp = df.copy()
    tmp["pred"] = pred
    rows = []
    for subcat, part in tmp.groupby("subcategory"):
        rows.append(
            {
                "dataset": dataset,
                "subcategory": subcat,
                "n": int(len(part)),
                "positives": int(part["label"].sum()),
                "precision": float(precision_score(part["label"], part["pred"], zero_division=0)),
                "recall": float(recall_score(part["label"], part["pred"], zero_division=0)),
                "F1": float(f1_score(part["label"], part["pred"], zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


def v1_v2_comparison(zero, random_ft, grouped_family, grouped_id, hard_eval):
    rows = []
    def pick(frame, dataset=None, experiment_contains=None):
        sub = frame
        if dataset:
            sub = sub[sub["dataset"] == dataset]
        if experiment_contains:
            sub = sub[sub["experiment"].astype(str).str.contains(experiment_contains)]
        return sub.iloc[0]
    for label, frame, dataset, exp in [
        ("municipal_v1_zero_shot", zero, "municipal_v1", None),
        ("municipal_v2_zero_shot", zero, "municipal_v2", None),
        ("municipal_v1_fine_tuned_random", random_ft, "municipal_v1", None),
        ("municipal_v2_fine_tuned_random", random_ft, "municipal_v2", None),
        ("municipal_v2_grouped_template_family", grouped_family, "municipal_v2", None),
        ("municipal_v2_grouped_template_id", grouped_id, "municipal_v2", None),
    ]:
        r = pick(frame, dataset, exp)
        rows.append({"result": label, "precision": r["precision"], "recall": r["recall"], "F1": r["F1"], "PR-AUC": r["PR-AUC"]})
    hard_fp = hard_eval["false_positive_rate"].max()
    rows.append({"result": "municipal_v2_hard_negative_max_fp_rate", "precision": np.nan, "recall": np.nan, "F1": hard_fp, "PR-AUC": np.nan})
    return pd.DataFrame(rows)


def plot_bar(df, label_col, value_col, title, out_path, ylabel="F1"):
    plt.figure(figsize=(10, 5.5))
    labels = df[label_col].astype(str).tolist()
    vals = df[value_col].astype(float).fillna(0).tolist()
    plt.bar(labels, vals, color="#4f6bed")
    plt.ylim(0, max(1.0, max(vals) * 1.15 if vals else 1.0))
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    for i, v in enumerate(vals):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_threshold(sens, out_path):
    pivot = sens.pivot(index="threshold_name", columns="dataset", values="F1")
    pivot.plot(kind="bar", figsize=(11, 5.8), color=["#4f6bed", "#238636", "#d9702f"])
    plt.ylabel("F1")
    plt.title("Threshold Sensitivity Across Target Corpora")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def paper_tables_tex(tables: dict[str, pd.DataFrame]) -> str:
    def esc(value) -> str:
        text = "" if pd.isna(value) else str(value)
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def simple_latex(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        spec = "l" * len(cols)
        lines = [f"\\begin{{tabular}}{{{spec}}}", "\\hline"]
        lines.append(" & ".join(esc(c) for c in cols) + r" \\")
        lines.append("\\hline")
        for _, row in df.iterrows():
            vals = []
            for c in cols:
                value = row[c]
                if isinstance(value, float):
                    vals.append("" if pd.isna(value) else f"{value:.3f}")
                else:
                    vals.append(esc(value))
            lines.append(" & ".join(vals) + r" \\")
        lines.extend(["\\hline", "\\end{tabular}"])
        return "\n".join(lines)

    chunks = []
    for title, df in tables.items():
        chunks.append(f"% {title}\n")
        chunks.append(simple_latex(df))
        chunks.append("\n")
    return "\n".join(chunks)


def write_reports(out_dir, summary, zero, sens, rand, gfam, gid, gsub, hard, vocab, comparison, missing_warnings):
    report = [
        "# Phase 1 V2 Domain-Shift Report",
        "",
        "This evaluation uses synthetic, template-generated target-domain corpora. It quantifies Enron-to-municipal/smart-building domain shift and adaptation behavior; it is not real-world validation.",
        "",
        "## Main Conclusion",
        "",
        "The Enron-trained classifier does not transfer reliably as a zero-shot municipal detector under its Enron-calibrated threshold. The target corpora still show ranking signal in ROC-AUC/PR-AUC, so the main failure mode is threshold calibration plus domain mismatch rather than total absence of signal. Target-domain calibration and fine-tuning are required.",
        "",
        "## Corpus Summary",
        "",
        markdown_table(summary),
        "",
    ]
    if missing_warnings:
        report.extend(["## Loader Warnings", "", *[f"- {w}" for w in missing_warnings], ""])
    report.extend(
        [
            "## Zero-Shot Transfer",
            "",
            markdown_table(zero),
            "",
            "## Threshold Sensitivity",
            "",
            markdown_table(sens),
            "",
            "## Fine-Tuning",
            "",
            markdown_table(pd.concat([rand, gfam, gid, gsub], ignore_index=True)),
            "",
            "## Hard Negatives",
            "",
            markdown_table(hard),
            "",
            "## Vocabulary Adaptation",
            "",
            markdown_table(vocab),
            "",
            "## V1 vs V2",
            "",
            markdown_table(comparison),
            "",
            "## What Goes Where",
            "",
            "Main paper: V2 zero-shot, threshold sensitivity, random fine-tuning, and the most conservative grouped/template-held-out result. Appendix/supplement: full per-subcategory tables, hard-negative top-score inspection, vocabulary variants, and Kurdi short-message stress test.",
            "",
            "## Revised Manuscript Paragraph",
            "",
            "To quantify whether the Enron-derived email-forensics component transfers to municipal smart-building communication, we evaluated an Enron-trained TF-IDF/logistic-regression classifier on synthetic municipal facilities corpora and a short-message smart-building stress-test corpus. The results show that Enron is useful as an initial calibration source but is not sufficient as a standalone detector: zero-shot performance depends strongly on threshold calibration, while target-domain threshold calibration and fine-tuning substantially improve target-domain performance. We therefore treat Enron as a source-domain calibration corpus and report target-domain adaptation results separately from evidence-gated SIEM/MAS results.",
            "",
            "## Response To Reviewers",
            "",
            "Reviewer 1 Q2: We agree that Enron alone is not representative of municipal smart-building operations. We therefore added a domain-shift evaluation using synthetic municipal facilities corpora, including V2 with hard benign near-miss messages and leakage-aware grouped splits. The results explicitly quantify the Enron-to-municipal gap rather than assuming transfer.",
            "",
            "Reviewer 1 Q5: We added V2 corpus metadata, methodology, template identifiers, template families, hard-negative labels, approval context, sender-domain type, and expected detection signals. These fields support reproducibility and grouped evaluation to reduce template leakage.",
            "",
            "Reviewer 2 Q2: The results show that the hybrid system is not merely compensating for Enron mismatch. The email classifier has partial ranking signal but requires target-domain calibration or fine-tuning; the SIEM/MAS layers remain necessary because municipal insider-risk evidence is broader than email text alone.",
            "",
            "Reviewer 3 Q3: We added explicit experiments showing Enron-trained transfer to municipal/smart-building corpora, including threshold sensitivity, target calibration, fine-tuning, grouped/template-held-out splits, and hard-negative evaluation. Low zero-shot performance is framed as confirmation of the reviewer concern and motivates adaptation.",
            "",
        ]
    )
    (out_dir / "phase1_v2_report.md").write_text("\n".join(report), encoding="utf-8")

    tables = {
        "Corpus Summary": summary,
        "Zero-Shot Transfer": zero,
        "Threshold Sensitivity": sens,
        "Fine-Tuned Random": rand,
        "Grouped Template Family": gfam,
        "Grouped Template ID": gid,
        "Hard Negative Evaluation": hard,
        "V1 vs V2 Comparison": comparison,
    }
    md = []
    for title, df in tables.items():
        md.extend([f"## {title}", "", markdown_table(df), ""])
    (out_dir / "phase1_v2_paper_tables.md").write_text("\n".join(md), encoding="utf-8")
    (out_dir / "phase1_v2_paper_tables.tex").write_text(paper_tables_tex(tables), encoding="utf-8")
    (out_dir / "v1_vs_v2_comparison.md").write_text("# V1 vs V2 Comparison\n\n" + markdown_table(comparison) + "\n", encoding="utf-8")

    threshold_lines = [
        "# Phase 1 V2 Threshold Sensitivity",
        "",
        "These results use synthetic target-domain corpora and diagnose threshold calibration under Enron-to-target transfer. The target-oracle rows are diagnostic upper bounds only and are not deployable results.",
        "",
        markdown_table(sens),
        "",
        "## Interpretation",
        "",
        "The Enron F1-optimized threshold is highly conservative on the target corpora and produces no positive predictions. Lower thresholds and target-domain calibration recover recall, showing that poor zero-shot F1 is driven by both calibration shift and target-domain mismatch. Target-domain threshold calibration or fine-tuning is therefore required before using the Enron-derived email classifier in the municipal setting.",
        "",
    ]
    (out_dir / "threshold_sensitivity_v2.md").write_text("\n".join(threshold_lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--out-dir", type=Path, default=Path("results/domain_shift_v2"))
    args = parser.parse_args()

    root = args.repo_root.resolve()
    out_dir = (root / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    enron = load_enron_spam(first_existing(root, ["enron_spam_data.csv"]))
    v1 = load_municipal(first_existing(root, ["data/municipal_facilities_emails.csv", "data/municipal_facilities_emails_v1.csv", "SyntheticDataset/municipal_facilities_emails_v1.csv", "SyntheticDataset/municipal_facilities_emails.csv"]))
    v2 = load_municipal_v2(first_existing(root, ["data/municipal_facilities_emails_v2.csv", "SyntheticDataset/municipal_facilities_emails_v2.csv"]))
    kurdi = load_kurdi(first_existing(root, ["data/Kurdi_cyber_insider_smart_building_muncipality.json"]))
    corpora = target_corpora(v1, v2, kurdi)
    cutoff = enron_cutoff(enron)
    missing_warnings = [f"municipal_v2 missing optional columns: {', '.join(v2.attrs.get('missing_optional_columns', []))}"] if v2.attrs.get("missing_optional_columns") else []

    summary = corpus_summary(corpora)
    zero = zero_shot(enron, corpora, cutoff)
    sens = threshold_sensitivity(enron, corpora, cutoff)
    rand = fine_tuned_random(enron, corpora)
    gfam = fine_tuned_grouped(enron, v2, "template_family")
    gid = fine_tuned_grouped(enron, v2, "template_id")
    gsub = fine_tuned_grouped(enron, v2, "subcategory")
    hard, hard_top = hard_negative_outputs(enron, v2, cutoff, rand[rand["dataset"] == "municipal_v2"], gfam)
    vocab = vocab_adaptation(enron, v2)
    zero_model = fit_enron_model(enron)
    per_muni = per_subcategory(v2, zero_model, cutoff, "municipal_v2")
    per_kurdi = per_subcategory(kurdi, zero_model, cutoff, "kurdi_smart_building")
    comparison = v1_v2_comparison(zero, rand, gfam, gid, hard)

    outputs = {
        "corpus_summary_v2.csv": summary,
        "transfer_zero_shot_v2.csv": zero,
        "threshold_sensitivity_v2.csv": sens,
        "transfer_fine_tuned_random_v2.csv": rand,
        "transfer_fine_tuned_grouped_template_family_v2.csv": gfam,
        "transfer_fine_tuned_grouped_template_id_v2.csv": gid,
        "transfer_fine_tuned_grouped_subcategory_v2.csv": gsub,
        "hard_negative_eval_v2.csv": hard,
        "hard_negative_top_scores_v2.csv": hard_top,
        "vocab_adaptation_v2.csv": vocab,
        "per_subcategory_municipal_v2.csv": per_muni,
        "per_category_recall_kurdi_v2.csv": per_kurdi,
        "v1_vs_v2_comparison.csv": comparison,
    }
    for filename, df in outputs.items():
        df.to_csv(out_dir / filename, index=False)

    write_reports(out_dir, summary, zero, sens, rand, gfam, gid, gsub, hard, vocab, comparison, missing_warnings)

    plot_bar(comparison[comparison["result"].str.contains("zero_shot|fine_tuned|grouped", regex=True)], "result", "F1", "V1 vs V2 F1", out_dir / "fig_v1_vs_v2_f1.png")
    plot_threshold(sens, out_dir / "fig_threshold_sensitivity_v2.png")
    plot_bar(pd.concat([rand[rand["dataset"] == "municipal_v2"], gfam, gid, gsub], ignore_index=True), "experiment", "F1", "Random vs Grouped Fine-Tuning V2", out_dir / "fig_random_vs_grouped_finetuning_v2.png")
    plot_bar(hard, "setting", "false_positive_rate", "Hard Negative False-Positive Rate", out_dir / "fig_hard_negative_fp_rate_v2.png", ylabel="False-positive rate")

    generated = sorted(str(p.relative_to(root)) for p in out_dir.iterdir() if p.is_file())
    print(json.dumps({"generated_files": generated}, indent=2))
    print("\nKey findings:")
    print(zero[["dataset", "F1", "ROC-AUC", "PR-AUC", "predicted_positive_rate"]].to_string(index=False))
    print(rand[["dataset", "F1", "ROC-AUC", "PR-AUC"]].to_string(index=False))
    print(gfam[["experiment", "F1", "ROC-AUC", "PR-AUC"]].to_string(index=False))
    print(hard[["setting", "false_positive_rate", "specificity_on_hard_negatives"]].to_string(index=False))


if __name__ == "__main__":
    main()
