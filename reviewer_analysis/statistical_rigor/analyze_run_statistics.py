#!/usr/bin/env python3
"""Reviewer-only statistical analysis from existing run-level result files.

This script is intentionally additive: it reads existing result files and writes
derived outputs under reviewer_analysis/statistical_rigor/outputs/. It does not
modify experiment code, model artifacts, or existing result directories.

The script fails gracefully when the repository only contains aggregate tables.
It never invents run-level statistics from means.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUT_DIR / ".mplconfig"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402


METRICS = [
    "actor_precision",
    "actor_recall",
    "actor_f1",
    "ttd_avg",
    "ttd_max",
    "confirmed_alerts",
    "confirmed_alert_precision",
    "confirmed_fp_per_run",
]

COMPARISONS = [
    ("LSC", "CE-SIEM"),
    ("LSC", "EG-SIEM"),
    ("LSC", "EG-SIEM-Enron"),
    ("EG-SIEM", "EG-SIEM-Enron"),
]


def ci95(values: pd.Series) -> tuple[float, float]:
    vals = values.dropna().astype(float).to_numpy()
    n = len(vals)
    if n == 0:
        return math.nan, math.nan
    if n == 1:
        return float(vals[0]), float(vals[0])
    mean = float(np.mean(vals))
    sem = float(stats.sem(vals))
    half = float(stats.t.ppf(0.975, n - 1) * sem)
    return mean - half, mean + half


def canonical_row(row: dict[str, Any], variant: str, source_file: str, note: str) -> dict[str, Any]:
    """Normalize metric names across existing result formats."""
    return {
        "variant": variant,
        "run": row.get("run", row.get("run_id")),
        "seed": row.get("seed", row.get("random_seed")),
        "actor_precision": row.get("actor_precision", row.get("precision")),
        "actor_recall": row.get("actor_recall", row.get("recall")),
        "actor_f1": row.get("actor_f1", row.get("f1")),
        "ttd_avg": row.get("ttd_avg", row.get("ttd_avg_steps", row.get("ttd_avg_conf"))),
        "ttd_max": row.get("ttd_max", row.get("ttd_max_steps", row.get("ttd_max_conf"))),
        "confirmed_alerts": row.get("confirmed_alerts", row.get("conf_total", row.get("conf_alerts_total"))),
        "confirmed_alert_precision": row.get(
            "confirmed_alert_precision", row.get("conf_prec", row.get("conf_alert_precision"))
        ),
        "confirmed_fp_per_run": row.get("confirmed_fp_per_run", row.get("conf_fp", row.get("conf_alerts_fp"))),
        "source_file": source_file,
        "source_note": note,
    }


def load_eg_siem_scalability() -> list[dict[str, Any]]:
    path = REPO_ROOT / "results" / "scalability" / "mesa_scalability_raw.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    needed = {"agent_count_human", "simulation_steps", "status"}
    if not needed.issubset(df.columns):
        return []
    subset = df[
        (df["agent_count_human"] == 42)
        & (df["simulation_steps"] == 240)
        & (df["status"].astype(str).str.lower() == "completed")
    ]
    note = "EG-SIEM rich config from scalability runner, 42 human agents, 240 steps; 3 run-level rows."
    return [canonical_row(r, "EG-SIEM", str(path), note) for r in subset.to_dict("records")]


def load_eg_siem_enron_json() -> list[dict[str, Any]]:
    path = REPO_ROOT / "results_eg_siem_enron_fixed.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    per_run = data.get("per_run")
    if not isinstance(per_run, list):
        return []
    cfg = data.get("config", {})
    note = (
        "EG-SIEM-Enron run-level JSON. Important: config reports "
        f"preset={cfg.get('preset')!r}, forensics_mode={cfg.get('forensics_mode')!r}, "
        f"population_kwargs={cfg.get('population_kwargs')!r}; compare cautiously."
    )
    return [canonical_row(r, "EG-SIEM-Enron", str(path), note) for r in per_run]


def load_existing_run_level() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.extend(load_eg_siem_scalability())
    rows.extend(load_eg_siem_enron_json())
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for metric in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    return df


def write_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not df.empty:
        for variant, part in df.groupby("variant"):
            for metric in METRICS:
                vals = part[metric].dropna()
                lo, hi = ci95(vals)
                rows.append(
                    {
                        "variant": variant,
                        "metric": metric,
                        "n": int(vals.shape[0]),
                        "mean": float(vals.mean()) if not vals.empty else math.nan,
                        "std": float(vals.std(ddof=1)) if vals.shape[0] > 1 else 0.0 if vals.shape[0] == 1 else math.nan,
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "source_files": "; ".join(sorted(part["source_file"].dropna().unique())),
                        "source_note": " | ".join(sorted(part["source_note"].dropna().unique())),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "summary_mean_sd_ci.csv", index=False)
    return out


def holm_bonferroni(pvals: list[float]) -> list[float]:
    """Return adjusted p-values in original order."""
    if not pvals:
        return []
    order = np.argsort(pvals)
    adjusted = [math.nan] * len(pvals)
    running = 0.0
    m = len(pvals)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * pvals[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def run_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    raw_p_indexes = []
    raw_p_values = []
    for left, right in COMPARISONS:
        for metric in METRICS:
            ldf = df[df["variant"] == left]
            rdf = df[df["variant"] == right]
            base = {"comparison": f"{left} vs {right}", "metric": metric}
            if ldf.empty or rdf.empty:
                rows.append(
                    {
                        **base,
                        "test_family": "not_run",
                        "test_name": "missing_run_level_data",
                        "paired": False,
                        "n_left": int(ldf.shape[0]),
                        "n_right": int(rdf.shape[0]),
                        "n_pairs": 0,
                        "statistic": math.nan,
                        "p_value": math.nan,
                        "p_holm": math.nan,
                        "valid": False,
                        "note": "Run-level records missing for one or both variants; aggregate tables were not used.",
                    }
                )
                continue
            merged = ldf[["seed", metric]].merge(rdf[["seed", metric]], on="seed", suffixes=("_left", "_right"))
            same_setup = left == "EG-SIEM" and right == "EG-SIEM-Enron"
            setup_note = (
                "Matched seeds exist, but EG-SIEM-Enron JSON uses a forensics_primary preset and n_takeover=0, "
                "so paired comparison to EG-SIEM scalability rows is not design-matched."
                if same_setup
                else ""
            )
            if merged.shape[0] >= 2 and not same_setup:
                diffs = merged[f"{metric}_left"] - merged[f"{metric}_right"]
                try:
                    wil = stats.wilcoxon(merged[f"{metric}_left"], merged[f"{metric}_right"])
                    rows.append(
                        {
                            **base,
                            "test_family": "paired",
                            "test_name": "wilcoxon_signed_rank",
                            "paired": True,
                            "n_left": int(ldf.shape[0]),
                            "n_right": int(rdf.shape[0]),
                            "n_pairs": int(merged.shape[0]),
                            "statistic": float(wil.statistic),
                            "p_value": float(wil.pvalue),
                            "p_holm": math.nan,
                            "valid": True,
                            "note": "Matched seeds available.",
                        }
                    )
                    raw_p_indexes.append(len(rows) - 1)
                    raw_p_values.append(float(wil.pvalue))
                except ValueError as exc:
                    rows.append({**base, "test_family": "paired", "test_name": "wilcoxon_signed_rank", "paired": True,
                                 "n_left": int(ldf.shape[0]), "n_right": int(rdf.shape[0]), "n_pairs": int(merged.shape[0]),
                                 "statistic": math.nan, "p_value": math.nan, "p_holm": math.nan, "valid": False,
                                 "note": f"Wilcoxon not valid: {exc}"})
                tt = stats.ttest_rel(merged[f"{metric}_left"], merged[f"{metric}_right"])
                rows.append(
                    {
                        **base,
                        "test_family": "paired",
                        "test_name": "paired_t_test",
                        "paired": True,
                        "n_left": int(ldf.shape[0]),
                        "n_right": int(rdf.shape[0]),
                        "n_pairs": int(merged.shape[0]),
                        "statistic": float(tt.statistic),
                        "p_value": float(tt.pvalue),
                        "p_holm": math.nan,
                        "valid": True,
                        "note": "Matched seeds available.",
                    }
                )
                raw_p_indexes.append(len(rows) - 1)
                raw_p_values.append(float(tt.pvalue))
            else:
                rows.append(
                    {
                        **base,
                        "test_family": "not_run",
                        "test_name": "paired_tests_not_valid",
                        "paired": False,
                        "n_left": int(ldf.shape[0]),
                        "n_right": int(rdf.shape[0]),
                        "n_pairs": int(merged.shape[0]),
                        "statistic": math.nan,
                        "p_value": math.nan,
                        "p_holm": math.nan,
                        "valid": False,
                        "note": setup_note or "Matched seeds unavailable or fewer than two matched pairs.",
                    }
                )
    adjusted = holm_bonferroni(raw_p_values)
    for row_idx, adj in zip(raw_p_indexes, adjusted):
        rows[row_idx]["p_holm"] = adj
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "statistical_tests.csv", index=False)
    return out


def plot_box(df: pd.DataFrame, metric: str, filename: str, ylabel: str) -> None:
    plt.figure(figsize=(7.5, 4.5))
    if df.empty or metric not in df.columns:
        plt.text(0.5, 0.5, "No run-level data found", ha="center", va="center")
        plt.axis("off")
    else:
        variants = [v for v in ["LSC", "CE-SIEM", "EG-SIEM", "EG-SIEM-Enron"] if v in set(df["variant"])]
        data = [df[df["variant"] == v][metric].dropna().to_numpy() for v in variants]
        if data:
            plt.boxplot(data, labels=variants, showmeans=True)
            plt.ylabel(ylabel)
            plt.title(ylabel + " by variant")
            plt.grid(axis="y", alpha=0.25)
        else:
            plt.text(0.5, 0.5, "No run-level data found", ha="center", va="center")
            plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{filename}.png", dpi=200)
    plt.savefig(OUT_DIR / f"{filename}.pdf")
    plt.close()


def write_markdown(df: pd.DataFrame, summary: pd.DataFrame, tests: pd.DataFrame) -> None:
    found = sorted(df["variant"].unique()) if not df.empty else []
    missing = [v for v in ["LSC", "CE-SIEM", "EG-SIEM", "EG-SIEM-Enron"] if v not in found]
    paired_valid = bool(tests["valid"].any()) if not tests.empty else False
    lines = [
        "# Statistical Rigor Summary",
        "",
        "## Repository State",
        "",
        "Initial `git status --short` could not run because this folder is not a Git repository.",
        "",
        "## Run-Level Data Found",
        "",
        f"- Variants with run-level records: {', '.join(found) if found else 'None'}",
        f"- Variants missing run-level records: {', '.join(missing) if missing else 'None'}",
        "",
        "Files loaded:",
    ]
    if df.empty:
        lines.append("- No run-level files with required metrics were found.")
    else:
        for source in sorted(df["source_file"].dropna().unique()):
            lines.append(f"- `{source}`")
    lines.extend(
        [
            "",
            "## Validity of Paired Tests",
            "",
            (
                "Paired tests were valid for at least one comparison."
                if paired_valid
                else "Paired tests were not valid for the requested main comparisons because LSC and CE-SIEM run-level logs were not found, and the available EG-SIEM-Enron run-level JSON uses a different preset/population than the EG-SIEM scalability rows."
            ),
            "",
            "No statistics were inferred from aggregate-only tables.",
            "",
            "## Outputs",
            "",
            "- `run_level_metrics_combined.csv`",
            "- `summary_mean_sd_ci.csv`",
            "- `statistical_tests.csv`",
            "- `actor_f1_boxplot.png` and `.pdf`",
            "- `ttd_boxplot.png` and `.pdf`",
            "",
            "## Rerun Needed For Full Reviewer Statistics",
            "",
            "To compute valid SD/CI/Wilcoxon/paired t-tests for LSC vs CE-SIEM vs EG-SIEM vs EG-SIEM-Enron, rerun all four variants with matched seeds, the same 42-human population, 240 steps, 60 warm-up steps, and identical attack scenarios, and save one row per run with the required metrics.",
        ]
    )
    (OUT_DIR / "statistical_rigor_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = load_existing_run_level()
    df.to_csv(OUT_DIR / "run_level_metrics_combined.csv", index=False)
    summary = write_summary(df)
    tests = run_tests(df)
    plot_box(df, "actor_f1", "actor_f1_boxplot", "Actor F1")
    plot_box(df, "ttd_avg", "ttd_boxplot", "TTD average (steps)")
    write_markdown(df, summary, tests)
    print(f"Wrote statistical rigor outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
