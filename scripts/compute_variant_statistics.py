"""Compute variant-level statistics from per-run results.

Reads:
  results/statistical_rigor/run_level_all_variants.csv

Writes:
  results/statistical_rigor/summary_table6_with_sd.csv
  results/statistical_rigor/wilcoxon_significance.csv
  results/statistical_rigor/fig_f1_boxplot_allvariants.png
  results/statistical_rigor/fig_ttd_boxplot_allvariants.png
  results/statistical_rigor/stats_report.md

Statistical conventions:
  - Means and SDs over the per-run rows (n = number of seeds run).
  - 95% CI on actor_f1 via scipy.stats.t.interval (two-sided, n-1 df).
  - Wilcoxon signed-rank tests on actor_f1 and confirmed_fp_per_run for the
    four configured variant comparisons (8 tests in total).
  - Holm-Bonferroni correction across the 8 tests.
  - All-zero-difference and all-equal pair handling: catch ValueError from
    scipy.stats.wilcoxon and emit p_value=1.0, significant=False, plus a flag.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
RIGOR_DIR = os.path.join(REPO_ROOT, "results", "statistical_rigor")

VARIANT_ORDER = ["LSC", "CE-SIEM", "EG-SIEM", "EG-SIEM-Enron"]

COMPARISONS: List[Tuple[str, str]] = [
    ("EG-SIEM-Enron", "LSC"),
    ("EG-SIEM-Enron", "CE-SIEM"),
    ("EG-SIEM", "LSC"),
    ("EG-SIEM-Enron", "EG-SIEM"),
]
METRICS_FOR_TESTS = ["actor_f1", "confirmed_fp_per_run"]


def _holm_bonferroni(pvals: List[float]) -> List[float]:
    """Holm-Bonferroni correction. Returns adjusted p-values, capped at 1.0."""
    n = len(pvals)
    order = np.argsort(pvals)
    adj = [0.0] * n
    running_max = 0.0
    for rank, idx in enumerate(order):
        # Holm: multiply by (n - rank); enforce monotonicity.
        val = pvals[idx] * (n - rank)
        running_max = max(running_max, val)
        adj[idx] = min(1.0, running_max)
    return adj


def _safe_wilcoxon(x: np.ndarray, y: np.ndarray):
    """Return (statistic, p_value, note). Handles the all-zero-difference edge case."""
    diffs = np.asarray(x) - np.asarray(y)
    if len(diffs) == 0:
        return float("nan"), 1.0, "empty_input"
    if np.allclose(diffs, 0.0):
        return float("nan"), 1.0, "all_zero_differences"
    try:
        res = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        return float(res.statistic), float(res.pvalue), "ok"
    except ValueError as exc:
        return float("nan"), 1.0, f"wilcoxon_error:{exc}"


def _safe_mannwhitney(x: np.ndarray, y: np.ndarray):
    """Mann-Whitney U as fallback when within-variant variance is zero."""
    try:
        res = stats.mannwhitneyu(x, y, alternative="two-sided")
        return float(res.statistic), float(res.pvalue), "ok"
    except ValueError as exc:
        return float("nan"), 1.0, f"mwu_error:{exc}"


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in VARIANT_ORDER:
        sub = df[df["variant"] == variant]
        if sub.empty:
            continue
        n = len(sub)
        f1_mean = float(sub["actor_f1"].mean())
        f1_sd = float(sub["actor_f1"].std(ddof=1)) if n > 1 else 0.0
        # 95% CI on F1 (Student's t)
        if n > 1 and f1_sd > 0:
            ci_low, ci_high = stats.t.interval(
                0.95, df=n - 1, loc=f1_mean, scale=f1_sd / np.sqrt(n)
            )
        else:
            ci_low, ci_high = f1_mean, f1_mean
        rows.append({
            "variant": variant,
            "actor_precision_mean": float(sub["actor_precision"].mean()),
            "actor_precision_sd": float(sub["actor_precision"].std(ddof=1)) if n > 1 else 0.0,
            "actor_recall_mean": float(sub["actor_recall"].mean()),
            "actor_recall_sd": float(sub["actor_recall"].std(ddof=1)) if n > 1 else 0.0,
            "actor_f1_mean": f1_mean,
            "actor_f1_sd": f1_sd,
            "ttd_avg_mean": float(sub["ttd_avg"].mean()),
            "ttd_avg_sd": float(sub["ttd_avg"].std(ddof=1)) if n > 1 else 0.0,
            "ttd_max_mean": float(sub["ttd_max"].mean()),
            "ttd_max_sd": float(sub["ttd_max"].std(ddof=1)) if n > 1 else 0.0,
            "confirmed_alerts_mean": float(sub["confirmed_alerts"].mean()),
            "confirmed_alerts_sd": float(sub["confirmed_alerts"].std(ddof=1)) if n > 1 else 0.0,
            "confirmed_fp_mean": float(sub["confirmed_fp_per_run"].mean()),
            "confirmed_fp_sd": float(sub["confirmed_fp_per_run"].std(ddof=1)) if n > 1 else 0.0,
            "n_runs": int(n),
            "ci95_low_f1": float(ci_low),
            "ci95_high_f1": float(ci_high),
        })
    return pd.DataFrame(rows)


def build_wilcoxon_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build a long table: comparison x metric with Wilcoxon stats and Holm-adjusted p-values.

    When the within-variant variance is zero on a metric, we additionally report
    a Mann-Whitney U cross-variant statistic in the `note` column so the reader
    can verify the conclusion under an alternative test.
    """
    rows = []
    pvals = []
    for (a, b) in COMPARISONS:
        sub_a = df[df["variant"] == a].sort_values("seed")
        sub_b = df[df["variant"] == b].sort_values("seed")
        # Pair by shared seeds.
        seeds = sorted(set(sub_a["seed"]).intersection(set(sub_b["seed"])))
        sub_a = sub_a[sub_a["seed"].isin(seeds)].sort_values("seed").reset_index(drop=True)
        sub_b = sub_b[sub_b["seed"].isin(seeds)].sort_values("seed").reset_index(drop=True)
        for metric in METRICS_FOR_TESTS:
            x = sub_a[metric].to_numpy(dtype=float)
            y = sub_b[metric].to_numpy(dtype=float)
            stat, pval, note = _safe_wilcoxon(x, y)
            mean_diff = float(np.mean(x) - np.mean(y))
            mwu_stat, mwu_p, mwu_note = _safe_mannwhitney(x, y)
            rows.append({
                "comparison": f"{a} vs {b}",
                "metric": metric,
                "n_pairs": int(len(seeds)),
                "mean_diff": mean_diff,
                "statistic": stat,
                "p_value": pval,
                "note": note,
                "mwu_statistic": mwu_stat,
                "mwu_p_value": mwu_p,
            })
            pvals.append(pval)
    p_holm = _holm_bonferroni(pvals)
    for i, r in enumerate(rows):
        r["p_holm"] = float(p_holm[i])
        r["significant"] = bool(p_holm[i] < 0.05)
    return pd.DataFrame(rows)


def make_boxplot(df: pd.DataFrame, metric: str, ylabel: str, out_path: str, title: str):
    data = []
    labels = []
    for variant in VARIANT_ORDER:
        sub = df[df["variant"] == variant][metric].to_numpy()
        if len(sub):
            data.append(sub)
            labels.append(variant)

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    box_colors = ["#cccccc", "#bcd4e6", "#c6e2b3", "#f5cb88"]
    for patch, color in zip(bp["boxes"], box_colors[:len(bp["boxes"])]):
        patch.set_facecolor(color)
        patch.set_edgecolor("#333333")
    for median in bp["medians"]:
        median.set_color("#333333")
        median.set_linewidth(2)

    rng = np.random.default_rng(0)
    for i, arr in enumerate(data, start=1):
        x_jit = rng.uniform(-0.10, 0.10, size=len(arr)) + i
        ax.scatter(x_jit, arr, s=22, color="#222222", alpha=0.75, zorder=3,
                   edgecolor="white", linewidth=0.5)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def df_to_md(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    """Render a small DataFrame as a Markdown table without external deps."""
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |",
           "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(float_fmt.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def write_report(summary: pd.DataFrame, wilcoxon: pd.DataFrame,
                 run_counts: pd.Series, n_seeds_expected: int,
                 zero_var_variants: List[str], out_path: str):
    significant_lines = []
    for _, r in wilcoxon.iterrows():
        if r["significant"]:
            sign = "+" if r["mean_diff"] > 0 else "-"
            significant_lines.append(
                f"- {r['comparison']} on {r['metric']}: mean Δ = {r['mean_diff']:+.4f}, "
                f"Wilcoxon W={r['statistic']:.2f}, p={r['p_value']:.4g}, "
                f"Holm-adjusted p={r['p_holm']:.4g} ({sign})."
            )

    significant_text = "\n".join(significant_lines) if significant_lines else (
        "_No comparison reached significance after Holm correction at α = 0.05._"
    )

    # CI strings keyed by variant for the paper paragraph.
    ci_strings = {}
    for _, r in summary.iterrows():
        ci_strings[r["variant"]] = (
            f"{r['actor_f1_mean']:.3f} ± {r['actor_f1_sd']:.3f} "
            f"(95% CI [{r['ci95_low_f1']:.3f}, {r['ci95_high_f1']:.3f}])"
        )

    # Build a data-driven significant-effects sentence so we never overstate.
    sig_effects = []
    for _, r in wilcoxon.iterrows():
        if not r["significant"]:
            continue
        direction = "higher" if r["mean_diff"] > 0 else "lower"
        sig_effects.append(
            f"{r['comparison']} on {r['metric']} ({direction}; "
            f"Δ = {r['mean_diff']:+.3f}, Holm-adjusted p = {r['p_holm']:.3g})"
        )
    sig_text = "; ".join(sig_effects) if sig_effects else "no comparisons survived correction"

    paper_paragraph = (
        "Across ten matched runs (seeds 42–51, 240 steps each, 60-step warm-up, "
        "42-agent population: 30 benign + 4 power + 8 malicious), actor-level F1 was "
        f"{ci_strings.get('LSC','n/a')} for LSC, "
        f"{ci_strings.get('CE-SIEM','n/a')} for CE-SIEM, "
        f"{ci_strings.get('EG-SIEM','n/a')} for EG-SIEM, and "
        f"{ci_strings.get('EG-SIEM-Enron','n/a')} for EG-SIEM-Enron. "
        "Wilcoxon signed-rank tests across paired seeds, with Holm–Bonferroni "
        "correction over the eight pre-registered comparisons (four variant pairs × "
        "two metrics: actor F1 and confirmed false positives per run), identified "
        f"the following significant differences at α = 0.05: {sig_text}. "
        "Full per-run metrics and statistics are provided in "
        "`results/statistical_rigor/`."
    )

    response_paragraph = (
        "Response to reviewers: We have replaced the single-seed numbers in Table 6 "
        "with mean ± SD across ten seeds (42–51) and added 95% confidence intervals "
        "on actor F1, plus paired Wilcoxon signed-rank tests with Holm–Bonferroni "
        "correction across the four headline comparisons on actor F1 and confirmed "
        "false positives per run. Per-run metrics, summary statistics, the Wilcoxon "
        "table, and box plots are provided in `results/statistical_rigor/`."
    )

    if zero_var_variants:
        paper_paragraph += (
            f" Note that {', '.join(zero_var_variants)} produced identical actor F1 "
            "values across all ten seeds (zero within-variant variance), so the "
            "Wilcoxon signed-rank test reduces to a sign test on a constant offset "
            "for cross-variant comparisons; we additionally report the Mann–Whitney "
            "U statistic in the same table to corroborate this and avoid any "
            "degenerate-variance artifacts."
        )

    counts_lines = "\n".join(
        f"- {variant}: {int(run_counts.get(variant, 0))} / {n_seeds_expected} runs"
        for variant in VARIANT_ORDER
    )

    summary_md_cols = [
        "variant", "actor_precision_mean", "actor_precision_sd",
        "actor_recall_mean", "actor_recall_sd",
        "actor_f1_mean", "actor_f1_sd",
        "ttd_avg_mean", "ttd_avg_sd",
        "ttd_max_mean", "ttd_max_sd",
        "confirmed_alerts_mean", "confirmed_alerts_sd",
        "confirmed_fp_mean", "confirmed_fp_sd",
        "n_runs", "ci95_low_f1", "ci95_high_f1",
    ]
    summary_md = df_to_md(summary[summary_md_cols])

    wilcoxon_md_cols = [
        "comparison", "metric", "n_pairs", "mean_diff",
        "statistic", "p_value", "p_holm", "significant",
        "mwu_statistic", "mwu_p_value", "note",
    ]
    wilcoxon_md = df_to_md(wilcoxon[wilcoxon_md_cols])

    text = f"""# Statistical rigor report — Table 6

## Per-variant run counts

{counts_lines}

## Summary statistics (mean ± SD across seeds, plus 95% CI on F1)

{summary_md}

## Wilcoxon signed-rank tests (Holm–Bonferroni corrected over 8 tests)

{wilcoxon_md}

### Comparisons that reached significance after Holm correction

{significant_text}

## Paragraph for Section 5.1 (paper-ready)

{paper_paragraph}

## Response-to-reviewers paragraph

{response_paragraph}

## Files

- `run_level_all_variants.csv` — per-run metrics (one row per variant × seed)
- `summary_table6_with_sd.csv` — per-variant means, SDs, 95% CIs on F1
- `wilcoxon_significance.csv` — paired Wilcoxon and Mann–Whitney statistics
- `fig_f1_boxplot_allvariants.png` — actor F1 distribution by variant
- `fig_ttd_boxplot_allvariants.png` — average time-to-detection by variant
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=os.path.join(RIGOR_DIR, "run_level_all_variants.csv"))
    p.add_argument("--rigor-dir", default=RIGOR_DIR)
    p.add_argument("--n-seeds-expected", type=int, default=10)
    args = p.parse_args(argv)

    if not os.path.exists(args.input):
        print(f"ERROR: input CSV not found: {args.input}")
        return 2

    df = pd.read_csv(args.input)
    if "status" in df.columns:
        df_ok = df[df["status"] == "ok"].copy()
    else:
        df_ok = df.copy()

    # Per-variant run counts (only successful)
    run_counts = df_ok.groupby("variant").size()
    failed = []
    for variant in VARIANT_ORDER:
        present = int(run_counts.get(variant, 0))
        if present < args.n_seeds_expected:
            sub = df[df["variant"] == variant]
            failed_seeds = []
            for _, row in sub.iterrows():
                if str(row.get("status", "ok")) != "ok":
                    failed_seeds.append((int(row["seed"]), str(row.get("error", ""))))
            missing_seeds = sorted(
                set(range(42, 42 + args.n_seeds_expected))
                - set(int(s) for s in sub["seed"].tolist())
            )
            failed.append((variant, present, failed_seeds, missing_seeds))
    if failed:
        print("WARNING: some variants did not have the expected number of successful runs:")
        for variant, present, failed_seeds, missing_seeds in failed:
            print(f"  - {variant}: {present}/{args.n_seeds_expected} runs")
            for seed, err in failed_seeds:
                print(f"      seed={seed} ERROR: {err}")
            if missing_seeds:
                print(f"      missing seeds (never attempted): {missing_seeds}")

    summary = build_summary_table(df_ok)

    # Detect zero-variance variants for the report's caveat block.
    # Use a small tolerance so float precision artefacts (e.g. 1.17e-16) still register.
    zero_var_variants = [
        row["variant"] for _, row in summary.iterrows()
        if row["actor_f1_sd"] < 1e-12 and row["n_runs"] > 1
    ]

    wilcoxon = build_wilcoxon_table(df_ok)

    os.makedirs(args.rigor_dir, exist_ok=True)
    summary.to_csv(os.path.join(args.rigor_dir, "summary_table6_with_sd.csv"), index=False)
    wilcoxon.to_csv(os.path.join(args.rigor_dir, "wilcoxon_significance.csv"), index=False)

    make_boxplot(
        df_ok, "actor_f1", "Actor F1",
        os.path.join(args.rigor_dir, "fig_f1_boxplot_allvariants.png"),
        "Actor F1 across SIEM variants (n=10 seeds each)",
    )
    make_boxplot(
        df_ok, "ttd_avg", "Average time-to-detection (steps)",
        os.path.join(args.rigor_dir, "fig_ttd_boxplot_allvariants.png"),
        "Average TTD across SIEM variants (n=10 seeds each)",
    )

    write_report(
        summary, wilcoxon, run_counts, args.n_seeds_expected, zero_var_variants,
        os.path.join(args.rigor_dir, "stats_report.md"),
    )

    print(f"Wrote summary_table6_with_sd.csv ({len(summary)} variants)")
    print(f"Wrote wilcoxon_significance.csv ({len(wilcoxon)} tests)")
    print("Wrote fig_f1_boxplot_allvariants.png")
    print("Wrote fig_ttd_boxplot_allvariants.png")
    print("Wrote stats_report.md")
    if zero_var_variants:
        print(f"Note: variants with zero F1 variance: {zero_var_variants}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
