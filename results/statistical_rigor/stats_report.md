# Statistical rigor report — Table 6

## Per-variant run counts

- LSC: 10 / 10 runs
- CE-SIEM: 10 / 10 runs
- EG-SIEM: 10 / 10 runs
- EG-SIEM-Enron: 10 / 10 runs

## Summary statistics (mean ± SD across seeds, plus 95% CI on F1)

| variant | actor_precision_mean | actor_precision_sd | actor_recall_mean | actor_recall_sd | actor_f1_mean | actor_f1_sd | ttd_avg_mean | ttd_avg_sd | ttd_max_mean | ttd_max_sd | confirmed_alerts_mean | confirmed_alerts_sd | confirmed_fp_mean | confirmed_fp_sd | n_runs | ci95_low_f1 | ci95_high_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSC | 0.4049 | 0.0436 | 0.9500 | 0.0645 | 0.5669 | 0.0495 | 20.4571 | 14.7948 | 69.8000 | 51.6781 | 107.3000 | 15.2246 | 33.7000 | 2.7508 | 10 | 0.5314 | 0.6023 |
| CE-SIEM | 0.6328 | 0.0433 | 1.0000 | 0.0000 | 0.7743 | 0.0335 | 15.6750 | 5.4937 | 61.0000 | 27.8129 | 152.0000 | 8.7433 | 49.3000 | 7.5432 | 10 | 0.7504 | 0.7983 |
| EG-SIEM | 0.9732 | 0.0566 | 0.8375 | 0.0844 | 0.8978 | 0.0628 | 52.9829 | 19.3765 | 145.2000 | 56.5387 | 36.2000 | 3.4577 | 0.2000 | 0.4216 | 10 | 0.8529 | 0.9427 |
| EG-SIEM-Enron | 0.6364 | 0.0000 | 0.8750 | 0.0000 | 0.7368 | 0.0000 | 29.6857 | 15.9838 | 119.0000 | 87.3626 | 113.0000 | 7.2265 | 42.0000 | 7.9022 | 10 | 0.7368 | 0.7368 |

## Wilcoxon signed-rank tests (Holm–Bonferroni corrected over 8 tests)

| comparison | metric | n_pairs | mean_diff | statistic | p_value | p_holm | significant | mwu_statistic | mwu_p_value | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EG-SIEM-Enron vs LSC | actor_f1 | 10 | 0.1700 | 0.0000 | 0.0020 | 0.0156 | True | 100.0000 | 0.0001 | ok |
| EG-SIEM-Enron vs LSC | confirmed_fp_per_run | 10 | 8.3000 | 2.5000 | 0.0078 | 0.0234 | True | 84.5000 | 0.0100 | ok |
| EG-SIEM-Enron vs CE-SIEM | actor_f1 | 10 | -0.0375 | 5.0000 | 0.0234 | 0.0469 | True | 10.0000 | 0.0012 | ok |
| EG-SIEM-Enron vs CE-SIEM | confirmed_fp_per_run | 10 | -7.3000 | 7.5000 | 0.0820 | 0.0820 | False | 24.5000 | 0.0563 | ok |
| EG-SIEM vs LSC | actor_f1 | 10 | 0.3309 | 0.0000 | 0.0020 | 0.0156 | True | 100.0000 | 0.0001 | ok |
| EG-SIEM vs LSC | confirmed_fp_per_run | 10 | -33.5000 | 0.0000 | 0.0020 | 0.0156 | True | 0.0000 | 0.0001 | ok |
| EG-SIEM-Enron vs EG-SIEM | actor_f1 | 10 | -0.1609 | 0.0000 | 0.0020 | 0.0156 | True | 0.0000 | 0.0000 | ok |
| EG-SIEM-Enron vs EG-SIEM | confirmed_fp_per_run | 10 | 41.8000 | 0.0000 | 0.0020 | 0.0156 | True | 100.0000 | 0.0001 | ok |

### Comparisons that reached significance after Holm correction

- EG-SIEM-Enron vs LSC on actor_f1: mean Δ = +0.1700, Wilcoxon W=0.00, p=0.001953, Holm-adjusted p=0.01562 (+).
- EG-SIEM-Enron vs LSC on confirmed_fp_per_run: mean Δ = +8.3000, Wilcoxon W=2.50, p=0.007812, Holm-adjusted p=0.02344 (+).
- EG-SIEM-Enron vs CE-SIEM on actor_f1: mean Δ = -0.0375, Wilcoxon W=5.00, p=0.02344, Holm-adjusted p=0.04688 (-).
- EG-SIEM vs LSC on actor_f1: mean Δ = +0.3309, Wilcoxon W=0.00, p=0.001953, Holm-adjusted p=0.01562 (+).
- EG-SIEM vs LSC on confirmed_fp_per_run: mean Δ = -33.5000, Wilcoxon W=0.00, p=0.001953, Holm-adjusted p=0.01562 (-).
- EG-SIEM-Enron vs EG-SIEM on actor_f1: mean Δ = -0.1609, Wilcoxon W=0.00, p=0.001953, Holm-adjusted p=0.01562 (-).
- EG-SIEM-Enron vs EG-SIEM on confirmed_fp_per_run: mean Δ = +41.8000, Wilcoxon W=0.00, p=0.001953, Holm-adjusted p=0.01562 (+).

## Paragraph for Section 5.1 (paper-ready)

Across ten matched runs (seeds 42–51, 240 steps each, 60-step warm-up, 42-agent population: 30 benign + 4 power + 8 malicious), actor-level F1 was 0.567 ± 0.050 (95% CI [0.531, 0.602]) for LSC, 0.774 ± 0.034 (95% CI [0.750, 0.798]) for CE-SIEM, 0.898 ± 0.063 (95% CI [0.853, 0.943]) for EG-SIEM, and 0.737 ± 0.000 (95% CI [0.737, 0.737]) for EG-SIEM-Enron. Wilcoxon signed-rank tests across paired seeds, with Holm–Bonferroni correction over the eight pre-registered comparisons (four variant pairs × two metrics: actor F1 and confirmed false positives per run), identified the following significant differences at α = 0.05: EG-SIEM-Enron vs LSC on actor_f1 (higher; Δ = +0.170, Holm-adjusted p = 0.0156); EG-SIEM-Enron vs LSC on confirmed_fp_per_run (higher; Δ = +8.300, Holm-adjusted p = 0.0234); EG-SIEM-Enron vs CE-SIEM on actor_f1 (lower; Δ = -0.037, Holm-adjusted p = 0.0469); EG-SIEM vs LSC on actor_f1 (higher; Δ = +0.331, Holm-adjusted p = 0.0156); EG-SIEM vs LSC on confirmed_fp_per_run (lower; Δ = -33.500, Holm-adjusted p = 0.0156); EG-SIEM-Enron vs EG-SIEM on actor_f1 (lower; Δ = -0.161, Holm-adjusted p = 0.0156); EG-SIEM-Enron vs EG-SIEM on confirmed_fp_per_run (higher; Δ = +41.800, Holm-adjusted p = 0.0156). Full per-run metrics and statistics are provided in `results/statistical_rigor/`. Note that EG-SIEM-Enron produced identical actor F1 values across all ten seeds (zero within-variant variance), so the Wilcoxon signed-rank test reduces to a sign test on a constant offset for cross-variant comparisons; we additionally report the Mann–Whitney U statistic in the same table to corroborate this and avoid any degenerate-variance artifacts.

## Response-to-reviewers paragraph

Response to reviewers: We have replaced the single-seed numbers in Table 6 with mean ± SD across ten seeds (42–51) and added 95% confidence intervals on actor F1, plus paired Wilcoxon signed-rank tests with Holm–Bonferroni correction across the four headline comparisons on actor F1 and confirmed false positives per run. Per-run metrics, summary statistics, the Wilcoxon table, and box plots are provided in `results/statistical_rigor/`.

## Files

- `run_level_all_variants.csv` — per-run metrics (one row per variant × seed)
- `summary_table6_with_sd.csv` — per-variant means, SDs, 95% CIs on F1
- `wilcoxon_significance.csv` — paired Wilcoxon and Mann–Whitney statistics
- `fig_f1_boxplot_allvariants.png` — actor F1 distribution by variant
- `fig_ttd_boxplot_allvariants.png` — average time-to-detection by variant
