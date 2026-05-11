# Statistical Rigor Summary

## Repository State

Initial `git status --short` could not run because this folder is not a Git repository.

## Run-Level Data Found

- Variants with run-level records: EG-SIEM, EG-SIEM-Enron
- Variants missing run-level records: LSC, CE-SIEM

Files loaded:
- `/Users/firdous/Documents/SmartCitiesSimulation/Insider-Threat-Detection-MAS-SIEM-main/results/scalability/mesa_scalability_raw.csv`
- `/Users/firdous/Documents/SmartCitiesSimulation/Insider-Threat-Detection-MAS-SIEM-main/results_eg_siem_enron_fixed.json`

## Validity of Paired Tests

Paired tests were not valid for the requested main comparisons because LSC and CE-SIEM run-level logs were not found, and the available EG-SIEM-Enron run-level JSON uses a different preset/population than the EG-SIEM scalability rows.

No statistics were inferred from aggregate-only tables.

## Outputs

- `run_level_metrics_combined.csv`
- `summary_mean_sd_ci.csv`
- `statistical_tests.csv`
- `actor_f1_boxplot.png` and `.pdf`
- `ttd_boxplot.png` and `.pdf`

## Rerun Needed For Full Reviewer Statistics

To compute valid SD/CI/Wilcoxon/paired t-tests for LSC vs CE-SIEM vs EG-SIEM vs EG-SIEM-Enron, rerun all four variants with matched seeds, the same 42-human population, 240 steps, 60 warm-up steps, and identical attack scenarios, and save one row per run with the required metrics.