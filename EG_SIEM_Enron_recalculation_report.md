# EG-SIEM-Enron recalculation report

## Scope
This report documents a code audit, pipeline correction, retraining pass, and EG-SIEM-Enron rerun performed on 2026-03-19.

## What was wrong in the original code

### Training pipeline issues
1. The phishing classifier fit TF-IDF before splitting, which leaked vocabulary statistics from held-out data into training.
2. The script selected the “best” classifier using held-out test performance instead of a separate validation workflow.
3. The original script did not report enough metrics for reproducible evaluation.
4. The dataset handling lacked a safeguard to stop duplicate/near-exact duplicate email text from crossing split boundaries.

### Simulation evaluation issues
1. The original actor precision calculation mixed actor counts with confirmed-alert counts:
   `n_det / (n_det + conf_fp)`.
2. That formula is not actor-level precision because `n_det` counts actors while `conf_fp` counts alerts.
3. The paper definition is actor-based: an actor is detected if they generate at least one confirmed alert during test.

## Exactly what changed

### `enron_combined_training.py`
- Split logic was changed to a reproducible train/validation/test workflow with `random_state=42`.
- Raw texts are split before the vectorizer is fit.
- TF-IDF is fit only on the training partition used for model selection, then refit on train+validation before the final untouched test evaluation.
- Multiple classifiers are still compared, but selection now uses validation F1 instead of final test performance.
- The script now reports accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and split sizes.
- Duplicate safeguards were added by grouping identical normalized texts and forcing group-wise splits so the same normalized email text cannot appear in both training/validation and test.
- The corrected model is saved to `combined_forensics_model_fixed.pkl` instead of overwriting the original artifact.

### `mini_mesa_EG-SIEM_Enron.py`
- Confirmed-alert precision remains alert-level.
- Actor-level metrics are now computed from actor sets:
  - `malicious_actor_ids`
  - `detected_actor_ids`
  - `false_detected_actor_ids`
- Output labels are now explicit:
  - `actor_precision`
  - `actor_recall`
  - `actor_f1`
  - `confirmed_alert_precision`
  - `confirmed_fp_per_run`
- Time-to-detection is still first malicious activity to first confirmed alert per malicious actor.
- The script now saves a machine-readable JSON bundle to `results_eg_siem_enron_fixed.json`.

## Data availability note
- `emails.csv` was **not** available in the repository during this rerun.
- Therefore, full-corpus baseline calibration was **not recomputed**.
- Retraining was limited to the phishing/spam classifier using `enron_spam_data.csv`.
- The simulator rerun still used the corrected classifier artifact plus the corrected evaluation logic.

## Duplicate leakage findings in `enron_spam_data.csv`
- Raw usable examples after the script’s minimum-length filter: **33,218**.
- Exact duplicate emails: **17,775**.
- Unique normalized-text groups: **15,123**.
- Conflicting-label duplicate groups: **6**.
- Normalized overlap between train and validation after grouping: **0**.
- Normalized overlap between train+validation and test after grouping: **0**.

Interpretation: duplicate leakage risk was real and substantial. The grouped split prevents the same normalized text from appearing across held-out boundaries.

## Original Table 3 EG-SIEM-Enron values
- Actor Precision: **1.000**
- Actor Recall: **0.875**
- Actor F1: **0.933**
- TTD avg: **10.26**
- TTD max: **35.4**
- Confirmed alerts/run: **73.4**
- Confirmed-alert precision: **1.000**
- Confirmed FP/run: **0.0**

## Newly recalculated values after the fixes
Configuration used:
- Runs: **10**
- Steps/run: **240**
- Warmup: **60**
- Seeds: **42-51**
- Model: `combined_forensics_model_fixed.pkl`

Recalculated EG-SIEM-Enron averages:
- Actor Precision: **1.0000**
- Actor Recall: **0.8750**
- Actor F1: **0.9333**
- TTD avg: **6.26**
- TTD max: **24.60**
- Confirmed alerts/run: **35.5**
- Confirmed-alert precision: **1.0000**
- Confirmed FP/run: **0.0**

## Side-by-side comparison

| Metric | Original Table 3 | Recalculated | Change |
|---|---:|---:|---:|
| Actor Precision | 1.000 | 1.0000 | 0.0000 |
| Actor Recall | 0.875 | 0.8750 | 0.0000 |
| Actor F1 | 0.933 | 0.9333 | +0.0003 |
| TTD avg | 10.26 | 6.26 | -4.00 |
| TTD max | 35.4 | 24.60 | -10.80 |
| Confirmed alerts/run | 73.4 | 35.5 | -37.9 |
| Confirmed-alert precision | 1.000 | 1.0000 | 0.0000 |
| Confirmed FP/run | 0.0 | 0.0 | 0.0 |

## Short interpretation
- The actor-level metrics did **not** change because the simulator already produced zero confirmed false-positive alerts in these runs, so the incorrect actor-precision formula happened to collapse to the same numeric value.
- The corrected classifier retraining did **not** degrade actor detection in the simulator, suggesting the SIEM logic is dominated by rule anchors and scenario scripting rather than classifier quality.
- Time-to-detection and confirmed-alert volume are lower than the original Table 3 values, so the corrected rerun should **not** be described as reproducing the paper’s original table.

## Training-model results
Final untouched spam/phishing test-set metrics from the corrected retraining pass:
- Accuracy: **0.5981**
- Precision: **0.0000**
- Recall: **0.0000**
- F1: **0.0000**
- ROC-AUC: **0.8217**
- Confusion matrix: **[[2351, 80], [1500, 0]]**

Interpretation: once grouped duplicate leakage is blocked, the local spam/phishing data appears much less separable than implied by the prior pipeline.

## Why the simulator still appears unrealistically easy
The rerun suggests that even after fixing the measurement and retraining path, the simulator still strongly favors high detection performance.

### Evidence gating thresholds
- The confirmed threshold is low relative to the strong rule weights.
- The SIEM only needs a minimum evidence count of 2 and minimum evidence weight of 2.5 before confirmation logic can fire.
- Several individual evidence features have weights at or above that scale.

### External-email anchors
- External email with attachments is directly treated as an anchor condition.
- Malicious scenarios repeatedly send to a hard-coded external destination like `external@outside` with large attachments.
- That makes exfiltration behavior easy to separate from most benign activity.

### Unapproved-recipient logic
- Benign users are provisioned with explicit approved partner lists.
- The SIEM flags external recipients not in those lists as `unapproved`.
- That is a powerful shortcut for the scripted scenarios because malicious actors typically email unapproved external recipients.

### Exfiltration chain logic
- The rules explicitly look for query/export/email sequences and can assign a dedicated `exfil_chain` feature.
- The malicious scenarios are scripted around that same chain, which tightly matches the detector’s assumptions.

### Small number of malicious actors and highly scripted scenarios
- The default test world only has eight malicious actors per run.
- Their behaviors are drawn from a small set of scenario templates (`exfil`, `stealth`, `acct_takeover`, `staging_exfil`, `email_only`).
- Those scripts are repetitive, sparse, and closely aligned with the SIEM feature engineering.

Overall conclusion: even after the requested fixes, the simulator design likely still inflates precision/recall because the environment, evidence features, and malicious scripts are strongly coupled.

## Output files
- Corrected training script: `enron_combined_training.py`
- Corrected simulation script: `mini_mesa_EG-SIEM_Enron.py`
- Corrected model artifact: `combined_forensics_model_fixed.pkl` (generated by training, not committed to the repo)
- Machine-readable rerun results: `results_eg_siem_enron_fixed.json`
- This report: `EG_SIEM_Enron_recalculation_report.md`

## Final corrected EG-SIEM-Enron metrics
- actor_precision: **1.0000**
- actor_recall: **0.8750**
- actor_f1: **0.9333**
- ttd_avg: **6.26**
- ttd_max: **24.60**
- confirmed_alerts/run: **35.5**
- confirmed_alert_precision: **1.0000**
- confirmed_fp_per_run: **0.0**


## Marginal contribution rerun (minimal simulator patch)
A minimal simulator patch added explicit forensics ablation modes so the same experiment can be rerun as `full`, `keyword_only`, `model_only`, or `disabled`.

Average results across 10 runs:

| Forensics mode | Actor Precision | Actor Recall | Actor F1 | TTD avg | Confirmed alerts/run | Notes |
|---|---:|---:|---:|---:|---:|---|
| full | 1.0000 | 0.8750 | 0.9333 | 6.26 | 35.5 | model + keywords |
| keyword_only | 1.0000 | 0.8750 | 0.9333 | 6.26 | 35.5 | keywords only |
| model_only | 1.0000 | 0.8750 | 0.9333 | 6.26 | 35.5 | learned model only |
| disabled | 1.0000 | 0.8750 | 0.9333 | 7.03 | 34.9 | no forensics feature |

Interpretation:
- `full`, `keyword_only`, and `model_only` are numerically identical in this simulator configuration.
- Disabling forensics entirely only slightly worsens average TTD (6.26 -> 7.03) and confirmed alerts/run (35.5 -> 34.9), while actor detection remains unchanged.
- So the learned model's **marginal contribution is effectively zero** in the current simulator, and the broader forensics channel contributes only a small timing/volume effect on top of the rule-based detector.

Additional output file:
- Marginal-ablation bundle: `results_eg_siem_enron_marginal.json`


## Forensics-primary recalculation
To measure the simulator mainly on the forensic channel rather than the other rule families, the simulator now supports a `forensics_primary` preset. This preset disables policy/ToM/other score families, keeps only email-capable malicious scenarios (`n_takeover=0`), and scores detections mainly from `anchor_email` plus `forensics_phishing`.

Average results across 10 runs for the `forensics_primary` preset:

| Forensics mode | Actor Precision | Actor Recall | Actor F1 | TTD avg | Confirmed alerts/run | Detected actors/run |
|---|---:|---:|---:|---:|---:|---:|
| full | 1.0000 | 0.5714 | 0.7273 | 2.00 | 42.9 | 4.0 / 7.0 |
| keyword_only | 1.0000 | 0.5714 | 0.7273 | 2.00 | 42.9 | 4.0 / 7.0 |
| model_only | 0.8867 | 0.5714 | 0.6918 | 2.00 | 44.9 | 4.0 / 7.0 |
| disabled | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.0 | 0.0 / 7.0 |

Interpretation:
- In the forensics-primary preset, the forensic channel is now clearly necessary: disabling it drops actor recall from 0.5714 to 0.0000.
- The improved learned model alone (`model_only`) now detects 4 of 7 malicious email-capable actors on average, up from the previous 1 of 7, but it introduces some false detections (actor_precision 0.8867, confirmed_alert_precision 0.9574).
- `full` and `keyword_only` still remain identical, so the added lift in the overall preset still comes from the heuristic keyword component rather than raising the combined-mode result.
- This preset is therefore a better measurement of the forensic channel contribution, and it now shows that the model-specific path can be strengthened, but it still does not outperform the keyword path in this simulator.

Additional output file:
- Forensics-primary bundle: `results_eg_siem_enron_forensics_primary.json`
