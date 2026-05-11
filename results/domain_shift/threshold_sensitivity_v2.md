# Phase 1 V2 Threshold Sensitivity

These results use synthetic target-domain corpora and diagnose threshold calibration under Enron-to-target transfer. The target-oracle rows are diagnostic upper bounds only and are not deployable results.

| dataset              | threshold_name                    | threshold_value | precision | recall | F1     | predicted_positive_rate | ROC-AUC | PR-AUC |
| -------------------- | --------------------------------- | --------------- | --------- | ------ | ------ | ----------------------- | ------- | ------ |
| municipal_v1         | enron_f1_optimized                | 0.9800          | 0.0000    | 0.0000 | 0.0000 | 0.0000                  | 0.5867  | 0.1883 |
| municipal_v1         | default_0_50                      | 0.5000          | 0.0000    | 0.0000 | 0.0000 | 0.0000                  | 0.5867  | 0.1883 |
| municipal_v1         | mesa_runtime_0_002                | 0.0020          | 0.1500    | 1.0000 | 0.2609 | 1.0000                  | 0.5867  | 0.1883 |
| municipal_v1         | target_oracle_best_f1_upper_bound | 0.0037          | 0.1914    | 0.7400 | 0.3041 | 0.5800                  | 0.5867  | 0.1883 |
| municipal_v1         | target_calibrated_heldout         | 0.0037          | 0.1684    | 0.6667 | 0.2688 | 0.5940                  | 0.5282  | 0.1680 |
| municipal_v2         | enron_f1_optimized                | 0.9800          | 0.0000    | 0.0000 | 0.0000 | 0.0000                  | 0.4462  | 0.1280 |
| municipal_v2         | default_0_50                      | 0.5000          | 0.0000    | 0.0000 | 0.0000 | 0.0000                  | 0.4462  | 0.1280 |
| municipal_v2         | mesa_runtime_0_002                | 0.0020          | 0.1500    | 1.0000 | 0.2609 | 1.0000                  | 0.4462  | 0.1280 |
| municipal_v2         | target_oracle_best_f1_upper_bound | 0.0033          | 0.1549    | 0.9833 | 0.2676 | 0.9525                  | 0.4462  | 0.1280 |
| municipal_v2         | target_calibrated_heldout         | 0.0033          | 0.1515    | 0.9533 | 0.2614 | 0.9440                  | 0.4452  | 0.1278 |
| kurdi_smart_building | enron_f1_optimized                | 0.9800          | 0.0000    | 0.0000 | 0.0000 | 0.0000                  | 0.6325  | 0.8035 |
| kurdi_smart_building | default_0_50                      | 0.5000          | 0.0000    | 0.0000 | 0.0000 | 0.0000                  | 0.6325  | 0.8035 |
| kurdi_smart_building | mesa_runtime_0_002                | 0.0020          | 0.7143    | 1.0000 | 0.8333 | 1.0000                  | 0.6325  | 0.8035 |
| kurdi_smart_building | target_oracle_best_f1_upper_bound | 0.0027          | 0.7246    | 1.0000 | 0.8403 | 0.9857                  | 0.6325  | 0.8035 |
| kurdi_smart_building | target_calibrated_heldout         | 0.0027          | 0.7246    | 1.0000 | 0.8403 | 0.9857                  | 0.6760  | 0.8197 |

## Interpretation

The Enron F1-optimized threshold is highly conservative on the target corpora and produces no positive predictions. Lower thresholds and target-domain calibration recover recall, showing that poor zero-shot F1 is driven by both calibration shift and target-domain mismatch. Target-domain threshold calibration or fine-tuning is therefore required before using the Enron-derived email classifier in the municipal setting.
