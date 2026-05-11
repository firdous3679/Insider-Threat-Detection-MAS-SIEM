## Corpus Summary

| corpus_name          | total_emails | positive_count | negative_count | positive_rate | average_body_word_count | median_body_word_count | number_of_categories | number_of_subcategories | number_of_templates | number_of_template_families | hard_negative_count | emails_with_attachment | emails_with_external_link |
| -------------------- | ------------ | -------------- | -------------- | ------------- | ----------------------- | ---------------------- | -------------------- | ----------------------- | ------------------- | --------------------------- | ------------------- | ---------------------- | ------------------------- |
| municipal_v1         | 1000         | 150            | 850            | 0.1500        | 62.8440                 | 59.0000                | 5                    | 32                      |                     |                             |                     |                        |                           |
| municipal_v2         | 2000         | 300            | 1700           | 0.1500        | 50.4790                 | 47.0000                | 7                    | 26                      | 109.0000            | 26.0000                     | 300.0000            | 573.0000               | 234.0000                  |
| kurdi_smart_building | 140          | 100            | 40             | 0.7143        | 13.1929                 | 10.5000                | 47                   | 47                      |                     |                             |                     |                        |                           |

## Zero-Shot Transfer

| dataset              | experiment      | threshold | precision | recall | F1     | ROC-AUC | PR-AUC | predicted_positive_rate |
| -------------------- | --------------- | --------- | --------- | ------ | ------ | ------- | ------ | ----------------------- |
| municipal_v1         | zero_shot_enron | 0.9800    | 0.0000    | 0.0000 | 0.0000 | 0.5867  | 0.1883 | 0.0000                  |
| municipal_v2         | zero_shot_enron | 0.9800    | 0.0000    | 0.0000 | 0.0000 | 0.4462  | 0.1280 | 0.0000                  |
| kurdi_smart_building | zero_shot_enron | 0.9800    | 0.0000    | 0.0000 | 0.0000 | 0.6325  | 0.8035 | 0.0000                  |

## Threshold Sensitivity

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

## Fine-Tuned Random

| dataset              | experiment              | threshold | precision | recall | F1     | ROC-AUC | PR-AUC | predicted_positive_rate | train_size | test_size | positive_rate_train | positive_rate_test | threshold_method                |
| -------------------- | ----------------------- | --------- | --------- | ------ | ------ | ------- | ------ | ----------------------- | ---------- | --------- | ------------------- | ------------------ | ------------------------------- |
| municipal_v1         | fine_tuned_random_split | 0.0510    | 1.0000    | 1.0000 | 1.0000 | 1.0000  | 1.0000 | 0.1500                  | 800        | 200       | 0.1500              | 0.1500             | target_train_validation_best_f1 |
| municipal_v2         | fine_tuned_random_split | 0.0090    | 1.0000    | 1.0000 | 1.0000 | 1.0000  | 1.0000 | 0.1500                  | 1600       | 400       | 0.1500              | 0.1500             | target_train_validation_best_f1 |
| kurdi_smart_building | fine_tuned_random_split | 0.0260    | 0.9091    | 1.0000 | 0.9524 | 0.9625  | 0.9857 | 0.7857                  | 112        | 28        | 0.7143              | 0.7143             | target_train_validation_best_f1 |

## Grouped Template Family

| dataset      | experiment                            | threshold | precision | recall | F1     | ROC-AUC | PR-AUC | predicted_positive_rate | grouping_column | number_of_train_groups | number_of_test_groups | shared_group_count | split_strategy       | split_note | train_size | test_size |
| ------------ | ------------------------------------- | --------- | --------- | ------ | ------ | ------- | ------ | ----------------------- | --------------- | ---------------------- | --------------------- | ------------------ | -------------------- | ---------- | ---------- | --------- |
| municipal_v2 | fine_tuned_grouped_by_template_family | 0.0080    | 0.9492    | 1.0000 | 0.9739 | 1.0000  | 1.0000 | 0.1553                  | template_family | 20                     | 6                     | 0                  | strict_grouped_split |            | 1620       | 380       |

## Grouped Template ID

| dataset      | experiment                        | threshold | precision | recall | F1     | ROC-AUC | PR-AUC | predicted_positive_rate | grouping_column | number_of_train_groups | number_of_test_groups | shared_group_count | split_strategy       | split_note | train_size | test_size |
| ------------ | --------------------------------- | --------- | --------- | ------ | ------ | ------- | ------ | ----------------------- | --------------- | ---------------------- | --------------------- | ------------------ | -------------------- | ---------- | ---------- | --------- |
| municipal_v2 | fine_tuned_grouped_by_template_id | 0.0090    | 0.9859    | 1.0000 | 0.9929 | 1.0000  | 1.0000 | 0.1524                  | template_id     | 87                     | 22                    | 0                  | strict_grouped_split |            | 1534       | 466       |

## Hard Negative Evaluation

| setting                                                  | hard_negative_count | false_positive_count | false_positive_rate | specificity_on_hard_negatives | threshold |
| -------------------------------------------------------- | ------------------- | -------------------- | ------------------- | ----------------------------- | --------- |
| enron_zero_shot_enron_threshold                          | 300                 | 0                    | 0.0000              | 1.0000                        | 0.9800    |
| enron_zero_shot_target_calibrated_threshold              | 300                 | 192                  | 0.6400              | 0.3600                        | 0.0033    |
| fine_tuned_random_split_hard_negative_holdout            | 51                  | 0                    | 0.0000              | 1.0000                        | 0.0090    |
| fine_tuned_grouped_template_family_hard_negative_holdout | 104                 | 3                    | 0.0288              | 0.9712                        | 0.0080    |

## V1 vs V2 Comparison

| result                                 | precision | recall | F1     | PR-AUC |
| -------------------------------------- | --------- | ------ | ------ | ------ |
| municipal_v1_zero_shot                 | 0.0000    | 0.0000 | 0.0000 | 0.1883 |
| municipal_v2_zero_shot                 | 0.0000    | 0.0000 | 0.0000 | 0.1280 |
| municipal_v1_fine_tuned_random         | 1.0000    | 1.0000 | 1.0000 | 1.0000 |
| municipal_v2_fine_tuned_random         | 1.0000    | 1.0000 | 1.0000 | 1.0000 |
| municipal_v2_grouped_template_family   | 0.9492    | 1.0000 | 0.9739 | 1.0000 |
| municipal_v2_grouped_template_id       | 0.9859    | 1.0000 | 0.9929 | 1.0000 |
| municipal_v2_hard_negative_max_fp_rate |           |        | 0.6400 |        |
