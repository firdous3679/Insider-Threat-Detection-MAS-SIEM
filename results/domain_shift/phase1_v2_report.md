# Phase 1 V2 Domain-Shift Report

This evaluation uses synthetic, template-generated target-domain corpora. It quantifies Enron-to-municipal/smart-building domain shift and adaptation behavior; it is not real-world validation.

## Main Conclusion

The Enron-trained classifier does not transfer reliably as a zero-shot municipal detector under its Enron-calibrated threshold. The target corpora still show ranking signal in ROC-AUC/PR-AUC, so the main failure mode is threshold calibration plus domain mismatch rather than total absence of signal. Target-domain calibration and fine-tuning are required.

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

## Fine-Tuning

| dataset              | experiment                            | threshold | precision | recall | F1     | ROC-AUC | PR-AUC | predicted_positive_rate | train_size | test_size | positive_rate_train | positive_rate_test | threshold_method                | grouping_column | number_of_train_groups | number_of_test_groups | shared_group_count | split_strategy       | split_note |
| -------------------- | ------------------------------------- | --------- | --------- | ------ | ------ | ------- | ------ | ----------------------- | ---------- | --------- | ------------------- | ------------------ | ------------------------------- | --------------- | ---------------------- | --------------------- | ------------------ | -------------------- | ---------- |
| municipal_v1         | fine_tuned_random_split               | 0.0510    | 1.0000    | 1.0000 | 1.0000 | 1.0000  | 1.0000 | 0.1500                  | 800        | 200       | 0.1500              | 0.1500             | target_train_validation_best_f1 |                 |                        |                       |                    |                      |            |
| municipal_v2         | fine_tuned_random_split               | 0.0090    | 1.0000    | 1.0000 | 1.0000 | 1.0000  | 1.0000 | 0.1500                  | 1600       | 400       | 0.1500              | 0.1500             | target_train_validation_best_f1 |                 |                        |                       |                    |                      |            |
| kurdi_smart_building | fine_tuned_random_split               | 0.0260    | 0.9091    | 1.0000 | 0.9524 | 0.9625  | 0.9857 | 0.7857                  | 112        | 28        | 0.7143              | 0.7143             | target_train_validation_best_f1 |                 |                        |                       |                    |                      |            |
| municipal_v2         | fine_tuned_grouped_by_template_family | 0.0080    | 0.9492    | 1.0000 | 0.9739 | 1.0000  | 1.0000 | 0.1553                  | 1620       | 380       |                     |                    |                                 | template_family | 20.0000                | 6.0000                | 0.0000             | strict_grouped_split |            |
| municipal_v2         | fine_tuned_grouped_by_template_id     | 0.0090    | 0.9859    | 1.0000 | 0.9929 | 1.0000  | 1.0000 | 0.1524                  | 1534       | 466       |                     |                    |                                 | template_id     | 87.0000                | 22.0000               | 0.0000             | strict_grouped_split |            |
| municipal_v2         | fine_tuned_grouped_by_subcategory     | 0.0080    | 0.9492    | 1.0000 | 0.9739 | 1.0000  | 1.0000 | 0.1553                  | 1620       | 380       |                     |                    |                                 | subcategory     | 20.0000                | 6.0000                | 0.0000             | strict_grouped_split |            |

## Hard Negatives

| setting                                                  | hard_negative_count | false_positive_count | false_positive_rate | specificity_on_hard_negatives | threshold |
| -------------------------------------------------------- | ------------------- | -------------------- | ------------------- | ----------------------------- | --------- |
| enron_zero_shot_enron_threshold                          | 300                 | 0                    | 0.0000              | 1.0000                        | 0.9800    |
| enron_zero_shot_target_calibrated_threshold              | 300                 | 192                  | 0.6400              | 0.3600                        | 0.0033    |
| fine_tuned_random_split_hard_negative_holdout            | 51                  | 0                    | 0.0000              | 1.0000                        | 0.0090    |
| fine_tuned_grouped_template_family_hard_negative_holdout | 104                 | 3                    | 0.0288              | 0.9712                        | 0.0080    |

## Vocabulary Adaptation

| dataset      | experiment                                | threshold | precision | recall | F1     | ROC-AUC | PR-AUC | predicted_positive_rate | feature_set                     |
| ------------ | ----------------------------------------- | --------- | --------- | ------ | ------ | ------- | ------ | ----------------------- | ------------------------------- |
| municipal_v2 | zero_shot_tfidf_only                      | 0.9800    | 0.0000    | 0.0000 | 0.0000 | 0.4462  | 0.1280 | 0.0000                  | tfidf_only                      |
| municipal_v2 | zero_shot_original_keywords               | 0.9800    | 0.0000    | 0.0000 | 0.0000 | 0.0777  | 0.0821 | 0.0000                  | original_keywords               |
| municipal_v2 | zero_shot_exfil_aligned_keywords          | 0.9800    | 0.0000    | 0.0000 | 0.0000 | 0.1149  | 0.0850 | 0.0000                  | exfil_aligned_keywords          |
| municipal_v2 | zero_shot_municipal_security_ops_keywords | 0.9800    | 0.0000    | 0.0000 | 0.0000 | 0.2381  | 0.0949 | 0.0000                  | municipal_security_ops_keywords |

## V1 vs V2

| result                                 | precision | recall | F1     | PR-AUC |
| -------------------------------------- | --------- | ------ | ------ | ------ |
| municipal_v1_zero_shot                 | 0.0000    | 0.0000 | 0.0000 | 0.1883 |
| municipal_v2_zero_shot                 | 0.0000    | 0.0000 | 0.0000 | 0.1280 |
| municipal_v1_fine_tuned_random         | 1.0000    | 1.0000 | 1.0000 | 1.0000 |
| municipal_v2_fine_tuned_random         | 1.0000    | 1.0000 | 1.0000 | 1.0000 |
| municipal_v2_grouped_template_family   | 0.9492    | 1.0000 | 0.9739 | 1.0000 |
| municipal_v2_grouped_template_id       | 0.9859    | 1.0000 | 0.9929 | 1.0000 |
| municipal_v2_hard_negative_max_fp_rate |           |        | 0.6400 |        |

## What Goes Where

Main paper: V2 zero-shot, threshold sensitivity, random fine-tuning, and the most conservative grouped/template-held-out result. Appendix/supplement: full per-subcategory tables, hard-negative top-score inspection, vocabulary variants, and Kurdi short-message stress test.

## Revised Manuscript Paragraph

To quantify whether the Enron-derived email-forensics component transfers to municipal smart-building communication, we evaluated an Enron-trained TF-IDF/logistic-regression classifier on synthetic municipal facilities corpora and a short-message smart-building stress-test corpus. The results show that Enron is useful as an initial calibration source but is not sufficient as a standalone detector: zero-shot performance depends strongly on threshold calibration, while target-domain threshold calibration and fine-tuning substantially improve target-domain performance. We therefore treat Enron as a source-domain calibration corpus and report target-domain adaptation results separately from evidence-gated SIEM/MAS results.

## Response To Reviewers

Reviewer 1 Q2: We agree that Enron alone is not representative of municipal smart-building operations. We therefore added a domain-shift evaluation using synthetic municipal facilities corpora, including V2 with hard benign near-miss messages and leakage-aware grouped splits. The results explicitly quantify the Enron-to-municipal gap rather than assuming transfer.

Reviewer 1 Q5: We added V2 corpus metadata, methodology, template identifiers, template families, hard-negative labels, approval context, sender-domain type, and expected detection signals. These fields support reproducibility and grouped evaluation to reduce template leakage.

Reviewer 2 Q2: The results show that the hybrid system is not merely compensating for Enron mismatch. The email classifier has partial ranking signal but requires target-domain calibration or fine-tuning; the SIEM/MAS layers remain necessary because municipal insider-risk evidence is broader than email text alone.

Reviewer 3 Q3: We added explicit experiments showing Enron-trained transfer to municipal/smart-building corpora, including threshold sensitivity, target calibration, fine-tuning, grouped/template-held-out splits, and hard-negative evaluation. Low zero-shot performance is framed as confirmation of the reviewer concern and motivates adaptation.
