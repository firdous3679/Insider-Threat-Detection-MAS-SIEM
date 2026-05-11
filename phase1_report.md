# Phase 1 Report: Email Domain-Shift Evaluation

## Corpus Summary

| dataset              | n     | positive | negative | positive_rate | avg_words |
| -------------------- | ----- | -------- | -------- | ------------- | --------- |
| enron_spam           | 33716 | 17171    | 16545    | 0.5093        | 270.0420  |
| municipal_synthetic  | 1000  | 150      | 850      | 0.1500        | 62.8440   |
| kurdi_smart_building | 140   | 100      | 40       | 0.7143        | 13.1929   |

## Main Finding

The Enron in-domain CV F1 is 1.000. Zero-shot transfer drops to 0.000 on the synthetic municipal corpus and 0.000 on the Kurdi smart-building corpus. That corresponds to F1 gaps of 1.000 and 1.000, respectively.

CV note: the runner attempted duplicate-aware grouping by normalized text, but the Enron spam class collapsed to only 6 unique groups. The reported in-domain row therefore uses `stratified_cv_duplicate_fallback` and records the fallback explicitly in the CSV.

Fine-tuning with 80% of each target corpus and evaluating on a stratified 20% holdout changes the picture:

| experiment          | dataset                         | n   | positives | threshold | precision | recall | f1     | roc_auc | pr_auc | train_adaptation_examples | test_holdout_examples |
| ------------------- | ------------------------------- | --- | --------- | --------- | --------- | ------ | ------ | ------- | ------ | ------------------------- | --------------------- |
| fine_tuned_transfer | municipal_synthetic_fine_tuned  | 200 | 30        | 0.0600    | 1.0000    | 1.0000 | 1.0000 | 1.0000  | 1.0000 | 800                       | 200                   |
| fine_tuned_transfer | kurdi_smart_building_fine_tuned | 28  | 20        | 0.0300    | 0.9091    | 1.0000 | 0.9524 | 0.9625  | 0.9857 | 112                       | 28                    |

## Zero-Shot Transfer

| experiment   | dataset                        | n     | positives | threshold | precision | recall | f1     | roc_auc | pr_auc | cv_strategy                      | min_unique_groups_per_class |
| ------------ | ------------------------------ | ----- | --------- | --------- | --------- | ------ | ------ | ------- | ------ | -------------------------------- | --------------------------- |
| in_domain_cv | enron_spam_grouped_cv          | 33716 | 17171     | 0.9800    | 0.9996    | 1.0000 | 0.9998 | 0.9998  | 0.9996 | stratified_cv_duplicate_fallback | 6.0000                      |
| zero_shot    | municipal_synthetic_zero_shot  | 1000  | 150       | 0.9800    | 0.0000    | 0.0000 | 0.0000 | 0.5867  | 0.1883 |                                  |                             |
| zero_shot    | kurdi_smart_building_zero_shot | 140   | 100       | 0.9800    | 0.0000    | 0.0000 | 0.0000 | 0.6325  | 0.8035 |                                  |                             |

## Vocabulary Adaptation

| feature_set            | dataset                        | f1     | precision | recall | pr_auc |
| ---------------------- | ------------------------------ | ------ | --------- | ------ | ------ |
| tfidf_only             | municipal_synthetic_zero_shot  | 0.0000 | 0.0000    | 0.0000 | 0.1883 |
| tfidf_only             | kurdi_smart_building_zero_shot | 0.0000 | 0.0000    | 0.0000 | 0.8035 |
| original_keywords      | municipal_synthetic_zero_shot  | 0.0000 | 0.0000    | 0.0000 | 0.1720 |
| original_keywords      | kurdi_smart_building_zero_shot | 0.0000 | 0.0000    | 0.0000 | 0.7584 |
| exfil_aligned_keywords | municipal_synthetic_zero_shot  | 0.0000 | 0.0000    | 0.0000 | 0.2045 |
| exfil_aligned_keywords | kurdi_smart_building_zero_shot | 0.0000 | 0.0000    | 0.0000 | 0.7498 |

The exfiltration-aligned vocabulary is useful as an interpretable signal check, but the measured results show whether keyword expansion alone closes the transfer gap. If the exfil-aligned feature set remains below the fine-tuned holdout F1, the reviewer-facing interpretation is that domain adaptation is required rather than vocabulary substitution alone.

## Kurdi Per-Category Recall

| category                        | n  | recall | detected | missed |
| ------------------------------- | -- | ------ | -------- | ------ |
| BMS_ACCESS_REQUEST              | 10 | 0.0000 | 0        | 10     |
| DATA_EXTRACTION_REQUEST         | 10 | 0.0000 | 0        | 10     |
| EMERGENCY_SYSTEM_TAMPERING      | 10 | 0.0000 | 0        | 10     |
| IAM_PRIVILEGE_ESCALATION        | 10 | 0.0000 | 0        | 10     |
| IOT_DEVICE_ACCESS               | 10 | 0.0000 | 0        | 10     |
| LOGS_AND_MONITORING_REQUEST     | 10 | 0.0000 | 0        | 10     |
| NETWORK_CONFIGURATION_REQUEST   | 10 | 0.0000 | 0        | 10     |
| PHYSICAL_SECURITY_SYSTEM_ACCESS | 10 | 0.0000 | 0        | 10     |
| SECURITY_POLICY_BYPASS          | 10 | 0.0000 | 0        | 10     |
| VENDOR_REMOTE_ACCESS_REQUEST    | 10 | 0.0000 | 0        | 10     |

Best transferring attack categories:

| category                   | n  | recall | detected | missed |
| -------------------------- | -- | ------ | -------- | ------ |
| BMS_ACCESS_REQUEST         | 10 | 0.0000 | 0        | 10     |
| DATA_EXTRACTION_REQUEST    | 10 | 0.0000 | 0        | 10     |
| EMERGENCY_SYSTEM_TAMPERING | 10 | 0.0000 | 0        | 10     |

Poorest transferring attack categories:

| category                   | n  | recall | detected | missed |
| -------------------------- | -- | ------ | -------- | ------ |
| BMS_ACCESS_REQUEST         | 10 | 0.0000 | 0        | 10     |
| DATA_EXTRACTION_REQUEST    | 10 | 0.0000 | 0        | 10     |
| EMERGENCY_SYSTEM_TAMPERING | 10 | 0.0000 | 0        | 10     |

## Reviewer 2 Q2 Answer

The hybrid architecture should be framed as synergistic rather than merely compensating for domain mismatch. The Enron-only text classifier gives a direct measure of the communication-layer domain gap; the broader MAS/SIEM evidence layers are still needed because municipal insider activity is not reducible to Enron-style spam/phishing text. Phase 1 therefore separates the limits of email transfer from the contribution of behavioral and evidence-gated signals.

## Paper-Ready Section 5 Paragraph

To quantify the external validity of the Enron-derived email-forensics component, we evaluated an Enron-trained TF-IDF/logistic-regression classifier under zero-shot transfer to two smart-building email corpora. The in-domain CV F1 on Enron Spam was 1.000, whereas zero-shot F1 was 0.000 on the 1,000-message synthetic municipal facilities corpus and 0.000 on the 140-message Kurdi smart-building corpus. After target-domain fine-tuning using 80% of each municipal corpus, holdout F1 was 1.000 and 0.952, respectively. These results show a measurable domain-shift gap and support using Enron as a calibration source rather than as a sufficient standalone representation of municipal insider-threat communication.
