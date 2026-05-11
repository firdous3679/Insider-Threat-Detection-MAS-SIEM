# V1 vs V2 Comparison

| result                                 | precision | recall | F1     | PR-AUC |
| -------------------------------------- | --------- | ------ | ------ | ------ |
| municipal_v1_zero_shot                 | 0.0000    | 0.0000 | 0.0000 | 0.1883 |
| municipal_v2_zero_shot                 | 0.0000    | 0.0000 | 0.0000 | 0.1280 |
| municipal_v1_fine_tuned_random         | 1.0000    | 1.0000 | 1.0000 | 1.0000 |
| municipal_v2_fine_tuned_random         | 1.0000    | 1.0000 | 1.0000 | 1.0000 |
| municipal_v2_grouped_template_family   | 0.9492    | 1.0000 | 0.9739 | 1.0000 |
| municipal_v2_grouped_template_id       | 0.9859    | 1.0000 | 0.9929 | 1.0000 |
| municipal_v2_hard_negative_max_fp_rate |           |        | 0.6400 |        |
