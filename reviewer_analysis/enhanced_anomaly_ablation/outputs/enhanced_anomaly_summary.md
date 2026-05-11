# EnhancedRoleAnomalyModel Status

This reviewer-only check did not modify existing experiment files.

## What is implemented

- `mini_mesa_LSC.py` implements and instantiates `EnhancedRoleAnomalyModel`.
- `mini_mesa_CE-SIEM.py` implements and instantiates `EnhancedRoleAnomalyModel`.
- `mini_mesa_EG-SIEM.py` uses a simpler `MLModel` with `IsolationForest`.
- `mini_mesa_EG-SIEM_Enron.py` uses the EG-SIEM style simple anomaly component rather than the LSC/CE enhanced class.

## Evaluation status

EnhancedRoleAnomalyModel was not cleanly evaluated as a one-factor ablation here. A fair experiment would need a common adapter that swaps only the anomaly model while keeping all other SIEM layers and simulation settings fixed. No existing result table found in the inspected files reports that isolated comparison.

## Minimal future experiment

Create a new isolated reviewer-only variant that keeps the EG-SIEM population, thresholds, evidence gate, scoring weights, seeds, steps, and warm-up fixed, then replaces only the anomaly model through a common adapter. Save matched run-level rows for both anomaly components before computing paired statistics.

## Outputs

- `enhanced_anomaly_status.csv`
- `enhanced_anomaly_summary.csv`
- `enhanced_anomaly_summary.md`