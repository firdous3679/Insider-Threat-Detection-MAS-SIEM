# Mesa Scalability Code Map

## Main Mesa Simulation Files

- `mini_mesa_LSC.py`: Layered SIEM-Core baseline simulation.
- `mini_mesa_CE-SIEM.py`: Cognitive-enriched SIEM variant.
- `mini_mesa_EG-SIEM.py`: Evidence-gated SIEM variant reused by this scalability runner.
- `mini_mesa_EG-SIEM_Enron.py`: Evidence-gated SIEM with Enron forensics artifact.

## Agent Creation

`mini_mesa_EG-SIEM.py::InsiderModel.__init__` controls benign employees, power users, malicious insiders, and fixed monitor/SIEM agents. `scripts/run_mesa_scalability.py::scaled_roles` maps requested human-agent counts to the existing constructor arguments.

## Simulation Length

`mini_mesa_EG-SIEM.py::InsiderModel.step` advances one Mesa step. The new runner controls the number of steps with `--steps` or the default scalability configuration.

## SIEM / Evidence Gating

`mini_mesa_EG-SIEM.py::SIEMAgent` and `SIEMConfig` implement scoring, evidence-gating, peer normalization, regularity suppression, ToM, and forensics weighting. The scalability runner reuses the original rich config and does not tune thresholds.

## Metrics

`mini_mesa_EG-SIEM.py::evaluate` computes actor precision/recall/F1, confirmed alerts, false positives, early alerts, and TTD. The runner adds runtime, memory, event-throughput, status, and error fields.

## Results

`results/scalability/` contains raw run CSV, aggregate summary CSV, paper-ready Markdown table, report, code map, and plots.

## Files Modified

- `scripts/run_mesa_scalability.py`
