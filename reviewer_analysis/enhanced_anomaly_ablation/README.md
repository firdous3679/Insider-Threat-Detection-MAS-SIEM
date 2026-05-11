# Reviewer Enhanced Anomaly Ablation

This folder contains reviewer-only status/evaluation support for
`EnhancedRoleAnomalyModel`.

Run:

```bash
.venv/bin/python reviewer_analysis/enhanced_anomaly_ablation/run_enhanced_anomaly_ablation.py
```

The current script inspects the existing code and writes a status report. It
does not monkey-patch or modify existing experiment files. A clean one-factor
ablation is documented as future work because the enhanced anomaly model is
implemented inside LSC/CE-SIEM variants, while EG-SIEM/EG-SIEM-Enron use a
different simple ML interface; comparing those variants directly would confound
the anomaly model with other SIEM-layer differences.
