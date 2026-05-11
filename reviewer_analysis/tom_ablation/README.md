# Reviewer ToM Ablation

This folder contains reviewer-only ablation code. It imports the existing Mesa
experiment files but does not modify them.

Run:

```bash
.venv/bin/python reviewer_analysis/tom_ablation/run_tom_ablation.py
```

Outputs are written to `reviewer_analysis/tom_ablation/outputs/`.

The script runs only variants that are available through existing toggles or
existing baselines. It records unavailable requested variants explicitly rather
than creating a fake ablation. In particular, `mini_mesa_LSC.py` does not expose
ToM or email-forensics toggles, so `LSC + ToM only` and `LSC + email only`
require future code changes and are documented as unavailable.
