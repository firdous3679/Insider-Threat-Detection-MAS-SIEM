# ToM Code Support Report

This report is based only on code inspection. It does not edit the manuscript.

## Files Inspected

- `mini_mesa_LSC.py`
- `mini_mesa_CE-SIEM.py`
- `mini_mesa_EG-SIEM.py`
- `mini_mesa_EG-SIEM_Enron.py`
- `cert_eval/`
- `domain_shift/`
- `scripts/`
- `results/`

## Is ToM Already Used In SIEM Scoring?

Yes for CE-SIEM, EG-SIEM, and EG-SIEM-Enron. It is not present in LSC.

Code evidence:

- `mini_mesa_EG-SIEM.py` defines `TomAbdAgent` and ToM goal/plan state at lines 27-167.
- `mini_mesa_EG-SIEM.py::SIEMConfig` includes `use_tom=True`, `tom_weight=2.0`, and `tom_threshold=0.30` at lines 271, 291, and 292.
- `mini_mesa_EG-SIEM.py::extract_features` maps ToM assessment into `tom_intent` and `tom_plan` at lines 601-604.
- `mini_mesa_EG-SIEM.py::correlate` adds `tom_weight * tom_intent` to the score at lines 694-696.
- `mini_mesa_EG-SIEM_Enron.py` has equivalent ToM scoring support: `tom_threshold=0.30`, `tom_weight=2.0`, feature extraction at lines 825-831, and score contribution at lines 760-762.
- `mini_mesa_CE-SIEM.py` includes `use_tom=True`, `tom_weight=1.5`, `tom_intent_threshold=0.4`, and adds ToM risk when `tom_malicious_intent` is present at lines 533, 567-568, and 1066-1068.

## Variables And Functions Supporting ToM Scoring

- `TomAbdAgent.observe_action(...)`
- `TomAbdAgent._check_contradictions(...)`
- `TomAbdAgent._abductive_inference(...)`
- `TomAbdAgent._update_plans(...)`
- `TomAbdAgent.get_threat_assessment(...)`
- `TomAbdAgent.get_cognitive_features(...)`
- `SIEMConfig.use_tom`
- `SIEMConfig.tom_weight`
- `SIEMConfig.tom_threshold` or `tom_intent_threshold`
- SIEM feature names: `tom_intent`, `tom_plan`, `tom_malicious_intent`, `tom_active_plan`

## Does ToM Contribute To Evidence Gating?

Yes in EG-SIEM and EG-SIEM-Enron. In `mini_mesa_EG-SIEM.py::calc_evidence`, `tom_intent > 0.4` adds one evidence count and evidence weight 2.0. The same logic is present in `mini_mesa_EG-SIEM_Enron.py::calc_evidence`.

## Can ToM Alone Trigger Confirmed Alerts?

No under the rich EG-SIEM/EG-SIEM-Enron evidence-gated configuration.

Reason:

- Confirmed escalation requires `min_evidence_count=2` and `min_evidence_weight=2.5`.
- A ToM intent signal contributes one evidence count and weight 2.0.
- Therefore ToM alone is below both the evidence-count requirement and the evidence-weight requirement.
- A confirmed alert also requires the total score to exceed the trust/role-adjusted confirmed threshold and the cooldown condition to be satisfied.

## What Should Be Described As Current Implementation?

The current implementation should be described as feature-level ToM integration:

- ToM observes activity and infers intent/plan features.
- ToM contradiction context can reduce inferred goal probabilities.
- ToM intent and plan features are incorporated into SIEM scoring.
- In EG-SIEM/EG-SIEM-Enron, ToM intent can contribute to the evidence gate but cannot independently trigger a confirmed alert.

## What Should Be Described As Future Work?

The following should be framed as future work unless a new experiment is added:

- richer coordinated multi-actor ToM reasoning,
- ToM-only alerting,
- independent ToM-only ablation for LSC without adding new code,
- deployment-grade compliance override logic tied to ToM contradiction handling,
- ToM models beyond the current feature-level intent and plan-completion signals.
