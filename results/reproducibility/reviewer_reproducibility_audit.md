# Reviewer Reproducibility and Safeguards Audit

Audited repository: `/Users/firdous/Documents/SmartCitiesSimulation/Insider-Threat-Detection-MAS-SIEM-main`

Manuscript text inspected:
- `/Users/firdous/Documents/SmartCitiesSimulation/Hybrid Insider Threat Detection for Smart Building Operations-Final Copy.pdf`
- `/Users/firdous/Documents/SmartCitiesSimulation/Insider-Threat-Detection-MAS-SIEM-main/Smart_Building_Management_Integrating_Multi_Agent_Simulation-1.pdf`
- `/Users/firdous/Documents/SmartCitiesSimulation/Manuscript_Revision_Plan.docx`
- `/Users/firdous/Documents/SmartCitiesSimulation/Unified Revision Plan.docx`
- `/Users/firdous/Documents/SmartCitiesSimulation/Reviewers Comments.docx`

## Short Summary

The current EG-SIEM and EG-SIEM-Enron code implements evidence-gated confirmation, trust-adaptive confirmed thresholds, policy/risk discounts for approved operational context, ToM contradiction handling, peer normalization, EWMA baselines, and role-based Isolation Forest anomaly scoring. Trust is updated after confirmed alerts, but trust decay toward baseline is not implemented in the EG-SIEM or EG-SIEM-Enron experiments. `mini_mesa_LSC.py` contains a `decay_trust()` method, but its configured decay rate is `0.00`, so there is no gradual decay in current experiments.

The manuscript sentence saying that trust “gradually decays toward the baseline” is not supported by the reported EG-SIEM/EG-SIEM-Enron code and should be revised.

The manuscript sentence saying that a “compliance override” is incorporated to ignore confirmed escalation overclaims the code. The implemented behavior is contradiction-aware handling and policy-based score reduction/suppression, not a complete compliance-override safeguard with independent approval, immutable audit logging, time bounds, or exploit testing. Events are still emitted into the simulator event log, but a real deployment override mechanism is not implemented.

## Source Audit Table

| Parameter / item | Value found | Source file | Function/class/line number | Notes |
|---|---:|---|---|---|
| Baseline human agents | 42 | `mini_mesa_EG-SIEM.py` | `InsiderModel.__init__`, lines 716-730 | 30 benign employees + 4 power users + 8 malicious insiders. |
| Benign users | 30 | `mini_mesa_EG-SIEM.py` | line 716, 722 | Default `n_emp=30`. |
| Power users | 4 | `mini_mesa_EG-SIEM.py` | line 716, 723 | Default `n_power=4`; report interval selected from 24, 36, 48. |
| Malicious insiders | 8 | `mini_mesa_EG-SIEM.py` | line 716, 725-729 | 3 exfil, 2 stealth, 1 takeover, 1 staging, 1 email-only. |
| System agents | 4 | `mini_mesa_EG-SIEM.py` | line 730 | DBMonitor, EmailMonitor, AuthMonitor, SIEMAgent. |
| Total Mesa agents | 46 | `mini_mesa_EG-SIEM.py`; manuscript | lines 716-730; manuscript final-copy lines 564-567 | 42 human + 4 system agents. |
| Baseline steps | 240 | `mini_mesa_EG-SIEM.py`; `mini_mesa_EG-SIEM_Enron.py` | EG line 781; Enron lines 1043, 1148 | Default `T=240` / `--steps 240`. |
| Warm-up steps | 60 | `mini_mesa_EG-SIEM.py`; `mini_mesa_EG-SIEM_Enron.py` | EG lines 716, 781; Enron lines 1043, 1149 | Attacks enabled when `steps >= warmup`. |
| Baseline number of runs | 10 | `mini_mesa_EG-SIEM.py`; `mini_mesa_EG-SIEM_Enron.py` | EG line 781; Enron lines 1043, 1147 | Default simulation reruns. |
| Seeds | 42-51 for 10-run EG/Enron experiments | `mini_mesa_EG-SIEM.py`; `mini_mesa_EG-SIEM_Enron.py` | EG line 785; Enron lines 1065-1073 | Seed = 42 + run index. |
| Scalability human-agent counts | 42, 100, 250, 500, 1000 | `scripts/run_mesa_scalability.py` | lines 31-38 | Also includes 100 at both 240 and 480 steps. |
| Scalability step counts | 240 and 480 | `scripts/run_mesa_scalability.py` | lines 31-38 | 42/240, 100/240, 100/480, 250/480, 500/480, 1000/480. |
| Scalability runs | 3 except 1000-agent run = 1 | `scripts/run_mesa_scalability.py` | lines 31-38 | Results saved in `results/scalability/mesa_scalability_summary.csv`. |
| Scalability scaling rule | malicious = max(8, round(humans * 0.15)); power = max(1, round(humans * 0.10)) | `scripts/run_mesa_scalability.py` | lines 52-81 | Remaining humans are benign employees. |
| Early threshold | 2.0 | `mini_mesa_EG-SIEM.py`; `mini_mesa_EG-SIEM_Enron.py` | EG line 276; Enron line 409 | Rich EG-SIEM/Enron config. |
| Confirmed base threshold | 4.0 | `mini_mesa_EG-SIEM.py`; `mini_mesa_EG-SIEM_Enron.py` | EG line 277; Enron line 410 | Used by confirmed threshold formula. |
| Confirmed threshold formula | `base + role_adj + trust_slope * (trust - 0.5)` | `mini_mesa_EG-SIEM.py`; `mini_mesa_EG-SIEM_Enron.py` | EG lines 646-651; Enron lines 682-685 | Enron implementation reads `self.trust[u]`; initialized via default dict elsewhere in class. |
| Role adjustments | staff 0, analyst 0.8, admin 1.2 | `mini_mesa_EG-SIEM.py`; `mini_mesa_EG-SIEM_Enron.py` | EG line 294; Enron line 427 | Raises confirmed threshold for analyst/admin roles. |
| Cooldown | 6 steps | `mini_mesa_EG-SIEM.py`; `mini_mesa_EG-SIEM_Enron.py` | EG line 279; Enron line 412 | `_can_fire` enforces step gap. |
| Sliding window | 48 events/steps of recent activity | `mini_mesa_EG-SIEM.py`; `mini_mesa_EG-SIEM_Enron.py` | EG line 278 and 685; Enron line 411 | Used for recent activity correlation. |
| EWMA alpha | 0.08 | `mini_mesa_EG-SIEM.py`; `mini_mesa_EG-SIEM_Enron.py` | EG line 282; Enron line 415 | EWMA mean and variance update in EG-SIEM baseline. |
| Z-score clipping | 6.0 | `mini_mesa_EG-SIEM.py`; `mini_mesa_EG-SIEM_Enron.py` | EG line 283; Enron line 416 | Baseline z-score capped at 6.0. |
| Baseline anomaly weight | 0.9 | `mini_mesa_EG-SIEM.py`; `mini_mesa_EG-SIEM_Enron.py` | EG line 284; Enron line 417 | Baseline contribution = zsum * 0.9. |
| Scoring feature weights | anchor_email 1.5; anchor_login 1.5; export_large 2.5; export_small 1.0; staging_export 2.0; after_hours 1.2; sens_burst 1.2; login_burst 1.0; unapproved 3.0; email_burst 1.0; exfil_chain 4.0; tom_intent 3.0; tom_plan 2.5; forensics_phishing 2.5; peer_dev 2.0; irreg 1.5; stealth_pattern 3.0; ext_email_count 0.0 | `mini_mesa_EG-SIEM.py`; `mini_mesa_EG-SIEM_Enron.py` | EG lines 297-302; Enron lines 430-435 | Rich EG-SIEM/Enron weights. |
| Minimum evidence count | 2 | `mini_mesa_EG-SIEM.py`; `mini_mesa_EG-SIEM_Enron.py` | EG line 280; Enron line 413 | Confirmed alert skipped if count is below 2. |
| Minimum evidence weight | 2.5 | `mini_mesa_EG-SIEM.py`; `mini_mesa_EG-SIEM_Enron.py` | EG line 281; Enron line 414 | Confirmed alert skipped if evidence weight below 2.5. |
| Evidence families and weights | export/staging 1.5; unapproved 2.0; exfil_chain 2.5; ToM intent >0.4 weight 2.0; forensics_phishing >0.5 weight 1.5; unauthorized after-hours 1.0; email burst + unapproved 1.5; forensics >0.3 + unapproved 2.0; ext_email_count >=2 weight 1.5; stealth + unapproved 2.5 | `mini_mesa_EG-SIEM.py`; `mini_mesa_EG-SIEM_Enron.py` | EG lines 540-556; Enron lines 687-709 | Evidence gate is separate from weighted risk score. |
| Conditions for confirmed escalation | score >= confirmed threshold, cooldown satisfied, and evidence count/weight gate satisfied | `mini_mesa_EG-SIEM.py`; `mini_mesa_EG-SIEM_Enron.py` | EG lines 705-711; Enron lines 777-787 | Confirmed alerts trigger trust update. |
| Can a single evidence type trigger confirmed? | No under rich EG-SIEM/Enron config | `mini_mesa_EG-SIEM.py`; `mini_mesa_EG-SIEM_Enron.py` | EG lines 280-281, 705-708; Enron lines 413-414, 777-780 | Requires at least 2 evidence events and weight >=2.5. |
| Initial trust | 0.70 | `mini_mesa_EG-SIEM.py`; `mini_mesa_EG-SIEM_Enron.py` | EG line 285; Enron line 418 | Used when user has no prior trust entry. |
| Minimum trust | 0.10 | `mini_mesa_EG-SIEM.py`; `mini_mesa_EG-SIEM_Enron.py` | EG line 286; Enron line 419 | Clipping lower bound. |
| Maximum trust | 0.95 | `mini_mesa_EG-SIEM.py`; `mini_mesa_EG-SIEM_Enron.py` | EG line 287; Enron line 420 | Clipping upper bound. |
| Trust slope | 1.2 | `mini_mesa_EG-SIEM.py`; `mini_mesa_EG-SIEM_Enron.py` | EG line 288; Enron line 421 | Confirmed threshold adapts by 1.2 * (trust - 0.5). |
| Trust update for true positive | -0.18 in EG-SIEM/Enron | `mini_mesa_EG-SIEM.py`; `mini_mesa_EG-SIEM_Enron.py` | EG line 289; Enron line 422 | Malicious confirmed alert decreases trust. |
| Trust update for false positive | +0.05 in EG-SIEM/Enron | `mini_mesa_EG-SIEM.py`; `mini_mesa_EG-SIEM_Enron.py` | EG line 290; Enron line 423 | Benign confirmed alert increases trust. |
| Trust decay | Not implemented in current EG-SIEM/EG-SIEM-Enron experiments | searched `mini_mesa_EG-SIEM.py`, `mini_mesa_EG-SIEM_Enron.py`, `mini_mesa_LSC.py`, `mini_mesa_CE-SIEM.py` | `rg trust_decay|decay_trust`; LSC lines 80, 1204-1211 | LSC has `trust_decay=0.00`; the method returns without change when `d <= 0.0`. |
| Does trust return toward baseline over time? | No for current EG-SIEM/Enron experiments; LSC mechanism configured to no-op | `mini_mesa_LSC.py` | lines 1204-1211 | Manuscript claim should be corrected. |
| ToM contradiction decay | Inferred goal probability multiplied by `0.85 ** len(contras)` | `mini_mesa_EG-SIEM.py` | lines 77-81 | This is ToM belief decay, not user-trust decay. |
| ToM abductive rule boosts | 0.4, 0.5, 0.3, 0.35 | `mini_mesa_EG-SIEM.py` | lines 60-65 | Rules: data theft/exfiltration/recon/data theft after-hours. |
| ToM intent threshold | 0.30 | `mini_mesa_EG-SIEM.py`; `mini_mesa_EG-SIEM_Enron.py` | EG line 292; Enron line 425 | Feature active above threshold. |
| ToM weight | 2.0 | `mini_mesa_EG-SIEM.py`; `mini_mesa_EG-SIEM_Enron.py` | EG line 291; Enron line 424 | Additional score contribution when intent present. |
| ToM plan threshold | 0.4 | `mini_mesa_EG-SIEM.py`; `mini_mesa_EG-SIEM_Enron.py` | EG line 604; Enron line 830 | Plan feature active above threshold. |
| Forensics weight | 1.5 | `mini_mesa_EG-SIEM.py`; `mini_mesa_EG-SIEM_Enron.py` | EG line 293; Enron line 426 | Additional score contribution. |
| Forensics phishing feature threshold in SIEM | Average phishing score >0.4 in EG-SIEM; >0.20 in EG-SIEM-Enron | `mini_mesa_EG-SIEM.py`; `mini_mesa_EG-SIEM_Enron.py` | EG lines 605-609; Enron lines 833-839 | Feature threshold differs between core and Enron variant. |
| Enron full corpus count used | Not found in current code/results; full `emails.csv` not included | `README.md`; `EG_SIEM_Enron_recalculation_report.md` | README lines 97-104; report lines 46-50 | Full Enron corpus unavailable in current repo; report says full-corpus baseline calibration was not recomputed. |
| Enron spam raw rows | 33,716 | `enron_spam_data.csv`; `phase1_report.md` | phase1 report lines 7-8 | Current CSV shape verified locally. |
| Enron spam/ham after training filter | 17,171 spam; 16,047 ham; 33,218 usable | `enron_combined_training.py`; `EG_SIEM_Enron_recalculation_report.md` | loader filter lines 287-310; report lines 52-56 | Filter is label valid and at least 20 words; duplicate report says 33,218 usable examples. |
| Exact duplicate emails | 17,775 | `EG_SIEM_Enron_recalculation_report.md` | lines 52-56 | Duplicate leakage risk documented. |
| Unique normalized-text groups | 15,123 | `EG_SIEM_Enron_recalculation_report.md` | lines 52-56 | Used for grouped split/retraining report. |
| TF-IDF feature dimension | 10,000 | `enron_combined_new.py`; `domain_shift/run_transfer.py`; saved `combined_forensics_model.pkl` | training lines 624-629, 719-724; domain lines 97-104 | Runtime pickle reports `vocabulary_size=10000`. |
| Selected classifier | Logistic regression | saved `combined_forensics_model.pkl`; manuscript | manuscript lines 540-559 | Saved pickle `classifier_name=logistic_regression`. |
| Logistic regression params | max_iter=2000, solver=liblinear, class_weight=balanced, random_state=42 | `enron_combined_new.py`; `domain_shift/run_transfer.py` | training lines 588-596; domain lines 119-124 | CalibratedClassifierCV wraps LR in `enron_combined_new.py` with sigmoid, cv=3. |
| Grouped-CV metrics for runtime forensics model | accuracy 0.7139, precision 0.7462, recall 0.9167, F1 0.8054, ROC-AUC 0.6512, threshold 0.002 | saved `combined_forensics_model.pkl`; manuscript | manuscript lines 540-559 | Values verified from pickle and final manuscript. |
| Runtime threshold | 0.002 | saved `combined_forensics_model.pkl`; manuscript; `domain_shift/run_transfer_v2.py` | manuscript line 546; domain v2 line 59 | Original Mesa runtime forensics threshold. |
| Phase-1 Enron F1-optimized threshold | 0.98 | `phase1_report.md`; `results/domain_shift_v2/threshold_sensitivity_v2.csv` | phase1 report lines 28-35 | Causes zero positive predictions on V1/V2/Kurdi in zero-shot transfer. |
| Municipal V2 target-calibrated threshold | 0.003329566888943843 | `results/domain_shift_v2/threshold_sensitivity_v2.csv` | row `municipal_v2,target_calibrated_heldout` | Calibration split only; held-out evaluation uses test split. |
| Municipal V2 target-oracle threshold | 0.0032638534180122235 | `results/domain_shift_v2/threshold_sensitivity_v2.csv` | row `municipal_v2,target_oracle_best_f1_upper_bound` | Diagnostic upper bound only. |
| Isolation Forest parameters, EG-SIEM | n_estimators=200, contamination=0.02, random_state=42 | `mini_mesa_EG-SIEM.py` | lines 327-329 | Requires role buffer length >= ml_min and refit interval. |
| Isolation Forest parameters, LSC enhanced model | n_estimators=250, max_samples=auto, contamination=0.01, random_state=42, bootstrap=True | `mini_mesa_LSC.py` | lines 312-318 | EnhancedRoleAnomalyModel implementation. |
| Isolation Forest parameters, CERT baseline | contamination=auto, random_seed CLI default 42 | `cert_eval/cert_baselines.py`; `cert_eval/run_cert_experiments.py` | baseline lines 21-28; runner lines 63-66 | CERT external benchmark baseline. |
| One-Class SVM baseline params | gamma=scale, nu=0.05 | `cert_eval/cert_baselines.py` | lines 27-29 | CERT baseline. |
| EG-SIEM ML minimum samples | 25 | `mini_mesa_EG-SIEM.py`; `mini_mesa_EG-SIEM_Enron.py` | EG line 295; Enron line 428 | Model fit blocked until buffer reaches 25. |
| EG-SIEM ML refit interval | 15 steps | `mini_mesa_EG-SIEM.py`; `mini_mesa_EG-SIEM_Enron.py` | EG line 296; Enron line 429 | Also only checks fit on even steps in EG-SIEM. |
| LSC online learning rate | 0.04 | `mini_mesa_LSC.py` | line 85 | Applies to feature-weight online update. |
| LSC online L2 | 1e-4 | `mini_mesa_LSC.py` | line 86 | Applies to feature-weight online update. |
| LSC online weight clip | 6.0 | `mini_mesa_LSC.py` | line 87 | Feature weights clipped to +/-6.0. |
| LSC ML train sample every | 1 default; 2 in tuned config block | `mini_mesa_LSC.py` | lines 96-99; tuned lines 441-444 | Tuned config block is present but default run uses `SIEMConfig()`. |
| EnhancedRoleAnomalyModel implemented? | Yes | `mini_mesa_LSC.py` | lines 126-151, 310-318 | Rich 24-feature model. |
| EnhancedRoleAnomalyModel used in reported EG-SIEM/Enron experiments? | Not found in current EG-SIEM/EG-SIEM-Enron experiments | searched `mini_mesa_EG-SIEM.py`, `mini_mesa_EG-SIEM_Enron.py`, `mini_mesa_LSC.py`, manuscript | manuscript lines 495-502 | Manuscript correctly frames it as planned upgrade at lines 501-502, but elsewhere should avoid presenting it as evaluated EG-SIEM component. |
| Compliance override function | Not found in current code | searched `override`, `compliance_override`, `ComplianceExploiter`, `approved_export`, `compliance_training` | search returned only contradiction/policy terms | No time-bounded, independently approved override function found. |
| Approved/compliance indicators reduce ToM/risk? | Yes | `mini_mesa_EG-SIEM.py`; `mini_mesa_EG-SIEM_Enron.py`; `mini_mesa_LSC.py` | EG ToM lines 100-107 and policy lines 617-629; Enron ToM lines 332-340; LSC policy lines 1104-1161 | Contradictions and policy multipliers/offsets reduce scores or remove email anchor in LSC. |
| Alerts deleted? | Not found | searched alert emit/delete/suppress terms | EG emit lines 702-711; LSC lines 1302-1305 | Code either emits or does not emit an alert; events remain in event log. No alert-deletion mechanism found. |
| LSC approved partner email anchor suppression | Approved external partner email sets `anchor_email=0.0`; if no anchor remains, correlation is skipped | `mini_mesa_LSC.py` | lines 1150-1152 and 1302-1305 | This is suppression, not audited override. |
| Compliance exploit test | Not implemented in current experiments | `Unified Revision Plan.docx`; repo search | revision-plan text says skip 3E; no `ComplianceExploiter` found | Should be stated as limitation/future deployment safeguard. |

## Paper-Ready LaTeX Hyperparameter Table

```latex
\begin{table*}[t]
\centering
\caption{Hyperparameters and reproducibility settings for the reported experiments.}
\label{tab:hyperparameters_reproducibility}
\small
\begin{tabular}{llll}
\toprule
Component & Parameter & Value & Role in experiment \\
\midrule
\multicolumn{4}{l}{\textit{Simulation and evaluation}} \\
Mesa population & Human agents & 42 & Baseline municipal smart-building simulation scale. \\
Mesa population & Benign employees / power users / malicious insiders & 30 / 4 / 8 & Ground-truth actor composition. \\
Mesa population & System agents & 4 & DBMonitor, EmailMonitor, AuthMonitor, SIEMAgent. \\
Mesa duration & Steps per run & 240 & Main simulation length. \\
Mesa duration & Warm-up steps & 60 & Baseline/anomaly warm-up before attack-enabled test phase. \\
Mesa replication & Runs and seeds & 10 runs, seeds 42--51 & Repeated stochastic simulation runs. \\
Scalability & Human-agent counts & 42, 100, 250, 500, 1000 & Scalability stress-test configurations. \\
Scalability & Step counts / runs & 240 or 480 steps; 3 runs except 1000-agent run = 1 & Runtime/memory scalability evaluation. \\
\midrule
\multicolumn{4}{l}{\textit{SIEM thresholding and scoring}} \\
Alerting & Early threshold & 2.0 & Threshold for early triage alerts. \\
Alerting & Confirmed base threshold & 4.0 & Base value for confirmed escalation. \\
Alerting & Confirmed threshold formula & base + role\_adj + 1.2(trust - 0.5) & Trust- and role-adaptive confirmed threshold. \\
Alerting & Role adjustments & staff 0, analyst 0.8, admin 1.2 & Role-specific confirmed-threshold offsets. \\
Alerting & Cooldown & 6 steps & Minimum gap between same-tier alerts for an actor. \\
Correlation & Sliding window & 48 & Recent activity window for feature extraction. \\
EWMA baseline & Alpha / z clip / weight & 0.08 / 6.0 / 0.9 & Baseline anomaly smoothing and score contribution. \\
Feature score & Weights & See released SIEMConfig dictionary & Weighted risk scoring over anchors, exfiltration, ToM, forensics, peer, and regularity features. \\
\midrule
\multicolumn{4}{l}{\textit{Evidence-gated confirmation}} \\
Evidence gate & Minimum evidence count & 2 & Prevents single-signal confirmed escalation. \\
Evidence gate & Minimum evidence weight & 2.5 & Requires sufficient total evidence weight. \\
Evidence gate & Evidence families & export/staging, unapproved destination, exfil chain, ToM intent, phishing forensics, unauthorized after-hours, email burst, stealth pattern & Independent evidence categories for confirmed alerts. \\
Evidence gate & Single evidence can trigger confirmed? & No & Rich EG-SIEM requires count and weight gates before confirmation. \\
\midrule
\multicolumn{4}{l}{\textit{Trust adaptation}} \\
Trust & Initial / min / max & 0.70 / 0.10 / 0.95 & Bounds user trust used in thresholds. \\
Trust & True-positive update & -0.18 & Confirmed malicious alert lowers trust. \\
Trust & False-positive update & +0.05 & Confirmed benign alert raises trust. \\
Trust & Decay toward baseline & Not implemented in EG-SIEM/Enron; LSC decay configured as 0.00 & No trust return-to-baseline effect in current reported experiments. \\
\midrule
\multicolumn{4}{l}{\textit{ToM and communication evidence}} \\
ToM & Intent threshold / weight & 0.30 / 2.0 & Adds cognitive-intent evidence to SIEM score. \\
ToM & Plan completion threshold & 0.4 & Adds ToM plan feature when plan completion exceeds threshold. \\
ToM & Contradiction factor & 0.85 per contradiction & Approved/scheduled/work-hours context reduces inferred goal probability. \\
Forensics & Forensics weight & 1.5 & Adds phishing forensics score to SIEM score. \\
\midrule
\multicolumn{4}{l}{\textit{Email forensics}} \\
Enron data & Full Enron corpus count & Not found; emails.csv not included & Full-corpus style calibration was not recomputed in current repo. \\
Enron spam data & Usable spam / ham examples & 17,171 / 16,047 & Supervised spam/phishing training after 20-word filter. \\
Enron spam data & Duplicate report & 17,775 exact duplicates; 15,123 normalized groups & Duplicate-leakage audit. \\
Text model & TF--IDF dimension & 10,000 & Email classifier feature dimension. \\
Text model & Classifier & Logistic regression & Selected runtime phishing/spam classifier. \\
Text model & Logistic regression settings & max\_iter=2000, solver=liblinear, class\_weight=balanced, random\_state=42 & Reproducible classifier settings. \\
Text model & Grouped-CV metrics & F1 0.8054, ROC--AUC 0.6512, threshold 0.002 & Runtime forensics model operating point. \\
Domain shift V2 & Enron F1 threshold / Mesa runtime threshold & 0.98 / 0.002 & Threshold-sensitivity comparison settings. \\
Domain shift V2 & Municipal V2 target-calibrated threshold & 0.003329566888943843 & Held-out target calibration result. \\
\midrule
\multicolumn{4}{l}{\textit{Anomaly and online learning}} \\
EG-SIEM ML & Isolation Forest & 200 trees, contamination 0.02, random\_state 42 & Role-based anomaly score. \\
EG-SIEM ML & Minimum samples / refit interval & 25 / 15 steps & Controls online role-model fitting. \\
LSC online update & Learning rate / L2 / weight clip & 0.04 / 1e-4 / 6.0 & Implemented in LSC online feature-weight update. \\
Enhanced anomaly model & EnhancedRoleAnomalyModel & Implemented in LSC; not used in EG-SIEM/Enron reported experiments & Should be framed as planned or separate variant unless evaluated. \\
CERT baseline & Isolation Forest / One-Class SVM & contamination=auto; gamma=scale, nu=0.05 & External CERT r4.2 baselines. \\
\midrule
\multicolumn{4}{l}{\textit{Reproducibility settings}} \\
Randomness & Global seed used in email/domain scripts & 42 & Train/test splits and classifiers. \\
Cross-validation & Enron Phase-1 CV & 5-fold stratified grouped CV when feasible; fallback recorded & Prevents duplicate leakage where group counts allow. \\
Municipal V2 split & Random fine-tuning & Stratified 80/20, random\_state 42 & Target-domain adaptation test. \\
Municipal V2 split & Grouped fine-tuning & GroupShuffleSplit, 200 attempts, 20\% test, random\_state 42 & Template-held-out adaptation test. \\
\bottomrule
\end{tabular}
\end{table*}
```

## Paper-Ready Paragraph for Section 4 / Appendix

To make the simulation and email-forensics experiments reproducible, Table~\ref{tab:hyperparameters_reproducibility} reports the hyperparameters and fixed configuration values used in the released code and result files. Unless otherwise noted, all SIEM variants are evaluated under matched Mesa simulation conditions, with the same baseline population, the same 60-step warm-up period, and the same ground-truth insider assignments across runs. The table records the alert thresholds, trust-adaptive threshold formula, trust update bounds and deltas, evidence-gating conditions, EWMA baseline settings, ToM and communication-forensics weights, anomaly-model parameters, email-classifier settings, and seed/split configuration used for reproducibility. Values that were not implemented in the current experiments are explicitly marked as not implemented rather than inferred.

## Section 4.2 Subsection: Compliance-Related Contradiction Handling and Safeguards

### Compliance-related contradiction handling and safeguards

The current implementation does not use compliance context as an unconditional alert deletion mechanism. Instead, approved operational context is treated as contradiction or policy evidence that can reduce risk confidence. For example, ToM goal probabilities are reduced when actions carry operationally benign context such as tickets, approved partners, scheduled reporting cycles, or work-hours activity. The SIEM policy layer can also discount scores for scheduled partner-reporting behavior with ticket metadata and avoid treating approved-partner email as an unapproved external transfer. These mechanisms reduce intent/risk confidence when approved operational context exists, but they do not remove the underlying event from the simulator event log or prevent later correlation with other suspicious behavior.

For deployment, this contradiction-handling layer should be guarded so that compliance context cannot become a blind spot. A compliance-related suppression should require independent approval, a valid ticket or change-control identifier, and a bounded change window; all suppressed or discounted events should be written to immutable audit logs. The suppression should not apply to high-severity chains that combine privilege escalation with exfiltration, unauthorized BMS control-plane modification, or suspicious remote-access behavior. Repeated reliance on compliance-context suppression by the same actor or vendor should itself become a risk signal and lower the confidence assigned to future compliance claims. These safeguards were not fully implemented or evaluated in the current experiments and should be described as deployment requirements rather than completed validation.

## Response-Letter Text

We agree that the original manuscript did not provide enough methodological detail for reproduction and that the compliance-override wording could imply an unsafe alert-deletion mechanism. In the revision, we added a hyperparameter and reproducibility table in Section 5 / Appendix A reporting the simulation population, run length, warm-up period, seeds, SIEM thresholds, trust-adaptive threshold formula, base and slope values, trust bounds and update deltas, EWMA settings, evidence-gating count and weight requirements, Isolation Forest settings, email-forensics classifier settings, and target-domain calibration thresholds. We also revised Section 4.2 to clarify that the current code implements compliance-related contradiction handling and policy score reduction, not unconditional alert deletion. Approved operational context can reduce risk confidence, but underlying events remain available for audit and future correlation. We added deployment safeguards as requirements: independent approval, valid ticket/change window, immutable logging, exclusion for high-severity privilege-escalation/exfiltration or unauthorized BMS-control chains, and monitoring repeated override use as suspicious. We also corrected Section 4.4 to avoid overstating trust decay: trust is updated after confirmed alerts and bounded between 0.10 and 0.95, but decay toward baseline is not active in the reported EG-SIEM/EG-SIEM-Enron experiments.

## Manuscript Edit Suggestions

### Section 4.2 SIEM Layering

Insert the hyperparameter cross-reference after the paragraph describing early and confirmed thresholds. Replace the current compliance sentence:

> Lastly, a compliance override is incorporated to ignore alerts indicating confirmed escalation in the presence of strong compliance factors.

with:

> The current implementation treats approved operational context as contradiction and policy evidence rather than as unconditional alert deletion. Ticketed scheduled reporting or approved-partner communication can reduce risk confidence, while the underlying event remains available for audit and later correlation. Deployment use of this mechanism requires additional safeguards, including independent approval, bounded change windows, immutable logging, and exclusion for high-severity privilege-escalation, exfiltration, or unauthorized BMS-control chains.

### Section 4.4 Theory-of-Mind and Trust Modeling

Revise the sentence:

> Trust is bounded between 0.10 and 0.95. Over time, trust gradually decays toward the baseline.

to:

> Trust is bounded between 0.10 and 0.95. In the reported EG-SIEM and EG-SIEM-Enron experiments, trust changes only after confirmed alert outcomes: a true positive applies a negative trust delta and a false positive applies a positive trust delta. Although the LSC code includes a decay hook, its configured decay rate is 0.00, so decay toward baseline is not active in the reported experiments.

Also revise:

> While our current SIEM score is primarily feature-driven, we plan to integrate these ToM signals into future alert models...

because ToM is already integrated into EG-SIEM scoring (`tom_intent`, `tom_plan`, and `tom_weight`). Suggested replacement:

> The current EG-SIEM implementation incorporates ToM intent and plan-completion signals as weighted evidence in the SIEM score and evidence gate; future work will evaluate richer coordinated-intent models beyond these feature-level ToM signals.

### Section 5 Implementation and Results / Appendix A

Add the LaTeX table above as Appendix A or immediately after the implementation paragraph that reports 42 human agents, 240 steps, and 60 warm-up steps. The table should be introduced by the paper-ready paragraph above.

### Limitations

Add:

> Compliance-context handling was evaluated as rule-based contradiction and risk-score reduction, not as a complete deployment-grade compliance override. The present experiments do not include an adversarial compliance-exploiter scenario, independent approval workflow, immutable audit log, or time-bounded change-window enforcement. These safeguards are required before deployment and remain future work.

### Current Text That Should Be Revised

- The trust-decay sentence in Section 4.4 is unsupported by the current reported EG-SIEM/EG-SIEM-Enron experiments.
- The compliance-override sentence in Section 4.2 overstates implementation and should be rewritten as contradiction-aware policy handling plus proposed safeguards.
- The ToM “future alert models” sentence conflicts with code that already uses ToM features in scoring; revise to distinguish current feature-level ToM integration from future richer models.
- The online-learning description should specify scope: logistic online feature-weight update is implemented in LSC/CE variants, while the reported EG-SIEM-Enron code primarily uses the Enron-trained communication-forensics classifier plus role-based anomaly scoring. If the paper claims online logistic learning for EG-SIEM-Enron, that should be corrected or explicitly limited to the LSC/CE implementation.
