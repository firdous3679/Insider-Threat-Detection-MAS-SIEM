# Insider Threat Detection MAS-SIEM

A research-driven implementation of insider threat detection using **Multi-Agent Simulation (MAS)** and a layered **SIEM** correlation pipeline. This repository includes multiple framework variants to study how **cognitive reasoning (ToM)**, **communication/email forensics**, and **evidence-gated validation** affect detection quality, alert precision, false-positive rates, and time-to-detection.

> Part of an ongoing research effort on hybrid insider-threat detection that combines behavioral analytics, trust-aware learning, Theory-of-Mind (TomAbd) inference, and role-aware anomaly modeling.

---

## Repository Overview

| File | Description |
|------|-------------|
| `mini_mesa_LSC.py` | Baseline **Layered SIEM-Core (LSC)** implementation |
| `mini_mesa_CE-SIEM.py` | **Cognitive-Enriched SIEM (CE-SIEM)** variant |
| `mini_mesa_EG-SIEM.py` | **Evidence-Gated SIEM (EG-SIEM)** variant |
| `mini_mesa_EG-SIEM_Enron.py` | **EG-SIEM-Enron** variant (loads a pretrained Enron-based email forensics model) |
| `enron_combined_training.py` | Trains the **combined email forensics model** (baseline calibration + spam/phishing classifier) |
| `combined_forensics_model.pkl` | Pretrained combined forensics model artifact used by EG-SIEM-Enron |
| `enron_spam_data.csv` | Labeled Enron spam/ham dataset used for supervised phishing/spam training |
| `sweep_thresholds_ablation_LSC.py` | Threshold sensitivity sweep for LSC |
| `threshold_sweep_ablation_LSC.csv` | Output data for the LSC threshold sweep |
| `README.md` | This document |

---

## Implementation Variants

We evaluate **four** configurations of the insider threat detection framework:

1. **Layered SIEM-Core (LSC)**  
   Traditional SIEM pipeline with policy checks, EWMA behavioral scoring, static confirmation thresholds, and anomaly detection. No cognitive reasoning or communication-derived evidence.

2. **Cognitive-Enriched SIEM (CE-SIEM)**  
   Builds on LSC by integrating **Theory-of-Mind (TomAbd)** intent indicators and **communication/email forensics** (phishing-oriented NLP features and AI-text/authorship consistency cues).

3. **Evidence-Gated SIEM (EG-SIEM)**  
   Adds precision-oriented mechanisms on top of CE-SIEM:  
   - Evidence-gated accumulation (multiple independent signals before confirmation)  
   - Peer-group normalization for role baselines  
   - Regularity suppression for routine/scheduled behaviors  
   - Contradiction-aware ToM validation  
   - Role-adaptive thresholds

4. **Enron-enabled Evidence-Gated SIEM (EG-SIEM-Enron)**  
   Extends EG-SIEM by loading a **pretrained Enron-based email forensics module** at runtime (`combined_forensics_model.pkl`). This module provides calibrated phishing likelihood and style-baseline deviation signals (derived from Enron corpora) that are fused into the evidence-gated alert scoring and correlation.

---

## Results (Summary)

Averaged over **10 simulation runs** with **8 malicious insiders**:

- **LSC (θ = 4)**  
  Actor F1 = 0.521 (P = 0.369, R = 0.888)  
  Confirmed-alert precision = 0.543, Confirmed FP ≈ 31.6/run

- **CE-SIEM**  
  Actor F1 = 0.774 (P = 0.633, R = 1.000)  
  Higher alert volume (≈152 confirmed alerts/run), confirmed-alert precision = 0.677

- **EG-SIEM**  
  Actor F1 = 0.922 (P = 0.975, R = 0.875)  
  Confirmed-alert precision = 0.997, Confirmed FP ≈ 0.2/run

- **EG-SIEM-Enron**  
  Actor F1 = 0.933 (P = 1.000, R = 0.875)  
  Confirmed-alert precision = 1.000, Confirmed FP = 0.0/run  
  Faster detection: average TTD ≈ 10.26 steps (vs. 15.20 for EG-SIEM)

Overall: cognitive and communication evidence improves sensitivity, evidence gating reduces false positives, and Enron-calibrated email forensics further improves timeliness while preserving a low-noise alert posture.

---

##  How to Run

### 1) Install dependencies
Recommended: Python 3.10+.

Key packages:
- `mesa==3.3.1` (required)
- `numpy`, `pandas`
- `scikit-learn`

Example:
```bash
pip install mesa==3.3.1 numpy pandas scikit-learn
python mini_mesa_LSC.py
python mini_mesa_CE-SIEM.py
python mini_mesa_EG-SIEM.py
python mini_mesa_EG-SIEM_Enron.py --model combined_forensics_model.pkl

**If you want to retrain the pretrained artifact:**
python enron_combined_training.py

### 2) Required inputs

enron_spam_data.csv (included in the repo)

emails.csv (Full Enron Email Corpus) not included in this repository due to size

Download the full Enron corpus from Kaggle:
https://www.kaggle.com/datasets/wcukierski/enron-email-dataset

Place emails.csv in the repository root (or update the file path inside enron_combined_training.py).

The training script produces:

combined_forensics_model.pkl
