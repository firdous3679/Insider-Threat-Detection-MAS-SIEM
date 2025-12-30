# Insider Threat Detection MAS-SIEM

A research-driven implementation of insider threat detection using Multi-Agent Simulation (MAS) and a layered SIEM framework. This repository contains three variants of the framework to analyze the impact of cognitive reasoning, communication forensics, and evidence-gated validation on detection quality, alert precision, and operational workload.

> This work is part of a broader research project evaluating hybrid insider threat detection systems that combine behavioral analytics, trust-aware learning, Theory-of-Mind (ToM) inference, and role-aware anomaly modeling.

---

## Repository Overview

| File | Description |
|------|-------------|
| `mini_mesa_LSC.py` | Baseline Layered SIEM-Core implementation |
| `mini_mesa_CE-SIEM.py` | Cognitive-Enriched SIEM variant |
| `mini_mesa_EG-SIEM.py` | Evidence-Gated SIEM variant |
| `sweep_thresholds_ablation_LSC.py` | Threshold sensitivity sweep for LSC |
| `threshold_sweep_ablation_LSC.csv` | Swept data for baseline analysis |
| `README.md` | This document |

---

## Implementation Variants

We evaluate **three configurations** of the insider threat detection framework:

1. **Layered SIEM-Core (LSC)**  
   Traditional SIEM pipeline with policy checks, EWMA behavioral scoring, static thresholds, and anomaly detection — *no cognitive or communication cues*.

2. **Cognitive-Enriched SIEM (CE-SIEM)**  
   Builds on LSC by integrating **Theory-of-Mind (TomAbd)** intent indicators and **communication forensics** (phishing-oriented NLP and AI-text/authorship validation).

3. **Evidence-Gated SIEM (EG-SIEM)**  
   Adds **precision-oriented mechanisms** over CE-SIEM:  
   ✓ Evidence accumulation (requires multiple independent signals before confirmation)  
   ✓ Peer-group normalization for role baselines  
   ✓ Regularity suppression for scheduled behaviors  
   ✓ Contradiction-aware ToM validation  
   ✓ Role-adaptive thresholds

---

##  Results (Summary)

Across 10 simulation runs involving eight insider actors:

- **LSC Baseline**  
  F1 ≈ 0.521, Precision ≈ 0.369, Recall ≈ 0.888  
  Confirmed precision ≈ 0.543 at threshold = 4.

- **CE-SIEM**  
  Improved detection sensitivity with F1 ≈ 0.774, perfect recall (1.000), but higher alert volume.

- **EG-SIEM**  
  Best operational profile: F1 ≈ 0.922, Precision ≈ 0.975, Recall ≈ 0.875  
  Confirmed alert precision ≈ 0.997 with minimal false positives and comparable detection latency.

These results show cognitive context improves sensitivity, while evidence gating dramatically reduces false positives and alert burden.



---

## 🧪 How to Run

1. Install Python and dependencies (e.g., `mesa`, `pandas`, `numpy`, `scikit-learn`).  Needs Mesa 3.3.1 version to execute it.
2. Run the baseline or variant script:

```bash
python mini_mesa_LSC.py
python mini_mesa_CE-SIEM.py
python mini_mesa_EG-SIEM.py
