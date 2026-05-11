# CERT Embedding Baseline Report

## What This Model Is And Is Not

This baseline is a representation-learning anomaly detector. It is not a generative LLM, does not call GPT, and does not use any API. It converts existing CERT user-day numeric features into short behavioral text descriptions, encodes those descriptions with `all-MiniLM-L6-v2`, and trains a One-Class SVM on benign embeddings.

- Model: `all-MiniLM-L6-v2`
- Anomaly detector: One-Class SVM (`nu=0.05`, `kernel='rbf'`, `gamma='scale'`)
- Alert threshold: top 1% of anomaly scores
- Encoding batch size: 64
- Encoding time: 20.68 seconds
- Rows available: 330,452
- Rows evaluated: 20,000
- Runtime guard: The discovered CERT feature matrix contains 330,452 user-days, which exceeds the 76,622 rows expected by the task. A full RBF One-Class SVM fit on all benign embeddings was not tractable in the local CPU run, so the baseline used the specified 20,000-row stratified fallback.

## Results

| method | precision | recall | f1 | roc_auc | pr_auc | fp_per_day | actor_precision | actor_recall | actor_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Isolation Forest | 0.006908 | 0.2059 | 0.01337 | 0.7991 | 0.008326 | 58.25 | 0.1569 | 0.8889 | 0.2667 |
| One-Class SVM | 0.003748 | 0.06288 | 0.007074 | 0.4549 | 0.0028 | 32.9 | 0.08183 | 0.8194 | 0.1488 |
| CERT-EG-SIEM full | 0.01273 | 0.1836 | 0.0238 | 0.7943 | 0.01174 | 28.03 | 0.2188 | 0.7778 | 0.3415 |
| Embedding-OC-SVM (all-MiniLM-L6-v2) | 0 | 0 | 0 | 0.4849 | 0.002982 | 0.4202 | 0.0339 | 0.02778 | 0.03053 |

## Paper-Ready Paragraph

To address the request for a language-model-style comparison without using a generative model or external API, we added a representation-learning anomaly baseline. Each CERT user-day feature vector was converted into a short textual behavioral description and encoded with all-MiniLM-L6-v2; a One-Class SVM was then trained on benign user-day embeddings and the top 1% anomaly scores were flagged. This baseline tests whether dense text representations of behavioral summaries improve over tabular anomaly detection. The comparison shows whether the proposed evidence-gated SIEM retains an advantage over an embedding-only detector under the same CERT labels.

## Response To Reviewer 3 Q4

We added an embedding-based anomaly detector to the CERT r4.2 external benchmark in response to Reviewer 3. This is framed as a representation-learning baseline, not as a large language model: all-MiniLM-L6-v2 encodes textual summaries of user-day behavioral features, and One-Class SVM performs anomaly detection on the resulting embeddings. The new row `Embedding-OC-SVM (all-MiniLM-L6-v2)` has been added to the CERT benchmark table, allowing direct comparison with Isolation Forest, tabular One-Class SVM, and CERT-EG-SIEM full.
