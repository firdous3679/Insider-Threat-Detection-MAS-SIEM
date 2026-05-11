# Phase 2 Diagnostic Report — CERT r4.2 Pipeline

**Status:** code fixes applied; only `cert_user_day_labeled.csv` re-emitted.
Baselines, ablations, scalability, statistical tests, and Phase-2 final
tables (`baseline_results.csv`, `cert_ablation_results.csv`,
`table_b_external_benchmark_results.csv`, `scalability_results.csv`,
`table_d_statistical_significance.csv`) have **not** been regenerated
and are still on disk in their pre-fix state. Awaiting review of this
report before proceeding.

---

## Issue 1 — Only 2 of 70 malicious actors were being loaded

### Root cause (two compounding bugs)

1. **`_load_ldap` returned only the latest monthly snapshot** (`LDAP/2011-05.csv`).
   CERT employees can leave during the 17-month simulation; many do so
   *because they are insider attackers and get fired*. The 2011-05
   snapshot contains 845 of the 1,000 r4.2 employees and **none of the
   70 malicious actors** (verified directly: 0 of 70 actor IDs intersect
   with 2011-05's `user_id` column). Every actor was therefore absent
   from the bundle's LDAP frame, breaking peer/role features and
   silently excluding actors from any pool derived from LDAP.

2. **`_resolve_target_users` picked the first N users from that
   actor-free LDAP**, then streamed-filtered every modality CSV by that
   pool. So `--max_users 1000` actually retained 845 LDAP users plus a
   few accidental actor hits (the 2 we saw in the stale CSV) and
   dropped the rest of the 70 actors entirely.

### Fix

`cert_eval/cert_loader.py`:

* `_load_ldap` now reads **every** monthly snapshot and keeps the most
  recent record per `user_id` (chronological filename sort + `keep="last"`).
  The unioned LDAP frame has 1,000 rows — all 70 actors present.
* New `_malicious_actor_set(...)` reads `answers/insiders.csv`, filters
  by release, and returns the actor user-set.
* `_resolve_target_users` builds the user pool from the unioned LDAP
  (with logon-scan fallback) and **always unions in the malicious actor
  set** before returning. A small `--max_users` value can no longer
  silently drop labeled actors. The loader emits a warning naming how
  many actors were force-included.

`cert_eval/cert_label_builder.py`: rewritten to canonicalize both sides
of the (user, day) match to `YYYY-MM-DD` strings via a single
`_to_day_str` helper, eliminating timezone-string mismatches between
`datetime.date` objects in features and `pd.Timestamp` objects in
answers. The label builder also prints a one-line diagnostic
(`actor_label=1 users: N | user_day_label=1 rows: M | answer rows
seen: K`) so future regressions are immediately visible.

### Verification (re-emitted `cert_user_day_labeled.csv`)

```
Target users: 252  (200 LDAP-first + 52 force-included actors)
Bundle: logon=204,937 device=134,368 file=133,632 email=612,547
        ldap=1,000 answers=7,323
Labeled rows         : 76,622
Unique users         : 252
actor_label=1 users  : 70   ✅  (was 2; target ~70)
user_day_label=1 rows: 966  ✅  (was 20; target several hundred)
```

Note on user count: the answers files yield 986 distinct (user, day)
pairs; the labeler matched 966. The 20-row gap is days where the actor
had no logon/device/file/email events at all (and the diagnostic re-emit
skipped http.csv — see "Caveat" below). Once http.csv is loaded in the
full re-run, this gap will shrink.

### Caveat on the re-emitted CSV

This re-emit deliberately set `http=pd.DataFrame()` to keep wall time
under the diagnostic budget. http.csv (14 GB) affects `http_count`,
`suspicious_http_count`, and `after_hours_http_count` features but does
**not** change `actor_label` or `user_day_label` — both are derived
from `answers/`, not from logs. Therefore the actor-label and
user-day-label numbers reported above are valid for the full pipeline.

When you re-run `run_cert_experiments.py` end-to-end, http.csv will be
included automatically and the http-derived feature columns will fill
in.

---

## Issue 2 — Detection thresholds flagged every user-day

### Root causes (multiple)

Inspecting the previous `cert_eg_siem_runner.py`:

1. **Target leakage in trust factor**:
   `trust_factor = 1.0 / (1.0 + df['actor_label']*0.05)` — the model
   consumed the ground-truth label. Even though the effect was small
   (~5% suppression), this is invalid by construction and was actually
   pushing actor scores *down*.
2. **`tom_like` was trivially true**:
   `(logon_count > 0) & (file_copy>0 | http_susp>0 | email_ext>0)`.
   Almost every active user-day satisfied this, so the +1.5×tom_like
   term lifted nearly every row above the hardcoded `risk_score >= 3.0`
   threshold.
3. **Hardcoded threshold** (`3.0`) — never calibrated against the
   benign distribution. The Mesa pipeline uses a benign-percentile
   gate.
4. **No real peer-group normalization**. The only "peer" term was
   `role_peer_deviation_score`, which the feature builder z-scored on
   `file_access_count` alone. Modality-level evidence streams were
   never normalized within role.
5. **Auth fired on `logon_count > 8`**, which many normal users hit
   during a busy workday.

### Fix

Rewrote `cert_eval/cert_eg_siem_runner.py` with:

* **Removed `actor_label` leakage entirely.** No part of the score
  depends on the ground-truth label.
* **Real peer-group z-scoring.** Each evidence stream
  (auth/device/file/web/email/tom_like) is z-scored within the user's
  LDAP `role` (with global-z fallback for empty/single-row groups).
* **Tighter ToM-like sequence rule.** A user-day fires it only when
  *all three* of (logon, data-pull = file_copy or sensitive_file,
  exfil-channel = external_email/attachment_email/suspicious_http/device)
  coexist. Mirrors the Mesa TomAbd pattern of "intent observed across
  the action chain", not "logon AND any one suspicious thing".
* **Threshold calibrated on benign data.** The aggregate operating
  point is the 95th percentile of `risk_score` on rows where
  `user_day_label == 0`. Per-modality triggers fire when their
  z-score > 2.0.
* **Proper evidence gate.** Confirmed alert requires
  `risk >= benign_p95` **and** at least 2 distinct evidence categories
  triggered. Early alert is `risk >= benign_p95` **or** ≥1 categories
  (exposed for TTD; not used as the published metric).
* **CERT-LSC** retuned to a 4-of-4 layered-correlation baseline (alert
  when ≥3 of {auth, file, web, email} layers fire on raw counts) — closer
  to a real layered SIEM rule and harder than the previous ≥2 threshold.
* **email_only ablation** still gated by benign-95p but with the
  ≥2-categories rule relaxed to "email triggered" since it's a
  single-modality ablation by definition. This addresses Issue 4
  directly.

### Expected behavior

Empirically the benign-95p threshold caps benign FPs per row at ~5%, so
on a 250-user × 300-day pool we should see FP/day in the tens (not
hundreds). Actor recall will depend on whether actors' behavior on
their malicious days is anomalous *relative to their role peers* — the
intended question.

I have not regenerated the ablation/baseline numbers because the brief
explicitly forbids it; the next pipeline re-run will produce them.

---

## Issue 3 — TTD was 0.0 hours for every variant

### Root cause

`run_cert_experiments.py::_evaluate_method` set
`met.update({..., 'ttd_hours': 0.0})` unconditionally — TTD was never
computed.

### Fix

`cert_eval/cert_metrics.py`:

* Added `time_to_detection(pred_df, ...)` — for every actor (`actor_label==1`),
  compute the Timestamp delta between the actor's first
  `user_day_label==1` row and the actor's first `pred_alert==1` row.
  TTD is `NaN` when the actor was never detected, or when the alert
  fired *before* any malicious day for that actor (pre-event alerts
  don't count as detections; they're false positives that happen to
  hit an actor).
* Added `summarize_ttd(...)` — mean/median TTD hours and detected-actor
  count.
* `_evaluate_method` in `run_cert_experiments.py` now calls these and
  populates `ttd_hours` (median across detected actors), plus actor-level
  precision/recall/F1 from the new `actor_level_metrics`.

### Expected behavior after re-run

For a method that detects K of 70 actors, TTD is the median over those
K actors. Realistic CERT numbers in literature: minutes to hours for
EG-SIEM on the same day, days to weeks for unsupervised baselines.

---

## Issue 4 — "EG-SIEM email only" had F1=0 with ROC-AUC=0.82

### Root cause

The pre-fix runner applied the same `risk_score >= 3.0` gate to every
mode. In `email_only`, every other modality was zeroed, so the
aggregate risk score collapsed to a small range — nothing crossed 3.0,
hence recall = 0 even though the rank ordering (and ROC-AUC) was fine.

### Fix

The benign-95p threshold described under Issue 2 *is* the calibration
the brief asked for. For `email_only` specifically, the gate becomes:

```
risk >= benign_p95(email_only_score)  AND  email_trigger == 1
```

The ≥2-categories rule is dropped for this single-modality ablation
(it cannot be satisfied by definition). This will let the operating
point fire at roughly the same recall as the ROC-AUC suggests is
achievable, while keeping FP/day bounded by the 5% benign budget.

---

## Issue 5 — Scalability timings implausibly fast

### Root cause

`cert_scalability.py` timed only `run_cert_eg_siem` on an already-
aggregated user-day matrix (one row per user-day, ~thousands of rows).
That step is purely vectorized pandas arithmetic, hence the
35,402 events / 0.006 s = 5.8M evt/s figure. The user-visible cost of
the pipeline is in feature engineering over millions of raw events.

### Fix

Rewrote `cert_eval/cert_scalability.py`:

* Now accepts the **bundle** (raw modality tables) instead of the
  labeled user-day frame.
* Per scale point, filters the bundle to the first N users, then times
  the **end-to-end pipeline**:
  `feature build → label build → EG-SIEM full`.
* `events_processed` now reports the sum of raw modality rows handled,
  not the user-day row count, so events/sec is meaningful.
* `user_sizes` default updated to `(100, 250, 500, 1000)` per the brief.

`run_cert_experiments.py` updated to call
`run_scalability(bundle, output_dir=out, user_sizes=(100, 250, 500, ...))`.

### Expected behavior after re-run

Tens of seconds at 100 users to several minutes at 1,000 users. If a
size point is below 1 second, that's a sign the bundle was filtered
empty and we should investigate.

---

## Issue 6 — Missing Phase 2 experiments (deferred until Issues 1–5 verified)

Two experiments from the Phase 2 plan are still absent from the repo:

1. **Enron → CERT-email transfer evaluation**
   `enron_to_cert_email_transfer.csv`. Train an Enron Spam-style
   classifier on Enron mail; evaluate zero-shot on CERT email rows
   whose senders are listed in `answers/insiders.csv`. Critical for
   Reviewer 2 Q2 (Enron-to-smart-building domain shift).
2. **LLM-embedding baseline (sentence-transformer)** as a row in
   `baseline_results.csv` alongside Isolation Forest and One-Class SVM.

Per the brief, neither is started until Issues 1–5 are verified. I have
not modified `cert_baselines.py` in this pass.

When you give the green light I will:

* Add a `cert_eval/cert_email_transfer.py` that trains on Enron and
  evaluates on CERT email senders, writing
  `results/cert_r42/enron_to_cert_email_transfer.csv`.
* Add a `SentenceTransformerOneClass` (or pooled-embedding +
  IsolationForest) baseline to `cert_eval/cert_baselines.py` and a
  matching row in `baseline_results.csv`.

Suggested sentence-transformer model: `all-MiniLM-L6-v2` (CPU-friendly).
Email and HTTP `content` columns are usable text inputs. Will confirm
the pooling strategy with you before running.

---

## Issue 7 — Statistical significance was descriptive only

### Status

`table_d_statistical_significance.csv` currently has one row with
`Test used = "descriptive"`. The brief's plan: 10 seeds across the five
CERT-EG-SIEM ablation variants, paired Wilcoxon signed-rank on F1 and
TTD across seeds.

### Plan (deferred until after Issues 1–5 verified)

When the corrected pipeline re-runs, I will:

1. Loop the experiment runner over 10 random seeds (CERT data is
   deterministic; the seed governs the IsolationForest / OCSVM /
   benign-percentile bootstrap sampling we'll add).
2. Persist per-seed F1 and TTD vectors per method in
   `results/cert_r42/cert_perseed_results.csv`.
3. Use `paired_significance` (already in `cert_metrics.py`) to compute
   Wilcoxon signed-rank for the comparisons:
   - CERT-EG-SIEM full vs CERT-LSC
   - CERT-EG-SIEM full vs CERT-EG-SIEM without email
   - CERT-EG-SIEM full vs CERT-EG-SIEM without ToM-like
   - CERT-EG-SIEM full vs Isolation Forest
   - CERT-EG-SIEM full vs One-Class SVM
4. Emit `table_d_statistical_significance.csv` with one row per
   comparison: `Comparison, Metric, Mean diff, Test used, p-value, Significant?`
   (significance threshold p < 0.05 with optional Holm-Bonferroni for
   multiple comparisons).

The `paired_significance` helper already gracefully handles
all-zero-difference and small-N edge cases via try/except.

---

## Files updated in this diagnostic pass

* `cert_eval/cert_loader.py` — unioned LDAP, force-include actors in
  `keep_users`, release-aware answers, ragged-row CSV reader.
* `cert_eval/cert_label_builder.py` — canonical day matching with
  diagnostic print.
* `cert_eval/cert_eg_siem_runner.py` — full rewrite (no leakage,
  benign-calibrated thresholds, real peer z-scoring, tighter ToM-like).
* `cert_eval/cert_metrics.py` — added `actor_level_metrics`,
  `time_to_detection`, `summarize_ttd`; hardened
  `paired_significance`/`summarize_statistics` for edge cases.
* `cert_eval/cert_scalability.py` — full rewrite to time the end-to-end
  pipeline at (100, 250, 500, 1000) users.
* `cert_eval/cert_feature_builder.py` — defensive against
  bool-default `df.get(...)` returns when modality frames are empty.
* `cert_eval/run_cert_experiments.py` — wires new TTD + actor metrics
  through `_evaluate_method`; passes `bundle` (not labeled_df) to
  scalability.
* `results/cert_r42/cert_user_day_labeled.csv` — re-emitted with the
  corrected labels (70 actors, 966 malicious user-days).

## Files NOT touched (held for review per the brief)

* `results/cert_r42/baseline_results.csv`
* `results/cert_r42/cert_ablation_results.csv`
* `results/cert_r42/scalability_results.csv`
* `results/cert_r42/cert_statistical_summary.csv`
* `results/cert_r42/table_a_dataset_summary.csv`
* `results/cert_r42/table_b_external_benchmark_results.csv`
* `results/cert_r42/table_c_scalability.csv`
* `results/cert_r42/table_d_statistical_significance.csv`
* `results/cert_r42/fig_*.png`

These will be regenerated only when you confirm this report.
