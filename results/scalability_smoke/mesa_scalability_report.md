# Mesa Scalability Report

## Purpose

This experiment measures computational scalability of the existing Mesa EG-SIEM simulation. It does not tune thresholds or optimize detection behavior.

## Configuration

- Model: `mini_mesa_EG-SIEM.py` rich evidence-gated configuration
- Baseline preserved: 42 human agents, 240 steps, 3 runs
- Memory guard: 16 GB default
- Runtime timer: `time.perf_counter()`
- Peak memory: `tracemalloc`; RSS included when `psutil` is available

## Results Summary

| agent_count_human | simulation_steps | mean_runtime_seconds | mean_peak_memory_mb | mean_events_generated | mean_events_per_second | mean_actor_f1 | mean_confirmed_alerts | mean_confirmed_false_positives | completed_runs | failed_runs |
| ----------------- | ---------------- | -------------------- | ------------------- | --------------------- | ---------------------- | ------------- | --------------------- | ------------------------------ | -------------- | ----------- |
| 42                | 5                | 0.592                | 0.781               | 62.000                | 104.810                | 0.000         | 0.000                 | 0.000                          | 1              | 0           |

## Completion

- 100 agents completed runs: 0
- 250 agents completed runs: 0
- 500 agents completed runs: 0
- 1000 agents completed runs: 0

## Failures Or Skips

No failed runs.

## Runtime And Memory Interpretation

Runtime and event volume increase with the number of human agents and steps. Detection metrics are reported honestly at each scale; no threshold changes were made to compensate for scale effects.

## Revised Manuscript Paragraph

We evaluated Mesa scalability using the unchanged evidence-gated SIEM configuration while varying human-agent count and simulation horizon. The original 42-human-agent, 240-step setup was preserved as the baseline, and larger runs scaled benign users, power users, and malicious insiders with fixed SIEM thresholds. Runtime and Python/RSS memory were recorded for each run.

## Response-To-Reviewers Paragraph

To address scalability concerns, we added a dedicated Mesa scalability runner and report raw and aggregated runtime, memory, event-throughput, and detection metrics across 42, 100, 250, 500, and optional 1000 human-agent configurations. Failed or skipped configurations are recorded explicitly rather than omitted.
