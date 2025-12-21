"""
sweep_thresholds_ablation_updated.py

Runs threshold sweeps across different SIEM layer configurations ("ablations")
for the MESA insider-threat Mini-MAS model (v5 adaptive layers).

Key fixes vs older sweep scripts:
- Robust module loading from a .py filename (works on Windows paths).
- Runs warmup + test (so online learning/baselines can warm up).
- Evaluates ONLY test-phase events when phase markers exist.
- Creates SIEMConfig safely even if fields differ (uses setattr when available).
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# -----------------------------
# Phase filtering (test only)
# -----------------------------
def _is_test_event(e: Dict[str, Any]) -> bool:
    m = e.get("meta") or {}
    return (e.get("phase") == "test") or (m.get("phase") == "test")


def filter_to_test(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """If phase markers exist, keep only test-phase events; otherwise return original."""
    has_phase = any(
        (e.get("phase") in ("train", "test")) or ((e.get("meta") or {}).get("phase") in ("train", "test"))
        for e in events
    )
    if not has_phase:
        return events

    test = [e for e in events if _is_test_event(e)]
    return test if test else events


# -----------------------------
# Evaluation (two tiers)
# -----------------------------
EARLY_TYPE = "alert_early"
CONF_TYPE = "alert_confirmed"


def eval_from_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    events = filter_to_test(events)

    actors = sorted({e["actor_id"] for e in events if "actor_id" in e})

    # Ground truth: malicious if actor ever emitted malicious-labeled event
    actor_truth = {a: "benign" for a in actors}
    first_mal_step: Dict[int, int] = {}
    for e in events:
        a = e.get("actor_id")
        if a is None:
            continue
        if e.get("label") == "malicious":
            actor_truth[a] = "malicious"
            first_mal_step[a] = min(first_mal_step.get(a, 10**9), int(e.get("step", 10**9)))

    # Collect alerts
    alerts_early = [e for e in events if e.get("event_type") == EARLY_TYPE]
    alerts_conf = [e for e in events if e.get("event_type") == CONF_TYPE]

    # First confirmed per actor
    first_conf_step: Dict[int, int] = {}
    conf_by_actor = defaultdict(list)
    for e in alerts_conf:
        a = e["actor_id"]
        conf_by_actor[a].append(e)
        first_conf_step[a] = min(first_conf_step.get(a, 10**9), int(e.get("step", 10**9)))

    # Actor-level confusion (confirmed alerts drive detection)
    tp = fp = tn = fn = 0
    ttd: List[int] = []

    for a in actors:
        truth = actor_truth[a]
        detected = a in conf_by_actor

        if truth == "malicious" and detected:
            tp += 1
            if a in first_mal_step and a in first_conf_step:
                ttd.append(first_conf_step[a] - first_mal_step[a])
        elif truth == "malicious" and not detected:
            fn += 1
        elif truth == "benign" and detected:
            fp += 1
        else:
            tn += 1

    actor_precision = tp / (tp + fp) if (tp + fp) else 0.0
    actor_recall = tp / (tp + fn) if (tp + fn) else 0.0
    actor_f1 = (2 * actor_precision * actor_recall / (actor_precision + actor_recall)) if (actor_precision + actor_recall) else 0.0

    # Alert-level precision (early + confirmed separately)
    early_tp = sum(1 for e in alerts_early if e.get("label") == "malicious")
    early_fp = sum(1 for e in alerts_early if e.get("label") != "malicious")
    early_prec = early_tp / (early_tp + early_fp) if (early_tp + early_fp) else 0.0

    conf_tp = sum(1 for e in alerts_conf if e.get("label") == "malicious")
    conf_fp = sum(1 for e in alerts_conf if e.get("label") != "malicious")
    conf_prec = conf_tp / (conf_tp + conf_fp) if (conf_tp + conf_fp) else 0.0

    return {
        "actor_precision": actor_precision,
        "actor_recall": actor_recall,
        "actor_f1": actor_f1,
        "ttd_avg_conf": (sum(ttd) / len(ttd)) if ttd else None,
        "ttd_max_conf": max(ttd) if ttd else None,
        "early_alert_precision": early_prec,
        "early_alerts_total": len(alerts_early),
        "early_alerts_fp": early_fp,
        "conf_alert_precision": conf_prec,
        "conf_alerts_total": len(alerts_conf),
        "conf_alerts_fp": conf_fp,
        "actors_total": len(actors),
    }


# -----------------------------
# Module loader (filename → module)
# -----------------------------
def load_module_from_py(py_file: Path):
    """Load a .py file as a real module (and register it in sys.modules).

    This extra sys.modules registration matters for dataclasses/typing internals,
    which sometimes look up the module by name during class decoration.
    """
    py_file = py_file.resolve()
    if not py_file.exists():
        raise FileNotFoundError(f"Model file not found: {py_file}")

    mod_name = py_file.stem  # e.g., mini_mas_mesa_optionB_v5_adaptive_layers_fixed
    spec = importlib.util.spec_from_file_location(mod_name, str(py_file))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {py_file}")

    mod = importlib.util.module_from_spec(spec)
    # CRITICAL: register before executing so dataclasses can resolve cls.__module__
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

# -----------------------------
# Safe config construction
# -----------------------------
def make_siem_cfg(SIEMConfig, threshold: float, flags: Dict[str, Any]):
    """
    Instantiate SIEMConfig (dataclass), then apply flags + threshold safely.
    Works even if the config class has gained/lost fields over time.
    """
    cfg = SIEMConfig()  # rely on defaults
    # Always set the base confirmed threshold we are sweeping
    if hasattr(cfg, "base_confirmed_threshold"):
        setattr(cfg, "base_confirmed_threshold", float(threshold))
    elif hasattr(cfg, "siem_threshold"):
        setattr(cfg, "siem_threshold", float(threshold))

    for k, v in flags.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    return cfg


def run_once(InsiderModel, SIEMConfig, threshold: float, seed: int, warmup_steps: int, test_steps: int, flags: Dict[str, Any]):
    siem_cfg = make_siem_cfg(SIEMConfig, threshold, flags)

    # Build kwargs that InsiderModel actually accepts
    init_sig = inspect.signature(InsiderModel.__init__)
    kwargs = {"seed": seed}
    if "siem_cfg" in init_sig.parameters:
        kwargs["siem_cfg"] = siem_cfg
    elif "siem_config" in init_sig.parameters:
        kwargs["siem_config"] = siem_cfg

    if "warmup_steps" in init_sig.parameters:
        kwargs["warmup_steps"] = warmup_steps
    if "test_steps" in init_sig.parameters:
        kwargs["test_steps"] = test_steps

    m = InsiderModel(**kwargs)

    total_steps = warmup_steps + test_steps
    for _ in range(total_steps):
        m.step()

    events = [e.to_dict() for e in m.event_log]
    return eval_from_events(events)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mini_mas_mesa_optionB_v5_adaptive_layers.py",
                    help="Path to the model .py file (default: v5 fixed).")
    ap.add_argument("--test_steps", type=int, default=240, help="Number of TEST steps to simulate.")
    ap.add_argument("--warmup_steps", type=int, default=60, help="Number of WARMUP (train) steps.")
    ap.add_argument("--threshold_min", type=int, default=3)
    ap.add_argument("--threshold_max", type=int, default=7)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="threshold_sweep_ablation.csv")
    args = ap.parse_args()

    mod = load_module_from_py(Path(args.model))
    InsiderModel = getattr(mod, "InsiderModel")
    SIEMConfig = getattr(mod, "SIEMConfig")

    thresholds = range(args.threshold_min, args.threshold_max + 1)
    seeds = range(1, args.seeds + 1)

    # Ablation configs (will only apply flags that exist in your SIEMConfig)
    ablations = [
        ("full", {"use_policy": True, "use_baseline": True, "use_trust": True, "use_online_learning": True, "use_ml": True}),
        ("full_no_ml", {"use_policy": True, "use_baseline": True, "use_trust": True, "use_online_learning": True, "use_ml": False}),
        ("policy", {"use_policy": True, "use_baseline": False, "use_trust": False, "use_online_learning": False, "use_ml": False}),
        ("policy+baseline", {"use_policy": True, "use_baseline": True, "use_trust": False, "use_online_learning": False, "use_ml": False}),
        ("policy+baseline+trust", {"use_policy": True, "use_baseline": True, "use_trust": True, "use_online_learning": False, "use_ml": False}),
        ("rules_only", {"use_policy": False, "use_baseline": False, "use_trust": False, "use_online_learning": False, "use_ml": False}),
    ]

    rows = []
    for cfg_name, flags in ablations:
        for thr in thresholds:
            for seed in seeds:
                r = run_once(
                    InsiderModel=InsiderModel,
                    SIEMConfig=SIEMConfig,
                    threshold=float(thr),
                    seed=seed,
                    warmup_steps=args.warmup_steps,
                    test_steps=args.test_steps,
                    flags=flags,
                )
                r["config"] = cfg_name
                r["threshold"] = thr
                r["seed"] = seed
                rows.append(r)

    df = pd.DataFrame(rows)

    summary = df.groupby(["config", "threshold"]).agg(
        actor_precision=("actor_precision", "mean"),
        actor_recall=("actor_recall", "mean"),
        actor_f1=("actor_f1", "mean"),
        ttd_avg_conf=("ttd_avg_conf", "mean"),
        ttd_max_conf=("ttd_max_conf", "mean"),
        early_alert_precision=("early_alert_precision", "mean"),
        early_alerts_total=("early_alerts_total", "mean"),
        early_alerts_fp=("early_alerts_fp", "mean"),
        conf_alert_precision=("conf_alert_precision", "mean"),
        conf_alerts_total=("conf_alerts_total", "mean"),
        conf_alerts_fp=("conf_alerts_fp", "mean"),
        actors_total=("actors_total", "mean"),
    ).reset_index()

    print(summary.to_string(index=False))
    summary.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()