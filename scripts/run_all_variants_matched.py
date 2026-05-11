"""Run all four SIEM variants under matched conditions and emit per-run metrics.

Reuses the four mini_mesa_*.py simulation modules as-is (no changes to thresholds,
weights, or hyperparameters). For each variant x seed combination, this script
constructs the variant's InsiderModel with the matched population (30 benign +
4 power + 8 malicious = 42 humans), runs the requested number of steps with the
matched warm-up, then computes per-run metrics in a uniform way from the
event_log so that all variants are scored on the same definitions.

Output:
  results/statistical_rigor/run_level_all_variants.csv

Columns:
  variant, seed, actor_precision, actor_recall, actor_f1,
  ttd_avg, ttd_max, confirmed_alerts, confirmed_alert_precision,
  confirmed_fp_per_run, status, error
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

VARIANT_FILES = {
    "LSC": "mini_mesa_LSC.py",
    "CE-SIEM": "mini_mesa_CE-SIEM.py",
    "EG-SIEM": "mini_mesa_EG-SIEM.py",
    "EG-SIEM-Enron": "mini_mesa_EG-SIEM_Enron.py",
}

DEFAULT_VARIANTS = ["LSC", "CE-SIEM", "EG-SIEM", "EG-SIEM-Enron"]
DEFAULT_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]


def _load_module(module_name: str, file_path: str):
    """Load a Python file as a module (handles file names with hyphens)."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    # Make sure the variant module can find sibling modules + data files in the repo root.
    saved_cwd = os.getcwd()
    saved_sys_path = list(sys.path)
    os.chdir(REPO_ROOT)
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    try:
        spec.loader.exec_module(module)
    finally:
        os.chdir(saved_cwd)
        sys.path[:] = saved_sys_path
    return module


def _evaluate_event_log(event_log) -> Dict[str, float]:
    """Uniform evaluator that mirrors the EG-SIEM-Enron evaluation function.

    Computed entirely from the post-warmup ('test' phase) event_log so it can be
    applied identically to all four variants without altering simulation logic.
    """
    test = [
        e for e in event_log
        if (getattr(e, "phase", None) == "test"
            or (getattr(e, "meta", None) or {}).get("phase") == "test")
    ]

    malicious_actor_ids = set()
    first_malicious_step: Dict[int, int] = {}
    skip_event_types = {"alert_early", "alert_confirmed", "siem_ingest", "analyst_verdict"}
    for e in test:
        if e.label == "malicious" and e.event_type not in skip_event_types:
            malicious_actor_ids.add(e.actor_id)
            first_malicious_step.setdefault(e.actor_id, e.step)

    confirmed_alerts = [e for e in test if e.event_type == "alert_confirmed"]
    confirmed_tp_alerts = [a for a in confirmed_alerts if a.label == "malicious"]
    confirmed_fp_alerts = [a for a in confirmed_alerts if a.label != "malicious"]

    detected_actor_ids = {a.actor_id for a in confirmed_tp_alerts}
    false_detected_actor_ids = {a.actor_id for a in confirmed_fp_alerts}
    tp_detected_actor_ids = detected_actor_ids & malicious_actor_ids

    denom = len(tp_detected_actor_ids) + len(false_detected_actor_ids)
    actor_precision = len(tp_detected_actor_ids) / denom if denom else 0.0
    actor_recall = (
        len(tp_detected_actor_ids) / len(malicious_actor_ids)
        if malicious_actor_ids else 0.0
    )
    actor_f1 = (
        2 * actor_precision * actor_recall / (actor_precision + actor_recall)
        if (actor_precision + actor_recall) else 0.0
    )

    ttd_by_actor: Dict[int, int] = {}
    for a in confirmed_tp_alerts:
        if a.actor_id in first_malicious_step and a.actor_id not in ttd_by_actor:
            ttd_by_actor[a.actor_id] = a.step - first_malicious_step[a.actor_id]
    ttd_vals = list(ttd_by_actor.values())

    confirmed_alert_precision = (
        len(confirmed_tp_alerts) / len(confirmed_alerts) if confirmed_alerts else 0.0
    )

    return {
        "actor_precision": float(actor_precision),
        "actor_recall": float(actor_recall),
        "actor_f1": float(actor_f1),
        "ttd_avg": float(np.mean(ttd_vals)) if ttd_vals else 0.0,
        "ttd_max": float(max(ttd_vals)) if ttd_vals else 0.0,
        "confirmed_alerts": int(len(confirmed_alerts)),
        "confirmed_alert_precision": float(confirmed_alert_precision),
        "confirmed_fp_per_run": int(len(confirmed_fp_alerts)),
    }


# ---- per-variant model construction (population + config exactly as in each module) ----

def _build_model_LSC(module, seed: int, warmup: int, agents: int):
    cfg = module.SIEMConfig()
    return module.InsiderModel(
        n_employees=30, n_power_users=4,
        n_malicious_exfil=3, n_malicious_stealth=2,
        n_malicious_acct_takeover=1, n_malicious_staging_exfil=1,
        n_malicious_email_only=1,
        seed=seed, siem_cfg=cfg, warmup_steps=warmup,
    )


def _build_model_CE(module, seed: int, warmup: int, agents: int):
    cfg = module.SIEMConfig(
        use_policy=True, use_baseline=True, use_trust=True,
        use_online_learning=True, use_ml=True,
        use_tom=True, use_forensics=True,
        early_threshold=2.5, base_confirmed_threshold=3.5,
    )
    return module.InsiderModel(
        n_employees=30, n_power_users=4,
        n_malicious_exfil=3, n_malicious_stealth=2,
        n_malicious_acct_takeover=1, n_malicious_staging_exfil=1,
        n_malicious_email_only=1,
        seed=seed, siem_cfg=cfg, warmup_steps=warmup,
    )


def _build_model_EG(module, seed: int, warmup: int, agents: int):
    cfg = module.SIEMConfig(
        use_policy=True, use_baseline=True, use_trust=True, use_ml=True,
        use_tom=True, use_forensics=True,
        use_evidence_gate=True, use_peer_norm=True, use_regularity=True,
        early_threshold=2.0, base_confirmed_threshold=4.0,
        min_evidence_count=2, min_evidence_weight=2.5,
        tom_weight=2.0, tom_threshold=0.30, forensics_weight=1.5,
    )
    return module.InsiderModel(seed=seed, siem_cfg=cfg, warmup=warmup)


def _build_model_Enron(module, seed: int, warmup: int, agents: int, diag: bool = False):
    cfg = module.build_siem_config("full")
    pop_kwargs = module.build_population_kwargs("full")
    forensics_path = os.path.join(REPO_ROOT, "combined_forensics_model.pkl")
    resolved_exists = os.path.exists(forensics_path)
    forensics_mode = "full"
    if not resolved_exists:
        forensics_path = None
    if diag:
        print(f"[diag-enron] cwd={os.getcwd()}")
        print(f"[diag-enron] REPO_ROOT={REPO_ROOT}")
        print(f"[diag-enron] resolved forensics_model_path={forensics_path!r}")
        print(f"[diag-enron] file_exists={resolved_exists}")
        print(f"[diag-enron] forensics_mode={forensics_mode!r}")
    model = module.InsiderModel(
        seed=seed, siem_cfg=cfg, warmup=warmup,
        forensics_model_path=forensics_path,
        forensics_mode=forensics_mode,
        **pop_kwargs,
    )
    if diag:
        # Inspect the constructed forensics agent inside the EmailMonitor.
        EmailMonitor = getattr(module, "EmailMonitor", None)
        forensics_agent = None
        if EmailMonitor is not None:
            for a in model.agents:
                if isinstance(a, EmailMonitor):
                    forensics_agent = getattr(a, "forensics", None)
                    break
        if forensics_agent is None:
            print("[diag-enron] forensics_agent: not located on EmailMonitor")
        else:
            classifier = getattr(forensics_agent, "classifier", None)
            vectorizer = getattr(forensics_agent, "vectorizer", None)
            mode = getattr(forensics_agent, "mode", None)
            acc = getattr(forensics_agent, "classifier_accuracy", None)
            phrase_w = getattr(forensics_agent, "learned_phrase_weights", {}) or {}
            print(f"[diag-enron] forensics_agent.mode={mode!r}")
            print(f"[diag-enron] forensics_agent.classifier_loaded={classifier is not None}")
            print(f"[diag-enron] forensics_agent.vectorizer_loaded={vectorizer is not None}")
            print(f"[diag-enron] forensics_agent.classifier_accuracy={acc!r}")
            print(f"[diag-enron] forensics_agent.learned_phrase_weights_count={len(phrase_w)}")
    return model


VARIANT_BUILDERS = {
    "LSC": _build_model_LSC,
    "CE-SIEM": _build_model_CE,
    "EG-SIEM": _build_model_EG,
    "EG-SIEM-Enron": _build_model_Enron,
}


def run_one(variant: str, module, seed: int, steps: int, warmup: int, agents: int,
            diag_enron: bool = False) -> Dict[str, Any]:
    builder = VARIANT_BUILDERS[variant]
    saved_cwd = os.getcwd()
    os.chdir(REPO_ROOT)  # forensics model and any relative resources resolve here
    try:
        if variant == "EG-SIEM-Enron":
            model = builder(module, seed=seed, warmup=warmup, agents=agents, diag=diag_enron)
        else:
            model = builder(module, seed=seed, warmup=warmup, agents=agents)
        for _ in range(steps):
            model.step()
        metrics = _evaluate_event_log(model.event_log)
    finally:
        os.chdir(saved_cwd)
    metrics["variant"] = variant
    metrics["seed"] = seed
    metrics["status"] = "ok"
    metrics["error"] = ""
    return metrics


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run all SIEM variants under matched conditions.")
    p.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS,
                   choices=DEFAULT_VARIANTS,
                   help="Variants to run (default: all four).")
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                   help="Random seeds (default: 42..51).")
    p.add_argument("--steps", type=int, default=240, help="Number of simulation steps.")
    p.add_argument("--warmup", type=int, default=60, help="Warm-up steps.")
    p.add_argument("--agents", type=int, default=42,
                   help="Total human agents (informational; populations are fixed by variant).")
    p.add_argument("--out", type=str,
                   default=os.path.join(REPO_ROOT, "results", "statistical_rigor",
                                        "run_level_all_variants.csv"))
    p.add_argument("--quiet", action="store_true", help="Suppress per-run prints.")
    p.add_argument("--append", action="store_true",
                   help="Append rows to an existing CSV instead of overwriting it.")
    p.add_argument("--diag-enron", action="store_true",
                   help="Print Enron forensics-model load diagnostics on each Enron build.")
    args = p.parse_args(argv)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Load each variant module once.
    modules = {}
    load_failures = []
    for variant in args.variants:
        try:
            path = os.path.join(REPO_ROOT, VARIANT_FILES[variant])
            modules[variant] = _load_module(f"_variant_{variant.replace('-', '_')}", path)
            if not args.quiet:
                print(f"[load] {variant} OK")
        except Exception as exc:  # noqa: BLE001
            load_failures.append((variant, str(exc)))
            if not args.quiet:
                print(f"[load] {variant} FAILED: {exc}")

    runnable = [v for v in args.variants if v in modules]
    # Halt only when the caller requested the full multi-variant sweep but fewer
    # than 3 of those variants could be loaded. Single-variant chunked calls are allowed.
    if len(args.variants) >= 3 and len(runnable) < 3:
        print("FATAL: fewer than 3 variants loaded successfully; halting before writing outputs.")
        for v, msg in load_failures:
            print(f"  - {v}: {msg}")
        return 2
    if not runnable:
        print("FATAL: no variants could be loaded; halting before writing outputs.")
        for v, msg in load_failures:
            print(f"  - {v}: {msg}")
        return 2

    rows: List[Dict[str, Any]] = []
    columns = [
        "variant", "seed",
        "actor_precision", "actor_recall", "actor_f1",
        "ttd_avg", "ttd_max",
        "confirmed_alerts", "confirmed_alert_precision", "confirmed_fp_per_run",
        "status", "error",
    ]

    for variant in runnable:
        module = modules[variant]
        for seed in args.seeds:
            t0 = time.time()
            try:
                row = run_one(variant, module, seed=seed, steps=args.steps,
                              warmup=args.warmup, agents=args.agents,
                              diag_enron=args.diag_enron)
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc().splitlines()[-1]
                row = {c: 0.0 for c in columns}
                row.update({
                    "variant": variant, "seed": seed,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc} | {tb}",
                })
            elapsed = time.time() - t0
            rows.append(row)
            if not args.quiet:
                if row["status"] == "ok":
                    print(
                        f"[{variant} seed={seed}] f1={row['actor_f1']:.3f} "
                        f"P={row['actor_precision']:.3f} R={row['actor_recall']:.3f} "
                        f"ttd_avg={row['ttd_avg']:.2f} conf={row['confirmed_alerts']} "
                        f"FP={row['confirmed_fp_per_run']} ({elapsed:.1f}s)"
                    )
                else:
                    print(f"[{variant} seed={seed}] ERROR: {row['error']}")

    file_exists = os.path.exists(args.out)
    mode = "a" if (args.append and file_exists) else "w"
    with open(args.out, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if mode == "w":
            writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in columns})

    verb = "Appended" if mode == "a" else "Wrote"
    print(f"\n{verb} {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
