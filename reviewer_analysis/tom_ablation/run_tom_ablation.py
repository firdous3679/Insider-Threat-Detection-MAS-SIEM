#!/usr/bin/env python3
"""Reviewer-only ToM ablation runner.

This script imports the existing Mesa model files without modifying them and
runs only feasible ablations. Variants that require non-existent toggles are
recorded as unavailable in the Markdown summary rather than faked.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = list(range(42, 52))
STEPS = 240
WARMUP = 60


def load_module(name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def evaluate_event_log(event_log: list[Any], warmup: int) -> dict[str, float]:
    """Common actor-level evaluator for LSC/CE/EG event objects."""
    test_events = [
        e for e in event_log
        if getattr(e, "phase", None) == "test" or ((getattr(e, "meta", None) or {}).get("phase") == "test")
    ]
    confirmed = [e for e in test_events if getattr(e, "event_type", None) == "alert_confirmed"]
    early = [e for e in test_events if getattr(e, "event_type", None) == "alert_early"]
    malicious = [
        e for e in test_events
        if getattr(e, "label", None) == "malicious"
        and getattr(e, "event_type", None) not in {"siem_ingest", "analyst_verdict"}
    ]

    mal_actors = {getattr(e, "actor_id", None) for e in malicious}
    conf_actors = {getattr(e, "actor_id", None) for e in confirmed}
    detected = mal_actors & conf_actors
    tp_conf = sum(1 for e in confirmed if getattr(e, "label", None) == "malicious")
    fp_conf = sum(1 for e in confirmed if getattr(e, "label", None) != "malicious")
    tp_early = sum(1 for e in early if getattr(e, "label", None) == "malicious")

    actor_precision = len(detected) / len(conf_actors) if conf_actors else 0.0
    actor_recall = len(detected) / len(mal_actors) if mal_actors else 0.0
    actor_f1 = (
        2 * actor_precision * actor_recall / (actor_precision + actor_recall)
        if actor_precision + actor_recall > 0 else 0.0
    )
    ttd = []
    for actor in detected:
        first_mal = min((getattr(e, "step", math.inf) for e in malicious if getattr(e, "actor_id", None) == actor), default=None)
        first_alert = min((getattr(e, "step", math.inf) for e in confirmed if getattr(e, "actor_id", None) == actor), default=None)
        if first_mal is not None and first_alert is not None:
            ttd.append(first_alert - first_mal)

    tom_detected = 0
    for e in confirmed:
        meta = getattr(e, "meta", None) or {}
        tom_payload = meta.get("tom") or meta.get("tom_assessment") or {}
        if tom_payload.get("has_malicious_intent"):
            tom_detected += 1

    return {
        "actor_precision": actor_precision,
        "actor_recall": actor_recall,
        "actor_f1": actor_f1,
        "ttd_avg": float(np.mean(ttd)) if ttd else 0.0,
        "ttd_max": float(max(ttd)) if ttd else 0.0,
        "confirmed_alerts": float(len(confirmed)),
        "confirmed_alert_precision": tp_conf / len(confirmed) if confirmed else 0.0,
        "confirmed_fp_per_run": float(fp_conf),
        "early_alerts": float(len(early)),
        "early_tp": float(tp_early),
        "actors_total": float(len(mal_actors)),
        "actors_detected": float(len(detected)),
        "tom_assisted_detections": float(tom_detected),
    }


def eg_config(module, *, use_tom: bool, use_forensics: bool = True):
    return module.SIEMConfig(
        use_policy=True,
        use_baseline=True,
        use_trust=True,
        use_ml=True,
        use_tom=use_tom,
        use_forensics=use_forensics,
        use_evidence_gate=True,
        use_peer_norm=True,
        use_regularity=True,
        early_threshold=2.0,
        base_confirmed_threshold=4.0,
        min_evidence_count=2,
        min_evidence_weight=2.5,
        tom_weight=2.0,
        tom_threshold=0.30,
        forensics_weight=1.5,
    )


def ce_config(module, *, use_tom: bool, use_forensics: bool):
    return module.SIEMConfig(
        use_policy=True,
        use_baseline=True,
        use_trust=True,
        use_online_learning=True,
        use_ml=True,
        use_tom=use_tom,
        use_forensics=use_forensics,
        early_threshold=2.5,
        base_confirmed_threshold=3.5,
    )


def run_lsc(seed: int) -> dict[str, float]:
    module = load_module("reviewer_lsc", "mini_mesa_LSC.py")
    model = module.InsiderModel(seed=seed, siem_cfg=module.SIEMConfig(), warmup_steps=WARMUP)
    for _ in range(STEPS):
        model.step()
    return evaluate_event_log(model.event_log, WARMUP)


def run_ce(seed: int, *, use_tom: bool, use_forensics: bool) -> dict[str, float]:
    module = load_module("reviewer_ce", "mini_mesa_CE-SIEM.py")
    cfg = ce_config(module, use_tom=use_tom, use_forensics=use_forensics)
    model = module.InsiderModel(seed=seed, siem_cfg=cfg, warmup_steps=WARMUP)
    for _ in range(STEPS):
        model.step()
    return module.evaluate_run(model.event_log, WARMUP)


def run_eg(seed: int, *, use_tom: bool, use_forensics: bool = True) -> dict[str, float]:
    module = load_module("reviewer_eg", "mini_mesa_EG-SIEM.py")
    cfg = eg_config(module, use_tom=use_tom, use_forensics=use_forensics)
    model = module.InsiderModel(seed=seed, siem_cfg=cfg, warmup=WARMUP)
    for _ in range(STEPS):
        model.step()
    raw = module.evaluate(model.event_log, WARMUP)
    return {
        "actor_precision": raw.get("precision", 0.0),
        "actor_recall": raw.get("recall", 0.0),
        "actor_f1": raw.get("f1", 0.0),
        "ttd_avg": raw.get("ttd_avg", 0.0),
        "ttd_max": raw.get("ttd_max", 0.0),
        "confirmed_alerts": raw.get("conf_total", 0.0),
        "confirmed_alert_precision": raw.get("conf_prec", 0.0),
        "confirmed_fp_per_run": raw.get("conf_fp", 0.0),
        "actors_total": raw.get("actors_total", 0.0),
        "actors_detected": raw.get("actors_detected", 0.0),
        "tom_assisted_detections": raw.get("tom_detected", 0.0),
    }


FEASIBLE_VARIANTS = [
    {
        "variant": "LSC baseline: no ToM, no email forensics",
        "runner": lambda seed: run_lsc(seed),
        "note": "LSC has no ToM/email-forensics toggles in current code; this is the baseline-only variant.",
    },
    {
        "variant": "CE-SIEM: ToM + email/forensics",
        "runner": lambda seed: run_ce(seed, use_tom=True, use_forensics=True),
        "note": "CE-SIEM exposes use_tom and use_forensics toggles.",
    },
    {
        "variant": "CE-SIEM: email/forensics only",
        "runner": lambda seed: run_ce(seed, use_tom=False, use_forensics=True),
        "note": "Reviewer-only approximation of email/forensics contribution in CE-SIEM; not LSC+email.",
    },
    {
        "variant": "EG-SIEM without ToM validation",
        "runner": lambda seed: run_eg(seed, use_tom=False, use_forensics=True),
        "note": "Uses existing EG-SIEM toggle use_tom=False; source file unchanged.",
    },
    {
        "variant": "EG-SIEM with ToM validation",
        "runner": lambda seed: run_eg(seed, use_tom=True, use_forensics=True),
        "note": "Uses existing EG-SIEM rich config with use_tom=True; source file unchanged.",
    },
]

UNAVAILABLE_VARIANTS = [
    {
        "variant": "LSC + ToM only",
        "reason": "mini_mesa_LSC.py does not include a ToM/TomAbd layer or use_tom toggle.",
        "minimal_future_change": "Add a ToM/TomAbd module and a use_tom toggle to LSC, then rerun matched seeds.",
    },
    {
        "variant": "LSC + email only",
        "reason": "mini_mesa_LSC.py does not include an email-forensics layer or use_forensics toggle.",
        "minimal_future_change": "Add the email-forensics monitor/scoring path and a use_forensics toggle to LSC, then rerun matched seeds.",
    },
]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "actor_precision",
        "actor_recall",
        "actor_f1",
        "ttd_avg",
        "ttd_max",
        "confirmed_alerts",
        "confirmed_alert_precision",
        "confirmed_fp_per_run",
        "tom_assisted_detections",
    ]
    rows = []
    for variant, part in df.groupby("variant"):
        row = {"variant": variant, "runs": int(part.shape[0]), "seeds": ",".join(map(str, sorted(part["seed"].unique())))}
        for metric in metrics:
            row[f"{metric}_mean"] = float(part[metric].mean()) if metric in part else float("nan")
            row[f"{metric}_sd"] = float(part[metric].std(ddof=1)) if metric in part and part.shape[0] > 1 else 0.0
        row["notes"] = " | ".join(sorted(part["note"].dropna().unique()))
        rows.append(row)
    return pd.DataFrame(rows)


def write_markdown(run_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    lines = [
        "# ToM Ablation Summary",
        "",
        "This reviewer-only script imported existing model files and did not modify them.",
        "",
        f"Settings: {len(SEEDS)} runs, seeds {SEEDS[0]}-{SEEDS[-1]}, {STEPS} steps, {WARMUP} warm-up steps.",
        "",
        "## Feasible Variants Run",
        "",
        summary_df.to_markdown(index=False),
        "",
        "## Requested Variants Not Safely Available",
        "",
    ]
    for item in UNAVAILABLE_VARIANTS:
        lines.extend(
            [
                f"### {item['variant']}",
                f"- Reason: {item['reason']}",
                f"- Minimal future change: {item['minimal_future_change']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Outputs",
            "",
            "- `tom_ablation_run_level.csv`",
            "- `tom_ablation_summary.csv`",
            "- `tom_ablation_summary.md`",
        ]
    )
    (OUT_DIR / "tom_ablation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows: list[dict[str, Any]] = []
    for variant in FEASIBLE_VARIANTS:
        for run_index, seed in enumerate(SEEDS, start=1):
            print(f"[ToM ablation] {variant['variant']} seed={seed}")
            start = time.perf_counter()
            metrics = variant["runner"](seed)
            elapsed = time.perf_counter() - start
            rows.append(
                {
                    "variant": variant["variant"],
                    "run": run_index,
                    "seed": seed,
                    "steps": STEPS,
                    "warmup": WARMUP,
                    "runtime_seconds": elapsed,
                    "note": variant["note"],
                    **metrics,
                }
            )
    run_df = pd.DataFrame(rows)
    run_df.to_csv(OUT_DIR / "tom_ablation_run_level.csv", index=False)
    summary_df = summarize(run_df)
    summary_df.to_csv(OUT_DIR / "tom_ablation_summary.csv", index=False)
    write_markdown(run_df, summary_df)
    print(f"Wrote ToM ablation outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
