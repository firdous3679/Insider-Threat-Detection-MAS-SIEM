#!/usr/bin/env python3
"""Reviewer-only status check for EnhancedRoleAnomalyModel.

The current repository contains EnhancedRoleAnomalyModel implementations, but a
clean one-factor replacement experiment is not available without changing or
monkey-patching existing SIEM classes. This script documents that status and
writes machine-readable summary files for the paper/rebuttal workflow.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


FILES = [
    "mini_mesa_LSC.py",
    "mini_mesa_CE-SIEM.py",
    "mini_mesa_EG-SIEM.py",
    "mini_mesa_EG-SIEM_Enron.py",
]


def find_lines(path: Path, pattern: str) -> list[int]:
    rx = re.compile(pattern)
    out = []
    for idx, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if rx.search(line):
            out.append(idx)
    return out


def main() -> None:
    rows = []
    for rel in FILES:
        path = REPO_ROOT / rel
        if not path.exists():
            rows.append({"file": rel, "status": "missing", "evidence": ""})
            continue
        enhanced = find_lines(path, r"class EnhancedRoleAnomalyModel|EnhancedRoleAnomalyModel\(")
        simple = find_lines(path, r"class MLModel|MLModel\(")
        isolation = find_lines(path, r"IsolationForest\(")
        rows.append(
            {
                "file": rel,
                "has_enhanced_role_anomaly_model": bool(enhanced),
                "enhanced_lines": ",".join(map(str, enhanced)),
                "has_simple_ml_model": bool(simple),
                "simple_model_lines": ",".join(map(str, simple)),
                "isolation_forest_lines": ",".join(map(str, isolation)),
            }
        )
    status_df = pd.DataFrame(rows)
    status_df.to_csv(OUT_DIR / "enhanced_anomaly_status.csv", index=False)

    summary_rows = [
        {
            "comparison": "current role-aware Isolation Forest vs EnhancedRoleAnomalyModel",
            "evaluated": False,
            "reason": (
                "Not cleanly evaluated in this run. EG-SIEM/EG-SIEM-Enron use the simple MLModel interface, "
                "while LSC/CE-SIEM instantiate EnhancedRoleAnomalyModel inside different SIEM implementations. "
                "Comparing EG-SIEM to LSC/CE-SIEM would confound anomaly-model changes with many other SIEM-layer changes."
            ),
            "minimal_future_experiment": (
                "Create a dedicated reviewer-only model variant that keeps the same EG-SIEM population, thresholds, "
                "evidence gates, scoring weights, and seeds, and swaps only the anomaly component through a common adapter. "
                "Then run matched seeds 42-51 and report run-level actor metrics and runtime."
            ),
        }
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "enhanced_anomaly_summary.csv", index=False)

    lines = [
        "# EnhancedRoleAnomalyModel Status",
        "",
        "This reviewer-only check did not modify existing experiment files.",
        "",
        "## What is implemented",
        "",
        "- `mini_mesa_LSC.py` implements and instantiates `EnhancedRoleAnomalyModel`.",
        "- `mini_mesa_CE-SIEM.py` implements and instantiates `EnhancedRoleAnomalyModel`.",
        "- `mini_mesa_EG-SIEM.py` uses a simpler `MLModel` with `IsolationForest`.",
        "- `mini_mesa_EG-SIEM_Enron.py` uses the EG-SIEM style simple anomaly component rather than the LSC/CE enhanced class.",
        "",
        "## Evaluation status",
        "",
        "EnhancedRoleAnomalyModel was not cleanly evaluated as a one-factor ablation here. A fair experiment would need a common adapter that swaps only the anomaly model while keeping all other SIEM layers and simulation settings fixed. No existing result table found in the inspected files reports that isolated comparison.",
        "",
        "## Minimal future experiment",
        "",
        "Create a new isolated reviewer-only variant that keeps the EG-SIEM population, thresholds, evidence gate, scoring weights, seeds, steps, and warm-up fixed, then replaces only the anomaly model through a common adapter. Save matched run-level rows for both anomaly components before computing paired statistics.",
        "",
        "## Outputs",
        "",
        "- `enhanced_anomaly_status.csv`",
        "- `enhanced_anomaly_summary.csv`",
        "- `enhanced_anomaly_summary.md`",
    ]
    (OUT_DIR / "enhanced_anomaly_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote EnhancedRoleAnomalyModel status outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
