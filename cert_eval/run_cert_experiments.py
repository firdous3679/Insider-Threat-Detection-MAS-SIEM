"""End-to-end CERT r4.2 experiment runner.

Defaults target CERT r4.2 (the user has switched from r5.2 to r4.2 to
fit experiments in memory). The runner:

1. Streams + filters the CERT CSVs by user inside ``load_cert_data``
   (peak memory is bounded by ``--max_users``).
2. Builds user-day features and ground-truth labels.
3. Runs unsupervised baselines, the CERT-EG-SIEM ablations, and the
   scalability sweep.
4. Writes paper-ready tables and figures into ``--output_dir``.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from cert_eval.cert_loader import load_cert_data
from cert_eval.cert_schema import build_normalized_events
from cert_eval.cert_feature_builder import build_user_day_features
from cert_eval.cert_label_builder import build_labels
from cert_eval.cert_baselines import run_baselines
from cert_eval.cert_eg_siem_runner import run_cert_eg_siem
from cert_eval.cert_scalability import run_scalability
from cert_eval.cert_metrics import (
    compute_classification_metrics,
    summarize_statistics,
    create_plots,
    actor_level_metrics,
    time_to_detection,
    summarize_ttd,
)


def _evaluate_method(df, method, pred_col="pred_alert", score_col="risk_score"):
    """Compute classification + alert + actor + TTD metrics for one mode."""
    y = df["user_day_label"].astype(int)
    pred = df[pred_col].astype(int)
    score = df[score_col] if score_col in df else pred
    met = compute_classification_metrics(y, pred, score)
    fp = int(((pred == 1) & (y == 0)).sum())
    days = max(1, df["day"].nunique())
    met["fp_per_day"] = fp / days

    # Actor-level metrics.
    met.update(actor_level_metrics(df, pred_col=pred_col))

    # Real TTD: median hours between first malicious user-day and first
    # confirmed alert, computed per actor.
    ttd_df = time_to_detection(df, pred_col=pred_col)
    met.update(summarize_ttd(ttd_df))
    met["ttd_hours"] = met.get("ttd_median_hours", float("nan"))

    met["method"] = method
    return met


def main():
    ap = argparse.ArgumentParser()
    # Default to r4.2 because r5.2's 14+ GB http.csv made the loader OOM on
    # typical workstations. r4.2 has the same schema with ~1k users instead
    # of ~2k, so the rest of the pipeline is unchanged.
    ap.add_argument("--data_dir", default="data/r4.2")
    ap.add_argument("--output_dir", default="results/cert_r42")
    ap.add_argument("--max_users", type=int, default=1000)
    ap.add_argument("--random_seed", type=int, default=42)
    ap.add_argument(
        "--release",
        default="r4.2",
        help="CERT release tag, used to filter the answers/ folder (default: r4.2).",
    )
    ap.add_argument(
        "--chunksize",
        type=int,
        default=500_000,
        help="Pandas chunksize for streaming CSV reads.",
    )
    ap.add_argument(
        "--write_normalized_events",
        action="store_true",
        help="Write cert_normalized_events.csv (can be large).",
    )
    ap.add_argument(
        "--sort_normalized_events",
        action="store_true",
        help="Sort normalized events by timestamp before writing.",
    )
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Streaming + user-filtered load. Peak memory is bounded by max_users,
    # so even r4.2's 14 GB http.csv stays tractable.
    bundle = load_cert_data(
        args.data_dir,
        max_users=args.max_users,
        release=args.release,
        chunksize=args.chunksize,
    )
    print(
        f"[CERT-RUNNER] Loaded CERT {args.release} tables "
        f"(subset to <= {args.max_users} users). Building user-day features..."
    )

    # Only build the normalized event stream when the user actually wants it
    # on disk; nothing downstream consumes it (features come straight from
    # the bundle), so building it unconditionally just inflates peak memory.
    if args.write_normalized_events:
        events = build_normalized_events(
            bundle, sort_events=args.sort_normalized_events
        )
        events.to_csv(out / "cert_normalized_events.csv", index=False)
        print("[CERT-RUNNER] Wrote cert_normalized_events.csv")
        del events  # release before the heavy feature step

    feats = build_user_day_features(bundle, out)
    print("[CERT-RUNNER] Wrote cert_user_day_features.csv")
    labeled = build_labels(feats, bundle.answers, out)
    labeled = labeled.sort_values(["user", "day"]).reset_index(drop=True)
    print("[CERT-RUNNER] Wrote cert_user_day_labeled.csv")

    baseline_df = run_baselines(labeled, random_seed=args.random_seed)
    baseline_df.to_csv(out / "baseline_results.csv", index=False)

    modes = [
        ("CERT-LSC", "lsc"),
        ("CERT-EG-SIEM without email", "without_email"),
        ("CERT-EG-SIEM email only", "email_only"),
        ("CERT-EG-SIEM without ToM-like evidence", "without_tom"),
        ("CERT-EG-SIEM full", "full"),
    ]
    ablation = []
    for mname, mode in modes:
        pred_df = run_cert_eg_siem(labeled, mode=mode)
        met = _evaluate_method(pred_df, mname)
        ablation.append(met)
    abl_df = pd.DataFrame(ablation)
    abl_df.to_csv(out / "cert_ablation_results.csv", index=False)

    # Scalability sweep now times the FULL feature->label->EG-SIEM
    # pipeline, not just the user-day scoring step. Sweep matches the
    # Phase 2 plan (100 / 250 / 500 / 1000). The largest point is the
    # *bundle's* user count, not args.max_users — internally
    # ``run_scalability`` caps each n at total_users so requesting 1000
    # on a 252-user bundle just measures the full 252.
    scal_df = run_scalability(
        bundle,
        output_dir=out,
        user_sizes=(100, 250, 500, 1000),
    )
    scal_df.to_csv(out / "scalability_results.csv", index=False)

    all_results = pd.concat([baseline_df, abl_df], ignore_index=True, sort=False)
    summary = []
    for metric in ["precision", "recall", "f1", "roc_auc", "pr_auc", "fp_per_day"]:
        s = summarize_statistics(all_results[metric].fillna(0).tolist())
        s["metric"] = metric
        summary.append(s)
    pd.DataFrame(summary).to_csv(out / "cert_statistical_summary.csv", index=False)
    create_plots(all_results, scal_df, out)

    # Paper-ready tables. ``Dataset`` is now derived from the --release flag
    # so switching releases doesn't require code edits.
    dataset_label = f"CERT {args.release}"
    pd.DataFrame(
        [
            {
                "Dataset": dataset_label,
                "Users": labeled["user"].nunique(),
                "Duration": f"{labeled['day'].nunique()} days",
                "Modalities": "logon/device/file/http/email/LDAP",
                "Insider scenarios": (
                    bundle.answers.get("scenario", pd.Series()).nunique()
                    if not bundle.answers.empty
                    else 0
                ),
                "Evaluation unit": "user-day",
                "Purpose in this paper": "External benchmark generalization",
            }
        ]
    ).to_csv(out / "table_a_dataset_summary.csv", index=False)
    all_results.to_csv(out / "table_b_external_benchmark_results.csv", index=False)
    scal_df.to_csv(out / "table_c_scalability.csv", index=False)
    pd.DataFrame(
        [
            {
                "Comparison": "CERT-EG-SIEM full vs CERT-LSC",
                "Metric": "F1",
                "Mean difference": float(
                    (
                        all_results[all_results.method == "CERT-EG-SIEM full"][
                            "f1"
                        ].mean()
                        - all_results[all_results.method == "CERT-LSC"][
                            "f1"
                        ].mean()
                    )
                    if "method" in all_results
                    else 0
                ),
                "Test used": "descriptive",
                "p-value": None,
                "Significant?": "N/A",
            }
        ]
    ).to_csv(out / "table_d_statistical_significance.csv", index=False)


if __name__ == "__main__":
    main()
