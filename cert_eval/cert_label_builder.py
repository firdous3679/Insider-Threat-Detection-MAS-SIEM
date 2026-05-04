from __future__ import annotations
from pathlib import Path
import pandas as pd


def build_labels(features: pd.DataFrame, answers: pd.DataFrame, output_dir: str | Path) -> pd.DataFrame:
    df = features.copy()
    df["user_day_label"] = 0
    df["actor_label"] = 0
    df["scenario"] = ""
    if not answers.empty:
        ucol = "user" if "user" in answers.columns else None
        daycol = "day" if "day" in answers.columns else None
        if ucol:
            actors = set(answers[ucol].dropna().astype(str))
            df.loc[df["user"].astype(str).isin(actors), "actor_label"] = 1
        if ucol and daycol:
            pairs = set(zip(answers[ucol].astype(str), answers[daycol].astype(str)))
            df["user_day_label"] = [1 if (str(u), str(d)) in pairs else 0 for u, d in zip(df["user"], df["day"])]
        if "scenario" in answers.columns and ucol:
            smap = answers.groupby(ucol)["scenario"].agg(lambda x: ";".join(sorted(set(x.astype(str))))).to_dict()
            df["scenario"] = df["user"].astype(str).map(smap).fillna("")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "cert_user_day_labeled.csv", index=False)
    return df
