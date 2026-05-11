"""Build user-day and actor-level labels from CERT answer files.

Two labels are produced:

* ``actor_label``: 1 if the user is a malicious insider anywhere in the
  evaluation period (any row in ``answers`` mentions them).
* ``user_day_label``: 1 if the user performed a malicious activity on
  that day (an ``answers`` row exists with a matching ``user`` and
  ``day``).

Robustness notes:

- We normalize *both* sides of the (user, day) join to the canonical
  ``YYYY-MM-DD`` string, regardless of whether the input dtype is
  ``datetime.date``, ``pd.Timestamp``, ``str``, or ``object``. Earlier
  versions used naive ``str(...)`` on whatever object came in, which
  produced hard-to-debug mismatches like ``"2010-10-23"`` vs
  ``"2010-10-23 00:00:00+00:00"``.
- We emit a small diagnostic block so reproducibility checks can
  confirm the actor counts at a glance.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _to_day_str(value) -> str:
    """Coerce any ``date``/``Timestamp``/``str`` to ``YYYY-MM-DD``.

    Empty / NaT becomes the empty string so it can never match a real
    answer day.
    """
    if value is None or pd.isna(value):
        return ""
    # ``pd.to_datetime`` handles datetime.date, pd.Timestamp, and string
    # variants like '2010-10-23' or '2010-10-23 00:00:00+00:00'.
    try:
        ts = pd.to_datetime(value, errors="coerce", utc=True)
    except Exception:
        return str(value)
    if pd.isna(ts):
        return ""
    return ts.strftime("%Y-%m-%d")


def build_labels(
    features: pd.DataFrame,
    answers: pd.DataFrame,
    output_dir: str | Path,
) -> pd.DataFrame:
    df = features.copy()
    df["user_day_label"] = 0
    df["actor_label"] = 0
    df["scenario"] = ""

    n_actor_users = 0
    n_user_day_label = 0

    if not answers.empty and "user" in answers.columns:
        # ----- actor-level labels --------------------------------------------
        actors = set(answers["user"].dropna().astype(str))
        df.loc[df["user"].astype(str).isin(actors), "actor_label"] = 1
        n_actor_users = int(df.loc[df["actor_label"] == 1, "user"].nunique())

        # ----- user-day labels ----------------------------------------------
        if "day" in answers.columns:
            ans_user = answers["user"].astype(str)
            ans_day = answers["day"].apply(_to_day_str)
            pairs = set(zip(ans_user, ans_day))
            feat_day_str = df["day"].apply(_to_day_str)
            feat_user_str = df["user"].astype(str)
            df["user_day_label"] = [
                1 if (u, d) in pairs else 0
                for u, d in zip(feat_user_str, feat_day_str)
            ]
            n_user_day_label = int(df["user_day_label"].sum())

        # ----- scenario annotation ------------------------------------------
        if "scenario" in answers.columns:
            smap = (
                answers.groupby("user")["scenario"]
                .agg(lambda x: ";".join(sorted(set(x.astype(str)))))
                .to_dict()
            )
            df["scenario"] = df["user"].astype(str).map(smap).fillna("")

    print(
        f"[CERT-LABELS] actor_label=1 users: {n_actor_users} | "
        f"user_day_label=1 rows: {n_user_day_label} | "
        f"answer rows seen: {len(answers)}"
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "cert_user_day_labeled.csv", index=False)
    return df
