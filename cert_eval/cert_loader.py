"""CERT r4.2 / r5.2 data loading utilities.

This module is intentionally defensive because CERT releases can vary in
column names and directory layouts. It loads files if available,
standardizes timestamps/columns, and emits warnings (not crashes) for
missing files to keep experiments reproducible across environments.

Memory-aware design (important for r4.2)
----------------------------------------
CERT r4.2 ships ``http.csv`` at ~14 GB and ``email.csv`` at ~1.3 GB. A
naive ``pd.read_csv`` of those files exhausts RAM on a typical laptop
and the OS kills the process. Instead, this loader:

1. Resolves a target user set up front (from LDAP if present, else
   from logon.csv) and respects an optional ``max_users`` cap.
2. Streams every modality CSV with ``chunksize`` and keeps only rows
   whose ``user`` is in the target set before concatenation.

That bounds peak memory by the *retained* user subset rather than the
raw on-disk size.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set

import pandas as pd


# ----- Bundle ----------------------------------------------------------------


@dataclass
class CertDataBundle:
    """Container for loaded CERT tables."""

    logon: pd.DataFrame
    device: pd.DataFrame
    file: pd.DataFrame
    http: pd.DataFrame
    email: pd.DataFrame
    ldap: pd.DataFrame
    answers: pd.DataFrame


# ----- Small helpers ---------------------------------------------------------


def _warn(msg: str) -> None:
    print(f"[CERT-LOADER WARNING] {msg}")


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find the first column in ``df`` whose lowercased name is in ``candidates``."""
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None


def _add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add timestamp/day/hour/after_hours/weekend columns in place.

    CERT r4.2 uses ``date`` while some r5.2 files use ``timestamp``;
    this helper handles either.
    """
    if df.empty:
        return df
    ts_col = _find_column(df, ["timestamp", "date", "time", "datetime"])
    if ts_col is None:
        _warn("No timestamp-like column found; adding empty time fields.")
        df["timestamp"] = pd.NaT
    else:
        df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)

    df["day"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour.fillna(-1).astype(int)
    df["after_hours"] = df["hour"].apply(lambda h: int(h < 8 or h > 18 if h >= 0 else 0))
    df["weekend"] = (
        df["timestamp"].dt.dayofweek.fillna(-1).astype(int).isin([5, 6]).astype(int)
    )
    return df


def _normalize_user_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a canonical ``user`` column exists.

    LDAP files in r4.2 use ``user_id``; activity files use ``user``.
    Downstream code only knows about ``user``, so we copy if needed.
    """
    if df.empty:
        return df
    if "user" not in df.columns:
        if "user_id" in df.columns:
            df["user"] = df["user_id"]
        elif "employee_name" in df.columns:
            df["user"] = df["employee_name"]
    return df


# ----- Streaming CSV reader --------------------------------------------------


def _read_csv_filtered(
    path: Path,
    keep_users: Optional[Set[str]],
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Read a CERT activity CSV, keeping only rows whose ``user`` is retained.

    Streaming is essential for r4.2's http.csv (~14 GB) and email.csv
    (~1.3 GB). When ``keep_users`` is None we fall back to a single
    read for small files (logon/device).
    """
    if not path.exists():
        _warn(f"Missing file: {path}")
        return pd.DataFrame()

    # Small files: read in one go to keep code simple.
    try:
        size_bytes = path.stat().st_size
    except OSError:
        size_bytes = 0
    small_file = size_bytes < 200 * 1024 * 1024  # < 200 MB

    if keep_users is None and small_file:
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.lower() for c in df.columns]
        df = _normalize_user_column(df)
        return _add_time_columns(df)

    # Otherwise stream and filter per chunk.
    frames: List[pd.DataFrame] = []
    try:
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            chunk.columns = [c.lower() for c in chunk.columns]
            chunk = _normalize_user_column(chunk)
            if keep_users is not None and "user" in chunk.columns:
                chunk = chunk[chunk["user"].astype(str).isin(keep_users)]
            if not chunk.empty:
                frames.append(chunk)
    except Exception as exc:
        _warn(f"Failed streaming {path.name}: {exc}")
        return pd.DataFrame()

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return _add_time_columns(df)


# ----- LDAP ------------------------------------------------------------------


def _load_ldap(data_dir: Path) -> pd.DataFrame:
    """Load LDAP user/role metadata, unioned across monthly snapshots.

    Important r4.2 quirk: CERT employees can leave during the simulation
    (some of them precisely because they are insider attackers and get
    fired). The latest LDAP snapshot ``2011-05.csv`` contains only ~845
    users and **none of the 70 malicious actors**. If we used only the
    latest snapshot we would silently drop every actor from peer/role
    features and from any user-pool subsetting.

    We therefore read every monthly snapshot, concatenate, and keep the
    *most recent* record per ``user_id`` (snapshot files are named
    ``YYYY-MM.csv`` so a lexicographic sort is chronological). The
    resulting frame has ~1,000 rows — one per CERT employee that ever
    appeared in LDAP — with their final role/department.
    """
    ldap_dir = data_dir / "LDAP"
    if not ldap_dir.exists():
        _warn(f"Missing LDAP directory: {ldap_dir}")
        return pd.DataFrame()
    files = sorted(ldap_dir.glob("*.csv"))
    if not files:
        _warn("No LDAP csv files found.")
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as exc:
            _warn(f"Failed loading LDAP file {f}: {exc}")
            continue
        df.columns = [c.lower() for c in df.columns]
        df["source_file"] = f.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()

    # Most recent record per user_id wins (chronological order from
    # sorted file names is preserved by pd.concat).
    union = pd.concat(frames, ignore_index=True)
    if "user_id" in union.columns:
        union = union.drop_duplicates(subset=["user_id"], keep="last")
    union = _normalize_user_column(union)
    return union.reset_index(drop=True)


# ----- Answers ---------------------------------------------------------------


def _read_headerless_event_csv(path: Path) -> pd.DataFrame:
    """Parse a per-user malicious-activity CSV from ``answers/<release>-X/``.

    These files have **no header** and **variable column counts** per row
    (a logon row has 5 fields, a file row 6, an http row 12). The first
    five fields are reliably ``event_type, id, date, user, pc`` across
    every event type, so we read line-by-line with the stdlib csv reader
    and keep only those. ``pd.read_csv`` cannot handle the ragged rows
    without dropping data.
    """
    import csv

    rows = []
    try:
        with open(path, "r", newline="", encoding="utf-8", errors="replace") as fh:
            reader = csv.reader(fh)
            for r in reader:
                if len(r) < 4:
                    continue
                rows.append(
                    {
                        "event_type": r[0],
                        "id": r[1],
                        "date": r[2],
                        "user": r[3],
                        "pc": r[4] if len(r) > 4 else "",
                    }
                )
    except Exception as exc:
        _warn(f"Failed reading malicious activity file {path}: {exc}")
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _load_answers(data_dir: Path, release: str = "r4.2") -> pd.DataFrame:
    """Build a long-form answers frame: one row per (user, day, scenario).

    Strategy:
    - Prefer per-user activity CSVs under ``answers/<release>-<scenario>/``
      because they list exact malicious days. Headerless format.
    - Fall back to ``answers/insiders.csv`` (start/end window) when the
      detailed activity files are missing.
    - Ignore answer files that belong to other CERT releases (r2, r3.x,
      r4.1, r5.x) so they don't pollute labels.
    """
    answers_dir = data_dir / "answers"
    if not answers_dir.exists():
        alt = data_dir / "Answers"
        if alt.exists():
            answers_dir = alt
        else:
            _warn(f"Missing answers directory: {answers_dir}")
            return pd.DataFrame()

    frames: List[pd.DataFrame] = []

    # 1) Per-user activity CSVs in subdirectories named like 'r4.2-1/'.
    scenario_dirs = sorted(p for p in answers_dir.iterdir() if p.is_dir() and p.name.startswith(release + "-"))
    for sdir in scenario_dirs:
        scenario_id = sdir.name  # e.g. 'r4.2-1'
        for f in sorted(sdir.glob("*.csv")):
            df = _read_headerless_event_csv(f)
            if df.empty:
                continue
            df["scenario"] = scenario_id
            df["source_file"] = f.name
            df = _add_time_columns(df)
            frames.append(df)

    # 2) Fallback: insiders.csv at the answers root, filtered to this release.
    insiders_path = answers_dir / "insiders.csv"
    if not frames and insiders_path.exists():
        try:
            ins = pd.read_csv(insiders_path)
            ins.columns = [c.lower() for c in ins.columns]
            # Filter to the requested release. CERT writes the dataset as
            # e.g. '4.2'; tolerate both '4.2' and 'r4.2'.
            rel_token = release.replace("r", "")
            if "dataset" in ins.columns:
                ins = ins[ins["dataset"].astype(str).str.replace("r", "", regex=False) == rel_token]
            if not ins.empty:
                # Expand each (user, start, end) into one row per day in window.
                ins["start"] = pd.to_datetime(ins.get("start"), errors="coerce", utc=True)
                ins["end"] = pd.to_datetime(ins.get("end"), errors="coerce", utc=True)
                rows = []
                for _, r in ins.iterrows():
                    if pd.isna(r["start"]) or pd.isna(r["end"]):
                        continue
                    days = pd.date_range(r["start"].normalize(), r["end"].normalize(), freq="D")
                    for d in days:
                        rows.append(
                            {
                                "user": r.get("user"),
                                "timestamp": d,
                                "day": d.date(),
                                "hour": -1,
                                "after_hours": 0,
                                "weekend": int(d.dayofweek in (5, 6)),
                                "scenario": str(r.get("scenario", "")),
                                "source_file": "insiders.csv",
                                "event_type": "malicious_window",
                            }
                        )
                if rows:
                    frames.append(pd.DataFrame(rows))
        except Exception as exc:
            _warn(f"Failed loading insiders.csv: {exc}")

    if not frames:
        _warn(f"No answer rows found for release {release} under {answers_dir}")
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ----- Target user resolution ------------------------------------------------


def _malicious_actor_set(data_dir: Path, release: str) -> Set[str]:
    """Return the set of user IDs marked malicious for ``release`` in
    ``answers/insiders.csv``. Empty set if the file is missing.

    We force-include this set into the keep_users pool so a small
    ``--max_users`` cap can never drop the labeled actors — without
    them, recall is undefined and the whole evaluation is meaningless.
    """
    insiders = data_dir / "answers" / "insiders.csv"
    if not insiders.exists():
        return set()
    try:
        ins = pd.read_csv(insiders)
    except Exception as exc:
        _warn(f"Failed reading insiders.csv: {exc}")
        return set()
    ins.columns = [c.lower() for c in ins.columns]
    rel_token = release.replace("r", "")
    if "dataset" in ins.columns:
        ins = ins[ins["dataset"].astype(str).str.replace("r", "", regex=False) == rel_token]
    if "user" not in ins.columns:
        return set()
    return set(ins["user"].dropna().astype(str))


def _resolve_target_users(
    data_dir: Path,
    max_users: Optional[int],
    release: str = "r4.2",
) -> Optional[Set[str]]:
    """Decide which users to retain BEFORE streaming the big CSVs.

    Returning ``None`` means "keep everyone" (used when ``max_users`` is
    not supplied). Otherwise we:

    1. Build a deterministic user pool from the **unioned** LDAP
       snapshots so departed users (including all 70 r4.2 insiders, who
       are absent from the latest snapshot alone) are visible.
    2. Take the first ``max_users`` of that pool.
    3. **Always union in the malicious-actor set** from
       ``answers/insiders.csv`` so a small ``--max_users`` value can
       never accidentally exclude any labeled actor.

    The third step is essential: without it, sweeping ``--max_users``
    from 50 → 1000 silently changed which actors were evaluable, and
    the actor-level recall reported in the paper would depend entirely
    on alphabetical luck rather than on the detection logic.
    """
    actors = _malicious_actor_set(data_dir, release)
    if not max_users or max_users <= 0:
        return None

    # 1. Build the deterministic candidate pool from unioned LDAP.
    user_pool: List[str] = []
    ldap_dir = data_dir / "LDAP"
    if ldap_dir.exists():
        ldap_files = sorted(ldap_dir.glob("*.csv"))
        if ldap_files:
            try:
                # Union all monthly snapshots so departed users appear.
                seen: Set[str] = set()
                for f in ldap_files:
                    df = pd.read_csv(f, usecols=lambda c: c.lower() in {"user_id", "user"})
                    df.columns = [c.lower() for c in df.columns]
                    df = _normalize_user_column(df)
                    if "user" not in df.columns:
                        continue
                    for u in df["user"].dropna().astype(str):
                        if u not in seen:
                            seen.add(u)
                            user_pool.append(u)
            except Exception as exc:
                _warn(f"Failed reading LDAP for user resolution: {exc}")

    # 2. Fall back to logon.csv if LDAP is unusable.
    if not user_pool:
        logon_path = data_dir / "logon.csv"
        if logon_path.exists():
            seen_set: Set[str] = set()
            try:
                for chunk in pd.read_csv(
                    logon_path,
                    chunksize=200_000,
                    usecols=lambda c: c.lower() in {"user"},
                ):
                    chunk.columns = [c.lower() for c in chunk.columns]
                    if "user" not in chunk.columns:
                        continue
                    for u in chunk["user"].dropna().astype(str).unique():
                        if u not in seen_set:
                            seen_set.add(u)
                            user_pool.append(u)
            except Exception as exc:
                _warn(f"Failed scanning logon.csv for user resolution: {exc}")

    if not user_pool and not actors:
        return None

    keep = set(user_pool[:max_users])

    # 3. Always include every malicious actor for this release.
    forced = actors - keep
    if forced:
        _warn(
            f"Force-including {len(forced)} malicious actors that fell outside "
            f"the first {max_users} LDAP/logon users so labels remain meaningful."
        )
    keep |= actors
    return keep


# ----- Public API ------------------------------------------------------------


def load_cert_data(
    data_dir: str | Path,
    max_users: Optional[int] = None,
    release: str = "r4.2",
    chunksize: int = 500_000,
) -> CertDataBundle:
    """Load CERT tables with resilient schema handling and streaming reads.

    Parameters
    ----------
    data_dir : str | Path
        Path to the unpacked CERT release (containing ``logon.csv``,
        ``device.csv``, ``file.csv``, ``http.csv``, ``email.csv``,
        ``LDAP/`` and ``answers/``).
    max_users : int, optional
        If provided, only rows for the first N users (from LDAP order)
        are retained. **Streaming-filtered** during read so peak memory
        stays bounded even with multi-GB CSVs.
    release : str
        CERT release tag used to filter ``answers/`` (default ``r4.2``).
    chunksize : int
        Pandas ``chunksize`` for the streaming reads.
    """
    root = Path(data_dir)
    keep_users = _resolve_target_users(root, max_users, release=release)
    if keep_users is not None:
        print(
            f"[CERT-LOADER] Streaming-filtering CSVs to {len(keep_users)} users "
            f"(max_users={max_users}; malicious actors force-included)."
        )

    return CertDataBundle(
        logon=_read_csv_filtered(root / "logon.csv", keep_users, chunksize),
        device=_read_csv_filtered(root / "device.csv", keep_users, chunksize),
        file=_read_csv_filtered(root / "file.csv", keep_users, chunksize),
        http=_read_csv_filtered(root / "http.csv", keep_users, chunksize),
        email=_read_csv_filtered(root / "email.csv", keep_users, chunksize),
        ldap=_load_ldap(root),
        answers=_load_answers(root, release=release),
    )
