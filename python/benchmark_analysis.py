"""!
@brief Turns RunResultsMessage records into the course's sequential-vs-parallel comparison.
@see docs/modules/benchmark-analysis.md
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from module_interface import RunResultsMessage

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_PATH = REPO_ROOT / "data" / "runs.jsonl"

## Runs are compared only with runs of the same instance and the same settings.
_SAME_RUN = ["instance_id", "n", "k", "seed", "chains", "sweeps"]
## What speedup_table() returns, in this order.
SPEEDUP_COLUMNS = _SAME_RUN + ["platform", "runs", "t_kernel", "msteps_per_s", "accept_rate",
                               "speedup", "same_best"]
## Instance ids made by data_collectors.build_instance() carry the size: "..._n3_k3_seed1234".
_SIZE_IN_ID = re.compile(r"_n(\d+)_k(\d+)")


def append_run(record: RunResultsMessage, path: Path = RUNS_PATH) -> None:
    """!
    @brief Appends one RunResultsMessage as a JSON line.
    @param record Result to record.
    @param path File to append to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(asdict(record)) + "\n")


def load_runs(path: Path = RUNS_PATH) -> pd.DataFrame:
    """!
    @brief Loads every recorded run.
    @param path File to read.
    @return All run records as a DataFrame (empty if path doesn't exist).
    """
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def _with_problem_size(df: pd.DataFrame) -> pd.DataFrame:
    """!
    @brief Makes sure every row has n and k. Records written before RunResultsMessage had
           those fields get them from the instance id ("..._n3_k3_..."), or 0 if it has none.
    @param df Run records.
    @return A copy of df with integer n and k columns.
    """
    df = df.copy()
    parsed = df["instance_id"].astype(str).str.extract(_SIZE_IN_ID).astype(float)
    for column, part in (("n", 0), ("k", 1)):
        recorded = pd.to_numeric(df[column], errors="coerce") if column in df else np.nan
        recorded = pd.Series(recorded, index=df.index).where(lambda v: v > 0)
        df[column] = recorded.fillna(parsed[part]).fillna(0).astype(int)
    return df


def speedup_table(df: pd.DataFrame, baseline: str = "scalar") -> pd.DataFrame:
    """!
    @brief Compares every platform's kernel time with the baseline platform's, for runs of
           the same instance with the same settings (seed, chains, sweeps).
    Repeated runs of one platform are summarized by their median t_kernel. Runs with no
    timing (t_kernel <= 0) are left out.
    @param df Run records, as returned by load_runs().
    @param baseline Platform every other platform's speedup is computed relative to.
    @return One row per (instance, settings, platform), columns SPEEDUP_COLUMNS:
            runs (how many runs were summarized), t_kernel (median seconds), msteps_per_s
            (millions per second of the run's "proposals": swaps proposed for SA,
            combinations scored for exact search; NaN when there were none), accept_rate
            (SA only), speedup (baseline t_kernel / this t_kernel; NaN when
            the group has no baseline run) and same_best (whether the best team equals the
            baseline's; None when there is no baseline run). Sorted by n, k, then the
            baseline first.
    @see docs/modules/benchmark-analysis.md
    """
    if df.empty or "t_kernel" not in df:
        return pd.DataFrame(columns=SPEEDUP_COLUMNS)
    runs = _with_problem_size(df)
    runs = runs[pd.to_numeric(runs["t_kernel"], errors="coerce") > 0]
    if runs.empty:
        return pd.DataFrame(columns=SPEEDUP_COLUMNS)
    runs = runs.assign(best=runs["best"].map(lambda team: tuple(sorted(team))))

    table = (runs.groupby(_SAME_RUN + ["platform"], as_index=False)
                 .agg(runs=("t_kernel", "size"), t_kernel=("t_kernel", "median"),
                      proposals=("proposals", "median"), accepts=("accepts", "median"),
                      best=("best", "first")))
    proposed = table["proposals"] > 0
    annealing = ~table["platform"].astype(str).str.startswith("exact")
    table["msteps_per_s"] = (table["proposals"] / table["t_kernel"] / 1e6).where(proposed)
    table["accept_rate"] = (table["accepts"] / table["proposals"]).where(proposed & annealing)

    base = (table[table["platform"] == baseline][_SAME_RUN + ["t_kernel", "best"]]
            .rename(columns={"t_kernel": "t_baseline", "best": "best_baseline"}))
    table = table.merge(base, on=_SAME_RUN, how="left")
    table["speedup"] = table["t_baseline"] / table["t_kernel"]
    table["same_best"] = [None if not isinstance(theirs, tuple) else ours == theirs
                          for ours, theirs in zip(table["best"], table["best_baseline"])]

    table["_order"] = (table["platform"] != baseline).astype(int)
    table = table.sort_values(["n", "k", "instance_id", "seed", "chains", "sweeps", "_order",
                               "platform"])
    return table[SPEEDUP_COLUMNS].reset_index(drop=True)


def time_to_answer(df: pd.DataFrame) -> pd.DataFrame:
    """!
    @brief TODO: for SA runs, time until a chain first reaches the known-best energy.
    @param df Run records, as returned by load_runs().
    @return One time-to-answer row per (n, k, platform).
    @see docs/modules/benchmark-analysis.md
    """
    raise NotImplementedError("TODO")
