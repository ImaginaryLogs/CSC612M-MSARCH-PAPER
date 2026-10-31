"""!
@brief Turns RunResultsMessage records into the course's sequential-vs-parallel comparison.
@see docs/modules/benchmark-analysis.md
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from module_interface import RunResultsMessage

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_PATH = REPO_ROOT / "data" / "runs.jsonl"


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


def speedup_table(df: pd.DataFrame, baseline: str = "scalar") -> pd.DataFrame:
    """!
    @brief TODO: groups by (n, k, solver), computes t_kernel speedup of each platform vs. baseline.
    @param df Run records, as returned by load_runs().
    @param baseline Platform every other platform's speedup is computed relative to.
    @return One speedup row per (n, k, platform).
    @see docs/modules/benchmark-analysis.md
    """
    raise NotImplementedError("TODO: group by instance size and compute speedup vs. baseline")


def time_to_answer(df: pd.DataFrame) -> pd.DataFrame:
    """!
    @brief TODO: for SA runs, time until a chain first reaches the known-best energy.
    @param df Run records, as returned by load_runs().
    @return One time-to-answer row per (n, k, platform).
    @see docs/modules/benchmark-analysis.md
    """
    raise NotImplementedError("TODO")
