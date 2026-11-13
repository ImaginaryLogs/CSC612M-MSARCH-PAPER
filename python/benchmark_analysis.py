"""!
@brief Turns RunResultsMessage records into the course's sequential-vs-parallel comparison.
@see docs/modules/benchmark-analysis.md
"""
from __future__ import annotations

import json
import math
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
## Platforms that return the exact optimum every run (success probability 1 by construction).
EXACT_PLATFORMS = ("exact_cpu", "exact_cuda", "exact_python")


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


def _size_of(row: pd.Series) -> tuple[float, float]:
    """!
    @brief (n, k) of one record: from extra["n"]/extra["k"] when present, else parsed from instance_id.
    @param row One run record.
    @return (n, k), NaN where neither source has it.
    """
    extra = row.get("extra") if isinstance(row.get("extra"), dict) else {}
    if "n" in extra and "k" in extra:
        return float(extra["n"]), float(extra["k"])
    match = _SIZE_IN_ID.search(str(row.get("instance_id", "")))
    if match:
        return float(match.group(1)), float(match.group(2))
    return math.nan, math.nan


def prepare(df: pd.DataFrame, tolerance: float = 1e-6) -> pd.DataFrame:
    """!
    @brief Adds the columns every analysis below needs: n, k, e_ref, gap, hit, accept_rate, ns_per_proposal.
           For exact platforms "proposals" counts teams evaluated, so ns_per_proposal is time per
           team and accept_rate is NaN.
    @param df Run records, as returned by load_runs().
    @param tolerance How close to e_ref (energy units) counts as reaching the optimum.
    @return A copy of df with the extra columns.
    @details e_ref is the reference (best) energy of each instance: extra["e_optimal"] when any
             record has it (an exact solver or --brute-force-check), otherwise the lowest valid
             e_judge seen on that instance (best known, not proven optimal). Energies are taken
             from e_judge, so a solver is never scored on its own number; invalid records never hit.
    """
    out = df.copy()
    if out.empty:
        return out
    sizes = out.apply(_size_of, axis=1, result_type="expand")
    out["n"], out["k"] = sizes[0], sizes[1]

    def optimal_of(extra) -> float:
        return float(extra.get("e_optimal", math.nan)) if isinstance(extra, dict) else math.nan

    out["e_optimal"] = out["extra"].map(optimal_of) if "extra" in out else math.nan
    exact_energy = out.loc[out["platform"].isin(EXACT_PLATFORMS) & out["valid"], ["instance_id", "e_judge"]]
    proven = pd.concat([out[["instance_id", "e_optimal"]].rename(columns={"e_optimal": "e"}),
                        exact_energy.rename(columns={"e_judge": "e"})]).groupby("instance_id")["e"].min()
    best_seen = out[out["valid"]].groupby("instance_id")["e_judge"].min()
    reference = proven.combine_first(best_seen)
    out["e_ref"] = out["instance_id"].map(reference)
    out["ref_is_proven"] = out["instance_id"].map(proven).notna()
    out["gap"] = out["e_judge"] - out["e_ref"]
    out["hit"] = out["valid"] & (out["gap"] <= tolerance)
    proposals = out["proposals"].where(out["proposals"] > 0)
    out["accept_rate"] = (out["accepts"] / proposals).where(~out["platform"].isin(EXACT_PLATFORMS))
    out["ns_per_proposal"] = out["t_kernel"] * 1e9 / proposals
    return out


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


def quality_table(df: pd.DataFrame, tolerance: float = 1e-6) -> pd.DataFrame:
    """!
    @brief How often and how closely each configuration reaches the reference energy.
    @param df Run records, as returned by load_runs().
    @param tolerance How close to the reference energy counts as a hit.
    @return One row per (n, k, platform, chains, sweeps): runs, success rate, mean/max gap,
            median t_kernel, acceptance rate, ns per proposal.
    @see docs/modules/benchmark-analysis.md
    """
    data = prepare(df, tolerance)
    if data.empty:
        return data
    data.loc[data["platform"].isin(EXACT_PLATFORMS), ["chains", "sweeps"]] = 0  # not used by exact search
    keys = ["n", "k", "platform", "chains", "sweeps"]
    table = data.groupby(keys).agg(
        runs=("hit", "size"),
        success=("hit", "mean"),
        mean_gap=("gap", "mean"),
        max_gap=("gap", "max"),
        t_kernel_median=("t_kernel", "median"),
        accept_rate=("accept_rate", "mean"),
        ns_per_proposal=("ns_per_proposal", "median"),
        ref_is_proven=("ref_is_proven", "all"),
    ).reset_index()
    return table.astype({"n": int, "k": int, "chains": int, "sweeps": int})


def _runs_to_target(success: float, target: float) -> float:
    """!
    @brief Independent runs needed so at least one hits with probability target: ln(1-target)/ln(1-p).
    @param success Success probability of one run.
    @param target Wanted overall probability, e.g. 0.99.
    @return Number of runs (fractional, >= 1); inf if success is 0.
    """
    if success >= 1.0:
        return 1.0
    if success <= 0.0:
        return math.inf
    return max(1.0, math.log(1.0 - target) / math.log(1.0 - success))


def time_to_answer(df: pd.DataFrame, target: float = 0.99, tolerance: float = 1e-6) -> pd.DataFrame:
    """!
    @brief Time to reach the reference energy with probability target (TTS, the usual annealing metric).
    @param df Run records, as returned by load_runs().
    @param target Wanted probability of having reached the reference energy, e.g. 0.99.
    @param tolerance How close to the reference energy counts as a hit.
    @return One row per (n, k, platform): the sweep budget with the lowest TTS, its success rate,
            runs needed, median time per run, and tts = time per run * runs needed.
    @details A run with success probability p repeated R = ln(1-target)/ln(1-p) times reaches the
             answer with probability target, so TTS = R * median t_kernel. Exact platforms have
             p = 1 and TTS = one run. TTS is inf when no run of any budget hit. This is a
             per-budget estimate from repeated runs, not a measurement inside a single chain.
    @see docs/modules/benchmark-analysis.md
    """
    table = quality_table(df, tolerance)
    if table.empty:
        return table
    table["runs_needed"] = table["success"].map(lambda p: _runs_to_target(p, target))
    table["tts"] = table["runs_needed"] * table["t_kernel_median"]
    best = table.loc[table.groupby(["n", "k", "platform"])["tts"].idxmin()]
    columns = ["n", "k", "platform", "chains", "sweeps", "runs", "success", "runs_needed",
               "t_kernel_median", "tts", "ref_is_proven"]
    return best[columns].sort_values(["n", "k", "platform"]).reset_index(drop=True)


def speedup_tts(df: pd.DataFrame, baseline: str = "scalar", target: float = 0.99) -> pd.DataFrame:
    """!
    @brief Speedup of every platform relative to baseline by time to answer, per instance size.
    @param df Run records, as returned by load_runs().
    @param baseline Platform every other platform's speedup is computed relative to.
    @param target Probability used for time_to_answer().
    @return One row per (n, k, platform): time (the TTS), baseline time, speedup = baseline time / time
            (> 1 means faster than baseline). NaN where the baseline has no runs at that size.
    @details Time to answer is the fair comparison between an exact solver and a heuristic.
    @see docs/modules/benchmark-analysis.md
    """
    tts = time_to_answer(df, target)
    if tts.empty:
        return tts
    times = tts.rename(columns={"tts": "time"})[["n", "k", "platform", "runs", "time"]]
    base = times[times["platform"] == baseline].set_index(["n", "k"])["time"]
    times["baseline_time"] = [base.get((n, k), math.nan) for n, k in zip(times["n"], times["k"])]
    times["speedup"] = times["baseline_time"] / times["time"]
    return times.astype({"n": int, "k": int}).sort_values(["n", "k", "platform"]).reset_index(drop=True)
