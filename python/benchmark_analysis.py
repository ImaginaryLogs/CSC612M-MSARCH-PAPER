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


## Matches the size in instance ids such as "ad_combo_n3_k2_seed1234" or "synthetic_n20_k6_seed0".
_SIZE_IN_ID = re.compile(r"_n(\d+)_k(\d+)")

## Platforms that return the exact optimum every run (success probability 1 by construction).
EXACT_PLATFORMS = ("exact_cpu", "exact_cuda", "exact_python")


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


def speedup_table(df: pd.DataFrame, baseline: str = "scalar", metric: str = "t_kernel",
                  target: float = 0.99) -> pd.DataFrame:
    """!
    @brief Speedup of every platform relative to baseline, per instance size.
    @param df Run records, as returned by load_runs(). With metric="t_kernel", filter it to the
           configuration to compare first (e.g. one sweep budget), since all runs of a platform
           are pooled into one median.
    @param baseline Platform every other platform's speedup is computed relative to.
    @param metric "t_kernel" (median time of one run) or "tts" (time to answer, see time_to_answer()).
    @param target Probability used for metric="tts".
    @return One row per (n, k, platform): time, baseline time, speedup = baseline time / time
            (> 1 means faster than baseline). NaN where the baseline has no runs at that size.
    @see docs/modules/benchmark-analysis.md
    """
    if metric == "t_kernel":
        data = prepare(df)
        if data.empty:
            return data
        times = data.groupby(["n", "k", "platform"]).agg(
            runs=("t_kernel", "size"), time=("t_kernel", "median")).reset_index()
    elif metric == "tts":
        times = time_to_answer(df, target).rename(columns={"tts": "time"})
        if times.empty:
            return times
        times = times[["n", "k", "platform", "runs", "time"]]
    else:
        raise ValueError(f"metric must be 't_kernel' or 'tts', not {metric!r}")
    base = times[times["platform"] == baseline].set_index(["n", "k"])["time"]
    times["baseline_time"] = [base.get((n, k), math.nan) for n, k in zip(times["n"], times["k"])]
    times["speedup"] = times["baseline_time"] / times["time"]
    times.insert(0, "metric", metric)
    return times.astype({"n": int, "k": int}).sort_values(["n", "k", "platform"]).reset_index(drop=True)
