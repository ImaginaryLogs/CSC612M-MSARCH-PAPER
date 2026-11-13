"""!
@brief Benchmarks exact_cpu and scalar SA on synthetic instances of several sizes and prints the
       analysis tables from benchmark_analysis (quality, time to answer, speedup).

The real AD instance has only n = 3 drugs, too small to tell the solvers apart, so this builds
random instances shaped like it: rewards w in (0.1, 1], symmetric penalties P in [0, 1] with a zero
diagonal, lambda = 1. For every size it runs exact_cpu (the reference optimum) and then scalar SA
at several sweep budgets with many seeds, grades every result with judge(), and appends one JSON
line per run to --out.

Usage (from the repository root, after scripts/build_native.sh):
    uv run python scripts/bench_synthetic.py
    uv run python scripts/bench_synthetic.py --sizes 10 20 30 --seeds 20 --fresh

@see docs/running-and-testing.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

import benchmark_analysis as ba  # noqa: E402
from data_collectors import Instance  # noqa: E402
from module_interface import ModuleConfig, RunResultsMessage, run_module  # noqa: E402
from validation import Validator  # noqa: E402

PYTHON_CHECK_MAX_COMBINATIONS = 200_000  # above this the Python brute-force cross-check is skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].replace("@brief ", ""))
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 30], help="values of n")
    parser.add_argument("--k", type=int, default=None,
                        help="team size for every n (default: n // 3)")
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--instance-seed", type=int, default=0, help="seed for generating w and P")
    parser.add_argument("--seeds", type=int, default=20, help="SA runs per sweep budget")
    parser.add_argument("--sweeps", type=int, nargs="+", default=[10, 30, 100, 300, 1000, 3000])
    parser.add_argument("--chains", type=int, default=1,
                        help="chains per SA run (1 makes success rate a per-chain number)")
    parser.add_argument("--t0", type=float, default=2.0)
    parser.add_argument("--t1", type=float, default=0.01)
    parser.add_argument("--exact-repeats", type=int, default=3, help="exact_cpu runs per size, for timing")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "bench_runs.jsonl")
    parser.add_argument("--fresh", action="store_true", help="delete --out before running")
    parser.add_argument("--tables-dir", type=Path, default=None,
                        help="also write each table as CSV into this directory")
    return parser.parse_args()


def synthetic_instance(n: int, k: int, lam: float, seed: int) -> Instance:
    """!
    @brief A random instance with the same shape and scaling as the real one.
    @param n Number of drugs.
    @param k Team size.
    @param lam Overlap penalty weight.
    @param seed RNG seed.
    @return The Instance; its id carries n and k so benchmark_analysis can group by size.
    """
    rng = np.random.default_rng(seed)
    w = (0.1 + 0.9 * rng.random(n)).astype(np.float32)
    p = rng.random((n, n)).astype(np.float32)
    p = (p + p.T) / 2
    np.fill_diagonal(p, 0.0)
    return Instance(id=f"synthetic_n{n}_k{k}_seed{seed}", n=n, k=k, lam=lam, w=w, P=p)


def record(inst: Instance, result: RunResultsMessage, e_optimal: float, out: Path) -> None:
    """!
    @brief Grades a result with judge(), tags it with n, k and the optimum, and appends it to out.
    """
    Validator().grade(inst, result)
    result.extra.update({"n": inst.n, "k": inst.k, "e_optimal": e_optimal})
    ba.append_run(result, out)


def run_size(n: int, args: argparse.Namespace) -> None:
    k = args.k if args.k is not None else max(1, n // 3)
    inst = synthetic_instance(n, k, args.lam, args.instance_seed)

    exact = [run_module(inst, ModuleConfig("exact_cpu")) for _ in range(args.exact_repeats)]
    e_optimal = exact[0].e_solver
    combinations = int(np.prod([n - i for i in range(k)]) // np.prod(range(1, k + 1)))
    if combinations <= PYTHON_CHECK_MAX_COMBINATIONS:  # independent cross-check of the reference
        oracle = Validator().brute_force(inst)
        if abs(oracle.e_solver - e_optimal) > 1e-6:
            raise RuntimeError(f"n={n}: exact_cpu ({e_optimal}) disagrees with Python brute force "
                               f"({oracle.e_solver})")
    for result in exact:
        record(inst, result, e_optimal, args.out)

    for sweeps in args.sweeps:
        for seed in range(args.seeds):
            config = ModuleConfig("scalar", seed=seed, chains=args.chains, sweeps=sweeps,
                                  T0=args.t0, T1=args.t1)
            record(inst, run_module(inst, config), e_optimal, args.out)

    print(f"n={n:3d} k={k:2d}: C(n,k)={combinations:,} teams, E*={e_optimal:.6f}, "
          f"exact_cpu {np.median([r.t_kernel for r in exact]) * 1e3:.3f} ms, "
          f"{len(args.sweeps) * args.seeds} SA runs")


def show(title: str, table: pd.DataFrame, tables_dir: Path | None, name: str) -> None:
    print(f"\n== {title} ==")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4g}") if not table.empty else "(no data)")
    if tables_dir is not None:
        tables_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(tables_dir / f"{name}.csv", index=False)


def main() -> None:
    args = parse_args()
    if args.fresh and args.out.exists():
        args.out.unlink()
    for n in args.sizes:
        run_size(n, args)

    df = ba.load_runs(args.out)
    df = df[df["instance_id"].str.startswith("synthetic_")]
    pd.set_option("display.width", 200)

    show("SA quality per sweep budget (success = fraction of runs reaching the exact optimum)",
         ba.quality_table(df), args.tables_dir, "quality")
    show("Time to answer with 99% probability (best sweep budget per platform)",
         ba.time_to_answer(df), args.tables_dir, "time_to_answer")
    show("Speedup over exact_cpu, by time to answer (> 1: faster than exact search)",
         ba.speedup_tts(df, baseline="exact_cpu"), args.tables_dir, "speedup_tts")
    longest = max(args.sweeps)
    one_budget = df[(df["platform"] != "scalar") | (df["sweeps"] == longest)]
    show(f"Speedup over exact_cpu, one run each (scalar at {longest} sweeps)",
         ba.speedup_table(one_budget, baseline="exact_cpu"), args.tables_dir,
         "speedup_kernel")
    print(f"\nruns appended to {args.out.relative_to(REPO_ROOT) if args.out.is_relative_to(REPO_ROOT) else args.out}")


if __name__ == "__main__":
    main()
