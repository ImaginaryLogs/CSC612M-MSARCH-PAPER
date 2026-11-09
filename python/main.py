"""!
@brief CLI entry point: parses arguments, builds an OrchestratorConfig, runs it.
@see docs/modules/main.md
"""
from __future__ import annotations

import argparse

from orchestrator import Orchestrator, OrchestratorConfig


def parse_args() -> argparse.Namespace:
    """!
    @brief Parses the CLI arguments.
    @return Parsed arguments (platforms, k, lam, seed, chains, sweeps, t0, t1,
            brute_force_check, synthetic).
    """
    parser = argparse.ArgumentParser(description="AD drug-combination solver pipeline")
    parser.add_argument("--platforms", nargs="+",
                         default=["scalar", "avx2", "cuda", "qaoa"],
                         choices=["exact_cpu", "exact_cuda", "scalar", "avx2", "cuda", "qaoa"])
    parser.add_argument("--k", type=int, default=3, help="combination size")
    parser.add_argument("--lam", type=float, default=1.0, help="overlap penalty weight")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--sweeps", type=int, default=1000)
    parser.add_argument("--t0", type=float, default=None,
                         help="start temperature in energy units (required for scalar and avx2 SA)")
    parser.add_argument("--t1", type=float, default=None,
                         help="end temperature in energy units (required for scalar and avx2 SA)")
    parser.add_argument("--brute-force-check", action="store_true",
                         help="also compute the exact optimum for comparison (slow for large n)")
    parser.add_argument("--synthetic", type=int, default=None, metavar="N",
                         help="solve a random N-drug instance instead of the real data (no "
                              "download); for timing at sizes the candidate list can't reach yet")
    args = parser.parse_args()
    if args.synthetic is not None and not 1 <= args.k <= args.synthetic:
        parser.error(f"--synthetic {args.synthetic} needs 1 <= --k <= {args.synthetic}")
    return args


def main() -> None:
    args = parse_args()
    config = OrchestratorConfig(
        platforms=args.platforms, k=args.k, lam=args.lam, seed=args.seed,
        chains=args.chains, sweeps=args.sweeps, T0=args.t0, T1=args.t1,
        brute_force_check=args.brute_force_check, synthetic_n=args.synthetic,
    )
    orchestrator = Orchestrator(config)
    results = orchestrator.run()
    for r in results:
        print(f"{r.platform:10s} best={r.best} e_solver={r.e_solver:.4f} "
              f"e_judge={r.e_judge:.4f} valid={r.valid}")

    # Timing for this instance and these settings, over every run recorded so far
    # (data/runs.jsonl), so repeating the command adds runs to the medians.
    if results:
        table = orchestrator.analyze()
        mine = table[(table["instance_id"] == results[0].instance_id)
                     & (table["seed"] == config.seed) & (table["chains"] == config.chains)
                     & (table["sweeps"] == config.sweeps)]
        if not mine.empty:
            print(f"\nspeedup vs scalar (n={mine['n'].iloc[0]}, k={mine['k'].iloc[0]}, "
                  f"median of every recorded run):")
            print(mine[["platform", "runs", "t_kernel", "msteps_per_s", "accept_rate", "speedup",
                        "same_best"]].to_string(index=False))


if __name__ == "__main__":
    main()
