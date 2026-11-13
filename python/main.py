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
    @return Parsed arguments (platforms, k, lam, seed, chains, sweeps, t0, t1, brute_force_check).
    """
    parser = argparse.ArgumentParser(description="AD drug-combination solver pipeline")
    
    args_config = [
        ("--platforms", 
            {"nargs": "+", 
             "default": ["scalar", "avx2", "cuda", "qaoa"], 
             "choices": ["exact_cpu", "exact_cuda", "scalar", "avx2", "cuda", "qaoa"]}),
        ("--k", 
            {"type": int, 
             "default": 3, 
             "help": "combination size"}),
        ("--lam", 
            {"type": float, 
             "default": 1.0, 
             "help": "overlap penalty weight"}),
        ("--seed", 
            {"type": int, "default": 1234}),
        ("--chains", 
            {"type": int, "default": 4}),
        ("--sweeps", 
            {"type": int, "default": 1000}),
        ("--t0", 
            {"type": float, 
             "default": None, 
             "help": "start temperature in energy units (required for scalar SA)"}),
        ("--t1", 
            {"type": float, 
             "default": None, 
             "help": "end temperature in energy units (required for scalar SA)"}),
        ("--brute-force-check", 
            {"action": "store_true", 
             "help": "also compute the exact optimum for comparison (slow for large n)"}),
    ]
    
    for name_or_flags, kwargs in args_config:
        parser.add_argument(name_or_flags, **kwargs)
        
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    config = OrchestratorConfig(
        platforms=args.platforms, 
        k=args.k, 
        lam=args.lam, 
        seed=args.seed,
        chains=args.chains, 
        sweeps=args.sweeps, 
        T0=args.t0, 
        T1=args.t1,
        brute_force_check=args.brute_force_check,
    )
    results = Orchestrator(config).run()
    for r in results:
        print(f"{r.platform:10s} best={r.best} e_solver={r.e_solver:.4f} e_judge={r.e_judge:.4f} valid={r.valid}")


if __name__ == "__main__":
    main()
