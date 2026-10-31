"""!
@brief Cross-checks the native exact_cpu oracle (module_1) against the independent Python
       brute force (Validator.brute_force()) and against judge(), on synthetic instances.
@see docs/modules/module-interface.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from data_collectors import Instance  # noqa: E402
from module_interface import ModuleConfig, run_module  # noqa: E402
from validation import Validator  # noqa: E402

pytest.importorskip("module_1", reason="run scripts/build_native.sh first")


def _random_instance(n: int, k: int, seed: int) -> Instance:
    """!
    @brief Builds a synthetic Instance with random reward/penalty for testing.
    @param n Number of drugs.
    @param k Combination size.
    @param seed RNG seed.
    @return A solver-ready Instance.
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n).astype(np.float32)
    p = rng.random((n, n)).astype(np.float32)
    p = (p + p.T) / 2
    np.fill_diagonal(p, 0)
    return Instance(id=f"synthetic-n{n}-k{k}-seed{seed}", n=n, k=k, lam=1.0, w=w, P=p)


@pytest.mark.parametrize("n,k", [(6, 2), (10, 3), (12, 0), (5, 5)])
def test_exact_cpu_matches_python_oracle(n: int, k: int) -> None:
    """!
    @brief module_1's exact_cpu result must match Validator.brute_force() exactly: same
           selection (up to tie-breaking) and the same energy.
    @param n Number of drugs in the synthetic instance.
    @param k Combination size.
    """
    inst = _random_instance(n, k, seed=n * 100 + k)
    config = ModuleConfig(platform="exact_cpu", seed=1)

    native = run_module(inst, config)
    oracle = Validator().brute_force(inst)

    assert sorted(native.best) == sorted(oracle.best)
    assert native.e_solver == pytest.approx(oracle.e_solver, abs=1e-5)
    assert native.ties == oracle.ties
    assert native.proposals == pytest.approx(
        __import__("math").comb(n, k)
    )


def test_exact_cpu_result_passes_judge() -> None:
    """!
    @brief The native result must independently re-score as valid under judge().
    """
    inst = _random_instance(n=8, k=3, seed=99)
    config = ModuleConfig(platform="exact_cpu", seed=1)

    result = run_module(inst, config)
    graded = Validator().grade(inst, result)

    assert graded.valid
    assert graded.e_judge == pytest.approx(graded.e_solver, abs=1e-5)


def test_exact_cpu_infeasible_instance_raises() -> None:
    """!
    @brief n < k must raise ValueError, not silently return a result.
    """
    inst = _random_instance(n=2, k=3, seed=1)
    config = ModuleConfig(platform="exact_cpu", seed=1)

    with pytest.raises(ValueError):
        run_module(inst, config)


def test_unimplemented_platform_raises() -> None:
    """!
    @brief Platforms module_1 doesn't implement yet (scalar/avx2/cuda SA) must raise
           RuntimeError rather than returning fake data.
    """
    inst = _random_instance(n=6, k=2, seed=2)
    config = ModuleConfig(platform="scalar", seed=1)

    with pytest.raises(RuntimeError):
        run_module(inst, config)
