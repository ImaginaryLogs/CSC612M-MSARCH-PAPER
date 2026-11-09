"""!
@brief Checks the scalar SA backend (module_2) through the same Python boundary the
       pipeline uses: ModuleConfig in, RunResultsMessage out. The C core has its own
       standalone test in tests/sa_scalar_test.c; these tests cover the glue.
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

pytest.importorskip("module_2", reason="run scripts/build_native.sh first")


def _random_instance(n: int, k: int, seed: int) -> Instance:
    """!
    @brief Builds a synthetic Instance: rewards in (0.1, 1], symmetric penalties in [0, 1].
    @param n Number of drugs.
    @param k Team size.
    @param seed RNG seed.
    @return A solver-ready Instance.
    """
    rng = np.random.default_rng(seed)
    w = (0.1 + 0.9 * rng.random(n)).astype(np.float32)
    p = rng.random((n, n)).astype(np.float32)
    p = (p + p.T) / 2
    np.fill_diagonal(p, 0)
    return Instance(id=f"synthetic-n{n}-k{k}-seed{seed}", n=n, k=k, lam=1.0, w=w, P=p)


def _config(**overrides) -> ModuleConfig:
    """!
    @brief A scalar-SA config with temperatures that suit the synthetic instances.
    @param overrides Fields of ModuleConfig to change.
    @return The config.
    """
    fields = dict(platform="scalar", seed=7, chains=8, sweeps=300, T0=2.0, T1=0.05)
    fields.update(overrides)
    return ModuleConfig(**fields)


@pytest.mark.parametrize("n,k", [(8, 2), (10, 3), (12, 4)])
def test_scalar_finds_the_exact_optimum(n: int, k: int) -> None:
    """!
    @brief On small instances SA should reach the energy of the exact answer, and the
           independent judge() should agree with the energy SA reported.
    @param n Number of drugs.
    @param k Team size.
    """
    inst = _random_instance(n, k, seed=n * 100 + k)
    result = run_module(inst, _config())
    oracle = Validator().brute_force(inst)

    graded = Validator().grade(inst, result)
    assert graded.valid
    assert result.e_solver == pytest.approx(oracle.e_solver, abs=1e-6)
    assert len(set(result.best)) == k


def test_counters_and_timing_are_filled_in() -> None:
    """!
    @brief proposals must be chains * sweeps * n, and accepts cannot exceed it.
    """
    inst = _random_instance(n=10, k=3, seed=5)
    cfg = _config(chains=4, sweeps=50)
    result = run_module(inst, cfg)

    assert result.proposals == 4 * 50 * 10
    assert 0 < result.accepts <= result.proposals
    assert result.t_kernel > 0.0


def test_same_seed_repeats_exactly_and_other_seeds_differ() -> None:
    """!
    @brief Paper test T10: same seed, same run; different seed, different run.
    """
    inst = _random_instance(n=12, k=4, seed=3)
    first = run_module(inst, _config(chains=2, sweeps=40, trace_steps=40 * 12))
    again = run_module(inst, _config(chains=2, sweeps=40, trace_steps=40 * 12))
    other = run_module(inst, _config(chains=2, sweeps=40, trace_steps=40 * 12, seed=8))

    assert first.best == again.best
    assert first.accepts == again.accepts
    assert first.extra["trace"] == again.extra["trace"]
    assert first.extra["trace"] != other.extra["trace"]


def test_trace_shape_and_agreement_with_accept_counter() -> None:
    """!
    @brief The trace has one row per chain and one entry per traced step, and the number
           of accepted entries equals the accept counter when every step is traced.
    """
    inst = _random_instance(n=9, k=3, seed=11)
    result = run_module(inst, _config(chains=3, sweeps=20, trace_steps=20 * 9))
    trace = np.array(result.extra["trace"])

    assert trace.shape == (3, 20 * 9)
    assert trace.sum() == result.accepts


def test_trace_is_absent_by_default() -> None:
    """!
    @brief Without trace_steps nothing is traced, so results stay small and JSON-safe.
    """
    inst = _random_instance(n=8, k=2, seed=1)
    result = run_module(inst, _config())
    assert "trace" not in result.extra


@pytest.mark.parametrize("t0,t1", [(None, None), (2.0, None), (None, 0.05), (0.01, 0.5), (2.0, 0.0)])
def test_missing_or_invalid_temperatures_are_refused(t0, t1) -> None:
    """!
    @brief The scalar backend has no default schedule: missing, reversed or zero
           temperatures raise ValueError instead of guessing.
    @param t0 Start temperature to try.
    @param t1 End temperature to try.
    """
    inst = _random_instance(n=8, k=2, seed=1)
    with pytest.raises(ValueError):
        run_module(inst, _config(T0=t0, T1=t1))


def test_infeasible_instance_raises() -> None:
    """!
    @brief k > n must raise ValueError, not return a result.
    """
    inst = _random_instance(n=2, k=3, seed=1)
    with pytest.raises(ValueError):
        run_module(inst, _config())


def test_whole_set_and_empty_set_are_handled() -> None:
    """!
    @brief k = n has exactly one team and k = 0 has an empty one; neither should crash.
    """
    everyone = run_module(_random_instance(n=5, k=5, seed=2), _config())
    assert sorted(everyone.best) == [0, 1, 2, 3, 4]
    assert everyone.proposals == 0

    nobody = run_module(_random_instance(n=5, k=0, seed=2), _config())
    assert nobody.best == []
    assert nobody.e_solver == 0.0
