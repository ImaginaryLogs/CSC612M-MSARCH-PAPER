"""!
@brief Checks the AVX2 SA backend (module_2_avx, platform "avx2") through the same Python
       boundary the pipeline uses, and checks it against the scalar backend (module_2)
       decision for decision (paper test T7). The kernels have their own standalone test
       in tests/sa_avx2_test.c; these tests cover the glue and the cross-platform contract.
@see docs/modules/sa-avx2.md
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

module_2_avx = pytest.importorskip("module_2_avx", reason="run scripts/build_native.sh first")
if not module_2_avx.supported():
    pytest.skip("this CPU/OS has no AVX2", allow_module_level=True)


def _random_instance(n: int, k: int, seed: int, lam: float = 1.0) -> Instance:
    """!
    @brief Builds a synthetic Instance: rewards in (0.1, 1], symmetric penalties in [0, 1].
    @param n Number of drugs.
    @param k Team size.
    @param seed RNG seed.
    @param lam Weight on the overlap penalty.
    @return A solver-ready Instance.
    """
    rng = np.random.default_rng(seed)
    w = (0.1 + 0.9 * rng.random(n)).astype(np.float32)
    p = rng.random((n, n)).astype(np.float32)
    p = (p + p.T) / 2
    np.fill_diagonal(p, 0)
    return Instance(id=f"synthetic-n{n}-k{k}-seed{seed}", n=n, k=k, lam=lam, w=w, P=p)


def _config(platform: str = "avx2", **overrides) -> ModuleConfig:
    """!
    @brief An SA config with temperatures that suit the synthetic instances.
    @param platform "avx2" or "scalar".
    @param overrides Fields of ModuleConfig to change.
    @return The config.
    """
    fields = dict(platform=platform, seed=7, chains=8, sweeps=300, T0=2.0, T1=0.05)
    fields.update(overrides)
    return ModuleConfig(**fields)


@pytest.mark.parametrize("n,k", [(8, 2), (10, 3), (12, 4)])
def test_avx2_finds_the_exact_optimum(n: int, k: int) -> None:
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
    assert result.platform == "avx2"
    assert result.e_solver == pytest.approx(oracle.e_solver, abs=1e-6)
    assert len(set(result.best)) == k


@pytest.mark.parametrize(
    "n,k,lam,seed,t0,t1",
    [
        (9, 3, 1.0, 1, 2.0, 0.05),      # one sweep is one full batch of 8 plus one
        (37, 5, 0.5, 2, 2.0, 0.02),
        (64, 8, 1.0, 3, 8.0, 8.0),      # hot: nearly every proposal accepted
        (130, 12, 2.0, 4, 0.01, 0.01),  # cold: nearly every proposal rejected
        (300, 3, 1.0, 5, 1.0, 0.01),
    ],
)
def test_same_decisions_as_scalar(n: int, k: int, lam: float, seed: int, t0: float, t1: float) -> None:
    """!
    @brief Paper test T7: for the same instance, seed and settings the AVX2 and scalar
           backends accept and reject exactly the same proposals, and return the same
           best team, energy and counters.
    """
    pytest.importorskip("module_2", reason="the scalar backend is the reference here")
    inst = _random_instance(n, k, seed=seed, lam=lam)
    sweeps = 60
    common = dict(seed=seed, chains=3, sweeps=sweeps, T0=t0, T1=t1, trace_steps=sweeps * n)
    scalar = run_module(inst, _config("scalar", **common))
    avx2 = run_module(inst, _config("avx2", **common))

    trace_s = np.array(scalar.extra["trace"])
    trace_v = np.array(avx2.extra["trace"])
    assert trace_s.shape == trace_v.shape == (3, sweeps * n)
    if not np.array_equal(trace_s, trace_v):
        chain, step = np.argwhere(trace_s != trace_v)[0]
        pytest.fail(f"first different decision: chain {chain}, step {step}")
    assert avx2.best == scalar.best
    assert avx2.e_solver == scalar.e_solver
    assert avx2.accepts == scalar.accepts
    assert avx2.proposals == scalar.proposals == 3 * sweeps * n


def test_counters_and_timing_are_filled_in() -> None:
    """!
    @brief proposals must be chains * sweeps * n, and accepts cannot exceed it.
    """
    inst = _random_instance(n=10, k=3, seed=5)
    result = run_module(inst, _config(chains=4, sweeps=50))
    assert result.proposals == 4 * 50 * 10
    assert 0 < result.accepts <= result.proposals
    assert result.t_kernel > 0.0


def test_trace_shape_and_agreement_with_accept_counter() -> None:
    """!
    @brief One trace row per chain, and the trace's accepted entries add up to the
           accept counter when every step is traced.
    """
    inst = _random_instance(n=9, k=3, seed=11)
    result = run_module(inst, _config(chains=3, sweeps=20, trace_steps=20 * 9))
    trace = np.array(result.extra["trace"])
    assert trace.shape == (3, 20 * 9)
    assert trace.sum() == result.accepts


@pytest.mark.parametrize("t0,t1", [(None, None), (2.0, None), (None, 0.05), (0.01, 0.5), (2.0, 0.0)])
def test_missing_or_invalid_temperatures_are_refused(t0, t1) -> None:
    """!
    @brief Like the scalar backend, no default schedule: bad temperatures raise ValueError.
    """
    inst = _random_instance(n=8, k=2, seed=1)
    with pytest.raises(ValueError):
        run_module(inst, _config(T0=t0, T1=t1))


def test_infeasible_instance_raises() -> None:
    """!
    @brief k > n must raise ValueError, not return a result.
    """
    with pytest.raises(ValueError):
        run_module(_random_instance(n=2, k=3, seed=1), _config())


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


def test_wrong_platform_is_refused() -> None:
    """!
    @brief module_2_avx only serves "avx2"; asking it for another platform is an error.
    """
    with pytest.raises(RuntimeError):
        module_2_avx.solve(_random_instance(n=8, k=2, seed=1), _config("scalar"))
