"""!
@brief Checks data_collectors.synthetic_instance() and the pipeline's --synthetic path
       (OrchestratorConfig.synthetic_n): a random instance of any size, no download.
@see docs/modules/data-collectors.md
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import benchmark_analysis as ba  # noqa: E402
import data_collectors as dc  # noqa: E402
import orchestrator as orch  # noqa: E402


def test_shape_and_value_ranges_match_the_real_instance() -> None:
    """!
    @brief float32 w in [0.1, 1) and P in [0, 1), symmetric with a zero diagonal -- the
           ranges build_instance() normalizes the real data to.
    """
    inst = dc.synthetic_instance(n=40, k=5, lam=0.5, seed=3)
    assert (inst.n, inst.k, inst.lam) == (40, 5, 0.5)
    assert inst.w.shape == (40,) and inst.w.dtype == np.float32
    assert inst.P.shape == (40, 40) and inst.P.dtype == np.float32
    assert np.all((inst.w >= 0.1) & (inst.w < 1.0))
    assert np.all((inst.P >= 0.0) & (inst.P < 1.0))
    assert np.array_equal(inst.P, inst.P.T)
    assert np.all(np.diag(inst.P) == 0.0)
    assert len(inst.drug_ids) == 40


def test_same_seed_same_instance_other_seed_differs() -> None:
    """!
    @brief Reproducible from the seed, which is what lets repeated runs be compared.
    """
    a = dc.synthetic_instance(n=30, k=3, seed=7)
    b = dc.synthetic_instance(n=30, k=3, seed=7)
    c = dc.synthetic_instance(n=30, k=3, seed=8)
    assert np.array_equal(a.w, b.w) and np.array_equal(a.P, b.P)
    assert not np.array_equal(a.P, c.P)


def test_id_carries_the_size_for_the_speedup_table() -> None:
    """!
    @brief The id encodes n and k the way benchmark_analysis reads them.
    """
    inst = dc.synthetic_instance(n=512, k=8, seed=1234)
    assert inst.id == "synthetic_n512_k8_seed1234"
    assert ba._SIZE_IN_ID.search(inst.id).groups() == ("512", "8")


@pytest.mark.parametrize("n,k", [(0, 0), (3, 4), (5, -1)])
def test_impossible_sizes_are_refused(n: int, k: int) -> None:
    """!
    @brief No drugs, or a team size outside 0..n, raises ValueError.
    """
    with pytest.raises(ValueError):
        dc.synthetic_instance(n=n, k=k)


@pytest.mark.skipif(
    any(importlib.util.find_spec(m) is None for m in ("module_1", "module_2", "module_2_avx")),
    reason="needs the native extensions (run scripts/build_native.sh first)")
def test_pipeline_runs_a_synthetic_instance_without_downloading(monkeypatch) -> None:
    """!
    @brief Orchestrator.run() with synthetic_n solves the random instance on every platform
           and never calls the data collection stage. Run records are captured instead of
           being appended to data/runs.jsonl.
    """
    import module_2_avx
    if not module_2_avx.supported():
        pytest.skip("this CPU/OS has no AVX2")

    def no_download(self):
        raise AssertionError("collect_data() must not run for a synthetic instance")

    recorded = []
    monkeypatch.setattr(orch.Orchestrator, "collect_data", no_download)
    monkeypatch.setattr(orch.ba, "append_run", lambda record: recorded.append(record))

    config = orch.OrchestratorConfig(platforms=["exact_cpu", "scalar", "avx2"], k=3, seed=11,
                                     chains=4, sweeps=300, T0=2.0, T1=0.02, synthetic_n=14)
    results = orch.Orchestrator(config).run()

    assert [r.platform for r in results] == ["exact_cpu", "scalar", "avx2"]
    assert all(r.valid and r.n == 14 and r.k == 3 for r in results)
    assert all(r.instance_id == "synthetic_n14_k3_seed11" for r in results)
    assert results[1].best == results[2].best == results[0].best  # both SA reach the optimum
    assert results[1].accepts == results[2].accepts  # same decisions
    assert len(recorded) == 3
