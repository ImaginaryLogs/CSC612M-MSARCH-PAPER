"""!
@brief Scalar SA vs the exact solvers on the real extended AD instance (38 candidate drugs).

Needs the instance that scripts/real_data_check.py builds and caches; without it the tests are
skipped, so the suite never needs the network. Build it once with:
    uv run python scripts/real_data_check.py
@see docs/real-data-check.md
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from data_collectors import Instance  # noqa: E402
from module_interface import ModuleConfig, run_module  # noqa: E402
from validation import Validator  # noqa: E402

pytest.importorskip("module_1", reason="run scripts/build_native.sh first")
pytest.importorskip("module_2", reason="run scripts/build_native.sh first")

CACHED = [REPO_ROOT / "data" / "raw" / "extended" / variant / "instance_seed1234.npz"
          for variant in ("all_targets", "no_pk_genes")]
AVAILABLE = [p for p in CACHED if p.exists()]

pytestmark = pytest.mark.skipif(not AVAILABLE, reason="run scripts/real_data_check.py first")


def _load(path: Path, k: int) -> Instance:
    data = np.load(path, allow_pickle=False)
    names = [str(x) for x in data["names"]]
    return Instance(id=f"real_{path.parent.name}_k{k}", n=len(names), k=k, lam=1.0,
                    w=data["w"], P=data["P"], drug_ids=names,
                    sources=json.loads(str(data["sources"])))


@pytest.fixture(params=AVAILABLE, ids=[p.parent.name for p in AVAILABLE])
def instance_path(request) -> Path:
    return request.param


def test_instance_is_well_formed(instance_path: Path) -> None:
    inst = _load(instance_path, k=2)
    assert inst.n >= 10, "the point of this instance is more drugs than the 5-drug pipeline list"
    assert inst.w.shape == (inst.n,) and inst.P.shape == (inst.n, inst.n)
    assert np.all(inst.w > 0) and np.isclose(inst.w.max(), 1.0)
    assert np.allclose(inst.P, inst.P.T) and np.all(np.diag(inst.P) == 0)
    assert inst.P.min() >= 0 and inst.P.max() <= 1


@pytest.mark.parametrize("k", [2, 3, 4, "n//3"])
def test_scalar_matches_exact_on_real_data(instance_path: Path, k) -> None:
    n = _load(instance_path, k=2).n
    k = max(2, n // 3) if k == "n//3" else k
    inst = _load(instance_path, k)

    exact = run_module(inst, ModuleConfig("exact_cpu"))
    sa = run_module(inst, ModuleConfig("scalar", seed=1234, chains=4, sweeps=1000, T0=1.0, T1=0.01))

    assert Validator().grade(inst, sa).valid
    assert sa.e_solver == pytest.approx(exact.e_solver, abs=1e-6)
    if math.comb(n, k) <= 200_000:  # independent Python oracle where it is fast enough
        assert Validator().brute_force(inst).e_solver == pytest.approx(exact.e_solver, abs=1e-6)
