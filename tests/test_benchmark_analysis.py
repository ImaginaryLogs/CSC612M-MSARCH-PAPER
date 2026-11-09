"""!
@brief Checks benchmark_analysis: the run log round trip and speedup_table(). Most tests
       build run records by hand, so they need no native module; the last one times the
       real scalar and AVX2 backends when they are built.
@see docs/modules/benchmark-analysis.md
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import benchmark_analysis as ba  # noqa: E402
from data_collectors import Instance  # noqa: E402
from module_interface import ModuleConfig, RunResultsMessage, run_module  # noqa: E402


def _run(platform: str, t_kernel: float, *, instance_id: str = "synthetic", n: int = 50,
         k: int = 3, seed: int = 1, chains: int = 4, sweeps: int = 100,
         best: list[int] | None = None, proposals: int = 20_000,
         accepts: int = 5_000) -> RunResultsMessage:
    """!
    @brief One hand-made run record.
    @return A RunResultsMessage with the given timing and settings.
    """
    return RunResultsMessage(instance_id=instance_id, platform=platform, seed=seed,
                             chains=chains, sweeps=sweeps, best=best or [0, 1, 2],
                             e_solver=-1.0, n=n, k=k, t_kernel=t_kernel,
                             proposals=proposals, accepts=accepts)


def _table(runs: list[RunResultsMessage], tmp_path: Path, baseline: str = "scalar") -> pd.DataFrame:
    """!
    @brief Writes runs with append_run(), reads them back with load_runs(), and builds the table.
    """
    path = tmp_path / "runs.jsonl"
    for r in runs:
        ba.append_run(r, path)
    return ba.speedup_table(ba.load_runs(path), baseline=baseline)


def _row(table: pd.DataFrame, platform: str) -> pd.Series:
    rows = table[table["platform"] == platform]
    assert len(rows) == 1, f"expected one {platform} row, got {len(rows)}"
    return rows.iloc[0]


def test_speedup_is_baseline_time_over_platform_time(tmp_path: Path) -> None:
    """!
    @brief avx2 at 1/3 of scalar's time is a 3x speedup; the baseline row comes first at 1x.
    """
    table = _table([_run("scalar", 0.30), _run("avx2", 0.10)], tmp_path)
    assert list(table.columns) == ba.SPEEDUP_COLUMNS
    assert list(table["platform"]) == ["scalar", "avx2"]
    assert _row(table, "scalar")["speedup"] == pytest.approx(1.0)
    assert _row(table, "avx2")["speedup"] == pytest.approx(3.0)
    assert _row(table, "avx2")["msteps_per_s"] == pytest.approx(20_000 / 0.10 / 1e6)
    assert _row(table, "avx2")["accept_rate"] == pytest.approx(0.25)
    assert bool(_row(table, "avx2")["same_best"]) is True


def test_repeated_runs_use_the_median(tmp_path: Path) -> None:
    """!
    @brief Three avx2 runs (one slow outlier) count as their median time.
    """
    runs = [_run("scalar", 0.30), _run("avx2", 0.10), _run("avx2", 0.12), _run("avx2", 0.90)]
    avx2 = _row(_table(runs, tmp_path), "avx2")
    assert avx2["runs"] == 3
    assert avx2["t_kernel"] == pytest.approx(0.12)
    assert avx2["speedup"] == pytest.approx(0.30 / 0.12)


def test_only_same_instance_and_settings_are_compared(tmp_path: Path) -> None:
    """!
    @brief Different sizes, seeds or sweep counts form separate groups, each against its
           own baseline run.
    """
    runs = [
        _run("scalar", 0.30, n=50), _run("avx2", 0.10, n=50),
        _run("scalar", 2.00, n=500, instance_id="bigger"), _run("avx2", 0.50, n=500, instance_id="bigger"),
        _run("avx2", 0.01, sweeps=10),  # no scalar run with these settings
    ]
    table = _table(runs, tmp_path)
    assert len(table) == 5
    big = table[(table["n"] == 500) & (table["platform"] == "avx2")].iloc[0]
    assert big["speedup"] == pytest.approx(4.0)
    lone = table[table["sweeps"] == 10].iloc[0]
    assert math.isnan(lone["speedup"])
    assert lone["same_best"] is None


def test_different_answers_are_flagged(tmp_path: Path) -> None:
    """!
    @brief same_best is False when a platform's best team differs from the baseline's.
    """
    table = _table([_run("scalar", 0.3, best=[0, 1, 2]), _run("avx2", 0.1, best=[0, 1, 3])],
                   tmp_path)
    assert bool(_row(table, "avx2")["same_best"]) is False


def test_exact_search_has_throughput_but_no_accept_rate(tmp_path: Path) -> None:
    """!
    @brief For exact search "proposals" are combinations scored, so throughput is shown
           but an accept rate is not.
    """
    table = _table([_run("scalar", 0.3), _run("exact_cpu", 0.6, proposals=19_600, accepts=0)],
                   tmp_path)
    exact = _row(table, "exact_cpu")
    assert exact["msteps_per_s"] == pytest.approx(19_600 / 0.6 / 1e6)
    assert math.isnan(exact["accept_rate"])
    assert exact["speedup"] == pytest.approx(0.5)


def test_untimed_runs_and_empty_logs_give_an_empty_table(tmp_path: Path) -> None:
    """!
    @brief Runs with no timing are dropped; nothing to compare gives an empty table with
           the usual columns.
    """
    assert list(ba.speedup_table(pd.DataFrame()).columns) == ba.SPEEDUP_COLUMNS
    table = _table([_run("scalar", 0.0)], tmp_path)
    assert table.empty and list(table.columns) == ba.SPEEDUP_COLUMNS


def test_older_records_get_n_and_k_from_the_instance_id(tmp_path: Path) -> None:
    """!
    @brief Records written before RunResultsMessage had n and k still group by size: the
           size is read from ids such as "ad_combo_n3_k3_seed1234".
    """
    old = pd.DataFrame([
        {"instance_id": "ad_combo_n40_k5_seed1", "platform": p, "seed": 1, "chains": 4,
         "sweeps": 100, "best": [0, 1, 2, 3, 4], "t_kernel": t, "proposals": 16_000,
         "accepts": 4_000}
        for p, t in (("scalar", 0.2), ("avx2", 0.1))
    ])
    table = ba.speedup_table(old)
    assert set(table["n"]) == {40} and set(table["k"]) == {5}
    assert _row(table, "avx2")["speedup"] == pytest.approx(2.0)


@pytest.mark.skipif(
    importlib.util.find_spec("module_2") is None or importlib.util.find_spec("module_2_avx") is None,
    reason="needs the scalar and AVX2 extensions (run scripts/build_native.sh first)")
def test_real_scalar_and_avx2_runs(tmp_path: Path) -> None:
    """!
    @brief End to end: time the real backends through run_module(), log the runs, and
           check the table lines them up (same answer, n and k recorded, a positive speedup).
    """
    import module_2_avx
    if not module_2_avx.supported():
        pytest.skip("this CPU/OS has no AVX2")
    rng = np.random.default_rng(3)
    n, k = 200, 5
    p = rng.random((n, n)).astype(np.float32)
    p = (p + p.T) / 2
    np.fill_diagonal(p, 0)
    inst = Instance(id=f"synthetic-n{n}-k{k}", n=n, k=k, lam=1.0,
                    w=(0.1 + 0.9 * rng.random(n)).astype(np.float32), P=p)
    runs = [run_module(inst, ModuleConfig(platform=platform, seed=5, chains=2, sweeps=200,
                                          T0=2.0, T1=0.02))
            for platform in ("scalar", "avx2", "scalar", "avx2")]
    assert all(r.n == n and r.k == k for r in runs)

    table = _table(runs, tmp_path)
    assert list(table["platform"]) == ["scalar", "avx2"]
    avx2 = _row(table, "avx2")
    assert avx2["runs"] == 2
    assert bool(avx2["same_best"]) is True
    assert avx2["speedup"] > 0
    assert 0 < avx2["accept_rate"] < 1
