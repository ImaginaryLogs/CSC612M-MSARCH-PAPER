"""!
@brief Checks benchmark_analysis on hand-made run records (no native modules needed).
@see docs/modules/benchmark-analysis.md
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import benchmark_analysis as ba  # noqa: E402


def _row(platform: str, e: float, t: float, sweeps: int = 100, n: int = 20, k: int = 5,
         proposals: int = 0, accepts: int = 0, valid: bool = True, **extra) -> dict:
    return dict(instance_id=f"synthetic_n{n}_k{k}_seed0", platform=platform, seed=0, chains=1,
                sweeps=sweeps, best=[], e_solver=e, e_judge=e, valid=valid, ties=0, t_kernel=t,
                t_h2d=0.0, t_d2h=0.0, proposals=proposals, accepts=accepts, extra=dict(extra))


def _records() -> pd.DataFrame:
    rows = [_row("exact_cpu", -2.0, 0.010, proposals=1000)]
    # scalar at 100 sweeps: half the runs reach -2.0; at 1000 sweeps all do
    rows += [_row("scalar", -2.0 if i % 2 == 0 else -1.5, 0.001, sweeps=100,
                  proposals=2000, accepts=500) for i in range(4)]
    rows += [_row("scalar", -2.0, 0.004, sweeps=1000, proposals=20000, accepts=4000)
             for _ in range(4)]
    return pd.DataFrame(rows)


def test_prepare_uses_exact_energy_as_reference_and_parses_size() -> None:
    data = ba.prepare(_records())
    assert (data["e_ref"] == -2.0).all()
    assert data["ref_is_proven"].all()
    assert (data["n"] == 20).all() and (data["k"] == 5).all()
    assert math.isnan(data.loc[data["platform"] == "exact_cpu", "accept_rate"].iloc[0])


def test_invalid_runs_never_count_as_hits() -> None:
    df = pd.DataFrame([_row("exact_cpu", -2.0, 0.01), _row("scalar", -2.0, 0.001, valid=False)])
    assert not ba.prepare(df).loc[1, "hit"]


def test_quality_table_success_rates() -> None:
    table = ba.quality_table(_records()).set_index(["platform", "sweeps"])
    assert table.loc[("scalar", 100), "success"] == 0.5
    assert table.loc[("scalar", 1000), "success"] == 1.0
    assert table.loc[("scalar", 100), "accept_rate"] == 0.25


def test_time_to_answer_picks_cheapest_budget() -> None:
    tts = ba.time_to_answer(_records()).set_index("platform")
    # 100 sweeps: p = 0.5 -> ln(0.01)/ln(0.5) = 6.64 runs * 1 ms = 6.6 ms
    # 1000 sweeps: p = 1 -> one run of 4 ms, so 1000 sweeps wins
    assert tts.loc["scalar", "sweeps"] == 1000
    assert tts.loc["scalar", "tts"] == pytest.approx(0.004)
    assert tts.loc["exact_cpu", "tts"] == pytest.approx(0.010)


def test_speedup_is_baseline_time_over_platform_time() -> None:
    table = ba.speedup_table(_records(), baseline="scalar", metric="tts").set_index("platform")
    assert table.loc["scalar", "speedup"] == pytest.approx(1.0)
    assert table.loc["exact_cpu", "speedup"] == pytest.approx(0.4)


def test_runs_to_target_edge_cases() -> None:
    assert ba._runs_to_target(1.0, 0.99) == 1.0
    assert ba._runs_to_target(0.0, 0.99) == math.inf
    assert ba._runs_to_target(0.5, 0.99) == pytest.approx(math.log(0.01) / math.log(0.5))


def test_unknown_metric_is_refused() -> None:
    with pytest.raises(ValueError):
        ba.speedup_table(_records(), metric="wall")
