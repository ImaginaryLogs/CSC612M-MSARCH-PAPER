"""!
@brief Python's one boundary to the compiled C/AVX2/CUDA modules and CUDA-Q, via ModuleConfig in / RunResultsMessage out.
@see docs/modules/module-interface.md
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field


@dataclass
class ModuleConfig:
    platform: str  ##< "scalar" | "avx2" | "cuda" | "qaoa" | "exact_cpu" | "exact_cuda"
    seed: int = 0
    chains: int = 1
    sweeps: int = 1000
    T0: float | None = None  ##< None => a default schedule is used
    T1: float | None = None
    trace_steps: int = 0  ##< >0 only for the cross-platform agreement check


@dataclass
class RunResultsMessage:
    """!
    @brief What comes back from one solver run, whatever module produced it.
    @see docs/modules/module-interface.md
    """
    instance_id: str
    platform: str
    seed: int
    chains: int
    sweeps: int
    best: list[int]
    e_solver: float
    e_judge: float = math.nan
    valid: bool = False
    ties: int = 0
    t_kernel: float = 0.0
    t_h2d: float = 0.0
    t_d2h: float = 0.0
    proposals: int = 0
    accepts: int = 0
    extra: dict = field(default_factory=dict)


def _native_module():
    """!
    @brief TODO: becomes `import module_1` (the pybind11 extension) once built.
    @return The native extension module.
    @see docs/modules/module-interface.md
    """
    raise NotImplementedError(
        "the native module (pybind11 extension built from include/module.hpp "
        "+ src/module_1.cpp) does not exist yet -- see the TODO at the top "
        "of this file"
    )


def run_module(instance, config: ModuleConfig) -> RunResultsMessage:
    """!
    @brief Dispatches one solver run to the right backend (native extension, or CUDA-Q for "qaoa").
    @param instance Problem instance to solve (combo.instance.Instance).
    @param config Which platform to run and with what settings.
    @return The run's result, with timings and provenance filled in.
    @see docs/modules/module-interface.md
    """
    t_start = time.perf_counter()

    if config.platform == "qaoa":
        from quantum import qaoa  # TODO: python/quantum/qaoa.py, next pass
        result = qaoa.solve(instance, config)
    else:
        native = _native_module()
        result = native.solve(instance, config)  # TODO: exact call shape once module.hpp exists

    t_total = time.perf_counter() - t_start
    return RunResultsMessage(
        instance_id=instance.id,
        platform=config.platform,
        seed=config.seed,
        chains=config.chains,
        sweeps=config.sweeps,
        best=sorted(result["best"]),
        e_solver=result["e_solver"],
        ties=result.get("ties", 0),
        t_kernel=result.get("t_kernel", 0.0),
        t_h2d=result.get("t_h2d", 0.0),
        t_d2h=result.get("t_d2h", 0.0),
        proposals=result.get("proposals", 0),
        accepts=result.get("accepts", 0),
        extra={"t_total": t_total},
    )
