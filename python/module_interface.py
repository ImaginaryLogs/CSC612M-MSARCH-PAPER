"""!
@brief Python's one boundary to the compiled C/AVX2/CUDA modules and CUDA-Q, via ModuleConfig in / RunResultsMessage out.
@see docs/modules/module-interface.md
"""
from __future__ import annotations

import importlib
import math
import time
from dataclasses import dataclass, field


@dataclass
class ModuleConfig:
    platform: str  ##< "scalar" | "avx2" | "cuda" | "qaoa" | "exact_cpu" | "exact_cuda"
    seed: int = 0
    chains: int = 1
    sweeps: int = 1000
    T0: float | None = None  ##< start temperature, energy units; required by "scalar" and "avx2", unused by exact search
    T1: float | None = None  ##< end temperature; T0 >= T1 > 0
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
    n: int = 0  ##< drugs in the instance (0 = not recorded, as in records written before this field)
    k: int = 0  ##< team size
    ties: int = 0
    t_kernel: float = 0.0
    t_h2d: float = 0.0
    t_d2h: float = 0.0
    proposals: int = 0
    accepts: int = 0
    extra: dict = field(default_factory=dict)


## Which compiled extension implements each platform. Platforms not listed here have
#  no native module yet (cuda, exact_cuda).
_NATIVE_MODULES = {"exact_cpu": "module_1", "scalar": "module_2", "avx2": "module_2_avx"}


def _native_module(platform: str):
    """!
    @brief Imports the pybind11 extension that implements a platform.
    @param platform Which backend is wanted, e.g. "exact_cpu" or "scalar".
    @return The native extension module.
    @see docs/modules/module-interface.md
    """
    name = _NATIVE_MODULES.get(platform)
    if name is None:
        raise NotImplementedError(f"platform {platform!r} has no native module yet")
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise NotImplementedError(
            f"{name} (the pybind11 extension for {platform!r}) is not built -- run "
            "scripts/build_native.sh, which compiles it and copies it next to "
            "python/main.py"
        ) from exc


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
        native = _native_module(config.platform)
        result = native.solve(instance, config)

    t_total = time.perf_counter() - t_start
    extra = {"t_total": t_total}
    if "trace" in result:  # only when config.trace_steps > 0; nested lists keep it JSON-safe
        extra["trace"] = result["trace"].tolist()
    return RunResultsMessage(
        instance_id=instance.id,
        platform=config.platform,
        seed=config.seed,
        chains=config.chains,
        sweeps=config.sweeps,
        best=sorted(result["best"]),
        e_solver=result["e_solver"],
        n=instance.n,
        k=instance.k,
        ties=result.get("ties", 0),
        t_kernel=result.get("t_kernel", 0.0),
        t_h2d=result.get("t_h2d", 0.0),
        t_d2h=result.get("t_d2h", 0.0),
        proposals=result.get("proposals", 0),
        accepts=result.get("accepts", 0),
        extra=extra,
    )
