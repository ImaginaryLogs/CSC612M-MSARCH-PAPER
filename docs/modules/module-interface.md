# `python/module_interface.py`

The one place Python talks to the compiled C/AVX2/CUDA/CUDA-Q modules. Everything that crosses the boundary is a dataclass (`ModuleConfig` in, `RunResultsMessage` out) so nothing downstream ever has to know whether a result came from scalar C++, AVX2, CUDA, or CUDA-Q.

## The native extension

Built as a pybind11 extension module from `include/module.hpp` + `src/module_1.cpp`, compiled by `scripts/build_native.sh` (plain CMake, not yet folded into `uv sync` -- see the TODO at the top of `pyproject.toml`):

- `include/module.hpp` — the interface every backend implements (`Instance`/`Config`/`Result` structs, `energy()`, and one `solve_*()` per backend)
- `src/module_1.cpp` — built so far: `solve_exact()`, the `exact_cpu` brute-force oracle (tries every k-combination, `O(C(n,k) * k^2)`), bound as `module_1.solve(instance, config)`
- `src/module_2_avx.cpp` — TODO: AVX2 simulated annealing (NASM kernel + this wrapper)
- `cuda/module_3.cu` — TODO: CUDA simulated annealing
- CUDA-Q stays pure Python — it already has a clean API, no C++ needed

`module_1.solve()` raises `RuntimeError` for any platform other than `exact_cpu` (scalar/avx2/cuda SA, `exact_cuda`) rather than returning fake data, since those backends don't exist yet. If `module_1` isn't built at all, `_native_module()` raises a clear `NotImplementedError` pointing at `scripts/build_native.sh` instead of a bare `ImportError`.

`exact_cpu` is also the oracle `tests/test_module_1.py` cross-checks against `validation.Validator.brute_force()` (the independent Python implementation) — useful both as a build/interop smoke test and, once SA lands, as ground truth to grade it against (together with `OrchestratorConfig.brute_force_check`).

## `ModuleConfig`

- `platform` — `"scalar" | "avx2" | "cuda" | "qaoa" | "exact_cpu" | "exact_cuda"`
- `T0`/`T1` — `None` means a default schedule is used
- `trace_steps` — `>0` only for the cross-platform agreement check

## `RunResultsMessage`

What comes back from one solver run, whatever module produced it. `e_judge`/`valid` are left unset here — `validation.py` fills them in; a module never grades its own answer.

## `_native_module()`

Becomes `import module_1` (the pybind11 extension) once built. Kept as a function (not a module-level import) so the rest of this file — and anything that imports it — can be exercised before the native module exists.

## `run_module()`

Dispatches one solver run to the right backend. QAOA is the exception: it's plain Python calling CUDA-Q, no native extension involved, so it's wired here directly rather than through `_native_module()`.
