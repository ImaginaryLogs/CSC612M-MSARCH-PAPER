# `python/module_interface.py`

The one place Python talks to the compiled C/AVX2/CUDA/CUDA-Q modules. Everything that crosses the boundary is a dataclass (`ModuleConfig` in, `RunResultsMessage` out) so nothing downstream ever has to know whether a result came from scalar C++, AVX2, CUDA, or CUDA-Q.

## TODO: the native extension

Next pass, once the Python skeleton is settled, build the native side as a pybind11 extension module, driven by scikit-build-core from the root `pyproject.toml`:

- `include/module.hpp` — the interface every backend implements
- `src/module_1.cpp` — worked example (scalar C++) using `module.hpp`
- `src/module_2_avx.cpp` — AVX2 (NASM kernel + this wrapper)
- `cuda/module_3.cu` — CUDA
- CUDA-Q stays pure Python — it already has a clean API, no C++ needed

Until that extension exists, `import module_1` fails and `run_module()` raises a clear `NotImplementedError` instead of silently returning fake data.

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
