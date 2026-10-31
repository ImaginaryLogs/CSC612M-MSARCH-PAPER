# Overview

## The problem

Pick the best combination of *k* drugs against an Alzheimer's-disease (AD) gene module: each drug gets a reward `w_i` (network closeness to the AD module — the stronger its biological case), and each pair gets a penalty `P_ij` (how much their targets overlap — redundant coverage is worth less than complementary coverage). The objective solvers optimize is

```
E(selection) = -sum(w_i) + lambda * sum_{i<j in selection} P_ij
```

lower is better. This is a small combinatorial optimization problem (choose `k` of `n`), which makes it a clean testbed for comparing solver platforms: the same objective, same input, four different pieces of hardware/software to produce an answer.

The full derivation (closeness scores, separation, QUBO/Ising encoding) is in `docs/paper/01_drug_candidacy_proposal/appendices/formulation.tex`; this doc stays at the pipeline level.

## The pipeline

```
data_collectors.py  ->  Instance  ->  module_interface.py (per platform)  ->  validation.py  ->  benchmark_analysis.py
     (collect)          (consolidate)         (solve)                          (grade)              (compare timing)
```

`orchestrator.py` runs these in sequence; `main.py` is just the CLI front door. See `docs/architecture.md` for the file layout and `docs/modules/orchestrator.md` for the stage-by-stage contract.

1. **Collect** (`data_collectors.py`) — pulls drugs, their gene targets, the AD gene module, and a protein-protein interactome from live, cited public APIs (DGIdb, NCBI, STRING), caching every raw response under `data/raw/` with a SHA-256 manifest for provenance.
2. **Consolidate** (`data_collectors.build_instance()`) — turns that into one solver-ready `Instance`: `n` drugs, reward vector `w`, penalty matrix `P`, choose `k`.
3. **Solve** (`module_interface.py`) — dispatches the same `Instance` to whichever platforms were requested. Every backend speaks the same two dataclasses: `ModuleConfig` in, `RunResultsMessage` out. Nothing downstream needs to know whether a result came from C++, assembly, a GPU kernel, or a quantum circuit simulator.
4. **Grade** (`validation.py`) — `judge()` independently recomputes `E(selection)` for every result (a solver never grades its own answer), and `Validator(brute_force=True)` can additionally compute the true optimum to check a heuristic actually found it.
5. **Compare** (`benchmark_analysis.py`) — turns accumulated `RunResultsMessage` records into the sequential-vs-parallel timing tables/figures the course asks for.

## Platforms, and where each one stands

| Platform | What it is | Status |
|---|---|---|
| `exact_cpu` | C++ brute force: every `k`-combination, `O(C(n,k)*k^2)` — the verified oracle every other platform is checked against | **Built** (`include/module.hpp` + `src/module_1.cpp`) |
| `scalar` | Simulated annealing (swap moves), plain scalar C++ — the reference the SIMD/GPU versions must match decision-for-decision in deterministic mode | Planned next — see `docs/TODO.md` |
| `avx2` | The same SA loop, inner update written in NASM assembly (`ymm` registers), with a C++ intrinsics twin as a cross-check | Planned next |
| `cuda` | SA with many independent chains, one thread per chain (v1), then one warp per chain (v2) | Planned after Milestone 2 |
| `qaoa` | Standard QAOA on CUDA-Q's GPU state-vector simulator, penalty-encoded `k`-exactly constraint | Planned after Milestone 2 |
| `exact_cuda` | GPU exact search, cross-checked against `exact_cpu` | Planned after Milestone 2 |

`exact_cpu` exists specifically to make the rest of this list checkable: once `scalar`/`avx2`/`cuda` exist, `OrchestratorConfig.brute_force_check=True` runs the same instance through the oracle and flags any heuristic answer that isn't actually optimal.

## Why this split (Python orchestrates, C++/CUDA/CUDA-Q solve)

Keeping every solver behind one typed boundary (`ModuleConfig`/`RunResultsMessage`) means the orchestration, grading, and benchmarking code never changes as new platforms land — only `module_interface.py`'s dispatch and a new `src/module_*.cpp` (or `.cu`, or CUDA-Q script) are added. This is also what makes a fair sequential-vs-parallel timing comparison possible: every platform is handed the exact same `Instance` and graded by the exact same `judge()`.

## See also

- [`architecture.md`](architecture.md) — file structure and where logic lives
- [`modules/`](modules) — one doc per `python/` module
- [`TODO.md`](TODO.md) — build-out plan and schedule
- `docs/paper/01_drug_candidacy_proposal/` — the full academic writeup (problem, related work, formulation, evaluation plan)
