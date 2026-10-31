# TODO

Build-out plan, in the order things actually need to happen: **C++ scalar SA and AVX2 first** (Phase 1 — the reference every other platform is checked against), **then CUDA and CUDA-Q** (Phase 2, once that reference exists and Milestone 2 has passed). Dates follow the academic-week schedule in `docs/paper/01_drug_candidacy_proposal/sections/milestones.tex`, anchored to the actual proposal submission (Week 6 = 2026-10-12, this repo's initial commit). Today is 2026-10-31 — Week 8.

## Phase 0 — Problem setup (Week 6–7, Oct 12 – Oct 25)

- [x] Frozen, cited data pipeline (`data_collectors.py`): drugs, targets, AD module, interactome, all from live public APIs with a SHA-256 manifest
- [x] Solver-ready `Instance` consolidation (`build_instance()`)
- [x] `judge()` — the independent energy function every result is graded against
- [x] `ModuleConfig`/`RunResultsMessage` — the typed boundary every native backend speaks

## Phase 1 — C++ exact search + scalar/AVX2 SA (Week 7–9, Oct 19 – Nov 8)

The **must-have** priority tier: everything Milestone 2 (Week 10) demos.

- [x] `include/module.hpp` — shared `Instance`/`Config`/`Result` structs, the interface every backend implements
- [x] `src/module_1.cpp` — **exact_cpu**: C++ brute-force oracle, pybind11-bound, built via `scripts/build_native.sh` *(done 2026-10-31)*
- [x] `tests/test_module_1.py` — cross-checked against the independent Python oracle (`Validator.brute_force()`)
- [ ] **Target: Nov 1 (end of Week 8)** — scalar C++ SA (swap moves), integer deterministic mode (Philox-seeded, shared accept/reject lookup table) as the reference every later platform must match decision-for-decision
- [ ] **Target: Nov 8 (end of Week 9)** — AVX2 SA: NASM inner update loop (`ymm` registers) + a C++ intrinsics twin as a cross-check; unit-tested against the scalar loop
- [ ] **Target: Nov 8** — cross-platform agreement test (T7): scalar vs. AVX2 vs. exact_cpu produce identical decisions for a fixed seed (`ModuleConfig.trace_steps`)
- [ ] **Target: Nov 8** — first sequential-vs-parallel timing table for exact search and SA (`benchmark_analysis.speedup_table()`)
- [ ] **Target: Nov 8** — first best AD combinations for `k in {2, 3}` at two values of `lambda`

### Milestone 2 — mid-project demo (Week 10, Nov 9 – Nov 15)

- [ ] Exhaustive Python check over all 2^16 patterns on a 16-drug instance confirms the QUBO's best pattern is valid and matches the exact-search answer
- [ ] CPU exact search matches the independent oracle's best combination and tie count
- [ ] Scalar C++, AVX2, and (if ready) CUDA v1 SA make identical decisions for a fixed seed
- [ ] Live demo + first timing table

## Phase 2 — CUDA + CUDA-Q (Week 11–13, Nov 16 – Dec 6)

The **should-have**/**optional** tier, built only once Phase 1's reference exists (SA correctness depends on having something to check against first).

- [ ] **Target: Nov 15** — CUDA exact-search kernel, cross-checked against `exact_cpu`
- [ ] **Target: Nov 22** — CUDA SA v1 (one thread per chain), checked against scalar/AVX2 in deterministic mode
- [ ] **Target: Nov 22** — install CUDA-Q; standard QAOA (penalty-encoded `k`-exactly constraint) on `n <= 16`, checked against the exact-search answer
- [ ] **Target: Nov 29** — CUDA SA v2 (warp per chain) + Nsight profiling
- [ ] **Target: Nov 29** — QAOA for `n` up to 24; study the effect of `lambda` on the chosen combination
- [ ] **Target: Nov 29** — large-instance (hundreds–thousands of drugs) speed tests across scalar/AVX2/CUDA
- [ ] *(stretch, only if everything above lands by Nov 22)* XY-mixer QAOA from a Dicke state — avoids ever producing an invalid combination, but needs a harder circuit

### Milestone 3 — final submission (Week 13, Nov 30 – Dec 6)

- [ ] `benchmark_analysis.speedup_table()` / `time_to_answer()` — filled in once there's real multi-platform timing data
- [ ] `literature_crosscheck()` wired to a real `collect_validation_list()` (currently raises — TODO in `data_collectors.py`)
- [ ] `report.ipynb` — the sequential-vs-parallel narrative, figures F2–F5, tables T1–T2 (`evaluation.tex`)
- [ ] GitHub submission: source, build/benchmark scripts, parallel-algorithm discussion, timing results
- [ ] *(optional)* conference-paper writeup

## Out of scope this term

Real quantum hardware, safety/side-effect data, splitting large instances across multiple QAOA circuits, parallel tempering, multi-core CPU parallelism — see `docs/paper/01_drug_candidacy_proposal/appendices/limitations.tex` for the full, reasoned list.
