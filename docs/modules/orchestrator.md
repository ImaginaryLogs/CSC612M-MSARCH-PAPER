# `python/orchestrator.py`

The coordinator: calls data collection, then each solver module, then validation, then benchmark analysis, in that order. Deliberately NOT a GoF Composite (no tree of nested orchestrators) — just one object that runs four stages in sequence and passes typed data between them.

Stages 1-2 (`collect_data`, `build_instance`) hit live public APIs (DGIdb, NCBI, STRING, ClinicalTrials.gov — see `data_collectors.py`) and consolidate the result into a solver-ready `Instance`. Stage 3 (`run_modules`) runs the built native platforms (`exact_cpu`, `scalar`, `avx2`); `cuda`, `exact_cuda` and `qaoa` have no backend yet.

With `OrchestratorConfig.synthetic_n` set (`main.py --synthetic N`), stages 1-2 are replaced by `data_collectors.synthetic_instance()`: a random N-drug instance, no download. Use it to run and time the solvers at sizes the curated candidate list can't reach yet.

## `collect_data()`

Stage 1: gather inputs. Returns `(drugs, interactome_edges, ad_module_entrez_ids)` — everything `build_instance()` needs. See `data_collectors.py` for what each piece is and where it's a documented substitute for the full biological-scale source.

## `build_instance()`

Stage 2: consolidate collected data into the problem the solvers see.

## `run_modules()`

Stage 3: run every requested platform on the same instance.

## `validate()`

Stage 4: grade every result against the independent judge (and, if enabled, the brute-force oracle).

## `analyze()`

Stage 5: load everything written so far and summarize timing. Returns `benchmark_analysis.speedup_table()` over every recorded run, with `scalar` as the baseline; `main.py` prints the rows for the instance it just solved.
