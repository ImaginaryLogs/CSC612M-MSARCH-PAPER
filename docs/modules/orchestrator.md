# `python/orchestrator.py`

The coordinator: calls data collection, then each solver module, then validation, then benchmark analysis, in that order. Deliberately NOT a GoF Composite (no tree of nested orchestrators) — just one object that runs four stages in sequence and passes typed data between them.

Stages 1-2 (`collect_data`, `build_instance`) are real: they hit live public APIs (DGIdb, NCBI, STRING, ClinicalTrials.gov — see `data_collectors.py`) and consolidate the result into a solver-ready `Instance`. Stage 3 (`run_modules`) is still TODO — it needs the native C/AVX2/CUDA extension and the CUDA-Q module, neither of which exist yet — so running this end-to-end today gets past data collection and stops there, in `module_interface.py`, not here.

## `collect_data()`

Stage 1: gather inputs. Returns `(drugs, interactome_edges, ad_module_entrez_ids)` — everything `build_instance()` needs. See `data_collectors.py` for what each piece is and where it's a documented substitute for the full biological-scale source.

## `build_instance()`

Stage 2: consolidate collected data into the problem the solvers see.

## `run_modules()`

Stage 3: run every requested platform on the same instance.

## `validate()`

Stage 4: grade every result against the independent judge (and, if enabled, the brute-force oracle).

## `analyze()`

Stage 5: load everything written so far and summarize timing.
