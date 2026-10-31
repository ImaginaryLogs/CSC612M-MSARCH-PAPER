# `python/benchmark_analysis.py`

Turns a pile of `RunResultsMessage` records into the sequential-vs-parallel comparison the course requires (requirements PDF, Milestone 3 deliverable #3) and the figures/tables listed in `evaluation.tex` (F2-F5, T1-T2).

`append_run()` / `load_runs()` are real (just JSON lines, nothing to stub). The actual speedup/plotting logic is TODO until there's real timing data to look at.

## `speedup_table()`

TODO: group by `(n, k, solver)`, compute `t_kernel` speedup of each platform relative to `baseline`. This is table T2 / figures F2-F5 in `evaluation.tex`.

## `time_to_answer()`

TODO: for SA runs, time until a chain first reaches the known-best energy (`evaluation.tex`, "Three ways to measure speed").
