# `python/benchmark_analysis.py`

Turns a pile of `RunResultsMessage` records into the sequential-vs-parallel comparison the course requires (requirements PDF, Milestone 3 deliverable #3) and the figures/tables listed in `evaluation.tex` (F2-F5, T1-T2).

`append_run()` / `load_runs()` keep the run log: one JSON line per graded run in `data/runs.jsonl` (git-ignored). The orchestrator appends every run, so repeating a pipeline command adds more runs to compare.

## `speedup_table()`

`speedup_table(df, baseline="scalar")` compares each platform's kernel time with the baseline platform's. It only compares runs of the same instance with the same settings: `instance_id`, `n`, `k`, `seed`, `chains`, `sweeps`. Repeated runs of one platform count as their **median** `t_kernel`. Runs with no timing are dropped.

One row per (instance, settings, platform):

| Column | Meaning |
|---|---|
| `runs` | how many runs the row summarizes |
| `t_kernel` | median kernel time, seconds |
| `msteps_per_s` | millions per second of the run's `proposals`: swaps proposed for SA, combinations scored for exact search |
| `accept_rate` | accepted / proposed swaps (SA only) |
| `speedup` | baseline `t_kernel` ÷ this `t_kernel` (NaN when the group has no baseline run) |
| `same_best` | whether the best team equals the baseline's (None without a baseline run) |

Rows are sorted by `n`, `k`, then the baseline first. Older records without `n`/`k` fields get them from ids like `ad_combo_n3_k3_seed1234`. `main.py` prints the table for the instance it just solved, via `Orchestrator.analyze()`.

`t_kernel` is the whole native solve call, so for large `n` it includes the O(n²) setup. For the annealing loop alone, and for the kernels on their own, use `bench/sa_bench.c` (see [`sa-avx2.md`](sa-avx2.md)). This is table T2 / figures F2-F5 in `evaluation.tex`.

## `time_to_answer()`

TODO: for SA runs, time until a chain first reaches the known-best energy (`evaluation.tex`, "Three ways to measure speed"). The native modules don't report when each chain first reaches the best energy yet, so there is nothing to compute this from.
