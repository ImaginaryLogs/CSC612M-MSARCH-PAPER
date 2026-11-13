# `python/benchmark_analysis.py`

Turns a pile of `RunResultsMessage` records into the sequential-vs-parallel comparison the course requires (requirements PDF, Milestone 3 deliverable #3) and the figures/tables listed in `evaluation.tex` (F2-F5, T1-T2).

Every function takes a DataFrame from `load_runs()`, so it works the same on `data/runs.jsonl` (pipeline runs) and `data/bench_runs.jsonl` (`scripts/bench_synthetic.py`). Tests: `tests/test_benchmark_analysis.py`.

## `append_run()` / `load_runs()`

`append_run()` / `load_runs()` keep the run log: one JSON line per graded run in `data/runs.jsonl` (git-ignored). The orchestrator appends every run, so repeating a pipeline command adds more runs to compare. `load_runs()` returns them as a DataFrame (empty if the file doesn't exist).

## `speedup_table(df, baseline="scalar")`

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

`t_kernel` is the whole native solve call, so for large `n` it includes the O(n²) setup. For the annealing loop alone, and for the kernels on their own, use `bench/sa_bench.c` (see [`sa-avx2.md`](sa-avx2.md)).

## `prepare(df, tolerance=1e-6)`

Adds the columns the other functions use:

| Column | Meaning |
|---|---|
| `n`, `k` | from `extra["n"]`/`extra["k"]` if present, else parsed from `instance_id` (`..._n20_k6_...`) |
| `e_ref` | reference energy of the instance: the exact optimum if any record carries it (`extra["e_optimal"]`, or a valid `exact_*` run), otherwise the lowest valid `e_judge` seen (best known) |
| `ref_is_proven` | `True` when `e_ref` is a proven optimum, `False` when it is only best known |
| `gap` | `e_judge - e_ref` (energies come from the judge, never from the solver itself) |
| `hit` | `valid` and `gap <= tolerance` |
| `accept_rate` | `accepts / proposals`; NaN for exact platforms |
| `ns_per_proposal` | `t_kernel / proposals` in ns. For exact platforms `proposals` counts teams evaluated, so this is time per team |

## `quality_table(df)`

One row per `(n, k, platform, chains, sweeps)`: number of runs, `success` (fraction of runs that hit `e_ref`), mean and max gap, median `t_kernel`, mean acceptance rate, median ns per proposal. This is how you see how many sweeps SA needs at each size.

## `time_to_answer(df, target=0.99)`

Time to reach `e_ref` with probability `target` (TTS, the standard annealing metric). For each sweep budget with per-run success probability `p` and median run time `t`:

```
runs_needed = ln(1 - target) / ln(1 - p)     (1 if p = 1, inf if p = 0)
tts         = runs_needed * t
```

Returns one row per `(n, k, platform)` — the sweep budget with the lowest TTS. Exact platforms have `p = 1`, so their TTS is one run.

Caveats: `p` is estimated from a finite number of seeds (20 by default in the script), so `p = 1` means "every run we tried hit", and the cheapest budget that reached 20/20 is picked, which slightly favours small budgets. The estimate is per run, not measured inside a chain. This is the per-budget TTS; the in-chain version (time until a chain first reaches the best energy) is still a TODO, because the native modules don't report when each chain first reaches it yet (`evaluation.tex`, "Three ways to measure speed").

## `speedup_tts(df, baseline="scalar", target=0.99)`

One row per `(n, k, platform)`: `time` (the TTS from `time_to_answer()`), the baseline's `time` at the same size, and `speedup = baseline_time / time` (> 1 means faster than the baseline; NaN if the baseline has no runs at that size).

`scripts/bench_synthetic.py` and `scripts/real_data_check.py` use `baseline="exact_cpu"`: how much faster SA reaches the answer than exhaustive search. Time to answer is the fair comparison between an exact solver and a heuristic. Once `avx2`/`cuda` exist, `speedup_table(df, baseline="scalar")` at a fixed sweep budget is the sequential-vs-parallel table (T2): same algorithm, same number of proposals, different hardware.

## Example output

From `uv run python scripts/bench_synthetic.py --fresh`, which uses `exact_cpu` as the speedup baseline (random instances, `k = n // 3`, 20 seeds per budget, `T0 = 2`, `T1 = 0.01`; timings from one machine, will vary):

| n | k | C(n,k) | `exact_cpu` | `scalar` budget for 20/20 hits | `scalar` TTS (99%) | `scalar` speedup over `exact_cpu` (TTS) |
|---|---|---|---|---|---|---|
| 10 | 3 | 120 | 1 µs | 10 sweeps | 20 µs | 0.05 (exact 20× faster) |
| 20 | 6 | 38,760 | 0.52 ms | 100 sweeps | 65 µs | ≈ 8 |
| 30 | 10 | 30,045,015 | 0.93–0.96 s | 300 sweeps (90% at 100) | ≈ 0.2 ms | ≈ 5,000 |

Exact search grows with C(n, k); SA's cost grows roughly with n × sweeps, so SA overtakes exact search between n = 10 and n = 20. This is an algorithmic crossover, not a parallel speedup — both run on one CPU core.
