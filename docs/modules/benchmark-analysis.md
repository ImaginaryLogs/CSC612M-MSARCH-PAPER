# `python/benchmark_analysis.py`

Turns a pile of `RunResultsMessage` records into the sequential-vs-parallel comparison the course requires (requirements PDF, Milestone 3 deliverable #3) and the figures/tables listed in `evaluation.tex` (F2-F5, T1-T2).

Every function takes a DataFrame from `load_runs()`, so it works the same on `data/runs.jsonl` (pipeline runs) and `data/bench_runs.jsonl` (`scripts/bench_synthetic.py`). Tests: `tests/test_benchmark_analysis.py`.

## `append_run()` / `load_runs()`

One JSON line per `RunResultsMessage`; `load_runs()` returns them as a DataFrame (empty if the file doesn't exist).

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

Caveats: `p` is estimated from a finite number of seeds (20 by default in the script), so `p = 1` means "every run we tried hit", and the cheapest budget that reached 20/20 is picked, which slightly favours small budgets. The estimate is per run, not measured inside a chain.

## `speedup_table(df, baseline="scalar", metric="t_kernel" | "tts")`

One row per `(n, k, platform)`: `time`, the baseline's `time` at the same size, and `speedup = baseline_time / time` (> 1 means faster than the baseline; NaN if the baseline has no runs at that size).

- `metric="t_kernel"` pools every run of a platform into one median, so filter `df` to the configuration you want to compare first (e.g. one sweep budget).
- `metric="tts"` compares time to answer, which is the fair comparison between an exact solver and a heuristic.

`scripts/bench_synthetic.py` uses `baseline="exact_cpu"`: how much faster SA reaches the answer than exhaustive search. Once `avx2`/`cuda` exist, `speedup_table(df, baseline="scalar")` at a fixed sweep budget is the sequential-vs-parallel table (T2): same algorithm, same number of proposals, different hardware.

## Example output

From `uv run python scripts/bench_synthetic.py --fresh`, which uses `exact_cpu` as the speedup baseline (random instances, `k = n // 3`, 20 seeds per budget, `T0 = 2`, `T1 = 0.01`; timings from one machine, will vary):

| n | k | C(n,k) | `exact_cpu` | `scalar` budget for 20/20 hits | `scalar` TTS (99%) | `scalar` speedup over `exact_cpu` (TTS) |
|---|---|---|---|---|---|---|
| 10 | 3 | 120 | 1 µs | 10 sweeps | 20 µs | 0.05 (exact 20× faster) |
| 20 | 6 | 38,760 | 0.52 ms | 100 sweeps | 65 µs | ≈ 8 |
| 30 | 10 | 30,045,015 | 0.93–0.96 s | 300 sweeps (90% at 100) | ≈ 0.2 ms | ≈ 5,000 |

Exact search grows with C(n, k); SA's cost grows roughly with n × sweeps, so SA overtakes exact search between n = 10 and n = 20. This is an algorithmic crossover, not a parallel speedup — both run on one CPU core.
