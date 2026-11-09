# Running and testing

How to build the project, run the pipeline, and check the solvers against the exact answer: what goes in, the steps, and what comes out (on the terminal and on disk).

Platforms that work today: `exact_cpu` (C++ brute force, `module_1`) and `scalar` (scalar simulated annealing in C, `module_2`). `avx2`, `cuda`, `exact_cuda` and `qaoa` are not built yet; asking for them raises `NotImplementedError` (or `ModuleNotFoundError` for `qaoa`).

## 1. Requirements

| Tool | Why |
|---|---|
| [uv](https://docs.astral.sh/uv/) | Python 3.11+ environment and dependencies (`numpy`, `pandas`, `requests`, `networkx`; dev: `pytest`, `pybind11`, `jupyter`) |
| CMake ≥ 3.15 and a C99 / C++17 compiler (gcc or clang) | builds the native solvers |
| Internet access, first run only | the pipeline downloads its input data once and caches it (see §3) |

## 2. Build

```sh
uv sync --group dev          # Python deps, including pybind11 and pytest
bash scripts/build_native.sh # compiles the native modules, copies them into python/
```

`build_native.sh` configures CMake into `build/`, compiles everything, and copies the two extensions next to `python/main.py`. Expected tail of its output:

```
done -- module_1.cpython-312-x86_64-linux-gnu.so built into python/
done -- module_2.cpython-312-x86_64-linux-gnu.so built into python/
```

| File produced | What it is |
|---|---|
| `python/module_1.*.so` | `exact_cpu` solver (pybind11 extension) |
| `python/module_2.*.so` | `scalar` SA solver (pybind11 extension) |
| `build/` | CMake build tree, including the C test executables `sa_scalar_test` and `sa_scalar_checked_test` |

Re-run `build_native.sh` after editing anything under `src/` or `include/`. If it stops before the `done` lines, the copied `.so` files are stale and the Python tests for that module will be **skipped**, not failed (see §5).

## 3. Inputs

### 3.1 Command-line arguments (`python/main.py`)

| Flag | Type / default | Meaning |
|---|---|---|
| `--platforms` | one or more of `exact_cpu exact_cuda scalar avx2 cuda qaoa`; default `scalar avx2 cuda qaoa` | Which solvers to run, in order, on the same instance. **Always pass this** — the default includes unbuilt platforms. |
| `--k` | int, default `3` | How many drugs to choose. Must be ≤ the number of eligible drugs (currently 3). |
| `--lam` | float, default `1.0` | λ, the weight on the pairwise-overlap penalty. |
| `--seed` | int, default `1234` | Seeds both the z-score sampling that builds the instance and the SA random numbers. Same seed ⇒ same instance and same SA run. |
| `--chains` | int, default `4` | Independent SA runs; the best is returned. Ignored by `exact_cpu`. |
| `--sweeps` | int, default `1000` | SA length; one sweep = `n` proposed swaps. Ignored by `exact_cpu`. |
| `--t0`, `--t1` | float, no default | Start / end SA temperature in energy units, `t0 ≥ t1 > 0`. **Required when `scalar` is in `--platforms`**; ignored by `exact_cpu`. A reasonable start: `--t0 1.0 --t1 0.01` (energies are of order 1 because `w` and `P` are scaled to [0, 1]). |
| `--brute-force-check` | flag | Also solve the instance exactly (Python brute force) and record whether each solver found the optimum. Cost grows as C(n, k). |

### 3.2 Data the pipeline reads

On every run the pipeline rebuilds the problem from these sources. Each one is downloaded only if its cache file is missing, so after the first run it works offline.

| Cache file (`data/raw/`, git-ignored) | Source | Contents |
|---|---|---|
| `dgidb_<drug>.json` (one per candidate drug) | DGIdb GraphQL | The drug's gene interactions (its targets). |
| `ad_module.json` | NCBI E-utilities | The 16 curated AD genes (`CANONICAL_AD_GENES` in `data_collectors.py`) mapped to Entrez IDs. |
| `interactome_string.tsv` | STRING-db | Protein–protein network around the AD genes and drug targets (human, confidence ≥ 400, columns `preferredName_A`/`preferredName_B` used). |
| `gene_ids_cache.json` | NCBI E-utilities | Gene symbol → Entrez ID lookups shared by all collectors. |

The candidate drugs are hard-coded in `data_collectors.collect_candidate_list()`: donepezil, rivastigmine, galantamine, memantine, sildenafil. A drug only enters the problem if it has targets in the network's largest connected component and a finite negative closeness z-score. With the current cache, **3 drugs are eligible**, so the real instance is `n = 3`.

To refetch a source, delete its cache file. To use different drugs, edit `collect_candidate_list()`.

### 3.3 The problem every solver receives (`Instance`)

Built by `data_collectors.build_instance()`; this is also what you construct by hand for synthetic tests (§5.3).

| Field | Content |
|---|---|
| `id` | e.g. `ad_combo_n3_k2_seed1234` |
| `n`, `k` | number of candidate drugs, team size; need `0 ≤ k ≤ n` |
| `lam` | λ |
| `w` | `float32[n]`, reward per drug (closeness to the AD module), scaled so the largest is 1 |
| `P` | `float32[n, n]`, overlap penalty per pair, symmetric, zero diagonal, scaled to [0, 1] |
| `drug_ids` | DrugBank IDs of the eligible drugs, index-aligned with `w` and `P` |

Objective (lower is better): `E = -Σ w_i + λ · Σ_{i<j in team} P_ij`.

## 4. Running the pipeline

```sh
uv run python python/main.py --platforms exact_cpu scalar --k 2 --t0 1.0 --t1 0.01 --brute-force-check
```

Stages: collect data → build `Instance` → run each platform → grade each result with `judge()` (and brute force if asked) → append to `data/runs.jsonl`.

### 4.1 Terminal output

One line per platform:

```
exact_cpu  best=[0, 1] e_solver=-0.7906 e_judge=-0.7906 valid=True
scalar     best=[0, 1] e_solver=-0.7906 e_judge=-0.7906 valid=True
```

| Field | Meaning |
|---|---|
| `best` | indices (into `w`/`P`/`drug_ids`) of the chosen drugs, sorted |
| `e_solver` | energy the solver reported for its team |
| `e_judge` | energy recomputed independently by `validation.judge()` |
| `valid` | `True` if the two agree within 1e-3 — the solver's bookkeeping is trusted only if this holds |

What a good run looks like: every line `valid=True`, and `scalar` gives the same `e_solver` as `exact_cpu`.

Common errors (Python traceback ending in):

| Message | Cause / fix |
|---|---|
| `ValueError: platform 'scalar' needs ModuleConfig.T0 and ModuleConfig.T1 ...` | add `--t0` and `--t1` |
| `ValueError: scalar SA: bad argument` | `t0 < t1`, `t1 ≤ 0`, `chains < 1` or `sweeps < 1` |
| `ValueError: only 3 eligible drug(s) but k=4 ...` | lower `--k` |
| `NotImplementedError: platform 'avx2' has no native module yet` | drop that platform from `--platforms` |
| `NotImplementedError: module_2 ... is not built` | run `scripts/build_native.sh` |
| `requests.exceptions.ConnectionError` | a cache file in `data/raw/` is missing and there's no internet |

### 4.2 Files written by a run

| File | Tracked in git? | What changes |
|---|---|---|
| `data/runs.jsonl` | no (git-ignored) | **One JSON line appended per platform per run.** Never overwritten — delete it to start fresh. |
| `data/frozen/manifest.json` | **yes** | Rewritten every run: one entry per data source with `path`, `url`, `sha256`, `fetched_at`, `license_note`. Only `fetched_at` changes when the cache is unchanged; revert with `git checkout -- data/frozen/manifest.json` if you don't want that diff. A changed `sha256` means the input data changed. |
| `data/raw/*` | no | Created only when missing (§3.2). |

A record in `data/runs.jsonl` (pretty-printed here; on disk it is one line):

```json
{
  "instance_id": "ad_combo_n3_k2_seed1234",
  "platform": "scalar",
  "seed": 1234, "chains": 4, "sweeps": 1000,
  "best": [0, 1],
  "e_solver": -0.7906028032302856,
  "e_judge": -0.7906028032302856,
  "valid": true,
  "ties": 0,
  "t_kernel": 0.000297278, "t_h2d": 0.0, "t_d2h": 0.0,
  "proposals": 12000, "accepts": 3369,
  "extra": {"t_total": 0.000492552004288882,
            "optimal": [0, 1], "e_optimal": -0.7906028032302856, "is_optimal": true}
}
```

| Field | Meaning |
|---|---|
| `ties` | `exact_cpu` only: how many teams share the optimal energy. Always 0 for SA. |
| `t_kernel` | seconds inside the native solver (the number to compare across platforms) |
| `t_h2d`, `t_d2h` | GPU transfer time; 0 on CPU platforms |
| `proposals` | `scalar`: swaps proposed = `chains × sweeps × n` (0 if `k = 0` or `k = n`) |
| `accepts` | `scalar`: swaps accepted. The acceptance rate `accepts / proposals` shows whether the temperatures are sensible: near 1 means `t0`/`t1` are too hot (random walk), near 0 too cold (stuck). |
| `extra.t_total` | seconds for the whole `run_module()` call, including Python overhead |
| `extra.trace` | only when `ModuleConfig.trace_steps > 0` (not exposed on the CLI): `chains × trace_steps` grid of 1 = accepted / 0 = rejected, for comparing decisions step by step with future AVX2/CUDA versions |
| `extra.optimal`, `extra.e_optimal`, `extra.is_optimal` | only with `--brute-force-check`: the exact best team, its energy, and whether this solver's team is that team |

`is_optimal` compares *teams*, not energies. On an instance where several teams tie for the best energy, a solver can return an optimal-energy team and still get `is_optimal: false`; compare `e_solver` with `e_optimal` in that case.

To load all runs for analysis:

```python
import sys; sys.path.insert(0, "python")
import benchmark_analysis as ba
df = ba.load_runs()   # pandas DataFrame, one row per record
```

## 5. Testing

### 5.1 Python tests

```sh
uv run pytest -q tests
```

Expected: `28 passed` (plus 10 more once `scripts/real_data_check.py` has cached the real instances, §7). If you see `skipped`, a native module isn't built or copied — re-run `scripts/build_native.sh` and check it printed both `done` lines.

| File | What it checks |
|---|---|
| `tests/test_benchmark_analysis.py` | the analysis functions in `benchmark_analysis.py` on hand-made records (no native modules needed) |
| `tests/test_module_1.py` | `exact_cpu` matches the independent Python brute force (`Validator.brute_force()`) |
| `tests/test_scalar_sa.py` | `scalar` reaches the exact optimum on random instances (n ≤ 12); proposal/accept counters; same seed ⇒ identical run and trace, different seed ⇒ different trace; trace sum equals `accepts`; missing or invalid temperatures and `k > n` raise `ValueError`; `k = 0` and `k = n` work |

### 5.2 C tests for the SA core (no Python)

```sh
(cd build && ctest --output-on-failure)
```

Expected: `100% tests passed out of 2`. The two tests are the same program, `tests/sa_scalar_test.c`, built twice:

- `sa_scalar_test` — Philox random numbers match published values, a hand-worked toy example, agreement with brute force, determinism, temperature effect on acceptance, edge cases.
- `sa_scalar_checked_test` — the same, with `-DSA_CHECK_G`: every 1,000 accepted swaps the running `g`/`E` values are compared with a full recomputation, and the program aborts if they drifted (paper test T9).

Run one directly to see its output (`./build/sa_scalar_test`); it ends with `all checks passed`.

### 5.3 Comparing `scalar` with the exact answer

There are three exact references:

1. **`exact_cpu`** (`module_1`, C++) — run it alongside `scalar` with `--platforms exact_cpu scalar`.
2. **`--brute-force-check`** — a Python brute force (`Validator.brute_force()`, separate code from `exact_cpu`) run on the same instance; results go in `extra.optimal` / `extra.e_optimal` / `extra.is_optimal`.
3. **`judge()`** — not an optimum, but recomputes the energy of every answer, so a solver never grades itself (`e_judge`, `valid`).

Because the real instance has only 3 drugs, it can't show whether SA really searches well. Use a larger synthetic instance:

```python
# run from the repository root: uv run python compare.py
import sys
sys.path.insert(0, "python")
import numpy as np
from data_collectors import Instance
from module_interface import ModuleConfig, run_module
from validation import Validator

rng = np.random.default_rng(0)
n, k = 20, 5
w = (0.1 + 0.9 * rng.random(n)).astype(np.float32)          # rewards in (0.1, 1]
P = rng.random((n, n)).astype(np.float32)
P = (P + P.T) / 2                                            # must be symmetric
np.fill_diagonal(P, 0)                                       # with a zero diagonal
inst = Instance(id="synthetic-n20-k5", n=n, k=k, lam=1.0, w=w, P=P)

sa = run_module(inst, ModuleConfig("scalar", seed=1, chains=4, sweeps=500, T0=2.0, T1=0.01))
exact = run_module(inst, ModuleConfig("exact_cpu"))
oracle = Validator().brute_force(inst)
print("scalar    ", sa.best, round(sa.e_solver, 6))
print("exact_cpu ", exact.best, round(exact.e_solver, 6))
print("python    ", oracle.best, round(oracle.e_solver, 6))
```

Expected output:

```
scalar     [5, 7, 9, 10, 12] -0.72437
exact_cpu  [5, 7, 9, 10, 12] -0.72437
python     [5, 7, 9, 10, 12] -0.72437
```

This writes no files (`run_module()` doesn't touch `data/runs.jsonl`; only the orchestrator does). Brute force grows as C(n, k): it is instant at n = 20, k = 5, but impractical much beyond n ≈ 30. For bigger instances, check SA against itself instead — the best energy from many seeds with many sweeps — and watch the gap shrink as `sweeps` grows.

If `scalar` misses the optimum, try in this order: more `sweeps`, more `chains`, then adjust temperatures so the acceptance rate starts high (≈ 0.5–0.9) and ends near 0.

## 6. Benchmarking on synthetic instances (n = 10, 20, 30)

```sh
uv run python scripts/bench_synthetic.py --fresh
```

### 6.1 Inputs

No downloaded data: the script generates random instances shaped like the real one (`w` in (0.1, 1], symmetric `P` in [0, 1] with zero diagonal, λ = 1).

| Flag | Default | Meaning |
|---|---|---|
| `--sizes` | `10 20 30` | values of n |
| `--k` | `n // 3` | team size for every n (k = 3, 6, 10 by default) |
| `--lam` | `1.0` | λ |
| `--instance-seed` | `0` | seed for generating `w` and `P` |
| `--seeds` | `20` | SA runs per sweep budget (more seeds ⇒ better success-rate estimates) |
| `--sweeps` | `10 30 100 300 1000 3000` | SA sweep budgets to try |
| `--chains` | `1` | chains per SA run; 1 makes the success rate a per-chain number |
| `--t0`, `--t1` | `2.0`, `0.01` | SA temperatures |
| `--exact-repeats` | `3` | `exact_cpu` runs per size, for a median time |
| `--out` | `data/bench_runs.jsonl` | where records are appended |
| `--fresh` | off | delete `--out` first (otherwise new runs are added to old ones and pooled in the tables) |
| `--tables-dir` | none | also save each table as CSV in this directory |

Exact search grows as C(n, k): n = 30, k = 10 takes about 1 s per `exact_cpu` run; n = 40, k = 13 would take hours. For sizes where C(n, k) ≤ 200,000 the script also checks `exact_cpu` against the Python brute force and stops with `RuntimeError` if they disagree.

### 6.2 Steps

For each n: build the instance → run `exact_cpu` (the reference optimum) → run `scalar` for every sweep budget × seed → grade every result with `judge()` → append to `--out` with `extra.n`, `extra.k`, `extra.e_optimal`. Then load the file and print the tables. The default run takes a few seconds.

### 6.3 Terminal output

One line per size, then four tables (abridged):

```
n= 10 k= 3: C(n,k)=120 teams, E*=-1.989210, exact_cpu 0.001 ms, 120 SA runs
n= 20 k= 6: C(n,k)=38,760 teams, E*=0.833486, exact_cpu 0.523 ms, 120 SA runs
n= 30 k=10: C(n,k)=30,045,015 teams, E*=11.297543, exact_cpu 959.275 ms, 120 SA runs

== SA quality per sweep budget (success = fraction of runs reaching the exact optimum) ==
 n  k  platform  chains  sweeps  runs  success  mean_gap  max_gap  t_kernel_median  accept_rate  ns_per_proposal  ref_is_proven
30 10    scalar       1      10    20      0.1    0.5863    1.224        4.059e-05       0.2813            135.3           True
30 10    scalar       1     100    20      0.9   0.04194   0.4516        0.0001037       0.2201            34.58           True
30 10    scalar       1     300    20        1         0        0         0.000246       0.2186            27.33           True
...
== Time to answer with 99% probability (best sweep budget per platform) ==
== Speedup over exact_cpu, by time to answer (> 1: faster than exact search) ==
== Speedup over exact_cpu, one run each (scalar at 3000 sweeps) ==
```

| Table | Read it as |
|---|---|
| quality | how many sweeps SA needs at each n: `success` should rise to 1 as `sweeps` grows; `mean_gap` is how far off the misses are |
| time to answer | time to reach the optimum with 99% probability, at the cheapest sweep budget that does it |
| speedup (tts) | how much faster `scalar` reaches the optimum than `exact_cpu` (the baseline, always 1): about 0.05 at n = 10 (exact wins on tiny problems), about 8 at n = 20, about 5,000 at n = 30 |
| speedup (t_kernel) | the same, using one run at the longest sweep budget instead of time to answer: raw run time, whether or not SA hit |

Column meanings and caveats are in [`modules/benchmark-analysis.md`](modules/benchmark-analysis.md). `ns_per_proposal` is high at small budgets because fixed setup cost (allocation, temperature table) is spread over few proposals; it levels off around 24 ns at large budgets. `ref_is_proven = True` means the optimum came from `exact_cpu`, not just the best SA result seen.

### 6.4 Files written

| File | Tracked? | Content |
|---|---|---|
| `data/bench_runs.jsonl` | no (git-ignored) | one JSON line per run, same format as `data/runs.jsonl` (§4.2) plus `extra.n`, `extra.k`, `extra.e_optimal`. 369 lines for the defaults (3 sizes × (3 exact + 6 budgets × 20 seeds)). |
| `<tables-dir>/quality.csv`, `time_to_answer.csv`, `speedup_tts.csv`, `speedup_kernel.csv` | — | only with `--tables-dir` |

The script does not touch `data/runs.jsonl` or `data/frozen/manifest.json`.

To analyse the records yourself:

```python
import sys; sys.path.insert(0, "python")
import benchmark_analysis as ba
df = ba.load_runs(ba.REPO_ROOT / "data" / "bench_runs.jsonl")
ba.quality_table(df); ba.time_to_answer(df); ba.speedup_table(df, baseline="exact_cpu", metric="tts")
```

## 7. Real biological data with more drugs

`scripts/real_data_check.py` builds a real instance from 38 AD-relevant drugs (n = 16–21 eligible), checks `scalar` against `exact_cpu` and the Python brute force, and compares the top combinations with known combinations from the literature and ClinicalTrials.gov. `tests/test_real_data.py` re-checks SA against exact on the cached instance (skipped until the script has been run once). Inputs, outputs, results and caveats: [`real-data-check.md`](real-data-check.md).
