# `python/validation.py`

Checks solver output, two different ways:

- **`judge()`** — independent float64 energy recompute. Grades any single `RunResultsMessage` against its own claimed energy. Used on every run, always.
- **`Validator(brute_force=True)`** — the simple, obviously-correct exact solver (`itertools.combinations` — no C++, nothing clever), used to check a heuristic (SA/QAOA) actually found the true optimum, or to double-check the native `exact_cpu`/`exact_cuda` kernels once they exist.
- **`literature_crosscheck()`** — exploratory only: is a chosen combination's drugs already in the AD clinical-trial list? Not proof a combination works — see `limitations.tex`.

## `judge()`

`E(selection) = -sum(w_i) + lambda * sum_{i<j in selection} P_ij`.

## `Validator`

Grades `RunResultsMessage`s. Pass `brute_force=True` to also compute the true optimum for comparison — off by default since brute force gets slow past a few hundred combinations (see `benchmark_analysis.py` for when the native `exact_cpu`/`exact_cuda` kernels should be used instead).

### `Validator.grade()`

Fills in `e_judge`/`valid` on a copy of `result`. A module never grades its own answer — this is the one place that does.

### `Validator.brute_force()`

The exact solver: try every k-combination, keep the best. `O(C(n,k) * k^2)` in plain Python — fine up to a few hundred combinations, which is what this is for. Raises if `n < k` (infeasible instance) rather than silently returning no result.

## `literature_crosscheck()`

Flags which selected drugs are already known from AD clinical trials. Exploratory only — absence isn't evidence against a combination, presence isn't evidence it works (`limitations.tex`).
