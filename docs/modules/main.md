# `python/main.py`

CLI entry point. Parses arguments, builds an `OrchestratorConfig`, runs it, prints each platform's answer and then the speedup table for that instance (`benchmark_analysis.speedup_table()`). All the actual logic is in `orchestrator.py` and below.

```sh
uv run python/main.py --platforms exact_cpu scalar avx2 --k 2 --t0 2.0 --t1 0.02
```

That solves the real Alzheimer's instance, built from downloaded public data. Today's curated candidate list gives only 3 eligible drugs, too small to time the solvers. `--synthetic N` solves a random N-drug instance instead. It uses the same value ranges as the real one, needs no download, and is seeded by `--seed`:

```sh
uv run python/main.py --synthetic 512 --platforms scalar avx2 --k 5 --t0 2.0 --t1 0.02
```

On Windows, `run.bat` in the repository root wraps all of this in a menu, and `run.bat <options>` passes the options to `main.py` unchanged.

Always pass `--platforms`: the default list includes `cuda` and `qaoa`, which have no backend yet. `scalar` and `avx2` need `--t0`/`--t1`. `exact_cpu` and `--brute-force-check` try every k-combination, so keep them to small N.
