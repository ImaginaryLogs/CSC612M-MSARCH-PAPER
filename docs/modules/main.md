# `python/main.py`

CLI entry point. Parses arguments, builds an `OrchestratorConfig`, runs it. Nothing else lives here — all the actual logic is in `orchestrator.py` and below.

```sh
uv run python/main.py --platforms scalar avx2 cuda qaoa --k 3
```
