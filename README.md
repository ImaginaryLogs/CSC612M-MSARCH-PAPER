# AD Drug-Combination Solver — CSC612M Integrating Project

[![Course](https://img.shields.io/badge/course-CSC612M-blue)](docs/paper/611M_G6_ProjectProposal.pdf)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)](include/module.hpp)
[![AVX2](https://img.shields.io/badge/AVX2%20(NASM)-planned-lightgrey)](docs/TODO.md)
[![CUDA](https://img.shields.io/badge/CUDA-planned-76B900?logo=nvidia&logoColor=white)](docs/TODO.md)
[![CUDA--Q](https://img.shields.io/badge/CUDA--Q-QAOA%20planned-76B900?logo=nvidia&logoColor=white)](docs/TODO.md)
[![Build](https://img.shields.io/badge/build-CMake%20%2B%20uv-informational)](CMakeLists.txt)
[![Status](https://img.shields.io/badge/status-active%20development-yellow)](docs/TODO.md)
[![Last Commit](https://img.shields.io/github/last-commit/ImaginaryLogs/CSC612M-MSARCH-PAPER)](https://github.com/ImaginaryLogs/CSC612M-MSARCH-PAPER/commits)

Selects the best *k*-drug combination against an Alzheimer's-disease gene module, using network-proximity scoring (reward `w`, pairwise overlap penalty `P`) turned into a QUBO/Ising objective. The same objective is solved and benchmarked across four platforms — scalar C++, AVX2 assembly, CUDA, and CUDA-Q (QAOA) — against an exact C++ brute-force oracle that verifies every heuristic answer. Python orchestrates; the solvers are native extensions it calls through one typed boundary (`ModuleConfig` in, `RunResultsMessage` out) so nothing downstream has to know which backend produced a result.

Built for the CSC612M Computer Architecture integrating project (1st Term, SY 2026–2027).

## Start here

- [`docs/overview.md`](docs/overview.md) — what the pipeline does, stage by stage, and where each platform stands today
- [`docs/architecture.md`](docs/architecture.md) — file layout and where each piece of logic lives
- [`docs/modules/`](docs/modules) — one doc per `python/` module
- [`docs/TODO.md`](docs/TODO.md) — the build-out plan and schedule (C++ + AVX2 first, CUDA + CUDA-Q after Milestone 2)

## Running it

```sh
uv sync                     # Python deps
scripts/build_native.sh     # compiles the native solver extension(s)
uv run python/main.py --platforms exact_cpu --k 2 --brute-force-check
```

## Proposal

[Watch the proposal video](https://drive.google.com/file/d/1FQ-T5fcMrj9heX-RddqpMa3wIlaA1rqf/view) · [Read the full proposal](docs/paper/611M_G6_ProjectProposal.pdf)
