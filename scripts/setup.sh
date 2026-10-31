#!/usr/bin/env bash
# Sets up the Python environment with uv. The C++/CUDA build (CMake +
# pybind11, via scikit-build-core) is a separate script added once
# include/module.hpp and src/module_1.cpp exist -- see pyproject.toml's
# top comment.
set -euo pipefail
cd "$(dirname "$0")/.."

uv sync
echo "done -- run with: uv run python/main.py"
