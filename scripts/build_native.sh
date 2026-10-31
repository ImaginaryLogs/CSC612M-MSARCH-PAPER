#!/usr/bin/env bash
# Builds the native pybind11 extension(s) under src/ (currently just module_1, the
# exact_cpu brute-force oracle) and drops the compiled module next to python/main.py
# so `import module_1` works for `uv run python/main.py`.
#
# Not yet wired into `uv sync` -- see pyproject.toml's top comment for the
# scikit-build-core plan that would make that automatic.
set -euo pipefail
cd "$(dirname "$0")/.."

uv sync --group dev
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$(pwd)/.venv/bin/python3"
cmake --build build -j

native_module="$(find build -maxdepth 2 -name 'module_1*.so' -print -quit)"
if [[ -z "$native_module" ]]; then
  echo "build succeeded but module_1*.so was not found under build/" >&2
  exit 1
fi
cp "$native_module" python/
echo "done -- $(basename "$native_module") built into python/"
