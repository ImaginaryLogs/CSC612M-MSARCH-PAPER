#!/usr/bin/env bash
# Builds the native pybind11 extension(s) under src/ (module_1, the exact_cpu
# brute-force oracle; module_2, the scalar SA; module_2_avx, the AVX2 SA) and drops the
# compiled modules next to python/main.py so `import module_1` etc. work for
# `uv run python/main.py`.
#
# module_2_avx is Windows x64 only and needs NASM (https://www.nasm.us). Elsewhere, or
# without NASM, CMake skips it and platform "avx2" reports that it is not built.
#
# Works on Linux/macOS (.so, .venv/bin) and on Windows from Git Bash (.pyd,
# .venv/Scripts; any CMake generator, including Visual Studio's multi-config one).
#
# Not yet wired into `uv sync` -- see pyproject.toml's top comment for the
# scikit-build-core plan that would make that automatic.
set -euo pipefail
cd "$(dirname "$0")/.."

uv sync --group dev
python_exe="$(pwd)/.venv/bin/python3"
if [[ ! -x "$python_exe" ]]; then
  python_exe="$(pwd)/.venv/Scripts/python.exe" # Windows venv layout
fi
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$python_exe"
cmake --build build -j --config Release

for name in module_1 module_2 module_2_avx; do
  # "${name}.*" (with the dot) so that module_2 does not also match module_2_avx.
  native_module="$(find build -maxdepth 2 \( -name "${name}.*.so" -o -name "${name}.so" \
    -o -name "${name}.*.pyd" -o -name "${name}.pyd" \) -print -quit)"
  if [[ -z "$native_module" ]]; then
    if [[ "$name" == module_2_avx ]]; then
      echo "note: module_2_avx was not built (it needs Windows x64 and NASM);" \
        "platform 'avx2' is unavailable" >&2
      continue
    fi
    echo "build succeeded but ${name} (.so/.pyd) was not found under build/" >&2
    exit 1
  fi
  cp "$native_module" python/
  echo "done -- $(basename "$native_module") built into python/"
done
