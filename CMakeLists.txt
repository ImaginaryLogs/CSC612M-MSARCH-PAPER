cmake_minimum_required(VERSION 3.15)
project(ad_combo_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python COMPONENTS Interpreter Development REQUIRED)

# pybind11 ships its own CMake config inside the venv -- ask the interpreter for it
# rather than requiring a system install.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE pybind11_DIR_result
)
if(NOT pybind11_DIR_result EQUAL 0)
  message(FATAL_ERROR
    "could not locate pybind11 via ${Python_EXECUTABLE} -- run `uv sync` first "
    "(pybind11 is a dev dependency in pyproject.toml)")
endif()

find_package(pybind11 CONFIG REQUIRED)

# module_1: the exact_cpu brute-force oracle (src/module_1.cpp against module.hpp).
# scalar/avx2/cuda land as their own pybind11_add_module() calls here later.
pybind11_add_module(module_1 src/module_1.cpp)
target_include_directories(module_1 PRIVATE include)
