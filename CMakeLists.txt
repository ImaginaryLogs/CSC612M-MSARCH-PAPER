cmake_minimum_required(VERSION 3.15)
project(ad_combo_native LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(AD_BUILD_PYTHON "Build the pybind11 extension modules (needs Python and pybind11)" ON)

# sa_scalar: scalar simulated annealing, plain C (src/sa_scalar.c), plus its standalone
# test (tests/sa_scalar_test.c, no Python needed: run `ctest`).
# -O2 -fno-tree-vectorize keeps the compiler from quietly vectorizing the scalar baseline
# that the AVX2 version is timed against (paper decision D4).
enable_testing()
function(add_sa_scalar target)
  add_library(${target} STATIC src/sa_scalar.c)
  target_include_directories(${target} PUBLIC include)
  set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE -O2 -fno-tree-vectorize -Wall -Wextra)
  endif()
  if(NOT WIN32)
    target_link_libraries(${target} PUBLIC m)
  endif()

  add_executable(${target}_test tests/sa_scalar_test.c)
  target_link_libraries(${target}_test PRIVATE ${target})
  add_test(NAME ${target}_test COMMAND ${target}_test)
endfunction()

add_sa_scalar(sa_scalar)

# The same core with the drift check compiled in (paper test T9): every 1,000 accepted
# swaps it rebuilds g and E from scratch and aborts if the running values have drifted.
add_sa_scalar(sa_scalar_checked)
target_compile_definitions(sa_scalar_checked PUBLIC SA_CHECK_G)

if(AD_BUILD_PYTHON)
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
      "(pybind11 is a dev dependency in pyproject.toml), or configure with "
      "-DAD_BUILD_PYTHON=OFF to build only the C core and its test")
  endif()

  find_package(pybind11 CONFIG REQUIRED)

  # module_1: the exact_cpu brute-force oracle (src/module_1.cpp against module.hpp).
  pybind11_add_module(module_1 src/module_1.cpp)
  target_include_directories(module_1 PRIVATE include)

  # module_2: the scalar SA backend (src/module_2.cpp adapting src/sa_scalar.c).
  # avx2/cuda land as their own pybind11_add_module() calls here later.
  pybind11_add_module(module_2 src/module_2.cpp)
  target_link_libraries(module_2 PRIVATE sa_scalar)

  # A MinGW-built extension must not depend on MinGW runtime DLLs that the Python
  # process cannot find, so link the compiler runtime in statically.
  if(MINGW)
    foreach(mod module_1 module_2)
      target_link_options(${mod} PRIVATE -static -static-libgcc -static-libstdc++)
    endforeach()
  endif()
endif()
