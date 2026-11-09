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

# The same core with the drift check compiled in (paper test T9): every 10,000 accepted
# swaps it rebuilds g and E from scratch and aborts if the running values have drifted.
add_sa_scalar(sa_scalar_checked)
target_compile_definitions(sa_scalar_checked PUBLIC SA_CHECK_G)

# sa_avx2: the same simulated annealing with the hot loop in AVX2 assembly
# (src/asm/sa_avx2.asm, NASM, Windows x64 only) and a C driver (src/sa_avx2.c). Same
# decisions as sa_scalar for the same seed (paper test T7, checked by
# tests/sa_avx2_test.c). Needs NASM on PATH or in C:/Program Files/NASM (CMake looks
# there), or set CMAKE_ASM_NASM_COMPILER. Skipped on any other OS or CPU.
option(AD_BUILD_AVX2 "Build the AVX2 (NASM) simulated-annealing backend (Windows x64)" ON)
if(AD_BUILD_AVX2 AND NOT (WIN32 AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64|x64)$"))
  message(STATUS "AVX2 backend skipped: it is Windows x64 only")
  set(AD_BUILD_AVX2 OFF)
endif()
if(AD_BUILD_AVX2)
  include(CheckLanguage)
  check_language(ASM_NASM)
  if(CMAKE_ASM_NASM_COMPILER)
    # Say the object format outright: CMake guesses it from the C compiler's pointer
    # size, and falls back to win32 when that probe fails (seen with MinGW).
    set(CMAKE_ASM_NASM_OBJECT_FORMAT win64)
    enable_language(ASM_NASM)
  else()
    message(WARNING
      "NASM not found, so the AVX2 backend (sa_avx2, sa_avx2_test, sa_bench, module_2_avx) "
      "is not built. Install NASM (https://www.nasm.us) or configure with -DAD_BUILD_AVX2=OFF.")
    set(AD_BUILD_AVX2 OFF)
  endif()
endif()

if(AD_BUILD_AVX2)
  add_library(sa_avx2 STATIC src/sa_avx2.c src/asm/sa_avx2.asm)
  target_include_directories(sa_avx2 PUBLIC include)
  target_link_libraries(sa_avx2 PUBLIC sa_scalar) # shared types and sa_status_message
  set_target_properties(sa_avx2 PROPERTIES POSITION_INDEPENDENT_CODE ON)
  if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(sa_avx2 PRIVATE $<$<COMPILE_LANGUAGE:C>:-O2 -Wall -Wextra>)
  endif()

  add_executable(sa_avx2_test tests/sa_avx2_test.c)
  target_link_libraries(sa_avx2_test PRIVATE sa_avx2)
  add_test(NAME sa_avx2_test COMMAND sa_avx2_test)

  # sa_bench: scalar vs AVX2 timing tables (bench/sa_bench.c). Build it Release. With
  # GCC/Clang the scalar baseline is -O2 -fno-tree-vectorize (decision D4), and a second
  # copy of sa_scalar.c built -O3 -march=native is timed too, as "auto-vec". (MSVC has no
  # switch to turn its auto-vectorizer off, so use a GCC/Clang build for the paper.)
  add_executable(sa_bench bench/sa_bench.c)
  target_link_libraries(sa_bench PRIVATE sa_avx2)
  if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(sa_bench PRIVATE -O2 -fno-tree-vectorize)
    add_library(sa_scalar_autovec OBJECT src/sa_scalar.c)
    target_include_directories(sa_scalar_autovec PRIVATE include)
    target_compile_options(sa_scalar_autovec PRIVATE -O3 -march=native)
    if(MINGW)
      # MinGW cannot always 32-byte align the stack; let the assembler use unaligned
      # moves for any ymm spill the vectorizer emits.
      target_compile_options(sa_scalar_autovec PRIVATE -Wa,-muse-unaligned-vector-move)
    endif()
    target_compile_definitions(sa_scalar_autovec PRIVATE
      sa_scalar_solve=sa_scalar_autovec_solve
      sa_philox4x32_10=sa_scalar_autovec_philox
      sa_status_message=sa_scalar_autovec_status_message)
    target_sources(sa_bench PRIVATE $<TARGET_OBJECTS:sa_scalar_autovec>)
    target_compile_definitions(sa_bench PRIVATE SA_BENCH_AUTOVEC)
  endif()
endif()

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
  # cuda lands as its own pybind11_add_module() call here later.
  pybind11_add_module(module_2 src/module_2.cpp)
  target_link_libraries(module_2 PRIVATE sa_scalar)
  set(ad_python_modules module_1 module_2)

  # module_2_avx: the AVX2 SA backend, platform "avx2" (src/module_2_avx.cpp adapting
  # src/sa_avx2.c + src/asm/sa_avx2.asm).
  if(AD_BUILD_AVX2)
    pybind11_add_module(module_2_avx src/module_2_avx.cpp)
    target_link_libraries(module_2_avx PRIVATE sa_avx2)
    list(APPEND ad_python_modules module_2_avx)
  endif()

  # A MinGW-built extension must not depend on MinGW runtime DLLs that the Python
  # process cannot find, so link the compiler runtime in statically.
  if(MINGW)
    foreach(mod ${ad_python_modules})
      target_link_options(${mod} PRIVATE -static -static-libgcc -static-libstdc++)
    endforeach()
  endif()
endif()
