# AVX2 simulated annealing (`avx2`)

The swap-move simulated annealing of `src/sa_scalar.c`, with its hot loop written in x86-64 AVX2 assembly (NASM). It is **Windows x64 only**. For the same seed it makes **exactly the same accept/reject decisions** as the scalar backend and returns the same best team, energy and counters (paper test T7). Only the speed differs, so the timing comparison measures the hardware and nothing else.

| File | What it is |
|---|---|
| `src/asm/sa_avx2.asm` | The kernels: the whole annealing loop of one chain, 8-lane Philox, the g update, the g/E rebuild, a CPUID check |
| `src/sa_avx2.c` | C driver: argument checks, the padded/aligned memory layout, temperature schedule, starting teams, best of all chains (mirrors `sa_scalar.c` part by part) |
| `include/sa_avx2.h` | `sa_avx2_solve()` (same contract as `sa_scalar_solve()`) and the kernel entry points |
| `src/module_2_avx.cpp` | pybind11 extension `module_2_avx`, platform `"avx2"` (shares `src/sa_binding.hpp` with `module_2`) |
| `tests/sa_avx2_test.c` | Standalone test (`ctest`): kernels bit-for-bit vs C, and the whole solver decision-for-decision vs `sa_scalar` |
| `tests/test_avx2_sa.py` | The same through the Python boundary, including the T7 trace comparison |
| `bench/sa_bench.c` | Timing tables: scalar vs auto-vectorized scalar vs AVX2 |

## What runs on SIMD lanes

One `ymm` register holds 8 floats or 8 int32s. Every part of the algorithm that does the same thing to many independent values runs 8 lanes at a time:

| Step in `sa_scalar.c` | AVX2 version | Kind of parallelism |
|---|---|---|
| `g_update`: `g[v] += λP[j][v] − λP[i][v]` for every drug | `vsubps`/`vaddps` on aligned rows, 4 `ymm` (32 drugs) per iteration | data parallel over drugs |
| `compute_g` (start of a chain, and the refresh every 1000 accepts) | 32 drugs per pass over the team | data parallel over drugs |
| `draw_random`: one Philox4x32-10 block per step | 8 blocks per call, one step per lane, structure-of-arrays; `vpmuludq` + `vpblendd` for the 32×32→64 multiplies | parallel over steps (counter-based RNG has no state, so future steps' numbers can be drawn early) |
| `random_below`: pick the slots `a`, `b` | high halves of 8 products at once | parallel over steps |
| `T · L[r >> 20]`: acceptance thresholds | `vpsrld` + `vgatherdps` + `vmulps` | parallel over steps |
| `propose_swap` + `should_accept` | speculative pass: 5 gathers (`order`→i, j; `g[i]`, `g[j]`; `λP[i][j]`), 8 ΔE, 8 compares, `vmovmskps` + `tzcnt` to jump to the first accepted lane | parallel over steps, speculative |
| copying the best team | `vmovdqu` + `vpmaskmovd` for the tail | data parallel |

### Why judging 8 proposals at once is still exact

The random numbers for step *s* come from Philox(seed, chain, *s*), so they don't depend on the state. All 8 steps of a batch are drawn up front and judged against the current state. If lane *m* is the first one accepted, lanes 0…*m*−1 would also have been rejected by the scalar loop, because nothing changed before them. They are skipped together. Lane *m* is committed exactly as `commit_swap()` does it. The lanes after it are judged against the new state.

Floating point is the same too. Every float operation is the scalar one in the same order (`(g[j] − g[i]) − λP[i][j]`, `g + (λP[j] − λP[i])`, `T · L`), with no FMA contraction and no reassociation. A lane computes bit-for-bit what the scalar loop computes.

### Adaptive evaluation

Re-running the gathers after every accepted swap puts a two-gathers-deep chain (about 50 cycles) on the critical path. Early in the anneal, when most proposals are accepted, that is slower than plain loads. So `sa_avx2_run_chain` works like this:

- **Speculative 8-wide pass** when the previous batch accepted at most 1 proposal. In the cold, mostly-rejecting part of the anneal this one pass decides all 8 steps.
- **Lane-by-lane walk** otherwise, and always for the rest of a batch after a commit.

Both paths are exact, so the choice only changes speed. Measured on the machine below, the speculative pass is worth **+60%** throughput on rejection-dominated runs (380 vs 236 Msteps/s at T = 0.001). It is neutral on the mixed regime. The threshold (`SPECULATE_MAX_LAST_ACCEPTS`) barely matters between 0 and 4.

### Memory layout

Every row of λP, and `w` and `g`, is zero-padded to a multiple of 8 floats and starts on a 32-byte boundary, so rows are read with aligned `vmovaps`. λP is symmetric, so row *j* is also column *j*. The g update therefore streams two contiguous rows. Gathers index λP with a signed 32-bit element number, so `n · stride` must fit in an int32 (n up to 46,336, an 8.6 GB matrix).

### Calling convention

The file is assembled with `nasm -f win64` and uses the Microsoft x64 ABI: arguments arrive in `rcx`, `rdx`, `r8`, `r9`. Any other output format stops the assembler with an error. Each function saves the callee-saved registers it touches, including `xmm6–xmm15`, and ends with `vzeroupper`. There are no unwind tables. Nothing in the kernels can raise, but a debugger cannot walk the stack through them. `sa_avx2_solve()` returns `SA_ERR_UNSUPPORTED_CPU` (Python: `RuntimeError`) when CPUID/XGETBV report no AVX2.

## Building, testing, timing

```sh
scripts/build_native.sh            # also builds module_2_avx; needs NASM on PATH (or in C:\Program Files\NASM)
ctest --test-dir build -C Release  # sa_scalar_test, sa_scalar_checked_test, sa_avx2_test
build/sa_bench --quick             # build/Release/sa_bench.exe with the Visual Studio generator
uv run python/main.py --platforms scalar avx2 --k 3 --t0 2.0 --t1 0.02
```

CMake skips the AVX2 targets on anything other than Windows x64, and with a warning if NASM is missing (`-DAD_BUILD_AVX2=OFF` to skip on purpose). The scalar and exact-search backends stay portable. Tested with MSVC 14.51 (Ninja and Visual Studio generators) and MinGW-w64 GCC 16.1 (Ninja), NASM 3.02, CMake 4.3.

`sa_bench` options: `--quick`, `--repeats R`, `--chains C`, `--t0 T --t1 T`, `--csv FILE`, `--no-pin`. Build it **Release**. For the paper's numbers use a GCC/Clang build. There the scalar baseline is `-O2 -fno-tree-vectorize` (decision D4), and an `-O3 -march=native` copy of `sa_scalar.c` is timed as "auto-vec". MSVC has no switch to turn its auto-vectorizer off. On Windows the bench pins itself to a P-core (highest `EfficiencyClass`).

## Results

Intel Core Ultra 7 265KF (Arrow Lake, pinned to a P-core), Windows 11, MinGW-w64 GCC 16.1, NASM 3.02. Single chain, T0 = 2 → T1 = 0.02, λ = 1, about 4M proposals per run, best of 3. Random instances: w ∈ (0.1, 1], P ∈ [0, 1]. Every row returned the identical best team, energy and accept count on all three builds.

**Kernels alone**

| n | g update, scalar | g update, AVX2 | speedup |
|---:|---:|---:|---:|
| 64 | 15.8 ns | 3.7 ns | 4.2× |
| 256 | 54.9 ns | 7.9 ns | 7.0× |
| 1024 | 207.9 ns | 27.2 ns | 7.6× |
| 2048 | 410.3 ns | 59.0 ns | 7.0× |
| 4096 | 835.1 ns | 130.2 ns | 6.4× |
| 16384 | 4035 ns | 1009 ns | 4.0× (memory-bound) |

Philox4x32-10: 7.62 ns per block scalar, 2.07 ns per block 8-wide, **3.7×**.

**The annealing loop** (million proposals per second; "anneal" = full run minus a one-sweep run, which removes the O(n²) setup)

| n | k | accepted | scalar | auto-vec (`-O3 -march=native`) | AVX2 | AVX2 vs scalar | AVX2 vs auto-vec | end-to-end vs scalar |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 5 | 23.7% | 55.5 | 82.3 | 120.6 | 2.17× | 1.47× | 2.17× |
| 128 | 5 | 23.5% | 48.0 | 76.2 | 119.3 | 2.48× | 1.57× | 2.48× |
| 256 | 5 | 23.3% | 35.6 | 62.1 | 107.8 | 3.03× | 1.74× | 3.01× |
| 512 | 5 | 23.4% | 23.2 | 53.2 | 79.3 | 3.42× | 1.49× | 3.38× |
| 1024 | 5 | 23.4% | 13.1 | 34.9 | 46.9 | 3.58× | 1.35× | 3.42× |
| 2048 | 5 | 23.3% | 6.3 | 15.4 | 18.6 | 2.98× | 1.21× | 2.77× |

Across k ∈ {3, 5, 8} the annealing-loop speedup is 2.2–3.7× over the scalar baseline and 1.2–1.7× over GCC's auto-vectorized build. Two other regimes:

- **Cold** (T = 0.001, almost nothing accepted): 3.2–4.3× over scalar and 2.6–4.0× over auto-vec. Here the 8-wide Philox and the speculative pass do all the work.
- **Hot** (T = 8, ~95% accepted): 2.9–4.0× over scalar and 1.15–1.4× over auto-vec. Here the g update dominates.

**Where the speedup stops.** Only accepted steps do the O(n) g update, so its 7× applies to a fraction of the steps (Amdahl). The per-step work that remains (one proposal's bookkeeping, a commit's scattered writes) is serial by nature. At n = 2048 the λP matrix is 16 MB, larger than L2, so every accepted swap streams two rows from L3. That is memory-bound for every backend, and the gap to auto-vec narrows to 1.2×. End-to-end times at large n also include the O(n²) copy of λP, which no backend vectorizes.

## Not done (yet)

- The proposal's **C++ intrinsics twin** of the update loop. The scalar backend plays the cross-check role through T7.
- The proposal's **integer deterministic mode**. Like `sa_scalar.c`, this backend is float. Identical decisions are guaranteed here by doing the same float operations in the same order, not by integers.
- **More chains per call in parallel** (8 chains per `ymm`, or one thread per chain). Chains run one after another, as in the scalar backend.
