/*
 * sa_bench.c -- times the scalar C simulated annealing against the AVX2 assembly one.
 *
 *   sa_bench [--quick] [--repeats R] [--chains C] [--t0 T] [--t1 T] [--csv FILE] [--no-pin]
 *
 * Part 1 times the kernels on their own (the parts that were vectorized):
 *   - the g update   g += lambda*P[j,:] - lambda*P[i,:]   scalar loop vs ymm
 *   - Philox4x32-10                                       1 block vs 8 blocks per call
 * Part 2 times whole runs (paper figure F4): for each n and k the same instance, seed
 * and settings go to sa_scalar_solve() and sa_avx2_solve(), and the table reports the
 * best-of-R end-to-end time, the annealing loop's own throughput (full run minus a
 * one-sweep run, which removes the O(n^2) setup), the speedups, the share of accepted
 * steps, and whether the runs agree (same best team, energy and accept count).
 *
 * Built by CMake as `sa_bench` (target sa_bench). For a fair baseline the scalar code
 * is compiled with -O2 -fno-tree-vectorize (paper decision D4). When CMake also builds
 * an -O3 -march=native copy of sa_scalar.c (GCC/Clang only), it is timed as a third
 * column, "auto-vec", so the hand-written SIMD can be compared with the compiler's.
 *
 * Windows only. The thread is pinned to one of the fastest cores (on hybrid CPUs such
 * as Alder/Raptor/Arrow Lake the P-cores) and raised in priority, so the numbers do
 * not depend on where the scheduler happens to put it. --no-pin turns that off.
 */
#include "sa_avx2.h"

#include <malloc.h> /* _aligned_malloc, _aligned_free */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#ifdef SA_BENCH_AUTOVEC
/* sa_scalar.c compiled a second time with -O3 -march=native and its public names
   renamed (see CMakeLists.txt). */
SaStatus sa_scalar_autovec_solve(const SaProblem *problem, const SaSettings *settings,
                                 SaResult *result, uint8_t *trace);
#endif

/* ---- timing and pinning ------------------------------------------------ */

static double now_seconds(void)
{
    static LARGE_INTEGER frequency;
    LARGE_INTEGER t;
    if (frequency.QuadPart == 0) {
        QueryPerformanceFrequency(&frequency);
    }
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)frequency.QuadPart;
}

/* Pins this thread to the first logical processor of the core with the highest
   EfficiencyClass (a P-core on a hybrid CPU; any core otherwise). */
static void pin_to_fast_core(void)
{
    DWORD length = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &length);
    unsigned char *buffer = malloc(length);
    if (buffer == NULL ||
        !GetLogicalProcessorInformationEx(RelationProcessorCore,
                                          (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buffer,
                                          &length)) {
        free(buffer);
        return;
    }
    KAFFINITY chosen = 0;
    int best_class = -1;
    int fast_cores = 0;
    int cores = 0;
    for (DWORD offset = 0; offset < length;) {
        const PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info =
            (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buffer + offset);
        const PROCESSOR_RELATIONSHIP *core = &info->Processor;
        const int efficiency = core->EfficiencyClass;
        cores++;
        if (core->GroupMask[0].Group == 0 && efficiency > best_class) {
            best_class = efficiency;
            chosen = core->GroupMask[0].Mask & (~core->GroupMask[0].Mask + 1); /* lowest bit */
            fast_cores = 0;
        }
        if (efficiency == best_class) {
            fast_cores++;
        }
        offset += info->Size;
    }
    free(buffer);
    if (chosen != 0) {
        int index = 0;
        while (((chosen >> index) & 1) == 0) {
            index++;
        }
        SetThreadAffinityMask(GetCurrentThread(), chosen);
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
        printf("pinned to logical processor %d (efficiency class %d; %d of %d cores in it)\n",
               index, best_class, fast_cores, cores);
    }
}

/* ---- instances ----------------------------------------------------------- */

static uint64_t rng_state = 88172645463325252ULL;
static double next_uniform(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (double)(rng_state >> 11) / 9007199254740992.0;
}

static void random_instance(int n, float *w, float *P)
{
    for (int i = 0; i < n; i++) {
        w[i] = (float)(0.1 + 0.9 * next_uniform());
    }
    for (int i = 0; i < n; i++) {
        P[(size_t)i * n + i] = 0.0f;
        for (int j = i + 1; j < n; j++) {
            P[(size_t)i * n + j] = P[(size_t)j * n + i] = (float)next_uniform();
        }
    }
}

/* 32-byte aligned memory, as the kernels need. */
static void *alloc32(size_t bytes)
{
    return _aligned_malloc(bytes > 0 ? bytes : 32, 32);
}
static void free32(void *p)
{
    _aligned_free(p);
}

/* ---- part 1: kernels ----------------------------------------------------- */

/* The g update exactly as sa_scalar.c writes it (built without auto-vectorization). */
static void scalar_g_update(float *g, const float *row_j, const float *row_i, int n)
{
#if defined(_MSC_VER) && !defined(__clang__)
#pragma loop(no_vector)
#endif
    for (int v = 0; v < n; v++) {
        g[v] += row_j[v] - row_i[v];
    }
}

static void bench_g_update(int repeats, FILE *csv)
{
    printf("\n## Kernel: g update (g += lambda*P[j,:] - lambda*P[i,:])\n\n");
    printf("| n | scalar ns | AVX2 ns | speedup | AVX2 GB/s |\n");
    printf("|---:|---:|---:|---:|---:|\n");
    const int sizes[] = {64, 256, 1024, 2048, 4096, 16384};
    for (int s = 0; s < 6; s++) {
        const int n = sizes[s];
        float *g = alloc32((size_t)n * sizeof *g);
        float *rj = alloc32((size_t)n * sizeof *rj);
        float *ri = alloc32((size_t)n * sizeof *ri);
        for (int v = 0; v < n; v++) {
            g[v] = (float)next_uniform();
            rj[v] = (float)next_uniform();
            ri[v] = (float)next_uniform();
        }
        const long long updates = 40000000LL / n + 1000;
        double best_scalar = INFINITY;
        double best_avx2 = INFINITY;
        for (int r = 0; r < repeats; r++) {
            double t = now_seconds();
            for (long long u = 0; u < updates; u += 2) {
                scalar_g_update(g, rj, ri, n);
                scalar_g_update(g, ri, rj, n);
            }
            t = now_seconds() - t;
            best_scalar = t < best_scalar ? t : best_scalar;
            t = now_seconds();
            for (long long u = 0; u < updates; u += 2) {
                sa_avx2_g_update(g, rj, ri, n);
                sa_avx2_g_update(g, ri, rj, n);
            }
            t = now_seconds() - t;
            best_avx2 = t < best_avx2 ? t : best_avx2;
        }
        const double ns_scalar = 1e9 * best_scalar / (double)updates;
        const double ns_avx2 = 1e9 * best_avx2 / (double)updates;
        const double gbs = 4.0 * n * sizeof(float) / ns_avx2; /* 3 loads + 1 store */
        printf("| %d | %.1f | %.1f | %.2fx | %.1f |\n", n, ns_scalar, ns_avx2,
               ns_scalar / ns_avx2, gbs);
        if (csv != NULL) {
            fprintf(csv, "kernel_g_update,scalar,%d,,,,,,,%.3f,\n", n, ns_scalar);
            fprintf(csv, "kernel_g_update,AVX2,%d,,,,,,,%.3f,\n", n, ns_avx2);
        }
        volatile float sink = g[n / 2];
        (void)sink;
        free32(g);
        free32(rj);
        free32(ri);
    }
}

static void bench_philox(int repeats, FILE *csv)
{
    printf("\n## Kernel: Philox4x32-10 (one block = 4 random words = one SA step)\n\n");
    const uint64_t blocks = 8u << 20;
    const uint32_t key[2] = {1234u, 0u};
    double best_scalar = INFINITY;
    double best_avx2 = INFINITY;
    uint32_t check = 0;
    for (int r = 0; r < repeats; r++) {
        double t = now_seconds();
        for (uint64_t b = 0; b < blocks; b++) {
            const uint32_t counter[4] = {(uint32_t)b, (uint32_t)(b >> 32), 7u, 0u};
            uint32_t out[4];
            sa_philox4x32_10(counter, key, out);
            check += out[0] ^ out[2];
        }
        t = now_seconds() - t;
        best_scalar = t < best_scalar ? t : best_scalar;
        t = now_seconds();
        for (uint64_t b = 0; b < blocks; b += 8) {
            uint32_t out[32];
            sa_avx2_philox_x8(b, 7u, 1234u, out);
            check += out[0] ^ out[16];
        }
        t = now_seconds() - t;
        best_avx2 = t < best_avx2 ? t : best_avx2;
    }
    const double ns_scalar = 1e9 * best_scalar / (double)blocks;
    const double ns_avx2 = 1e9 * best_avx2 / (double)blocks;
    printf("| | ns per block | blocks per call |\n|---|---:|---:|\n");
    printf("| scalar | %.2f | 1 |\n| AVX2 | %.2f | 8 |\n", ns_scalar, ns_avx2);
    printf("\nspeedup %.2fx   (checksum %08x)\n", ns_scalar / ns_avx2, check);
    if (csv != NULL) {
        fprintf(csv, "kernel_philox,scalar,,,,,,,,%.3f,\n", ns_scalar);
        fprintf(csv, "kernel_philox,AVX2,,,,,,,,%.3f,\n", ns_avx2);
    }
}

/* ---- part 2: whole runs -------------------------------------------------- */

typedef SaStatus (*SolveFn)(const SaProblem *, const SaSettings *, SaResult *, uint8_t *);

typedef struct {
    const char *name;
    SolveFn solve;
} Backend;

static const Backend backends[] = {
    {"scalar", sa_scalar_solve},
#ifdef SA_BENCH_AUTOVEC
    {"auto-vec", sa_scalar_autovec_solve},
#endif
    {"AVX2", sa_avx2_solve},
};
enum { NUM_BACKENDS = (int)(sizeof backends / sizeof backends[0]) };

/* Best-of-`repeats` wall time of one solve. */
static double time_solve(SolveFn solve, const SaProblem *problem, const SaSettings *settings,
                         int repeats, int *best, SaResult *result)
{
    double best_time = INFINITY;
    for (int r = 0; r < repeats; r++) {
        result->best = best;
        const double t = now_seconds();
        const SaStatus status = solve(problem, settings, result, NULL);
        const double elapsed = now_seconds() - t;
        if (status != SA_OK) {
            fprintf(stderr, "solve failed: %s\n", sa_status_message(status));
            exit(1);
        }
        best_time = elapsed < best_time ? elapsed : best_time;
    }
    return best_time;
}

static int same_result(const SaResult *a, const SaResult *b, int k)
{
    return a->e_best == b->e_best && a->accepts == b->accepts &&
           a->proposals == b->proposals && memcmp(a->best, b->best, (size_t)k * sizeof(int)) == 0;
}

/* Each backend is timed twice: the full run (S sweeps) and a one-sweep run. The
   one-sweep run is almost all fixed cost (copying the O(n^2) matrix, building tables,
   starting the chains), so "anneal" = full - one-sweep is the time the S - 1 extra
   sweeps took: the annealing loop alone. Both numbers are reported. */
static void bench_runs(int quick, int repeats, int chains, double t0, double t1, FILE *csv)
{
    printf("\n## Whole runs: chains=%d, T0=%g, T1=%g, lambda=1, best of %d\n\n", chains, t0,
           t1, repeats);
    printf("End-to-end = one whole solve call (what Python sees). Anneal = the annealing "
           "loop alone (full run minus a one-sweep run), in million proposals per second.\n\n");
    printf("| n | k | steps | accepted |");
    for (int b = 0; b < NUM_BACKENDS; b++) {
        printf(" %s ms |", backends[b].name);
    }
    for (int b = 0; b < NUM_BACKENDS; b++) {
        printf(" %s anneal Msteps/s |", backends[b].name);
    }
    printf(" AVX2 speedup, end-to-end | AVX2 speedup, anneal |");
#ifdef SA_BENCH_AUTOVEC
    printf(" AVX2 vs auto-vec, anneal |");
#endif
    printf(" same answer |\n|---:|---:|---:|---:|");
    for (int c = 0; c < 2 * NUM_BACKENDS + 2; c++) {
        printf("---:|");
    }
#ifdef SA_BENCH_AUTOVEC
    printf("---:|");
#endif
    printf(":---:|\n");

    const int sizes_full[] = {64, 128, 256, 512, 1024, 2048};
    const int sizes_quick[] = {128, 512, 2048};
    const int *sizes = quick ? sizes_quick : sizes_full;
    const int num_sizes = quick ? 3 : 6;
    const int ks[] = {3, 5, 8};
    const long long target_steps = quick ? 1000000 : 4000000;

    for (int s = 0; s < num_sizes; s++) {
        const int n = sizes[s];
        float *w = malloc((size_t)n * sizeof *w);
        float *P = malloc((size_t)n * (size_t)n * sizeof *P);
        random_instance(n, w, P);
        for (int kk = 0; kk < 3; kk++) {
            const int k = ks[kk];
            const SaProblem problem = {n, k, 1.0, w, P};
            SaSettings settings;
            settings.seed = 2026;
            settings.chains = chains;
            settings.sweeps = (int)(target_steps / ((long long)n * chains));
            if (settings.sweeps < 20) {
                settings.sweeps = 20;
            }
            settings.t0 = t0;
            settings.t1 = t1;
            settings.trace_steps = 0;
            SaSettings one_sweep = settings;
            one_sweep.sweeps = 1;

            double total[NUM_BACKENDS];
            double setup[NUM_BACKENDS];
            double anneal_rate[NUM_BACKENDS];
            int same[NUM_BACKENDS];
            int best[NUM_BACKENDS][16];
            SaResult result[NUM_BACKENDS];
            const double anneal_steps = (double)(settings.sweeps - 1) * n * chains;
            for (int b = 0; b < NUM_BACKENDS; b++) {
                int scratch[16];
                SaResult ignored;
                total[b] = time_solve(backends[b].solve, &problem, &settings, repeats, best[b],
                                      &result[b]);
                setup[b] = time_solve(backends[b].solve, &problem, &one_sweep, repeats, scratch,
                                      &ignored);
                const double anneal = total[b] - setup[b];
                anneal_rate[b] = anneal > 0.0 ? anneal_steps / anneal / 1e6 : NAN;
                same[b] = same_result(&result[0], &result[b], k);
            }
            const int avx2 = NUM_BACKENDS - 1;
            const double steps = (double)result[0].proposals;
            const double accepted = (double)result[0].accepts / steps;
            int all_same = 1;
            for (int b = 0; b < NUM_BACKENDS; b++) {
                all_same = all_same && same[b];
            }

            printf("| %d | %d | %.0f | %.1f%% |", n, k, steps, 100.0 * accepted);
            for (int b = 0; b < NUM_BACKENDS; b++) {
                printf(" %.1f |", 1e3 * total[b]);
            }
            for (int b = 0; b < NUM_BACKENDS; b++) {
                printf(" %.1f |", anneal_rate[b]);
            }
            printf(" %.2fx | %.2fx |", total[0] / total[avx2], anneal_rate[avx2] / anneal_rate[0]);
#ifdef SA_BENCH_AUTOVEC
            printf(" %.2fx |", anneal_rate[avx2] / anneal_rate[1]);
#endif
            printf(" %s |\n", all_same ? "yes" : "NO");
            fflush(stdout);

            if (csv != NULL) {
                for (int b = 0; b < NUM_BACKENDS; b++) {
                    fprintf(csv, "sa_run,%s,%d,%d,%.0f,%.6f,%.6f,%.6f,%.3f,,%d\n",
                            backends[b].name, n, k, steps, accepted, total[b], setup[b],
                            anneal_rate[b], same[b]);
                }
            }
        }
        free(w);
        free(P);
    }
}

int main(int argc, char **argv)
{
    int quick = 0;
    int pin = 1;
    int repeats = 3;
    int chains = 1;
    double t0 = 2.0;
    double t1 = 0.02;
    const char *csv_path = NULL;
    for (int a = 1; a < argc; a++) {
        if (strcmp(argv[a], "--quick") == 0) {
            quick = 1;
        } else if (strcmp(argv[a], "--no-pin") == 0) {
            pin = 0;
        } else if (strcmp(argv[a], "--repeats") == 0 && a + 1 < argc) {
            repeats = atoi(argv[++a]);
        } else if (strcmp(argv[a], "--chains") == 0 && a + 1 < argc) {
            chains = atoi(argv[++a]);
        } else if (strcmp(argv[a], "--t0") == 0 && a + 1 < argc) {
            t0 = atof(argv[++a]);
        } else if (strcmp(argv[a], "--t1") == 0 && a + 1 < argc) {
            t1 = atof(argv[++a]);
        } else if (strcmp(argv[a], "--csv") == 0 && a + 1 < argc) {
            csv_path = argv[++a];
        } else {
            fprintf(stderr,
                    "usage: %s [--quick] [--repeats R] [--chains C] [--t0 T] [--t1 T] "
                    "[--csv FILE] [--no-pin]\n",
                    argv[0]);
            return 2;
        }
    }
    if (repeats < 1 || chains < 1 || !(t1 > 0.0 && t0 >= t1)) {
        fprintf(stderr, "need repeats >= 1, chains >= 1 and t0 >= t1 > 0\n");
        return 2;
    }
    if (!sa_avx2_supported()) {
        fprintf(stderr, "this CPU/OS has no AVX2\n");
        return 1;
    }
    if (pin) {
        pin_to_fast_core();
    }

    FILE *csv = NULL;
    if (csv_path != NULL) {
        csv = fopen(csv_path, "w");
        if (csv == NULL) {
            fprintf(stderr, "cannot write %s\n", csv_path);
            return 1;
        }
        fprintf(csv, "section,backend,n,k,steps,accept_rate,total_s,setup_s,"
                     "anneal_msteps_per_s,ns_per_call,same_as_scalar\n");
    }

    printf("# Scalar C vs AVX2 assembly simulated annealing\n");
    bench_g_update(repeats, csv);
    bench_philox(repeats, csv);
    bench_runs(quick, repeats, chains, t0, t1, csv);

    if (csv != NULL) {
        fclose(csv);
    }
    return 0;
}
