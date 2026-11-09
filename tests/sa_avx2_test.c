/*
 * sa_avx2_test.c -- standalone checks for the AVX2 simulated annealing
 * (src/sa_avx2.c + src/asm/sa_avx2.asm). No Python needed. Built and registered with
 * ctest by CMakeLists.txt; by hand on Windows with MinGW:
 *
 *   nasm -f win64 src/asm/sa_avx2.asm -o sa_avx2.obj
 *   gcc -std=c99 -O2 -fno-tree-vectorize -Wall -Wextra -pedantic -Iinclude ^
 *       src/sa_scalar.c src/sa_avx2.c sa_avx2.obj tests/sa_avx2_test.c -o sa_avx2_test
 *
 * The kernels are checked bit for bit against plain C, and the whole solver is checked
 * decision for decision against sa_scalar.c (paper test T7): same trace, same best
 * team, same energy, same counters, for many sizes, seeds and temperatures.
 */
#include "sa_avx2.h"

#include <malloc.h> /* _aligned_malloc, _aligned_free */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(condition, ...)                                   \
    do {                                                        \
        if (!(condition)) {                                     \
            failures++;                                         \
            printf("  FAIL line %d: ", __LINE__);               \
            printf(__VA_ARGS__);                                \
            printf("\n");                                       \
        }                                                       \
    } while (0)

/* ---- helpers ----------------------------------------------------------- */

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static uint64_t next_u64(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}
static double next_uniform(void)
{
    return (double)(next_u64() >> 11) / 9007199254740992.0;
}

static void random_instance(int n, float *w, float *P)
{
    for (int i = 0; i < n; i++) {
        w[i] = (float)(0.1 + 0.9 * next_uniform());
    }
    for (int i = 0; i < n; i++) {
        P[i * n + i] = 0.0f;
        for (int j = i + 1; j < n; j++) {
            P[i * n + j] = P[j * n + i] = (float)next_uniform();
        }
    }
}

static double team_energy(int n, double lambda, const float *w, const float *P,
                          const int *team, int k)
{
    double reward = 0.0;
    double penalty = 0.0;
    for (int a = 0; a < k; a++) {
        reward += w[team[a]];
        for (int b = a + 1; b < k; b++) {
            penalty += P[team[a] * n + team[b]];
        }
    }
    return -reward + lambda * penalty;
}

static void brute_force_walk(int n, int k, double lambda, const float *w, const float *P,
                             int start, int depth, int *team, double *best)
{
    if (depth == k) {
        const double e = team_energy(n, lambda, w, P, team, k);
        if (e < *best) {
            *best = e;
        }
        return;
    }
    for (int v = start; v < n; v++) {
        team[depth] = v;
        brute_force_walk(n, k, lambda, w, P, v + 1, depth + 1, team, best);
    }
}

static double brute_force_best(int n, int k, double lambda, const float *w, const float *P)
{
    int team[32];
    double best = INFINITY;
    brute_force_walk(n, k, lambda, w, P, 0, 0, team, &best);
    return best;
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

static SaSettings default_settings(void)
{
    SaSettings s;
    s.seed = 1234;
    s.chains = 8;
    s.sweeps = 300;
    s.t0 = 2.0;
    s.t1 = 0.02;
    s.trace_steps = 0;
    return s;
}

/* ---- kernel tests ------------------------------------------------------ */

/* The 8-lane Philox must give, in every lane, exactly what the scalar Philox gives for
   that lane's counter: the published known answers (lane 0), and random counters,
   including runs of 8 that carry from word 0 into word 1 and wrap past 2^64. */
static void test_philox_x8(void)
{
    printf("philox x8 vs scalar philox\n");
    struct { uint64_t first, chain_stream, seed; uint32_t expect[4]; } kat[] = {
        {0, 0, 0, {0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u}},
        {0xffffffffffffffffULL, 0xffffffffffffffffULL, 0xffffffffffffffffULL,
         {0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu}},
        {0x85a308d3243f6a88ULL, 0x0370734413198a2eULL, 0x299f31d0a4093822ULL,
         {0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u}},
    };
    for (int c = 0; c < 3; c++) {
        uint32_t out[32];
        sa_avx2_philox_x8(kat[c].first, kat[c].chain_stream, kat[c].seed, out);
        CHECK(out[0] == kat[c].expect[0] && out[8] == kat[c].expect[1] &&
                  out[16] == kat[c].expect[2] && out[24] == kat[c].expect[3],
              "known answer %d: got %08x %08x %08x %08x", c, out[0], out[8], out[16], out[24]);
    }

    const uint64_t firsts[] = {0, 1, 7, 12345, 0xfffffffaULL, 0xfffffffcULL,
                               0x1fffffff9ULL, 0xfffffffffffffffcULL};
    int mismatches = 0;
    for (int f = 0; f < 8 + 200; f++) {
        const uint64_t first = (f < 8) ? firsts[f] : next_u64();
        const uint64_t chain_stream = (f < 8) ? (uint64_t)f : next_u64();
        const uint64_t seed = (f < 8) ? 1234 + (uint64_t)f : next_u64();
        uint32_t out[32];
        sa_avx2_philox_x8(first, chain_stream, seed, out);
        for (int lane = 0; lane < 8; lane++) {
            const uint64_t index = first + (uint64_t)lane;
            const uint32_t counter[4] = {(uint32_t)index, (uint32_t)(index >> 32),
                                         (uint32_t)chain_stream, (uint32_t)(chain_stream >> 32)};
            const uint32_t key[2] = {(uint32_t)seed, (uint32_t)(seed >> 32)};
            uint32_t expect[4];
            sa_philox4x32_10(counter, key, expect);
            for (int word = 0; word < 4; word++) {
                mismatches += out[word * 8 + lane] != expect[word];
            }
        }
    }
    CHECK(mismatches == 0, "%d words differ from the scalar Philox", mismatches);
}

/* g += row_j - row_i must match the scalar loop bit for bit, for every length (the
   unrolled x4 part and the one-register tail). */
static void test_g_update(void)
{
    printf("g update vs scalar loop\n");
    const int max_count = 8 * 70;
    float *g = alloc32(max_count * sizeof *g);
    float *expect = alloc32(max_count * sizeof *expect);
    float *row_j = alloc32(max_count * sizeof *row_j);
    float *row_i = alloc32(max_count * sizeof *row_i);
    int bad_lengths = 0;
    for (int count = 0; count <= max_count; count += 8) {
        for (int v = 0; v < max_count; v++) {
            g[v] = expect[v] = (float)(next_uniform() * 20.0 - 10.0);
            row_j[v] = (float)next_uniform();
            row_i[v] = (float)next_uniform();
        }
        for (int v = 0; v < count; v++) {
            expect[v] += row_j[v] - row_i[v];
        }
        sa_avx2_g_update(g, row_j, row_i, count);
        bad_lengths += memcmp(g, expect, max_count * sizeof *g) != 0;
    }
    CHECK(bad_lengths == 0, "%d lengths differ (or wrote past count)", bad_lengths);
    free32(g);
    free32(expect);
    free32(row_j);
    free32(row_i);
}

/* The refresh kernel (vector compute_g + scalar team energy) must match the scalar
   formulas bit for bit, for team sizes 0..n and sizes around multiples of 8 and 32. */
static void test_refresh(void)
{
    printf("refresh (compute g, team energy) vs scalar formulas\n");
    const int sizes[] = {1, 2, 7, 8, 9, 31, 32, 33, 40, 64, 100};
    for (int s = 0; s < 11; s++) {
        const int n = sizes[s];
        const int stride = (n + 7) / 8 * 8;
        float *w = alloc32(stride * sizeof *w);
        float *lam_p = alloc32((size_t)n * stride * sizeof *lam_p);
        float *g = alloc32(stride * sizeof *g);
        int32_t *order = malloc(n * sizeof *order);
        memset(w, 0, stride * sizeof *w);
        memset(lam_p, 0, (size_t)n * stride * sizeof *lam_p);
        for (int i = 0; i < n; i++) {
            w[i] = (float)(0.1 + 0.9 * next_uniform());
            for (int j = i + 1; j < n; j++) {
                lam_p[i * stride + j] = lam_p[j * stride + i] = (float)(1.7 * next_uniform());
            }
            order[i] = i;
        }
        for (int i = n - 1; i > 0; i--) { /* a random order */
            const int u = (int)(next_u64() % (uint64_t)(i + 1));
            const int32_t tmp = order[i];
            order[i] = order[u];
            order[u] = tmp;
        }
        for (int k = 0; k <= n; k++) {
            SaAvx2Chain c;
            memset(&c, 0, sizeof c);
            c.lam_p = lam_p;
            c.w = w;
            c.g = g;
            c.order = order;
            c.stride = stride;
            c.n = n;
            c.k = k;
            sa_avx2_refresh(&c);

            int bad = 0;
            for (int v = 0; v < n; v++) {
                float sum = -w[v];
                for (int t = 0; t < k; t++) {
                    sum += lam_p[v * stride + order[t]];
                }
                bad += memcmp(&sum, &g[v], sizeof sum) != 0;
            }
            float energy = 0.0f;
            for (int a = 0; a < k; a++) {
                energy -= w[order[a]];
                for (int b = a + 1; b < k; b++) {
                    energy += lam_p[order[a] * stride + order[b]];
                }
            }
            CHECK(bad == 0, "n=%d k=%d: %d entries of g differ", n, k, bad);
            CHECK(memcmp(&energy, &c.energy, sizeof energy) == 0,
                  "n=%d k=%d: energy %.9g, expected %.9g", n, k, c.energy, energy);
        }
        free32(w);
        free32(lam_p);
        free32(g);
        free(order);
    }
}

/* ---- whole-solver tests ------------------------------------------------ */

/* Paper test T7: run the scalar and the AVX2 solver on the same instance and settings,
   trace every step, and require identical decisions and identical results. */
static int compare_with_scalar(int n, int k, double lambda, SaSettings settings,
                               const char *label)
{
    float *w = malloc((size_t)n * sizeof *w);
    float *P = malloc((size_t)n * (size_t)n * sizeof *P);
    random_instance(n, w, P);
    const SaProblem problem = {n, k, lambda, w, P};

    const long long steps = (long long)settings.sweeps * n;
    settings.trace_steps = (int)(steps < 2000000 ? steps : 2000000);
    const size_t trace_size = (size_t)settings.chains * (size_t)settings.trace_steps;
    uint8_t *trace_s = malloc(trace_size + 1);
    uint8_t *trace_v = malloc(trace_size + 1);
    int *best_s = malloc(((size_t)k + 1) * sizeof *best_s);
    int *best_v = malloc(((size_t)k + 1) * sizeof *best_v);
    SaResult rs = {best_s, 0, 0, 0};
    SaResult rv = {best_v, 0, 0, 0};

    const SaStatus ss = sa_scalar_solve(&problem, &settings, &rs, trace_s);
    const SaStatus sv = sa_avx2_solve(&problem, &settings, &rv, trace_v);

    const int failures_before = failures;
    CHECK(ss == SA_OK && sv == SA_OK, "%s: status scalar %d, avx2 %d", label, ss, sv);
    if (ss == SA_OK && sv == SA_OK) {
        if (memcmp(trace_s, trace_v, trace_size) != 0) {
            size_t t = 0;
            while (trace_s[t] == trace_v[t]) {
                t++;
            }
            CHECK(0, "%s: first different decision at chain %d step %d (scalar %d, avx2 %d)",
                  label, (int)(t / (size_t)settings.trace_steps),
                  (int)(t % (size_t)settings.trace_steps), trace_s[t], trace_v[t]);
        }
        CHECK(rs.accepts == rv.accepts, "%s: accepts scalar %lld, avx2 %lld", label,
              rs.accepts, rv.accepts);
        CHECK(rs.proposals == rv.proposals, "%s: proposals scalar %lld, avx2 %lld", label,
              rs.proposals, rv.proposals);
        CHECK(rs.e_best == rv.e_best, "%s: e_best scalar %.12f, avx2 %.12f", label, rs.e_best,
              rv.e_best);
        CHECK(k == 0 || memcmp(best_s, best_v, (size_t)k * sizeof *best_s) == 0,
              "%s: different best teams", label);
    }

    free(w);
    free(P);
    free(trace_s);
    free(trace_v);
    free(best_s);
    free(best_v);
    return failures == failures_before;
}

static void test_matches_scalar(void)
{
    printf("same decisions as the scalar backend (T7)\n");
    /* Sizes around the 8-lane batches and the 8/32-float blocks, teams around the
       8-drug best-team copy, and n - k = 1 / k = 1 extremes. */
    const int cases[][2] = {{2, 1},  {3, 1},  {3, 2},   {4, 2},   {5, 4},   {8, 3},
                            {9, 4},  {12, 4}, {16, 8},  {17, 1},  {17, 16}, {24, 12},
                            {31, 7}, {33, 9}, {40, 6},  {64, 5},  {65, 17}, {100, 10},
                            {129, 3}, {257, 9}, {600, 12}};
    const int num_cases = (int)(sizeof cases / sizeof cases[0]);
    const double lambdas[] = {0.5, 1.0, 2.0};
    const uint64_t seeds[] = {1, 1234, (1ULL << 40) + 7};
    /* standard anneal, always hot (almost every proposal accepted: the speculative
       batch is redone after nearly every lane), always cold (almost none accepted) */
    const double temps[][2] = {{2.0, 0.02}, {8.0, 8.0}, {0.001, 0.001}};

    int passed = 0;
    int total = 0;
    for (int c = 0; c < num_cases; c++) {
        const int n = cases[c][0];
        const int k = cases[c][1];
        for (int x = 0; x < 3; x++) {
            SaSettings s = default_settings();
            s.seed = seeds[x];
            s.chains = 3;
            s.sweeps = n >= 256 ? 40 : 150;
            s.t0 = temps[x][0];
            s.t1 = temps[x][1];
            char label[96];
            snprintf(label, sizeof label, "n=%d k=%d lambda=%.1f seed=%llu T=%g..%g", n, k,
                     lambdas[x], (unsigned long long)s.seed, s.t0, s.t1);
            passed += compare_with_scalar(n, k, lambdas[x], s, label);
            total++;
        }
    }
    printf("  %d of %d configurations identical\n", passed, total);
}

/* Hundreds of thousands of accepts: crosses the every-1000-accepts refresh many times,
   which is where float drift between the two would show up. */
static void test_long_run_matches_scalar(void)
{
    printf("long runs, many refreshes, same decisions\n");
    SaSettings s = default_settings();
    s.chains = 4;
    s.sweeps = 3000;
    compare_with_scalar(40, 6, 1.0, s, "n=40 k=6 long");
    s.sweeps = 400;
    s.t0 = 4.0;
    s.t1 = 0.5;
    compare_with_scalar(300, 20, 1.0, s, "n=300 k=20 hot");
}

/* Sideways moves (delta E exactly 0) are always accepted, even when T * L rounds to 0.
   With equal rewards and no penalties every swap has delta 0, and a denormal
   temperature makes T * L underflow to 0 for most of the L table, so only the
   "delta <= 0" half of the accept test can take them. */
static void test_sideways_moves_accepted(void)
{
    printf("sideways moves accepted at a vanishing temperature\n");
    enum { N = 13, K = 5 };
    float w[N];
    float P[N * N];
    for (int i = 0; i < N; i++) {
        w[i] = 0.5f;
    }
    memset(P, 0, sizeof P);
    const SaProblem problem = {N, K, 1.0, w, P};
    SaSettings s = default_settings();
    s.chains = 2;
    s.sweeps = 20;
    s.t0 = s.t1 = 1e-44; /* a float denormal */
    int best[K];
    SaResult result = {best, 0, 0, 0};
    CHECK(sa_avx2_solve(&problem, &s, &result, NULL) == SA_OK, "solve failed");
    CHECK(result.accepts == result.proposals, "accepted %lld of %lld sideways moves",
          result.accepts, result.proposals);
}

/* The four-drug example from Lesson 1: the best team is {0,3} at energy -1.4. */
static void test_toy_example(void)
{
    printf("toy example\n");
    const float w[4] = {0.9f, 0.8f, 0.6f, 0.5f};
    const float P[16] = {0.0f, 0.9f, 0.15f, 0.0f, 0.9f, 0.0f, 0.2f, 0.1f,
                         0.15f, 0.2f, 0.0f, 0.0f, 0.0f, 0.1f, 0.0f, 0.0f};
    const SaProblem problem = {4, 2, 1.0, w, P};
    const SaSettings settings = default_settings();
    int best[2];
    SaResult result = {best, 0, 0, 0};
    const SaStatus status = sa_avx2_solve(&problem, &settings, &result, NULL);
    CHECK(status == SA_OK, "status: %s", sa_status_message(status));
    CHECK(fabs(result.e_best - (-1.4)) < 1e-3, "energy %f", result.e_best);
    CHECK((best[0] == 0 && best[1] == 3) || (best[0] == 3 && best[1] == 0), "team {%d,%d}",
          best[0], best[1]);
    printf("  best {%d,%d}  E=%.4f  accepts=%lld/%lld\n", best[0], best[1], result.e_best,
           result.accepts, result.proposals);
}

static void test_matches_brute_force(void)
{
    printf("matches brute force\n");
    const int sizes[][2] = {{8, 2}, {10, 3}, {12, 4}, {12, 6}, {14, 3}};
    for (int c = 0; c < 5; c++) {
        const int n = sizes[c][0];
        const int k = sizes[c][1];
        for (int trial = 0; trial < 4; trial++) {
            float w[16];
            float P[16 * 16];
            random_instance(n, w, P);
            const double lambda = 0.5 + trial * 0.5;
            const SaProblem problem = {n, k, lambda, w, P};
            const SaSettings settings = default_settings();
            int best[16];
            SaResult result = {best, 0, 0, 0};
            const SaStatus status = sa_avx2_solve(&problem, &settings, &result, NULL);
            CHECK(status == SA_OK, "status: %s", sa_status_message(status));
            const double exact = brute_force_best(n, k, lambda, w, P);
            CHECK(fabs(result.e_best - exact) < 1e-9, "n=%d k=%d trial=%d: SA %.9f, exact %.9f",
                  n, k, trial, result.e_best, exact);
            for (int a = 0; a < k; a++) {
                CHECK(best[a] >= 0 && best[a] < n, "drug %d out of range", best[a]);
                for (int b = a + 1; b < k; b++) {
                    CHECK(best[a] != best[b], "drug %d repeated", best[a]);
                }
            }
        }
    }
}

/* Same edge cases and the same refusals as the scalar backend. */
static void test_edge_cases(void)
{
    printf("edge cases\n");
    const float w[3] = {0.5f, 0.25f, 0.75f};
    const float P[9] = {0, 0.1f, 0.2f, 0.1f, 0, 0.3f, 0.2f, 0.3f, 0};
    const SaSettings settings = default_settings();
    int best[3];
    SaResult result = {best, 0, 0, 0};

    SaProblem all = {3, 3, 1.0, w, P};
    CHECK(sa_avx2_solve(&all, &settings, &result, NULL) == SA_OK, "k=n failed");
    CHECK(fabs(result.e_best - team_energy(3, 1.0, w, P, best, 3)) < 1e-4, "k=n energy");
    CHECK(result.proposals == 0, "k=n should propose nothing");

    SaProblem none = {3, 0, 1.0, w, P};
    CHECK(sa_avx2_solve(&none, &settings, &result, NULL) == SA_OK, "k=0 failed");
    CHECK(result.e_best == 0.0, "k=0 energy %f", result.e_best);

    SaProblem empty = {0, 0, 1.0, NULL, NULL};
    CHECK(sa_avx2_solve(&empty, &settings, &result, NULL) == SA_OK, "n=0 failed");

    SaProblem too_big = {3, 4, 1.0, w, P};
    CHECK(sa_avx2_solve(&too_big, &settings, &result, NULL) == SA_ERR_BAD_ARGUMENT,
          "k>n should be refused");

    SaProblem fine = {3, 1, 1.0, w, P};
    SaSettings bad = settings;
    bad.t0 = -1.0;
    bad.t1 = -1.0;
    CHECK(sa_avx2_solve(&fine, &bad, &result, NULL) == SA_ERR_BAD_ARGUMENT,
          "missing temperatures should be refused");
    bad = settings;
    bad.t1 = 0.0;
    CHECK(sa_avx2_solve(&fine, &bad, &result, NULL) == SA_ERR_BAD_ARGUMENT,
          "t1 = 0 should be refused");
    bad = settings;
    bad.t0 = 0.01;
    bad.t1 = 0.5;
    CHECK(sa_avx2_solve(&fine, &bad, &result, NULL) == SA_ERR_BAD_ARGUMENT,
          "t0 < t1 should be refused");
    bad = settings;
    bad.t0 = NAN;
    CHECK(sa_avx2_solve(&fine, &bad, &result, NULL) == SA_ERR_BAD_ARGUMENT,
          "NaN temperature should be refused");
    bad = settings;
    bad.trace_steps = 10;
    CHECK(sa_avx2_solve(&fine, &bad, &result, NULL) == SA_ERR_BAD_ARGUMENT,
          "trace_steps without a trace buffer should be refused");

    const float nan_w[3] = {NAN, 1.0f, 1.0f};
    SaProblem not_a_number = {3, 1, 1.0, nan_w, P};
    CHECK(sa_avx2_solve(&not_a_number, &settings, &result, NULL) == SA_ERR_OUT_OF_RANGE,
          "NaN reward should be out of range");
    const float inf_w[3] = {INFINITY, 1.0f, 1.0f};
    SaProblem infinite = {3, 1, 1.0, inf_w, P};
    CHECK(sa_avx2_solve(&infinite, &settings, &result, NULL) == SA_ERR_OUT_OF_RANGE,
          "infinite reward should be out of range");
    SaProblem huge_lambda = {3, 1, 1e300, w, P};
    CHECK(sa_avx2_solve(&huge_lambda, &settings, &result, NULL) == SA_ERR_OUT_OF_RANGE,
          "lambda*P too large for a float should be out of range");

    SaSettings flat = settings;
    flat.t0 = flat.t1 = 0.5;
    CHECK(sa_avx2_solve(&fine, &flat, &result, NULL) == SA_OK, "constant temperature failed");
    SaSettings one_sweep = settings;
    one_sweep.sweeps = 1;
    CHECK(sa_avx2_solve(&fine, &one_sweep, &result, NULL) == SA_OK, "one sweep failed");
    SaSettings scorching = settings;
    scorching.t0 = 1e300;
    CHECK(sa_avx2_solve(&fine, &scorching, &result, NULL) == SA_ERR_OUT_OF_RANGE,
          "huge temperature should be out of range");
}

int main(void)
{
    if (!sa_avx2_supported()) {
        printf("this CPU/OS has no AVX2: nothing to test (skipped)\n");
        return 0;
    }
    test_philox_x8();
    test_g_update();
    test_refresh();
    test_sideways_moves_accepted();
    test_toy_example();
    test_matches_brute_force();
    test_matches_scalar();
    test_long_run_matches_scalar();
    test_edge_cases();

    if (failures == 0) {
        printf("\nall checks passed\n");
        return 0;
    }
    printf("\n%d check(s) failed\n", failures);
    return 1;
}
