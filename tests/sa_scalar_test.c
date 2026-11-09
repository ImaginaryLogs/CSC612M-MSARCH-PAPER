/*
 * sa_scalar_test.c -- standalone checks for src/sa_scalar.c. No Python needed.
 *
 * Build and run (from the repository root):
 *   cc -std=c99 -O2 -fno-tree-vectorize -Wall -Wextra -pedantic -Iinclude \
 *      src/sa_scalar.c tests/sa_scalar_test.c -lm -o sa_scalar_test && ./sa_scalar_test
 *
 * Add -DSA_CHECK_G to the build to also rebuild g and E from scratch every 10,000
 * accepted swaps and abort if they have drifted (paper test T9).
 */
#include "sa_scalar.h"

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

/* The reference energy, written the plain way (same formula as judge()). */
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

/* Tries every k-combination and returns the lowest energy. */
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

/* A small deterministic generator, only for building test instances. */
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
        P[i * n + i] = 0.0f;
        for (int j = i + 1; j < n; j++) {
            P[i * n + j] = P[j * n + i] = (float)next_uniform();
        }
    }
}

static SaSettings default_settings(void)
{
    SaSettings s;
    s.seed = 1234;
    s.chains = 8;
    s.sweeps = 300;
    s.t0 = 2.0;  /* hot enough to accept most uphill swaps at the start */
    s.t1 = 0.02; /* cold enough to accept almost none at the end */
    s.trace_steps = 0;
    return s;
}

/* ---- tests ------------------------------------------------------------- */

/* Published Philox4x32-10 known-answer values (Random123 kat_vectors). */
static void test_philox_known_answers(void)
{
    printf("philox known answers\n");
    struct { uint32_t ctr[4], key[2], expect[4]; } cases[] = {
        {{0, 0, 0, 0}, {0, 0},
         {0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u}},
        {{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu}, {0xffffffffu, 0xffffffffu},
         {0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu}},
        {{0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u}, {0xa4093822u, 0x299f31d0u},
         {0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u}},
    };
    for (int c = 0; c < 3; c++) {
        uint32_t out[4];
        sa_philox4x32_10(cases[c].ctr, cases[c].key, out);
        CHECK(memcmp(out, cases[c].expect, sizeof out) == 0,
              "case %d: got %08x %08x %08x %08x", c, out[0], out[1], out[2], out[3]);
    }
}

/* The four-drug example from Lesson 1: the best team is {0,3} at energy -1.4. */
static void test_toy_example(void)
{
    printf("toy example\n");
    const float w[4] = {0.9f, 0.8f, 0.6f, 0.5f};
    const float P[16] = {0.0f, 0.9f, 0.15f, 0.0f,
                         0.9f, 0.0f, 0.2f,  0.1f,
                         0.15f, 0.2f, 0.0f, 0.0f,
                         0.0f, 0.1f, 0.0f,  0.0f};
    const SaProblem problem = {4, 2, 1.0, w, P};
    const SaSettings settings = default_settings();
    int best[2];
    SaResult result = {best, 0, 0, 0};

    const SaStatus status = sa_scalar_solve(&problem, &settings, &result, NULL);
    CHECK(status == SA_OK, "status: %s", sa_status_message(status));
    CHECK(fabs(result.e_best - (-1.4)) < 1e-3, "energy %f", result.e_best);
    CHECK((best[0] == 0 && best[1] == 3) || (best[0] == 3 && best[1] == 0),
          "team {%d,%d}", best[0], best[1]);
    CHECK(result.proposals == (long long)settings.chains * settings.sweeps * 4,
          "proposals %lld", result.proposals);
    printf("  best {%d,%d}  E=%.4f  accepts=%lld/%lld\n", best[0], best[1], result.e_best,
           result.accepts, result.proposals);
}

/* On small random instances SA should reach the exact optimum, and the energy it
   reports should match an independent recomputation for the team it returns. */
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
            const SaStatus status = sa_scalar_solve(&problem, &settings, &result, NULL);
            CHECK(status == SA_OK, "status: %s", sa_status_message(status));

            const double exact = brute_force_best(n, k, lambda, w, P);
            const double recomputed = team_energy(n, lambda, w, P, best, k);
            CHECK(fabs(result.e_best - exact) < 1e-9,
                  "n=%d k=%d trial=%d: SA %.9f, exact %.9f", n, k, trial, result.e_best, exact);
            CHECK(fabs(result.e_best - recomputed) < 1e-9,
                  "n=%d k=%d: reported %.6f, recomputed %.6f", n, k, result.e_best, recomputed);

            /* The team must be k different drugs, all in range. */
            for (int a = 0; a < k; a++) {
                CHECK(best[a] >= 0 && best[a] < n, "drug %d out of range", best[a]);
                for (int b = a + 1; b < k; b++) {
                    CHECK(best[a] != best[b], "drug %d repeated", best[a]);
                }
            }
        }
    }
}

/* Same seed, same run (T10). Different seed, different run. The trace also has to
   agree with the accept counter. */
static void test_determinism_and_trace(void)
{
    printf("determinism and trace\n");
    enum { N = 12, K = 4 };
    float w[N];
    float P[N * N];
    random_instance(N, w, P);
    const SaProblem problem = {N, K, 1.0, w, P};

    SaSettings settings = default_settings();
    settings.chains = 3;
    settings.sweeps = 50;
    settings.trace_steps = settings.sweeps * N; /* trace every step */

    const size_t trace_size = (size_t)settings.chains * (size_t)settings.trace_steps;
    uint8_t *trace_a = malloc(trace_size);
    uint8_t *trace_b = malloc(trace_size);
    uint8_t *trace_c = malloc(trace_size);
    int best_a[K], best_b[K], best_c[K];
    SaResult result_a = {best_a, 0, 0, 0};
    SaResult result_b = {best_b, 0, 0, 0};
    SaResult result_c = {best_c, 0, 0, 0};

    sa_scalar_solve(&problem, &settings, &result_a, trace_a);
    sa_scalar_solve(&problem, &settings, &result_b, trace_b);
    settings.seed = 99;
    sa_scalar_solve(&problem, &settings, &result_c, trace_c);

    CHECK(memcmp(trace_a, trace_b, trace_size) == 0, "same seed gave different traces");
    CHECK(memcmp(best_a, best_b, sizeof best_a) == 0, "same seed gave different teams");
    CHECK(result_a.accepts == result_b.accepts, "same seed gave different accept counts");
    CHECK(memcmp(trace_a, trace_c, trace_size) != 0, "different seeds gave identical traces");

    long long traced_accepts = 0;
    for (size_t t = 0; t < trace_size; t++) {
        traced_accepts += trace_a[t];
    }
    CHECK(traced_accepts == result_a.accepts, "trace says %lld accepts, counter says %lld",
          traced_accepts, result_a.accepts);
    printf("  accepts %lld of %lld proposals\n", result_a.accepts, result_a.proposals);

    free(trace_a);
    free(trace_b);
    free(trace_c);
}

/* A long run: hundreds of thousands of accepted swaps, so g and E are refreshed many
   times. Built with -DSA_CHECK_G, every refresh first compares the running float values
   with a fresh rebuild and aborts if they have drifted. The reported energy is exact by
   construction (recomputed in double), so this test cannot see bookkeeping bugs by
   itself; the checked build is what catches them. */
static void test_long_run_consistency(void)
{
    printf("long run (g and E drift check when built with -DSA_CHECK_G)\n");
    enum { N = 40, K = 6 };
    float w[N];
    float P[N * N];
    random_instance(N, w, P);
    const SaProblem problem = {N, K, 1.0, w, P};
    SaSettings settings = default_settings();
    settings.chains = 4;
    settings.sweeps = 3000;
    int best[K];
    SaResult result = {best, 0, 0, 0};
    const SaStatus status = sa_scalar_solve(&problem, &settings, &result, NULL);
    CHECK(status == SA_OK, "status: %s", sa_status_message(status));
    const double recomputed = team_energy(N, 1.0, w, P, best, K);
    CHECK(fabs(result.e_best - recomputed) < 1e-9, "reported %.9f, recomputed %.9f",
          result.e_best, recomputed);
    printf("  accepts %lld of %lld\n", result.accepts, result.proposals);
}

/* Fraction of proposals accepted when the temperature is held constant. */
static double acceptance_rate(const SaProblem *problem, double temperature)
{
    SaSettings settings = default_settings();
    settings.chains = 4;
    settings.sweeps = 200;
    settings.t0 = temperature;
    settings.t1 = temperature;
    int best[16];
    SaResult result = {best, 0, 0, 0};
    const SaStatus status = sa_scalar_solve(problem, &settings, &result, NULL);
    CHECK(status == SA_OK, "status: %s", sa_status_message(status));
    return (double)result.accepts / (double)result.proposals;
}

/* Temperature has to control how many swaps are accepted: a hot run takes nearly all of
   them, a cold run almost none. This is the test that notices a broken accept rule.
   Finding the optimum on a tiny instance would not, because remembering the best team
   seen lets even a random walk find it. */
static void test_temperature_controls_acceptance(void)
{
    printf("temperature controls acceptance\n");
    enum { N = 20, K = 5 };
    float w[N];
    float P[N * N];
    random_instance(N, w, P);
    const SaProblem problem = {N, K, 1.0, w, P};

    const double hot = acceptance_rate(&problem, 8.0);
    const double warm = acceptance_rate(&problem, 0.3);
    const double cold = acceptance_rate(&problem, 0.001);
    printf("  accepted: hot %.3f, warm %.3f, cold %.3f\n", hot, warm, cold);

    CHECK(hot > 0.9, "hot run accepted only %.3f", hot);
    CHECK(cold < 0.05, "cold run accepted %.3f", cold);
    CHECK(hot > warm && warm > cold, "acceptance should fall as temperature falls");
}

static void test_edge_cases(void)
{
    printf("edge cases\n");
    const float w[3] = {0.5f, 0.25f, 0.75f};
    const float P[9] = {0, 0.1f, 0.2f, 0.1f, 0, 0.3f, 0.2f, 0.3f, 0};
    const SaSettings settings = default_settings();
    int best[3];
    SaResult result = {best, 0, 0, 0};

    /* k = n: the only team is everyone. */
    SaProblem all = {3, 3, 1.0, w, P};
    CHECK(sa_scalar_solve(&all, &settings, &result, NULL) == SA_OK, "k=n failed");
    CHECK(fabs(result.e_best - team_energy(3, 1.0, w, P, best, 3)) < 1e-4, "k=n energy");
    CHECK(result.proposals == 0, "k=n should propose nothing");

    /* k = 0: the empty team has energy 0. */
    SaProblem none = {3, 0, 1.0, w, P};
    CHECK(sa_scalar_solve(&none, &settings, &result, NULL) == SA_OK, "k=0 failed");
    CHECK(result.e_best == 0.0, "k=0 energy %f", result.e_best);

    /* k > n is refused. */
    SaProblem too_big = {3, 4, 1.0, w, P};
    CHECK(sa_scalar_solve(&too_big, &settings, &result, NULL) == SA_ERR_BAD_ARGUMENT,
          "k>n should be refused");

    /* The temperatures are required: missing, zero, reversed or NaN ones are refused. */
    SaProblem fine = {3, 1, 1.0, w, P};
    SaSettings bad = settings;
    bad.t0 = -1.0;
    bad.t1 = -1.0;
    CHECK(sa_scalar_solve(&fine, &bad, &result, NULL) == SA_ERR_BAD_ARGUMENT,
          "missing temperatures should be refused");
    bad = settings;
    bad.t1 = 0.0;
    CHECK(sa_scalar_solve(&fine, &bad, &result, NULL) == SA_ERR_BAD_ARGUMENT,
          "t1 = 0 should be refused");
    bad = settings;
    bad.t0 = 0.01;
    bad.t1 = 0.5;
    CHECK(sa_scalar_solve(&fine, &bad, &result, NULL) == SA_ERR_BAD_ARGUMENT,
          "t0 < t1 should be refused");
    bad = settings;
    bad.t0 = NAN;
    CHECK(sa_scalar_solve(&fine, &bad, &result, NULL) == SA_ERR_BAD_ARGUMENT,
          "NaN temperature should be refused");

    /* NaN, infinity, or a lambda*P too large for a float are refused, not computed on. */
    const float nan_w[3] = {NAN, 1.0f, 1.0f};
    SaProblem not_a_number = {3, 1, 1.0, nan_w, P};
    CHECK(sa_scalar_solve(&not_a_number, &settings, &result, NULL) == SA_ERR_OUT_OF_RANGE,
          "NaN reward should be out of range");
    const float inf_w[3] = {INFINITY, 1.0f, 1.0f};
    SaProblem infinite = {3, 1, 1.0, inf_w, P};
    CHECK(sa_scalar_solve(&infinite, &settings, &result, NULL) == SA_ERR_OUT_OF_RANGE,
          "infinite reward should be out of range");
    SaProblem huge_lambda = {3, 1, 1e300, w, P};
    CHECK(sa_scalar_solve(&huge_lambda, &settings, &result, NULL) == SA_ERR_OUT_OF_RANGE,
          "lambda*P too large for a float should be out of range");

    /* Equal temperatures (a constant-temperature run) and a single sweep are allowed. */
    SaSettings flat = settings;
    flat.t0 = flat.t1 = 0.5;
    CHECK(sa_scalar_solve(&fine, &flat, &result, NULL) == SA_OK, "constant temperature failed");
    SaSettings one_sweep = settings;
    one_sweep.sweeps = 1;
    CHECK(sa_scalar_solve(&fine, &one_sweep, &result, NULL) == SA_OK, "one sweep failed");

    /* A temperature too large for a float is refused. */
    SaSettings scorching = settings;
    scorching.t0 = 1e300;
    CHECK(sa_scalar_solve(&fine, &scorching, &result, NULL) == SA_ERR_OUT_OF_RANGE,
          "huge temperature should be out of range");
}

int main(void)
{
    test_philox_known_answers();
    test_toy_example();
    test_matches_brute_force();
    test_determinism_and_trace();
    test_long_run_consistency();
    test_temperature_controls_acceptance();
    test_edge_cases();

    if (failures == 0) {
        printf("\nall checks passed\n");
        return 0;
    }
    printf("\n%d check(s) failed\n", failures);
    return 1;
}
