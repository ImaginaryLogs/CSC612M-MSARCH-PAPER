/*
 * sa_avx2.c -- the C half of the AVX2 simulated annealing.
 *
 * The annealing itself (every proposal, accept test, commit and refresh) runs in
 * src/asm/sa_avx2.asm. This file does what happens once per run or once per chain:
 * argument checks, the memory layout the SIMD code needs, the temperature schedule,
 * the random starting team, and picking the best chain. It follows sa_scalar.c part by
 * part, and anything that can change a decision is computed exactly as it is there:
 *
 *   - the same float copies of w and lambda*P (lambda*P rounded once from double)
 *   - the same L table and the same float temperature per sweep
 *   - the same Philox counters: (seed; stream, chain, step)
 *   - the same float operations in the same order inside the kernels
 *
 * so for the same seed the two backends make identical accept/reject decisions.
 *
 * What is different is the layout. Every row of lambda*P, and w and g, is padded with
 * zeros to a multiple of 8 floats and starts on a 32-byte boundary, so one row is read
 * with aligned ymm loads. lambda*P is symmetric, so row j is also column j, and the g
 * update streams two contiguous rows. The gathers index lambda*P with a signed 32-bit
 * element number, so n * stride has to fit in an int32 (n up to 46,336, which would
 * already be an 8.6 GB matrix).
 *
 * Windows x64 only, like the assembly it drives.
 */
#ifndef _WIN32
#error "the AVX2 backend is Windows-only"
#endif

#include "sa_avx2.h"

#include <malloc.h> /* _aligned_malloc, _aligned_free */
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

enum {
    ACCEPT_TABLE_SIZE = 4096, /* must match sa_scalar.c */
    LANES = 8,                /* floats per ymm register */
    ALIGNMENT = 32,           /* bytes per ymm register */
    STREAM_START = 1          /* Philox counter word 3 for the starting team */
};

/* The assembly reads SaAvx2Chain by fixed offsets (C_* in sa_avx2.asm). A mismatch
   becomes a compile error here (negative array size) instead of a wrong answer. */
#define SA_LAYOUT_CHECK(field, offset) \
    typedef char sa_layout_check_##field[(offsetof(SaAvx2Chain, field) == (offset)) ? 1 : -1]
SA_LAYOUT_CHECK(lam_p, 0);
SA_LAYOUT_CHECK(w, 8);
SA_LAYOUT_CHECK(temps, 16);
SA_LAYOUT_CHECK(neg_ln_u, 24);
SA_LAYOUT_CHECK(trace, 32);
SA_LAYOUT_CHECK(g, 40);
SA_LAYOUT_CHECK(order, 48);
SA_LAYOUT_CHECK(best_team, 56);
SA_LAYOUT_CHECK(stride, 64);
SA_LAYOUT_CHECK(trace_steps, 72);
SA_LAYOUT_CHECK(seed, 80);
SA_LAYOUT_CHECK(accepts, 88);
SA_LAYOUT_CHECK(n, 96);
SA_LAYOUT_CHECK(k, 100);
SA_LAYOUT_CHECK(sweeps, 104);
SA_LAYOUT_CHECK(chain_id, 108);
SA_LAYOUT_CHECK(energy, 112);
SA_LAYOUT_CHECK(best_energy, 116);

/* 32-byte aligned memory (never a zero-byte request), freed with _aligned_free(). */
static void *alloc_aligned(size_t bytes)
{
    return _aligned_malloc(bytes > 0 ? bytes : ALIGNMENT, ALIGNMENT);
}

/* ------------------------------------------------------------------------- */
/* Model: the inputs in the layout the kernels want (sa_scalar.c's Model)     */
/* ------------------------------------------------------------------------- */

typedef struct {
    int n;
    int k;
    size_t stride;            /* floats per padded row: n rounded up to a multiple of 8 */
    const SaProblem *problem; /* the original inputs, for the exact final energy */
    float *w;                 /* stride rewards, zero padded, aligned */
    float *lam_p;             /* n rows of stride floats, lambda*P, zero padded, aligned */
    float *neg_ln_u;          /* the L table */
    float *temps;             /* one temperature per sweep */
} Model;

static void model_destroy(Model *m)
{
    _aligned_free(m->w);
    _aligned_free(m->lam_p);
    _aligned_free(m->neg_ln_u);
    free(m->temps);
    memset(m, 0, sizeof *m);
}

/* L[m] = -ln((m + 0.5) / 4096), as in sa_scalar.c. */
static void build_accept_table(float *table)
{
    for (int m = 0; m < ACCEPT_TABLE_SIZE; m++) {
        const double u = ((double)m + 0.5) / ACCEPT_TABLE_SIZE;
        table[m] = (float)(-log(u));
    }
}

static SaStatus model_create(const SaProblem *p, Model *m)
{
    memset(m, 0, sizeof *m);
    m->n = p->n;
    m->k = p->k;
    m->problem = p;

    const size_t n = (size_t)p->n;
    m->stride = (n + LANES - 1) / LANES * LANES;
    if (n > 0 && m->stride > (size_t)INT32_MAX / n) {
        return SA_ERR_BAD_ARGUMENT; /* i * stride + j would overflow a gather index */
    }

    m->w = alloc_aligned(m->stride * sizeof *m->w);
    m->lam_p = alloc_aligned(n * m->stride * sizeof *m->lam_p);
    m->neg_ln_u = alloc_aligned(ACCEPT_TABLE_SIZE * sizeof *m->neg_ln_u);
    if (m->w == NULL || m->lam_p == NULL || m->neg_ln_u == NULL) {
        model_destroy(m);
        return SA_ERR_NO_MEMORY;
    }
    /* Every real entry is written below; only the padding needs zeros. */
    memset(m->w, 0, m->stride * sizeof *m->w);
    for (size_t i = 0; i < n; i++) {
        memset(m->lam_p + i * m->stride + n, 0, (m->stride - n) * sizeof *m->lam_p);
    }

    for (size_t v = 0; v < n; v++) {
        m->w[v] = p->w[v];
        if (!isfinite(m->w[v])) {
            model_destroy(m);
            return SA_ERR_OUT_OF_RANGE;
        }
    }

    /* Upper triangle, mirrored: exactly symmetric with a zero diagonal, so row j is
       column j. lambda*P is computed in double and rounded to float once. */
    for (size_t i = 0; i < n; i++) {
        float *row_i = m->lam_p + i * m->stride;
        row_i[i] = 0.0f;
        for (size_t j = i + 1; j < n; j++) {
            const float value = (float)(p->lambda * (double)p->P[i * n + j]);
            if (!isfinite(value)) {
                model_destroy(m);
                return SA_ERR_OUT_OF_RANGE;
            }
            row_i[j] = value;
            m->lam_p[j * m->stride + i] = value;
        }
    }

    build_accept_table(m->neg_ln_u);
    return SA_OK;
}

/* T_s = T0 * (T1/T0)^(s/(S-1)), one float per sweep, as in sa_scalar.c. */
static SaStatus schedule_create(Model *m, double t0, double t1, int sweeps)
{
    m->temps = malloc((size_t)sweeps * sizeof *m->temps);
    if (m->temps == NULL) {
        return SA_ERR_NO_MEMORY;
    }
    for (int s = 0; s < sweeps; s++) {
        const double progress = (sweeps > 1) ? (double)s / (double)(sweeps - 1) : 0.0;
        const float temperature = (float)(t0 * pow(t1 / t0, progress));
        if (!isfinite(temperature)) {
            return SA_ERR_OUT_OF_RANGE;
        }
        m->temps[s] = temperature;
    }
    return SA_OK;
}

/* ------------------------------------------------------------------------- */
/* Chain buffers and start                                                    */
/* ------------------------------------------------------------------------- */

typedef struct {
    float *g;           /* stride floats, aligned */
    int32_t *order;     /* n slots */
    int32_t *best_team; /* k drugs */
} ChainBuffers;

static void chain_buffers_destroy(ChainBuffers *b)
{
    _aligned_free(b->g);
    free(b->order);
    free(b->best_team);
    memset(b, 0, sizeof *b);
}

static SaStatus chain_buffers_create(const Model *m, ChainBuffers *b)
{
    memset(b, 0, sizeof *b);
    b->g = alloc_aligned(m->stride * sizeof *b->g);
    b->order = malloc(((size_t)m->n + 1) * sizeof *b->order);
    b->best_team = malloc(((size_t)m->k + 1) * sizeof *b->best_team);
    if (b->g == NULL || b->order == NULL || b->best_team == NULL) {
        chain_buffers_destroy(b);
        return SA_ERR_NO_MEMORY;
    }
    memset(b->g, 0, m->stride * sizeof *b->g);
    return SA_OK;
}

static int random_below(uint32_t r, int bound)
{
    return (int)(((uint64_t)r * (uint64_t)bound) >> 32);
}

/* A random starting team, as chain_init() in sa_scalar.c: a partial Fisher-Yates
   shuffle driven by word 0 of Philox(seed; STREAM_START, chain, t). The draws do not
   depend on the shuffle, so they come 8 at a time from the vector Philox; the swaps
   themselves stay sequential. Then g and E are built by the vector refresh kernel. */
static void chain_init(const Model *m, uint64_t seed, SaAvx2Chain *c)
{
    for (int v = 0; v < m->n; v++) {
        c->order[v] = v;
    }

    const uint64_t chain_stream = (uint64_t)c->chain_id | ((uint64_t)STREAM_START << 32);
    for (int t0 = 0; t0 < m->k; t0 += LANES) {
        uint32_t words[4 * LANES];
        sa_avx2_philox_x8((uint64_t)t0, chain_stream, seed, words);
        const int lanes = (m->k - t0 < LANES) ? m->k - t0 : LANES;
        for (int lane = 0; lane < lanes; lane++) {
            const int t = t0 + lane;
            const int u = t + random_below(words[lane], m->n - t); /* word 0 of the lane */
            const int32_t tmp = c->order[t];
            c->order[t] = c->order[u];
            c->order[u] = tmp;
        }
    }

    sa_avx2_refresh(c); /* g and energy */
    c->best_energy = c->energy;
    memcpy(c->best_team, c->order, (size_t)m->k * sizeof *c->best_team);
}

/* Energy of a team from the caller's original inputs, summed in double, exactly as
   exact_team_energy() in sa_scalar.c. */
static double exact_team_energy(const Model *m, const int32_t *team)
{
    const SaProblem *p = m->problem;
    const size_t n = (size_t)m->n;
    double reward = 0.0;
    double penalty = 0.0;
    for (int a = 0; a < m->k; a++) {
        reward += (double)p->w[team[a]];
        for (int b = a + 1; b < m->k; b++) {
            penalty += (double)p->P[(size_t)team[a] * n + (size_t)team[b]];
        }
    }
    return -reward + p->lambda * penalty;
}

/* ------------------------------------------------------------------------- */
/* Public entry point                                                         */
/* ------------------------------------------------------------------------- */

int sa_avx2_supported(void)
{
    static int cached = -1; /* a benign race: every thread computes the same answer */
    if (cached < 0) {
        cached = sa_avx2_cpu_supported() ? 1 : 0;
    }
    return cached;
}

/* Same checks, in the same order, as sa_scalar.c. */
static SaStatus check_arguments(const SaProblem *p, const SaSettings *s,
                                const SaResult *r, const uint8_t *trace)
{
    if (p == NULL || s == NULL || r == NULL) {
        return SA_ERR_BAD_ARGUMENT;
    }
    if (p->n < 0 || p->k < 0 || p->k > p->n || !isfinite(p->lambda)) {
        return SA_ERR_BAD_ARGUMENT;
    }
    if (p->n > 0 && (p->w == NULL || p->P == NULL)) {
        return SA_ERR_BAD_ARGUMENT;
    }
    if (p->k > 0 && r->best == NULL) {
        return SA_ERR_BAD_ARGUMENT;
    }
    if (s->chains < 1 || s->sweeps < 1 || s->trace_steps < 0) {
        return SA_ERR_BAD_ARGUMENT;
    }
    if (s->trace_steps > 0 && trace == NULL) {
        return SA_ERR_BAD_ARGUMENT;
    }
    if (!(s->t1 > 0.0 && s->t0 >= s->t1 && isfinite(s->t0))) {
        return SA_ERR_BAD_ARGUMENT;
    }
    return SA_OK;
}

/* Runs every chain and keeps the best team; the lower-numbered chain wins ties. */
static SaStatus run_all_chains(Model *m, const SaSettings *s, SaResult *result,
                               uint8_t *trace)
{
    const bool can_swap = m->k > 0 && m->k < m->n;
    if (can_swap) {
        const SaStatus status = schedule_create(m, s->t0, s->t1, s->sweeps);
        if (status != SA_OK) {
            return status;
        }
    }
    if (trace != NULL) {
        memset(trace, 0, (size_t)s->chains * (size_t)s->trace_steps);
    }

    ChainBuffers buffers;
    const SaStatus status = chain_buffers_create(m, &buffers);
    if (status != SA_OK) {
        return status;
    }

    SaAvx2Chain c;
    memset(&c, 0, sizeof c);
    c.lam_p = m->lam_p;
    c.w = m->w;
    c.temps = m->temps;
    c.neg_ln_u = m->neg_ln_u;
    c.g = buffers.g;
    c.order = buffers.order;
    c.best_team = buffers.best_team;
    c.stride = (int64_t)m->stride;
    c.seed = s->seed;
    c.n = m->n;
    c.k = m->k;
    c.sweeps = s->sweeps;

    long long proposals = 0;
    long long accepts = 0;
    double overall_best = 0.0;
    for (int chain_id = 0; chain_id < s->chains; chain_id++) {
        c.chain_id = (uint32_t)chain_id;
        c.trace = (trace != NULL) ? trace + (size_t)chain_id * (size_t)s->trace_steps : NULL;
        c.trace_steps = (trace != NULL) ? s->trace_steps : 0;
        c.accepts = 0;

        chain_init(m, s->seed, &c);
        if (can_swap) {
            sa_avx2_run_chain(&c);
            proposals += (long long)s->sweeps * (long long)m->n;
            accepts += c.accepts;
        }

        const double chain_best = exact_team_energy(m, c.best_team);
        if (chain_id == 0 || chain_best < overall_best) {
            overall_best = chain_best;
            for (int t = 0; t < m->k; t++) {
                result->best[t] = c.best_team[t];
            }
        }
    }

    chain_buffers_destroy(&buffers);
    result->e_best = overall_best;
    result->proposals = proposals;
    result->accepts = accepts;
    return SA_OK;
}

SaStatus sa_avx2_solve(const SaProblem *problem, const SaSettings *settings,
                       SaResult *result, uint8_t *trace)
{
    SaStatus status = check_arguments(problem, settings, result, trace);
    if (status != SA_OK) {
        return status;
    }
    if (!sa_avx2_supported()) {
        return SA_ERR_UNSUPPORTED_CPU;
    }

    Model model;
    status = model_create(problem, &model);
    if (status != SA_OK) {
        return status;
    }
    status = run_all_chains(&model, settings, result, trace);
    model_destroy(&model);
    return status;
}
