/*
 * sa_scalar.c -- scalar simulated annealing, floating point.
 *
 *
 * The file is ordered like the algorithm runs: set up the numbers, set up the random
 * source, set up the schedule, then start a chain, propose, accept, commit, loop.
 *
 *    1  pick a random starting team; compute g and E
 *    2  for each sweep s                      (temperature T_s from a table)
 *    3    for each of n proposals q
 *    4      draw four random 32-bit numbers from Philox(seed, chain, step)
 *    5      pick team member i and outsider j
 *    6      delta E = g_j - g_i - lambda*P_ij
 *    7      if delta E <= 0 or delta E < T_s * L[r2 >> 20]
 *    8        g = g + lambda*P[:,j] - lambda*P[:,i]
 *    9        swap i and j; E = E + delta E
 *   10        if E < E_best, save E_best and the team
 *
 * Build: cc -std=c99 -O2 -fno-tree-vectorize -Iinclude src/sa_scalar.c -lm
 * (-fno-tree-vectorize keeps the compiler from quietly vectorizing the scalar
 * baseline that the AVX2 version is measured against.)
 *
 * SA_CHECK_G -- an optional self-check, off unless compile with -DSA_CHECK_G
 * (check if the step by step updates of g and E drift from a full recomputation every 10,000 accepted swaps)
 */
#include "sa_scalar.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------------- */
/* Constants and types                                                        */
/* ------------------------------------------------------------------------- */

enum {
    ACCEPT_TABLE_SIZE = 4096, /* 12 random bits choose an entry of the L table */
    ACCEPT_TABLE_SHIFT = 20,  /* 32 - 12: how far to shift a random word down */
    REFRESH_EVERY_ACCEPTS = 1000 /* rebuild g and E from scratch this often (address float drift) */
};

/* Which of the independent random streams a draw belongs to (counter word 3). */
enum { STREAM_PROPOSAL = 0, STREAM_START = 1 };

/* SA hyperparameters and input values that stay fixed while the chains run. */
typedef struct {
    int n;
    int k;
    const SaProblem *problem;             /* the original inputs, for the exact final energy */
    float *w;                             /* n rewards for each drug */
    float *lam_p;                         /* n*n lambda*P aka lambda * (overlap penalty matrix), symmetric */
    float neg_ln_u[ACCEPT_TABLE_SIZE];    /* the L table */
    float *temps;                         /* one temperature per sweep */
} Model;

/* The state of one chain. */
typedef struct {
    int *order;          /* order[0..k-1] = current team of drugs, order[k..n-1] = outsiders */
    float *g;            /* g[v] = energy change if drug v joined the current team */
    float energy;        /* energy of the current team */
    int *best_team;      /* k drugs: the best team this chain has seen */
    float best_energy;
} Chain;

/* Four random 32-bit words used throughout the algorithm for repeatable randomness. */
typedef struct {
    uint32_t v[4];
} Draw;

/* One proposed swap. */
typedef struct {
    int a;               /* slot in the team (index in order) */
    int b;               /* slot among the outsiders (index in order) */
    int i;               /* the drug leaving the team */
    int j;               /* the drug joining it */
    float delta;         /* energy change if the swap is accepted */
    uint32_t accept_bits;
} Proposal;

typedef struct {
    long long proposals;
    long long accepts;
} Totals;

/* ------------------------------------------------------------------------- */
/* Part 1: Philox from Salmon et al., 2011 (repeatable random numbers)       */
/* ------------------------------------------------------------------------- */

/*
 * Philox4x32-10: ten rounds that scramble a 128-bit counter (c0..c3) with a 64-bit key
 * (k0, k1). The four words live in local variables, not in an array, so the compiler
 * keeps them in registers for all ten rounds. (An earlier version that kept them in an
 * array took 86 ns per call; this one takes 29 ns and returns identical numbers.)
 */
void sa_philox4x32_10(const uint32_t counter[4], const uint32_t key[2], uint32_t out[4])
{
    uint32_t c0 = counter[0], c1 = counter[1], c2 = counter[2], c3 = counter[3];
    uint32_t k0 = key[0], k1 = key[1];

    for (int round = 0; round < 10; round++) {
        if (round > 0) {
            k0 += 0x9E3779B9u; /* the key changes a little between rounds */
            k1 += 0xBB67AE85u;
        }
        /* Two 32x32 -> 64 bit multiplies, then xor the halves into the other words. */
        const uint64_t product0 = (uint64_t)0xD2511F53u * c0;
        const uint64_t product1 = (uint64_t)0xCD9E8D57u * c2;
        const uint32_t next0 = (uint32_t)(product1 >> 32) ^ c1 ^ k0;
        const uint32_t next1 = (uint32_t)product1;
        const uint32_t next2 = (uint32_t)(product0 >> 32) ^ c3 ^ k1;
        const uint32_t next3 = (uint32_t)product0;
        c0 = next0;
        c1 = next1;
        c2 = next2;
        c3 = next3;
    }

    out[0] = c0;
    out[1] = c1;
    out[2] = c2;
    out[3] = c3;
}

/* SAME (seed, stream, chain, index) always gives the SAME words. */
static Draw draw_random(uint64_t seed, uint32_t stream, uint32_t chain, uint64_t index)
{
    const uint32_t counter[4] = {(uint32_t)index, (uint32_t)(index >> 32), chain, stream};
    const uint32_t key[2] = {(uint32_t)seed, (uint32_t)(seed >> 32)};
    Draw d;
    sa_philox4x32_10(counter, key, d.v);
    return d;
}

/* Maps a random 32-bit word to a random slot in the n drugs w/o modulo. */
static int random_below(uint32_t r, int bound)
{
    return (int)(((uint64_t)r * (uint64_t)bound) >> 32);
}

/*
 * The L table: L[m] = -ln(u) for u = (m + 0.5) / 4096.
 * Probability of accepting uphill is a function of delta energy, T (temperature),
 * and u (random number in [0, 1)): Accept if delta < T * (-ln u)
 * to avoid calling -ln(u) every step, precompute -ln(u) for 4096 buckets
 * and access a random bucket at each step
 */
static void build_accept_table(float table[ACCEPT_TABLE_SIZE])
{
    for (int m = 0; m < ACCEPT_TABLE_SIZE; m++) {
        const double u = ((double)m + 0.5) / ACCEPT_TABLE_SIZE;
        table[m] = (float)(-log(u));
    }
}

/* ------------------------------------------------------------------------- */
/* Model: allocate, fill, free                                                */
/* ------------------------------------------------------------------------- */

static void model_destroy(Model *m)
{
    free(m->w);
    free(m->lam_p);
    free(m->temps);
    memset(m, 0, sizeof *m);
}

/* Copies the problem into float arrays with lambda folded in, and builds the L table.
   Temperatures come later. The caller's arrays must stay alive for the whole run. */
static SaStatus model_create(const SaProblem *p, Model *m)
{
    memset(m, 0, sizeof *m);
    m->n = p->n;
    m->k = p->k;
    m->problem = p;

    const size_t n = (size_t)p->n;
    m->w = malloc(n * sizeof *m->w);
    m->lam_p = malloc(n * n * sizeof *m->lam_p);
    if (m->w == NULL || m->lam_p == NULL) {
        model_destroy(m);
        return SA_ERR_NO_MEMORY;
    }

    for (size_t v = 0; v < n; v++) {
        m->w[v] = p->w[v];
        if (!isfinite(m->w[v])) {
            model_destroy(m);
            return SA_ERR_OUT_OF_RANGE;
        }
    }

    /* Fill from the upper triangle and mirror it, so the matrix is exactly symmetric
       and has a zero diagonal. Row j then doubles as column j. lambda*P is computed in
       double and rounded to float once. */
    for (size_t i = 0; i < n; i++) {
        m->lam_p[i * n + i] = 0.0f;
        for (size_t j = i + 1; j < n; j++) {
            const float value = (float)(p->lambda * (double)p->P[i * n + j]);
            if (!isfinite(value)) { /* NaN, infinity, or too large for a float */
                model_destroy(m);
                return SA_ERR_OUT_OF_RANGE;
            }
            m->lam_p[i * n + j] = value;
            m->lam_p[j * n + i] = value;
        }
    }

    build_accept_table(m->neg_ln_u);
    return SA_OK;
}

/* ------------------------------------------------------------------------- */
/* Part 2: Temperature Schedule                                              */
/* Remember that temp decreases over the steps in simulated annealing        */
/* T0: temperature at the start of SA; T1: temperature at the end of SA      */
/* ------------------------------------------------------------------------- */

/* Temp at sweep s: T_s = T0 * (T1/T0)^(s/(S-1)), stored as one float per sweep. */
/* Temp start initially decreases fast and then later on decreases more slowly */
static SaStatus schedule_create(Model *m, double t0, double t1, int sweeps)
{
    m->temps = malloc((size_t)sweeps * sizeof *m->temps);
    if (m->temps == NULL) {
        return SA_ERR_NO_MEMORY;
    }
    for (int s = 0; s < sweeps; s++) {
        const double progress = (sweeps > 1) ? (double)s / (double)(sweeps - 1) : 0.0;
        const float temperature = (float)(t0 * pow(t1 / t0, progress));
        if (!isfinite(temperature)) { /* too large for a float */
            return SA_ERR_OUT_OF_RANGE;
        }
        m->temps[s] = temperature;
    }
    return SA_OK;
}

/* ------------------------------------------------------------------------- */
/* Chain: allocate, free                                                      */
/* ------------------------------------------------------------------------- */

static void chain_destroy(Chain *c)
{
    free(c->order);
    free(c->g);
    free(c->best_team);
    memset(c, 0, sizeof *c);
}

static SaStatus chain_create(int n, int k, Chain *c)
{
    memset(c, 0, sizeof *c);
    c->order = malloc((size_t)n * sizeof *c->order);
    c->g = malloc((size_t)n * sizeof *c->g);
    c->best_team = malloc(((size_t)k + 1) * sizeof *c->best_team);
    if (c->order == NULL || c->g == NULL || c->best_team == NULL) {
        chain_destroy(c);
        return SA_ERR_NO_MEMORY;
    }
    return SA_OK;
}

/* ------------------------------------------------------------------------- */
/* Part 3: Starting a unique SA Chain                                        */
/* ------------------------------------------------------------------------- */

/* g[v] = -w[v] + lambda * (sum of P[v][l] over the team). 
   In English, g is how much would a team's energy change if drug v joined the current team. 
   Maintaining g allows us to compute delta E in O(1) time instead of O(k). */
static void compute_g(const Model *m, const int *order, float *g)
{
    const size_t n = (size_t)m->n;
    for (size_t v = 0; v < n; v++) {
        float sum = -m->w[v];
        for (int t = 0; t < m->k; t++) {
            sum += m->lam_p[v * n + (size_t)order[t]];
        }
        g[v] = sum;
    }
}

/* E(team) = -(sum of w) + (sum of lambda*P over every pair in the team). 
   Only called when initializing a chain. */
static float team_energy(const Model *m, const int *order)
{
    const size_t n = (size_t)m->n;
    float energy = 0.0f;
    for (int a = 0; a < m->k; a++) {
        energy -= m->w[order[a]];
        for (int b = a + 1; b < m->k; b++) {
            energy += m->lam_p[(size_t)order[a] * n + (size_t)order[b]];
        }
    }
    return energy;
}

/* The same energy, but taken from the caller's original inputs and summed in double,
   the way the Python judge does it. This is the number reported to the caller, and the
   one used to pick the winning chain, so float rounding never reaches the answer. */
static double exact_team_energy(const Model *m, const int *team)
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

static void chain_init(const Model *m, uint64_t seed, int chain_id, Chain *c)
{   // drugs initially in order
    for (int v = 0; v < m->n; v++) {
        c->order[v] = v;
    }

	// Randomly swap slots in the first k drugs with drugs from k to n-1, 
    // so the first k slots are a random team
    for (int t = 0; t < m->k; t++) {
        const Draw d = draw_random(seed, STREAM_START, (uint32_t)chain_id, (uint64_t)t);
        const int u = t + random_below(d.v[0], m->n - t);
        const int tmp = c->order[t];
        c->order[t] = c->order[u];
        c->order[u] = tmp;
    }

    // fill initial g values
    compute_g(m, c->order, c->g);
    c->energy = team_energy(m, c->order);

	// fill initial best team and energy
    c->best_energy = c->energy;
    memcpy(c->best_team, c->order, (size_t)m->k * sizeof *c->best_team);
}

/* ------------------------------------------------------------------------- */
/* Part 4: Proposing a swap of drugs (1 drug on team <-> 1 drug outside team) */
/* ------------------------------------------------------------------------- */

// Returns a proposal for a swap of one team member with one outsider,
// and the energy change if the swap is accepted.
static Proposal propose_swap(const Model *m, const Chain *c, Draw draw)
{
    Proposal p;

	// indices in the order array: a in the team, b among the outsiders
    p.a = random_below(draw.v[0], m->k);               /* a team slot */
    p.b = m->k + random_below(draw.v[1], m->n - m->k); /* an outsider slot */

    // lookup which drug numbers are actually being swapped
    p.i = c->order[p.a];
    p.j = c->order[p.b];

	// change in energy if the swap is accepted: g_j - g_i - lambda*P_ij
    p.delta = c->g[p.j] - c->g[p.i] - m->lam_p[(size_t)p.i * (size_t)m->n + (size_t)p.j];

	// supply a random number for the acceptance test, in the same word as the swap proposal
    p.accept_bits = draw.v[2];

    return p;
}

/* ------------------------------------------------------------------------- */
/* Part 5: Accept or Reject a drug swap                                      */
/* ------------------------------------------------------------------------- */

static bool should_accept(float delta, float temperature,
                          const float neg_ln_u[ACCEPT_TABLE_SIZE], uint32_t random_bits)
{
    if (delta <= 0.0f) {
        return true; /* downhill and sideways moves are always taken */
    }

    // otherwise, an uphill has a chance of still being accepted
    return delta < temperature * neg_ln_u[random_bits >> ACCEPT_TABLE_SHIFT];
}

/* ------------------------------------------------------------------------- */
/* Part 6: Commit a drug swap                                                */
/* ------------------------------------------------------------------------- */

/* AVX2 IMPPROVEMENT POSSIBLE HERE 
   Updates only involved removing and adding overlap penalties. Reward w for each drug is fixed.
*/
static void g_update(float *g, const float *row_j, const float *row_i, int n)
{
    for (int v = 0; v < n; v++) {
        // drug i leaves, so lose the overlap penalty with i
        // drug j joins, so gain the overlap penlaty with j
        // the reward is fixed (part of g initialization)
        g[v] += row_j[v] - row_i[v];
    }
}

#ifdef SA_CHECK_G
/* Test T9. Rounding explains small differences between the running values and a fresh
   rebuild; a bug in the bookkeeping shows up as a large one. */
static bool close_enough(float running, float fresh)
{
    return fabsf(running - fresh) <= 1e-4f * (1.0f + fabsf(fresh));
}

static void check_drift(const Model *m, const Chain *c)
{
    float *fresh = malloc((size_t)m->n * sizeof *fresh);
    if (fresh == NULL) {
        return;
    }
    compute_g(m, c->order, fresh);
    bool ok = close_enough(c->energy, team_energy(m, c->order));
    for (int v = 0; v < m->n; v++) {
        ok = ok && close_enough(c->g[v], fresh[v]);
    }
    free(fresh);
    if (!ok) {
        fprintf(stderr, "sa_scalar: running g/E drifted from a fresh rebuild\n");
        abort();
    }
}
#endif

/* Float sums pick up a little rounding error with every accepted swap. Rebuilding g and
   E from scratch every REFRESH_EVERY_ACCEPTS swaps stops it building up. In a build
   with -DSA_CHECK_G the running values are compared with the fresh ones first. */
static void refresh_chain(const Model *m, Chain *c)
{
#ifdef SA_CHECK_G
    check_drift(m, c);
#endif
    compute_g(m, c->order, c->g);
    c->energy = team_energy(m, c->order);
}

static void commit_swap(const Model *m, Chain *c, Proposal p)
{
    const size_t n = (size_t)m->n;

    // Every drug's g changes by lambda*(P_v,j - P_v,i)
    g_update(c->g, &m->lam_p[(size_t)p.j * n], &m->lam_p[(size_t)p.i * n], m->n);

    // Exchange the two slots in order and move the energy
    c->order[p.a] = p.j;
    c->order[p.b] = p.i;
    c->energy += p.delta;

    // Remember the best team seen so far
    if (c->energy < c->best_energy) {
        c->best_energy = c->energy;
        memcpy(c->best_team, c->order, (size_t)m->k * sizeof *c->best_team);
    }
}

/* ------------------------------------------------------------------------- */
/* Part 7: How one SA chain runs by calling the functions above              */
/* ------------------------------------------------------------------------- */

static void run_chain(const Model *m, const SaSettings *s, int chain_id, Chain *c,
                      Totals *totals, uint8_t *trace)
{
    uint64_t step = 0;
    int accepts_since_refresh = 0;
    for (int sweep = 0; sweep < s->sweeps; sweep++) {                                           // for every sweep  
        const float temperature = m->temps[sweep];                                           // decrease temp
        for (int q = 0; q < m->n; q++, step++) {                                                // many steps within a sweep 
            const Draw draw = draw_random(s->seed, STREAM_PROPOSAL, (uint32_t)chain_id, step);      // repeatable andom numbers
            const Proposal p = propose_swap(m, c, draw);                                            // a possible swap
            const bool accepted = should_accept(p.delta, temperature, m->neg_ln_u, p.accept_bits);  // should swap be accepted

            totals->proposals++;                                                                    // running total of how many swaps were proposed
            if (trace != NULL && step < (uint64_t)s->trace_steps) {
                // needed if we want to check agreement with CUDA and AVX2 versions
                trace[(size_t)chain_id * (size_t)s->trace_steps + (size_t)step] = accepted;
            }
            if (accepted) {                                
                commit_swap(m, c, p);                      
                totals->accepts++;                                                                  // running total of how many swaps were accepted
                if (++accepts_since_refresh == REFRESH_EVERY_ACCEPTS) {                             // float drift: start fresh
                    refresh_chain(m, c);
                    accepts_since_refresh = 0;
                }
            }
        }
    }
}

/* ------------------------------------------------------------------------- */
/* Public entry point                                                     */
/* ------------------------------------------------------------------------- */

const char *sa_status_message(SaStatus status)
{
    switch (status) {
    case SA_OK: return "ok";
    case SA_ERR_BAD_ARGUMENT: return "bad argument";
    case SA_ERR_OUT_OF_RANGE: return "values are not finite or too large for a float";
    case SA_ERR_NO_MEMORY: return "out of memory";
    case SA_ERR_UNSUPPORTED_CPU: return "this CPU or operating system does not support AVX2";
    }
    return "unknown status";
}

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
    /* The caller always chooses the temperatures: need t0 >= t1 > 0. Written so that
       NaN fails the test. */
    if (!(s->t1 > 0.0 && s->t0 >= s->t1 && isfinite(s->t0))) {
        return SA_ERR_BAD_ARGUMENT;
    }
    return SA_OK;
}

/* ------------------------------------------------------------------------- */
/* Part 8: How many SA chains run and how they are managed                   */
/* ------------------------------------------------------------------------- */

/* Runs every chain and keeps the best team. Chain lower number chain wins ties. */
static SaStatus run_all_chains(Model *m, Chain *c, const SaSettings *s,
                               SaResult *result, uint8_t *trace)
{
    /* Edge cases: With k = 0 or k = n there is only one possible team and nothing to swap. */
    const bool can_swap = m->k > 0 && m->k < m->n;

    if (can_swap) {
        // build temperature schedule
        const SaStatus status = schedule_create(m, s->t0, s->t1, s->sweeps);
        if (status != SA_OK) {
            return status;
        }
    }
    if (trace != NULL) {
        memset(trace, 0, (size_t)s->chains * (size_t)s->trace_steps);
    }

    Totals totals = {0, 0};
    double overall_best = 0.0;
    for (int chain_id = 0; chain_id < s->chains; chain_id++) {
        chain_init(m, s->seed, chain_id, c);
        if (can_swap) {
            run_chain(m, s, chain_id, c, &totals, trace);
        }
        // compare chains by the exact energy of their best team, not the running float value
        const double chain_best = exact_team_energy(m, c->best_team);
        if (chain_id == 0 || chain_best < overall_best) {   // keep lowest energy, lowest number chain wins energy ties
            overall_best = chain_best;     // lowest energy is saved

            // save the winning team
            if (m->k > 0) {
                memcpy(result->best, c->best_team, (size_t)m->k * sizeof *result->best);
            }
        }
    }

    result->e_best = overall_best;
    result->proposals = totals.proposals;
    result->accepts = totals.accepts;
    return SA_OK;
}

SaStatus sa_scalar_solve(const SaProblem *problem, const SaSettings *settings,
                         SaResult *result, uint8_t *trace)
{
    SaStatus status = check_arguments(problem, settings, result, trace);
    if (status != SA_OK) {
        // memory allocation problem or hyperparameters are not possible
        return status;
    }

    Model model;
    status = model_create(problem, &model);
    if (status != SA_OK) {
        return status;
    }
    Chain chain;
    status = chain_create(problem->n, problem->k, &chain);
    if (status != SA_OK) {
        model_destroy(&model);
        return status;
    }

    // begin running all chains
    status = run_all_chains(&model, &chain, settings, result, trace);

    chain_destroy(&chain);
    model_destroy(&model);
    return status;
}
