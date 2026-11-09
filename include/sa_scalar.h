/*
 * sa_scalar.h -- scalar simulated annealing for the AD drug-combination problem.
 *
 * Picks k of n drugs to minimise
 *
 *     E(team) = -sum(w_i) + lambda * sum_{i<j in team} P_ij
 *
 * with swap moves (one drug out, one drug in), so the team always has exactly k
 * drugs. This is the reference implementation that the AVX2 and CUDA versions are
 * compared with. All the arithmetic is float. Float sums can round differently on
 * different platforms, so another platform is expected to agree with this one on
 * nearly every accept/reject decision.
*/
#ifndef SA_SCALAR_H
#define SA_SCALAR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int n;               /* number of candidate drugs */
    int k;               /* team size */
    double lambda;       /* weight on the pairwise overlap penalty */
    const float *w;      /* n rewards */
    const float *P;      /* n*n penalties, row-major, symmetric, zero diagonal */
} SaProblem;

typedef struct {
    uint64_t seed;
    int chains;          /* independent runs; the best of them is returned */
    int sweeps;          /* one sweep = n proposed swaps */
    double t0;           /* start temperature, in energy units; required, t0 >= t1 > 0 */
    double t1;           /* end temperature, in energy units; required */
    int trace_steps;     /* record the first trace_steps decisions of every chain */
} SaSettings;

typedef struct {
    int *best;           /* filled with the best team of all the SA chains */
    double e_best;       /* exact energy of that team (summed in double) */
    long long proposals; /* swaps proposed, summed over all chains */
    long long accepts;   /* swaps accepted, summed over all chains */
} SaResult;

typedef enum {
    SA_OK = 0,
    SA_ERR_BAD_ARGUMENT,
    SA_ERR_OUT_OF_RANGE, /* a value is NaN, infinite, or too large for a float */
    SA_ERR_NO_MEMORY
} SaStatus;

/* Runs every chain and fills result. trace may be NULL when trace_steps is 0. */
SaStatus sa_scalar_solve(const SaProblem *problem, const SaSettings *settings,
                         SaResult *result, uint8_t *trace);

const char *sa_status_message(SaStatus status);

/* Philox4x32-10, exposed so tests can check its output against the published
   known-answer values (Salmon et al., 2011). */
void sa_philox4x32_10(const uint32_t counter[4], const uint32_t key[2], uint32_t out[4]);

#ifdef __cplusplus
}
#endif

#endif /* SA_SCALAR_H */
