/*
 * sa_avx2.h -- simulated annealing for the AD drug-combination problem, with the hot
 * loop written in AVX2 assembly (src/asm/sa_avx2.asm, NASM). Windows x64 only.
 *
 * Same problem, same settings, same random numbers and the same float arithmetic in the
 * same order as sa_scalar.c, so for a given seed the two make identical accept/reject
 * decisions, return the identical best team, and report identical counters (paper test
 * T7). Only the speed differs. The SIMD work is:
 *
 *   - the g update after an accepted swap      8 floats per ymm, unrolled x4
 *   - Philox4x32-10 random numbers              8 steps per call, one step per lane
 *   - turning random words into swap slots      vpmuludq high halves, 8 lanes
 *   - the acceptance thresholds T * L[r >> 20]  vgatherdps from the L table
 *   - Delta E and the accept test               8 proposals evaluated at once with
 *                                               gathers; vmovmskps + tzcnt jump to the
 *                                               first accepted one
 *   - rebuilding g from the team (refresh)      8 drugs per ymm, unrolled x4
 *   - saving the best team                      vmovdqu / vpmaskmovd
 *
 * See docs/modules/sa-avx2.md for the design and the measured speedups.
 */
#ifndef SA_AVX2_H
#define SA_AVX2_H

#include <stdint.h>

#include "sa_scalar.h" /* SaProblem, SaSettings, SaResult, SaStatus, sa_status_message */

#ifdef __cplusplus
extern "C" {
#endif

/* Same contract as sa_scalar_solve(). Returns SA_ERR_UNSUPPORTED_CPU when the CPU or the
   operating system cannot run AVX2 code. */
SaStatus sa_avx2_solve(const SaProblem *problem, const SaSettings *settings,
                       SaResult *result, uint8_t *trace);

/* 1 if this CPU and OS can run the AVX2 kernels, 0 if not. */
int sa_avx2_supported(void);

/* ------------------------------------------------------------------------- */
/* The assembly kernels. Exposed for the tests and the benchmark; the solver  */
/* above is the normal way in.                                                */
/* ------------------------------------------------------------------------- */

/* Everything one chain's run needs. The assembly reads it by fixed byte offsets (the
   C_* defines in sa_avx2.asm), and sa_avx2.c checks those offsets at compile time, so
   change both together. Arrays marked "aligned" must start on a 32-byte boundary. */
typedef struct SaAvx2Chain {
    const float *lam_p;    /*   0  n rows of `stride` floats, lambda*P, aligned, zero padded */
    const float *w;        /*   8  `stride` rewards, aligned, zero padded */
    const float *temps;    /*  16  one temperature per sweep */
    const float *neg_ln_u; /*  24  the 4096-entry L table */
    uint8_t *trace;        /*  32  this chain's trace row, or NULL */
    float *g;              /*  40  `stride` floats, aligned */
    int32_t *order;        /*  48  n slots: [0, k) the team, [k, n) the outsiders */
    int32_t *best_team;    /*  56  k drugs */
    int64_t stride;        /*  64  floats per row, a multiple of 8 */
    int64_t trace_steps;   /*  72  steps to record; 0 when trace is NULL */
    uint64_t seed;         /*  80 */
    int64_t accepts;       /*  88  out: swaps this chain accepted */
    int32_t n;             /*  96 */
    int32_t k;             /* 100 */
    int32_t sweeps;        /* 104 */
    uint32_t chain_id;     /* 108 */
    float energy;          /* 112  in/out: energy of the current team */
    float best_energy;     /* 116  in/out */
} SaAvx2Chain;

/* CPUID + XGETBV: AVX and AVX2 present and ymm state enabled by the OS. */
int sa_avx2_cpu_supported(void);

/* Eight Philox4x32-10 blocks at once. Lane l uses the counter
   {lo(first + l), hi(first + l), lo(chain_stream), hi(chain_stream)} and the key
   {lo(seed), hi(seed)}; word w of lane l is written to out[w * 8 + l]. */
void sa_avx2_philox_x8(uint64_t first_index, uint64_t chain_stream, uint64_t seed,
                       uint32_t out[32]);

/* g[v] += row_j[v] - row_i[v] for v < count. count is a multiple of 8 and all three
   arrays are aligned. */
void sa_avx2_g_update(float *g, const float *row_j, const float *row_i, int64_t count);

/* Rebuilds chain->g from the team in chain->order[0, k) and sets chain->energy. */
void sa_avx2_refresh(SaAvx2Chain *chain);

/* Runs chain->sweeps sweeps of n proposals each, starting from the state in *chain
   (g, order, energy, best_energy and best_team already set). */
void sa_avx2_run_chain(SaAvx2Chain *chain);

#ifdef __cplusplus
}
#endif

#endif /* SA_AVX2_H */
