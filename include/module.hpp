#pragma once
// The interface every native backend (exact_cpu, scalar, avx2, cuda, ...) implements.
// Mirrors python/module_interface.py's ModuleConfig/RunResultsMessage so the pybind11
// binding layer in each src/module_*.cpp has a thin, mechanical job: unpack the Python
// objects into these structs, call a solve_*() below, pack the Result back into a dict.
// @see docs/modules/module-interface.md

#include <cstdint>
#include <string>
#include <vector>

namespace combo {

// One problem instance: n drugs, choose k, reward w, pairwise overlap penalty P.
// Mirrors data_collectors.Instance.
struct Instance {
    std::string id;
    int n = 0;
    int k = 0;
    double lam = 0.0;
    std::vector<float> w;  // size n
    std::vector<float> P;  // size n*n, row-major, symmetric, zero diagonal
};

// Mirrors module_interface.ModuleConfig. T0/T1 < 0 means "use default schedule"
// (ModuleConfig's None), since every backend here only ever needs non-negative
// temperatures.
struct Config {
    std::string platform;
    uint64_t seed = 0;
    int chains = 1;
    int sweeps = 1000;
    double T0 = -1.0;
    double T1 = -1.0;
    int trace_steps = 0;
};

// Mirrors the solver-facing fields of RunResultsMessage (instance_id/platform/seed/
// chains/sweeps are added back in by the Python side, which already has them).
struct Result {
    std::vector<int> best;
    double e_solver = 0.0;
    int ties = 0;
    double t_kernel = 0.0;
    double t_h2d = 0.0;
    double t_d2h = 0.0;
    long long proposals = 0;
    long long accepts = 0;
};

// E(selection) = -sum(w_i) + lambda * sum_{i<j in selection} P_ij.
// Must stay numerically identical to validation.judge() -- it is what every backend
// here is scored against.
double energy(const Instance& inst, const std::vector<int>& selection);

// The exact_cpu backend: tries every k-combination of [0, n), keeps the best.
// O(C(n,k) * k^2). Throws std::invalid_argument if n < k (infeasible instance).
Result solve_exact(const Instance& inst);

}  // namespace combo
