// Worked example of module.hpp: the exact_cpu brute-force oracle, exposed to Python
// as the `module_1` pybind11 extension. Scalar/AVX2/CUDA simulated annealing land in
// their own src/module_*.cpp files later, bound the same way.
// @see docs/modules/module-interface.md

#include "module.hpp"

#include <chrono>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;

namespace combo {

double energy(const Instance& inst, const std::vector<int>& selection) {
    double reward = 0.0;
    for (int i : selection) {
        reward += inst.w[static_cast<size_t>(i)];
    }
    double penalty = 0.0;
    for (size_t a = 0; a < selection.size(); ++a) {
        for (size_t b = a + 1; b < selection.size(); ++b) {
            const size_t i = static_cast<size_t>(selection[a]);
            const size_t j = static_cast<size_t>(selection[b]);
            penalty += inst.P[i * static_cast<size_t>(inst.n) + j];
        }
    }
    return -reward + inst.lam * penalty;
}

Result solve_exact(const Instance& inst) {
    if (inst.n < inst.k) {
        throw std::invalid_argument(
            "infeasible instance: n=" + std::to_string(inst.n) +
            " < k=" + std::to_string(inst.k) + ", no k-combination exists");
    }
    const int n = inst.n;
    const int k = inst.k;

    std::vector<int> idx(static_cast<size_t>(k));
    for (int i = 0; i < k; ++i) {
        idx[static_cast<size_t>(i)] = i;
    }

    bool have_best = false;
    double best_e = 0.0;
    std::vector<int> best_sel;
    long long ties = 0;
    long long proposals = 0;

    // Standard in-place k-combination advance over [0, n): same enumeration order as
    // Python's itertools.combinations(range(n), k), just without allocating per step.
    while (true) {
        ++proposals;
        const double e = energy(inst, idx);
        if (!have_best || e < best_e) {
            best_e = e;
            best_sel = idx;
            ties = 1;
            have_best = true;
        } else if (e == best_e) {
            ++ties;
        }

        int i = k - 1;
        while (i >= 0 && idx[static_cast<size_t>(i)] == n - k + i) {
            --i;
        }
        if (i < 0) {
            break;
        }
        ++idx[static_cast<size_t>(i)];
        for (int j = i + 1; j < k; ++j) {
            idx[static_cast<size_t>(j)] = idx[static_cast<size_t>(j - 1)] + 1;
        }
    }

    Result res;
    res.best = best_sel;
    res.e_solver = best_e;
    res.ties = static_cast<int>(ties);
    res.proposals = proposals;
    res.accepts = 0;  // no accept/reject step -- brute force, not a Markov chain
    return res;
}

}  // namespace combo

namespace {

combo::Instance instance_from_py(const py::object& instance) {
    combo::Instance inst;
    inst.id = instance.attr("id").cast<std::string>();
    inst.n = instance.attr("n").cast<int>();
    inst.k = instance.attr("k").cast<int>();
    inst.lam = instance.attr("lam").cast<double>();

    py::array_t<float, py::array::c_style | py::array::forcecast> w(instance.attr("w"));
    py::array_t<float, py::array::c_style | py::array::forcecast> p(instance.attr("P"));

    if (w.ndim() != 1 || static_cast<int>(w.shape(0)) != inst.n) {
        throw std::invalid_argument("instance.w must have shape (n,)");
    }
    if (p.ndim() != 2 || static_cast<int>(p.shape(0)) != inst.n ||
        static_cast<int>(p.shape(1)) != inst.n) {
        throw std::invalid_argument("instance.P must have shape (n, n)");
    }

    inst.w.assign(w.data(), w.data() + inst.n);
    inst.P.assign(p.data(), p.data() + static_cast<size_t>(inst.n) * inst.n);
    return inst;
}

combo::Config config_from_py(const py::object& config) {
    combo::Config cfg;
    cfg.platform = config.attr("platform").cast<std::string>();
    cfg.seed = config.attr("seed").cast<uint64_t>();
    cfg.chains = config.attr("chains").cast<int>();
    cfg.sweeps = config.attr("sweeps").cast<int>();
    cfg.trace_steps = config.attr("trace_steps").cast<int>();

    py::object t0 = config.attr("T0");
    py::object t1 = config.attr("T1");
    cfg.T0 = t0.is_none() ? -1.0 : t0.cast<double>();
    cfg.T1 = t1.is_none() ? -1.0 : t1.cast<double>();
    return cfg;
}

py::dict result_to_py(const combo::Result& res) {
    py::dict out;
    out["best"] = res.best;
    out["e_solver"] = res.e_solver;
    out["ties"] = res.ties;
    out["t_kernel"] = res.t_kernel;
    out["t_h2d"] = res.t_h2d;
    out["t_d2h"] = res.t_d2h;
    out["proposals"] = res.proposals;
    out["accepts"] = res.accepts;
    return out;
}

py::dict solve(py::object instance, py::object config) {
    const combo::Instance inst = instance_from_py(instance);
    const combo::Config cfg = config_from_py(config);

    if (cfg.platform != "exact_cpu") {
        throw std::runtime_error(
            "module_1 only implements platform='exact_cpu' so far (the brute-force "
            "oracle); '" + cfg.platform +
            "' (scalar/avx2/cuda SA, exact_cuda) is not built yet");
    }

    const auto t0 = std::chrono::steady_clock::now();
    combo::Result res = combo::solve_exact(inst);
    const auto t1 = std::chrono::steady_clock::now();
    res.t_kernel = std::chrono::duration<double>(t1 - t0).count();

    return result_to_py(res);
}

}  // namespace

PYBIND11_MODULE(module_1, m) {
    m.doc() = "Native C++ solver module: exact_cpu brute-force oracle today, "
              "scalar/avx2/cuda simulated annealing to follow.";
    m.def("solve", &solve, py::arg("instance"), py::arg("config"),
          "Solve one Instance under one ModuleConfig; returns a dict matching "
          "RunResultsMessage's solver-facing fields.");
}
