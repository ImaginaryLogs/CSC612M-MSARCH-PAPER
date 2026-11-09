// module_2: the scalar simulated-annealing backend, exposed to Python as the `module_2`
// pybind11 extension. This file is only an adapter. It unpacks the Python Instance and
// ModuleConfig, calls the C core in src/sa_scalar.c, and packs the answer into the dict
// that module_interface.run_module() expects. The algorithm itself lives in sa_scalar.c.
// @see docs/modules/module-interface.md

#include "sa_scalar.h"

#include <chrono>
#include <cstdint>
#include <new>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

// Turns a failure status from the C core into the Python exception that fits it.
void throw_for_status(SaStatus status) {
    switch (status) {
    case SA_OK:
        return;
    case SA_ERR_NO_MEMORY:
        throw std::bad_alloc();
    default:  // bad argument, or values that are NaN, infinite, or too large for a float
        throw std::invalid_argument(std::string("scalar SA: ") + sa_status_message(status));
    }
}

py::dict solve(py::object instance, py::object config) {
    const std::string platform = config.attr("platform").cast<std::string>();
    if (platform != "scalar") {
        throw std::runtime_error("module_2 only implements platform='scalar'; '" + platform +
                                 "' is handled by another module");
    }

    // The problem. w and P are borrowed from the numpy arrays, not copied.
    const int n = instance.attr("n").cast<int>();
    const int k = instance.attr("k").cast<int>();
    FloatArray w(instance.attr("w"));
    FloatArray p(instance.attr("P"));
    if (w.ndim() != 1 || static_cast<int>(w.shape(0)) != n) {
        throw std::invalid_argument("instance.w must have shape (n,)");
    }
    if (p.ndim() != 2 || static_cast<int>(p.shape(0)) != n || static_cast<int>(p.shape(1)) != n) {
        throw std::invalid_argument("instance.P must have shape (n, n)");
    }
    SaProblem problem;
    problem.n = n;
    problem.k = k;
    problem.lambda = instance.attr("lam").cast<double>();
    problem.w = w.data();
    problem.P = p.data();

    // The settings. The temperatures are required: this backend has no default schedule.
    py::object t0 = config.attr("T0");
    py::object t1 = config.attr("T1");
    if (t0.is_none() || t1.is_none()) {
        throw std::invalid_argument(
            "platform 'scalar' needs ModuleConfig.T0 and ModuleConfig.T1 (start and end "
            "temperature, in energy units, T0 >= T1 > 0)");
    }
    SaSettings settings;
    settings.seed = config.attr("seed").cast<uint64_t>();
    settings.chains = config.attr("chains").cast<int>();
    settings.sweeps = config.attr("sweeps").cast<int>();
    settings.t0 = t0.cast<double>();
    settings.t1 = t1.cast<double>();
    settings.trace_steps = config.attr("trace_steps").cast<int>();

    // Output buffers owned here and filled by the C core.
    std::vector<int> best(static_cast<size_t>(k > 0 ? k : 1));
    SaResult result{};
    result.best = best.data();

    std::vector<uint8_t> trace;
    if (settings.trace_steps > 0 && settings.chains > 0) {
        trace.resize(static_cast<size_t>(settings.chains) * static_cast<size_t>(settings.trace_steps));
    }

    SaStatus status;
    const auto start = std::chrono::steady_clock::now();
    {
        py::gil_scoped_release unlocked;  // a long run should not freeze other Python threads
        status = sa_scalar_solve(&problem, &settings, &result, trace.empty() ? nullptr : trace.data());
    }
    const auto stop = std::chrono::steady_clock::now();
    throw_for_status(status);

    py::dict out;
    out["best"] = std::vector<int>(best.begin(), best.begin() + k);
    out["e_solver"] = result.e_best;
    out["ties"] = 0;  // annealing has no tie count, unlike exhaustive search
    out["t_kernel"] = std::chrono::duration<double>(stop - start).count();
    out["t_h2d"] = 0.0;  // no device transfer on the CPU
    out["t_d2h"] = 0.0;
    out["proposals"] = static_cast<long long>(result.proposals);
    out["accepts"] = static_cast<long long>(result.accepts);
    if (!trace.empty()) {
        // One row per chain, one column per traced step: 1 = accepted, 0 = rejected.
        out["trace"] = py::array_t<uint8_t>(
            {static_cast<py::ssize_t>(settings.chains), static_cast<py::ssize_t>(settings.trace_steps)},
            trace.data());
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(module_2, m) {
    m.doc() = "Native scalar simulated annealing (float), platform 'scalar'.";
    m.def("solve", &solve, py::arg("instance"), py::arg("config"),
          "Solve one Instance under one ModuleConfig; returns a dict matching "
          "RunResultsMessage's solver-facing fields. Needs config.T0 and config.T1.");
}
