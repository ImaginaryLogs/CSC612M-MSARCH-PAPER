// module_2_avx: the AVX2 simulated-annealing backend, exposed to Python as the
// `module_2_avx` pybind11 extension (platform "avx2"). This file is only an adapter:
// src/sa_binding.hpp unpacks the Python Instance and ModuleConfig, calls the C driver in
// src/sa_avx2.c (whose hot loop is the NASM kernel in src/asm/sa_avx2.asm), and packs
// the answer into the dict that module_interface.run_module() expects.
//
// For the same seed it makes exactly the decisions module_2 ("scalar") makes, so the
// two can be compared step for step with ModuleConfig.trace_steps.
// @see docs/modules/sa-avx2.md

#include "sa_avx2.h"
#include "sa_binding.hpp"

namespace py = pybind11;

PYBIND11_MODULE(module_2_avx, m) {
    m.doc() = "Native AVX2 simulated annealing (float, NASM ymm kernel), platform 'avx2'.";
    m.def(
        "solve",
        [](py::object instance, py::object config) {
            return sa_binding::solve(instance, config, "module_2_avx", "avx2", sa_avx2_solve);
        },
        py::arg("instance"), py::arg("config"),
        "Solve one Instance under one ModuleConfig; returns a dict matching "
        "RunResultsMessage's solver-facing fields. Needs config.T0 and config.T1. Raises "
        "RuntimeError on a CPU without AVX2.");
    m.def(
        "supported", [] { return sa_avx2_supported() != 0; },
        "True if this CPU and OS can run the AVX2 kernel.");
}
