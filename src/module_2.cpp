// module_2: the scalar simulated-annealing backend, exposed to Python as the `module_2`
// pybind11 extension. This file is only an adapter: src/sa_binding.hpp unpacks the
// Python Instance and ModuleConfig, calls the C core in src/sa_scalar.c, and packs the
// answer into the dict that module_interface.run_module() expects. The algorithm itself
// lives in sa_scalar.c.
// @see docs/modules/module-interface.md

#include "sa_binding.hpp"

namespace py = pybind11;

PYBIND11_MODULE(module_2, m) {
    m.doc() = "Native scalar simulated annealing (float), platform 'scalar'.";
    m.def(
        "solve",
        [](py::object instance, py::object config) {
            return sa_binding::solve(instance, config, "module_2", "scalar", sa_scalar_solve);
        },
        py::arg("instance"), py::arg("config"),
        "Solve one Instance under one ModuleConfig; returns a dict matching "
        "RunResultsMessage's solver-facing fields. Needs config.T0 and config.T1.");
}
