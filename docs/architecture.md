
# File Structure

```.txt
/
├── CMakeLists.txt             # Builds the C++/CUDA shared library
├── pyproject.toml             # Python package configuration
├── scripts/                # simple build scripts
├── include/
│   ├── module.hpp          # only a skeleton
│   └── benchmark.hpp       # benchmark skeletons
├── src/
│   ├── module_1.cpp        # example of using module.hpp
│   └── benchmark.cpp       # benchmarking implementation 
└── python/
    ├── main.py
    ├── orchestrator.py
    ├── data_collectors.py
    ├── module_inteface.py
    ├── benchmark_analysis.py
    ├── report.ipynb
    └── validation.py
```

1. Main calls orchestrator, acts just as a cli.
2. Orchestrator is considered a kind of composite class that calls a performs the next steps in the grand scheme
    1. Data collectors collect drugs and stores them in data/ in whatever format. Maybe pkl.
    2. Orchestrator begins executing runs for each module in a loop.
    3. Module interface handles any io cpp / cuda interfacing.
    4. Orchestrator stores results, maybe has data cache facade class to group these functionality to store results.
    5. Validation checks results to see what is correct and not. This include a class of python code that brute forces the correct answers when flag is given
    6. Benchmark analyzes analyzes timing of different runs.
3. Report.ipynb just structures the data in a cohesive narrative.

Any cpp code are mades a cpp modular extension. Any related arguments that goes through between python modules should be wrapped in a data class. An example are the results coming from the cpp modules bundled in a data class called `RunResultsMessage` or something equivalent. Simplicity is key.