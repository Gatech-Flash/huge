# pyhuge Documentation

`pyhuge` is the Python interface to the R package `huge` for high-dimensional
undirected graph estimation and inference.

If you are new to both `huge` and `pyhuge`, follow this path:

1. [Start Here](getting-started.md)
2. [Installation](installation.md)
3. [Quick Start](quickstart.md)
4. [Beginner Workflow](beginner-workflow.md)
5. [Troubleshooting](troubleshooting.md)

## What pyhuge provides

- Core estimators: `huge`, `huge_mb`, `huge_glasso`, `huge_ct`, `huge_tiger`
- Model selection and preprocessing: `huge_select`, `huge_npn`
- Simulation and evaluation: `huge_generator`, `huge_roc`, `huge_inference`
- Plotting helpers: `huge_plot_sparsity`, `huge_plot_roc`,
  `huge_plot_graph_matrix`, `huge_plot_network`, `huge_plot`
- Dataset helper: `huge_stockdata`
- Runtime check helper: `pyhuge.test(require_runtime=False)`

## Runtime architecture

`pyhuge` is a bridge wrapper:

- Python code calls `rpy2`
- `rpy2` calls the R package `huge`
- R/C++ backend does the core computation

This means:

- You get mature `huge` behavior from Python.
- You must have a working local R runtime.
- Python and R architectures must match (`arm64` vs `x86_64`).

For architecture and environment details, see [Installation](installation.md)
and [Troubleshooting](troubleshooting.md).

## Documentation map

- Concept and first-use pages:
  - [Start Here](getting-started.md)
  - [Beginner Workflow](beginner-workflow.md)
  - [FAQ](faq.md)
- Task pages:
  - [Quick Start](quickstart.md)
  - [Tutorials](tutorials.md)
  - [Troubleshooting](troubleshooting.md)
- Reference pages:
  - [API Reference](api.md)
  - [Function Manual](man/index.md)
  - [Performance Notes](performance.md)
  - [Design Notes](design.md)
