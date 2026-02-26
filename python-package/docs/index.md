# pyhuge 0.3 Documentation

`pyhuge` 0.3 is a native Python package for high-dimensional undirected graph
estimation and inference workflows inspired by `huge`.

If you are new, follow this order:

1. [Start Here](getting-started.md)
2. [Installation](installation.md)
3. [Quick Start](quickstart.md)
4. [Beginner Workflow](beginner-workflow.md)
5. [Troubleshooting](troubleshooting.md)

## Package capabilities

- Estimators: `huge`, `huge_mb`, `huge_glasso`, `huge_ct`, `huge_tiger`
- Selection and preprocessing: `huge_select`, `huge_npn`
- Simulation and inference: `huge_generator`, `huge_roc`, `huge_inference`
- Plot helpers: `huge_plot_sparsity`, `huge_plot_roc`,
  `huge_plot_graph_matrix`, `huge_plot_network`, `huge_plot`
- Dataset helper: `huge_stockdata`
- Environment probe: `pyhuge.test(require_runtime=False)`, `pyhuge-doctor`

## Runtime model

`pyhuge` 0.3 is native Python:

- Core logic in Python + NumPy/SciPy/scikit-learn
- Optional C++ acceleration module (`pyhuge._native_core`) via pybind11
- No R runtime dependency

## Documentation map

- Concept pages: [Start Here](getting-started.md), [FAQ](faq.md)
- Practical pages: [Quick Start](quickstart.md), [Tutorials](tutorials.md), [Troubleshooting](troubleshooting.md)
- Reference pages: [API Reference](api.md), [Function Manual](man/index.md), [Design](design.md), [Performance](performance.md)
