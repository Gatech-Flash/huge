# pyhuge Documentation

`pyhuge` provides Python access to the `huge` R package for high-dimensional
undirected graph estimation.

## What you get

- Python-callable APIs:
  - `huge(...)`
  - `huge_mb(...)`
  - `huge_glasso(...)`
  - `huge_ct(...)`
  - `huge_tiger(...)`
  - `huge_select(...)`
  - `huge_npn(...)`
  - `huge_generator(...)`
  - `huge_inference(...)`
  - `huge_roc(...)`
  - `huge_summary(...)`
  - `huge_select_summary(...)`
  - `huge_plot_sparsity(...)`
  - `huge_plot_roc(...)`
  - `huge_plot_graph_matrix(...)`
  - `huge_plot_network(...)`
- Typed return objects:
  - `HugeResult`
  - `HugeSelectResult`
- Sparse outputs converted to SciPy sparse matrices where appropriate.

## Current architecture

This version is an `rpy2` bridge wrapper:
- Python -> `rpy2` -> R `huge` package.
- Existing R/C++ implementation remains the source of truth.

See `design.md` for details and roadmap.
