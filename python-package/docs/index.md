# pyhuge Documentation

`pyhuge` provides Python access to the `huge` R package for high-dimensional
undirected graph estimation.

## Quick links

- Installation: `installation.md`
- API: `api.md`
- Function manual (one page per function): `man/index.md`
- Tutorials: `tutorials.md`
- Performance: `performance.md`
- Troubleshooting: `troubleshooting.md`
- Release: `release.md`
- Documentation website deployment: `.github/workflows/python-package-docs.yml`

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
  - `test(...)`
- Typed return objects:
  - `HugeResult`
- `HugeSelectResult`
- Sparse outputs converted to SciPy sparse matrices where appropriate.
- A runtime probe helper:
  - `pyhuge.test(require_runtime=False)`

## Current architecture

This version is an `rpy2` bridge wrapper:
- Python -> `rpy2` -> R `huge` package.
- Existing R/C++ implementation remains the source of truth.

See `design.md` for details and roadmap.
