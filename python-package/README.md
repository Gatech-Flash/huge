# pyhuge Python Package

[![Python wrapper tests](https://github.com/HMJiangGatech/huge/actions/workflows/python-wrapper-tests.yml/badge.svg)](https://github.com/HMJiangGatech/huge/actions/workflows/python-wrapper-tests.yml)
[![Python docs](https://github.com/HMJiangGatech/huge/actions/workflows/python-package-docs.yml/badge.svg)](https://github.com/HMJiangGatech/huge/actions/workflows/python-package-docs.yml)
[![Python package release](https://github.com/HMJiangGatech/huge/actions/workflows/python-package-release.yml/badge.svg)](https://github.com/HMJiangGatech/huge/actions/workflows/python-package-release.yml)

`pyhuge` is the Python wrapper for the
[`huge`](https://cran.r-project.org/package=huge) package:
High-dimensional Undirected Graph Estimation.

## Table of contents

- [Background](#background)
- [Directory structure](#directory-structure)
- [What this package provides](#what-this-package-provides)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Documentation and tutorials](#documentation-and-tutorials)
- [Performance notes](#performance-notes)
- [Implemented APIs](#implemented-apis)
- [For developer](#for-developer)
- [Citation](#citation)
- [References](#references)

## Background

`huge` is widely used for sparse undirected graphical model estimation in high dimensions.
`pyhuge` keeps the original R/C++ backend and exposes a Python API via `rpy2` so Python users can:

- call the same core methods (`mb`, `glasso`, `ct`, `tiger`);
- run model selection (`ric`, `stars`, `ebic`);
- access simulation, ROC, and inference helpers;
- work with `numpy` and `scipy.sparse` objects directly.

## Directory structure

The Python package directory is organized as:

- `pyhuge/`: Python wrapper source code
- `tests/`: unit/runtime/e2e tests
- `examples/`: runnable scripts
- `docs/`: MkDocs documentation pages
- `scripts/`: release and docs build scripts

## What this package provides

- Thin and stable bridge to mature `huge` backend.
- Argument validation with explicit `PyHugeError` messages.
- Typed result objects.
- Plot helpers including node-edge network visualization (`huge_plot_network`).
- Tests, documentation website workflow, and release automation scripts.

## Requirements

- Python `>=3.9`
- R installed and available in PATH
- R package `huge` installed (with `Rcpp`, `RcppEigen`, `igraph`, etc.)
- Python and R architecture must match (both `arm64` or both `x86_64`)

## Installation

Install from source:

```bash
git clone https://github.com/HMJiangGatech/huge.git
cd huge/python-package
pip install -e .
```

Install directly from GitHub:

```bash
pip install "git+https://github.com/HMJiangGatech/huge.git#subdirectory=python-package"
```

Optional extras:

```bash
pip install -e ".[viz]"      # matplotlib + networkx
pip install -e ".[release]"  # build + twine
pip install -e ".[dev]"      # test + docs + release deps
```

Runtime check:

```bash
python -c "import pyhuge; print(pyhuge.test())"
```

If `runtime=False`, verify `R_LIBS_USER`, architecture match, and `huge` visibility in R.

## Usage

```python
import numpy as np
from pyhuge import huge, huge_npn, huge_select

rng = np.random.default_rng(1)
x = rng.normal(size=(120, 30))
x_npn = huge_npn(x, npn_func="shrinkage", verbose=False)

fit = huge(x_npn, method="mb", nlambda=8, verbose=False)
sel = huge_select(fit, criterion="ric", verbose=False)
print(fit.method, len(fit.path), sel.opt_lambda, sel.opt_sparsity)
```

Network visualization:

```python
import matplotlib.pyplot as plt
from pyhuge import huge_plot_network

fig, ax = plt.subplots(figsize=(5, 5))
huge_plot_network(fit, index=-1, ax=ax, layout="spring")
plt.show()
```

## Documentation and tutorials

- Website: <https://hmjianggatech.github.io/huge/>
- Docs source: `python-package/docs`
- Tutorial scripts: `python-package/examples`

Build docs locally:

```bash
cd python-package
mkdocs serve
```

Run tutorials/examples:

```bash
cd python-package
python examples/run_huge_mb.py
python examples/run_summary_and_plot.py
python examples/run_network_plot.py
```

If you use custom R library path:

```bash
R_LIBS_USER=/Users/tourzhao/Desktop/huge-master/.Rlib \
python examples/run_huge_mb.py
```

## Performance notes

`pyhuge` runtime is dominated by R backend solve time for medium/large problems.
For many tiny repeated calls, Python-R bridge overhead can matter.
Benchmark backend and end-to-end workflows separately when tuning performance.

## Implemented APIs

- `huge`, `huge_mb`, `huge_glasso`, `huge_ct`, `huge_tiger`
- `huge_select`, `huge_npn`
- `huge_generator`, `huge_inference`, `huge_roc`
- `huge_summary`, `huge_select_summary`
- `huge_plot_sparsity`, `huge_plot_roc`, `huge_plot_graph_matrix`, `huge_plot_network`
- `test`

## For developer

```bash
cd python-package
pip install -e ".[dev]"
pytest
mkdocs build --strict
bash scripts/build_docs.sh
bash scripts/build_dist.sh
```

Optional e2e run with local R runtime:

```bash
export R_LIBS_USER=/Users/tourzhao/Desktop/huge-master/.Rlib
export PYHUGE_REQUIRE_RUNTIME=1
pytest tests/test_e2e_optional.py -rA
```

Workflows:

- `.github/workflows/python-wrapper-tests.yml`
- `.github/workflows/python-package-docs.yml`
- `.github/workflows/python-package-release.yml`

## Citation

If you use `huge` / `pyhuge` in research, cite:

```bibtex
@article{zhao2012huge,
  title   = {The huge Package for High-dimensional Undirected Graph Estimation in R},
  author  = {Zhao, Tuo and Liu, Han and Roeder, Kathryn and Lafferty, John and Wasserman, Larry},
  journal = {Journal of Machine Learning Research},
  volume  = {13},
  pages   = {1059--1062},
  year    = {2012}
}
```

## References

- T. Zhao, H. Liu, K. Roeder, J. Lafferty, and L. Wasserman,
  *The huge Package for High-dimensional Undirected Graph Estimation in R*,
  JMLR 13:1059-1062, 2012.
- Huge CRAN page: <https://cran.r-project.org/package=huge>
- Repository: <https://github.com/HMJiangGatech/huge>
