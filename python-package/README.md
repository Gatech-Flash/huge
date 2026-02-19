# pyhuge

`pyhuge` is a Python wrapper for the
[`huge`](https://cran.r-project.org/package=huge) R package.

The current release is a bridge wrapper:
- it keeps the existing R implementation unchanged;
- it calls `huge` from Python through `rpy2`;
- it converts output graph objects to `scipy.sparse` / `numpy`.

## Status

Implemented APIs:
- `pyhuge.huge`
- `pyhuge.huge_mb`
- `pyhuge.huge_glasso`
- `pyhuge.huge_ct`
- `pyhuge.huge_tiger`
- `pyhuge.huge_select`
- `pyhuge.huge_npn`
- `pyhuge.huge_generator`
- `pyhuge.huge_inference`
- `pyhuge.huge_roc`
- `pyhuge.huge_summary`
- `pyhuge.huge_select_summary`
- `pyhuge.huge_plot_sparsity`
- `pyhuge.huge_plot_roc`
- `pyhuge.huge_plot_graph_matrix`
- `pyhuge.huge_plot_network`

Documentation includes:
- installation and environment requirements;
- quick-start examples;
- API contract and return objects;
- architecture and known limitations;
- troubleshooting guide.

## Installation

Prerequisites:
- Python `>=3.9`
- R installed and available in PATH
- R package `huge` installed (with `Rcpp`, `RcppEigen`, `igraph`, etc.)
- Python and R should use the same architecture (both arm64 or both x86_64)

Install:

```bash
cd python-package
pip install -e .
```

Optional visualization extras:

```bash
cd python-package
pip install -e ".[viz]"
```

## Quick start

```python
import numpy as np
from pyhuge import huge, huge_select, huge_npn

rng = np.random.default_rng(1)
x = rng.normal(size=(100, 20))

# optional nonparanormal transform
x_npn = huge_npn(x, npn_func="shrinkage", verbose=False)

# graph path estimation
fit = huge(x_npn, method="mb", nlambda=6, verbose=False)
print(fit.method, fit.lambda_path.shape, len(fit.path))

# model selection
sel = huge_select(fit, criterion="ric", verbose=False)
print(sel.criterion, sel.opt_lambda, sel.opt_sparsity)
```

## Full docs

See the local docs folder:
- `docs/index.md`
- `docs/installation.md`
- `docs/quickstart.md`
- `docs/api.md`
- `docs/design.md`
- `docs/troubleshooting.md`

Example scripts:
- `examples/run_huge_mb.py`
- `examples/run_huge_glasso.py`
- `examples/run_method_wrappers.py`
- `examples/run_sim_inference_roc.py`
- `examples/run_summary_and_plot.py`
- `examples/run_network_plot.py`

You can also build docs with MkDocs:

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

Contributor guide:
- `CONTRIBUTING.md`
- `CHANGELOG.md`

Release automation:
- `scripts/bump_version.py`
- `scripts/build_dist.sh`
- `scripts/release.sh`
- GitHub Actions: `.github/workflows/python-package-release.yml`

## Notes

- `rpy2` is required at runtime.
- `fit.path` and `select.refit` are converted to SciPy sparse matrices when possible.
- `fit.raw` / `select.raw` keep the original R objects for advanced users.
- This is not yet a pure-Python/native-C wrapper; it depends on local R.
