# Changelog

## 0.3.3

- Updated PyPI long description wording to remove stale "0.3 Native Line" phrasing.
- Replaced README in-page anchor table of contents with absolute documentation links that work on PyPI.
- Normalized release command examples in docs to use `<major.minor.patch>` placeholders.


## 0.3.2

- Made C++ acceleration build optional at install time; falls back to pure Python if native build fails.
- Switched release workflow to publish source distributions (`sdist`) to avoid invalid non-manylinux wheel uploads.
- Kept runtime API behavior unchanged (`pyhuge.test()`, core estimators, selection, plotting).


## 0.3.1

- Renamed the native package tree to `python-package/` and removed legacy `python-package-v0.3/` references.
- Removed obsolete `python-wrapper-tests` CI workflow and standardized on `python-package-tests`.
- Fixed parity CI dependency bootstrap for R (`Rcpp`, `RcppEigen`, `igraph`) in isolated `R_LIBS_USER`.
- Updated root repository README to describe native `pyhuge` 0.3 installation and workflow paths.
- Added manual trigger support (`workflow_dispatch`) for `python-package-tests`.


## 0.3.0

- Introduced native Python implementation (`pyhuge` 0.3 line).
- Added optional C++ acceleration module (`pyhuge._native_core`).
- Added packaged dataset loader: `huge_stockdata`.
- Added runtime diagnostics: `test()` compatibility status + `pyhuge-doctor` CLI.
- Added expanded plotting support including network graph view.
- Added docs/man skeleton parity with previous package structure.
- Added release/build helper scripts for wheel/sdist workflows.
