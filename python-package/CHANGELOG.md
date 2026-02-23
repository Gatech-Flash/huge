# Changelog

## Unreleased

- Make `rpy2` a runtime optional dependency (`pyhuge[runtime]`) so `pip install pyhuge` is beginner-friendly.
- Add `pyhuge.doctor()` and CLI `pyhuge-doctor` for one-command runtime diagnostics with actionable suggestions.
- Expand installation/troubleshooting/docs to include copy-paste setup flow for non-expert users.
- Add unit tests for doctor helpers and include them in CI unit workflow coverage.

## 0.2.1

- Fix CI e2e runtime dependency installation by installing `Rcpp`, `RcppEigen`, and `igraph` in `R_LIBS_USER`.
- Make e2e pass criteria robust to pytest output formatting changes.
- Migrate rpy2 conversion calls to `get_conversion()` API to avoid deprecation warnings.
- Decouple docs build from `rpy2`/R runtime requirements and restrict docs deploy to the upstream repository.
- Update repository/documentation links and docs deployment target to `Gatech-Flash/huge`.


## 0.2.0

- Prepare first publish-ready `pyhuge` release line.
- Reframe repository and package documentation to cover both R (`huge`) and Python (`pyhuge`) variants.
- Add `docs/man/` function manual pages (one page per public API), analogous to R `man/` lookup style.
- Add docs website deployment workflow (`python-package-docs`) and integrate manual pages into MkDocs navigation.
- Add runtime readiness helper `pyhuge.test(require_runtime=...)` and corresponding public API tests.
- Add packaging metadata and manifest improvements for release/distribution (`MANIFEST.in`, project URLs, include-package-data).
- Keep Python test matrix and optional R-runtime e2e path aligned with current wrapper scope.


## 0.1.0

- Initial Python wrapper package `pyhuge`.
- Implemented APIs:
  - `huge`
  - `huge_mb`
  - `huge_glasso`
  - `huge_ct`
  - `huge_tiger`
  - `huge_select`
  - `huge_npn`
  - `huge_generator`
  - `huge_inference`
  - `huge_roc`
- Added utility APIs:
  - `huge_summary`
  - `huge_select_summary`
  - `huge_plot_sparsity`
  - `huge_plot_roc`
  - `huge_plot_graph_matrix`
  - `huge_plot_network`
  - `test` (runtime readiness probe)
- Added strict argument validation with clear `PyHugeError` messages.
- Expanded optional end-to-end tests (`tests/test_e2e_optional.py`) to cover:
  - method wrappers (`mb`, `ct`, `tiger`, `glasso`)
  - model selection (`ric`, `stars`, `ebic`)
  - `huge_npn`, `huge_generator`, `huge_roc`, `huge_inference`
  - summary helpers and network plotting
- Added CI split:
  - `unit` matrix job (pure Python tests)
  - `e2e-r-runtime` job (R+rpy2 integration tests)
- Added Python package release automation:
  - `scripts/bump_version.py`
  - `scripts/build_dist.sh`
  - `scripts/release.sh`
  - `.github/workflows/python-package-release.yml`
- Added Python docs website workflow:
  - `.github/workflows/python-package-docs.yml`
  - strict MkDocs build and GitHub Pages deployment
- Added result dataclasses:
  - `HugeResult`
  - `HugeSelectResult`
- Added MkDocs documentation set, including release process and architecture troubleshooting.
- Added docs pages for tutorials, performance notes, citation, and changelog index.
- Added `docs/man/` function manual pages (one page per public API function), analogous to R `man/` style lookup.
- Added packaging support files:
  - `MANIFEST.in`
  - Python package `.gitignore`
- Added examples and pytest test suite.
