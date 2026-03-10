# Changelog

## 0.8.0

- Bumped package version to 0.8.0, aligned with R package huge 1.5.

## 0.3.3

- Bumped package version to 0.3.3.
- Removed hard dependency and fallback path to `scikit-learn` for `mb`, `tiger`, and `glasso`.
- Enforced native C++ core (`pyhuge._native_core`) for `mb`, `tiger`, and `glasso`.
- Updated docs/tests to reflect native-core runtime expectations.


## 0.3.0

- Introduced native Python implementation (`pyhuge` 0.3 line).
- Added optional C++ acceleration module (`pyhuge._native_core`).
- Added packaged dataset loader: `huge_stockdata`.
- Added runtime diagnostics: `test()` compatibility status + `pyhuge-doctor` CLI.
- Added expanded plotting support including network graph view.
- Added docs/man skeleton parity with previous package structure.
- Added release/build helper scripts for wheel/sdist workflows.
