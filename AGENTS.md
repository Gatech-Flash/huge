# Repository Guidelines

## Project Structure & Module Organization

`huge` provides R and Python interfaces for high-dimensional graphical modeling over a shared C++17 core.

- `R/`: R API; `src/`: native core and Rcpp adapters.
- `python-package/pyhuge/`: Python API; `python-package/cpp/`: mirrored core and pybind11 bindings.
- `tests/testthat/` and `python-package/tests/`: R and Python suites.
- `man/`, `vignettes/`, and `python-package/docs/`: reference documentation and guides.
- `data/` and `python-package/pyhuge/data/`: bundled datasets; `benchmark/`: performance experiments; `tools/`: validation scripts.

## Build, Test, and Development Commands

Native builds require a C++17 compiler and LP64 BLAS; OpenMP is optional. Install R dependencies from `DESCRIPTION` and use Python 3.9+.

From the repository root:

- `R CMD INSTALL .`: compile and install the local R package.
- `Rscript -e 'testthat::test_dir("tests/testthat")'`: run R tests after installation, with `testthat` installed.
- `R CMD build --no-build-vignettes .`: create an R source archive.
- `R CMD check --no-manual --no-vignettes huge_2.0.1.tar.gz`: check that archive; adjust the version to match `DESCRIPTION`.
- `cmake -S . -B build && cmake --build build`: build the standalone core with CMake 3.18+.
- `sh tools/check_cmake_install.sh`: validate static/shared library installation and consumption.

From `python-package/`:

- `python -m pip install -e ".[dev]"`: install editable Python code and development dependencies.
- `pytest`: run Python tests.
- `mkdocs serve` / `mkdocs build --strict`: preview or validate documentation.

## Coding Style & Naming Conventions

Match surrounding code: Python and core C++ use four-space indentation; R tests use two spaces, while older R sources use tabs. Preserve dotted R names (`huge.glasso`), Python snake_case (`huge_glasso`), and PascalCase dataclasses. No dedicated formatter or linter is configured. Update roxygen comments and corresponding `man/` documentation when changing R APIs.

## Testing Guidelines

Use testthat files named `test-*.R` and pytest files named `test_*.py`. Add regression tests for public behavior changes, seed randomized inputs, and use explicit numerical tolerances. No coverage percentage is configured. Python/R parity tests require the local R package installed.

Keep the three core mirror pairs byte-identical; run `sh tools/check_core_mirrors.sh` and rebuild both packages after core edits.

## Commit & Pull Request Guidelines

Git history is unavailable in this source snapshot, so commit conventions cannot be verified. Use concise, imperative subjects. PRs should describe behavior changes, link relevant issues, report validation and skipped tests, and update `NEWS.md` or `python-package/CHANGELOG.md` as applicable. Include before/after images for plotting changes.
