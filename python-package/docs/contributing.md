# Contributing

## Local setup

```bash
cd python-package
pip install -e ".[dev]"
```

## Run tests

```bash
pytest
```

Run the complete native suite without an installed R package:

```bash
python -m pytest -ra --ignore=tests/test_native_vs_r_parity.py \
  --ignore=tests/test_strict_r_parity.py \
  --ignore=tests/test_streaming_r_parity.py
```

CI discovers all non-R tests on each supported Python version. The minimum
dependency jobs compile without build isolation using pybind11 2.12.0 on
Python 3.9/3.12, 2.13.0 on 3.13 and 3.0.0 on 3.14. They test NumPy 1 and 2,
including NumPy/SciPy 1.23.0/1.9.0 and 2.0.2/1.13.1. Add regression tests under
`tests/` so these jobs collect them. Install the `viz` extra to include plotting
tests. Releases reuse these checks and R parity tests.

## Run parity checks against R huge (optional)

Requires local R with package `huge` installed.

```bash
cd python-package
python scripts/r_parity_report.py --out parity_report.json
```

## Build docs

```bash
mkdocs build --strict
```

## Build release artifacts

```bash
bash scripts/build_dist.sh
```

Check installed artifacts from outside the checkout using `python -I` and
`pyhuge-doctor`. Verify that `pyhuge.__file__` and the native extension path
belong to the installation. To run runtime tests against a wheel, copy those
tests outside the checkout; the repository's `conftest.py` intentionally adds
the source package to `sys.path`.

## Bump version

```bash
python scripts/bump_version.py 2.0.2
bash scripts/release.sh 2.0.2
```

## Code principles

- Keep public API aligned with huge-style semantics.
- Keep dataclass fields backward-stable where possible.
- Add tests for every new public behavior.
- Document behavior changes in `CHANGELOG.md`.
