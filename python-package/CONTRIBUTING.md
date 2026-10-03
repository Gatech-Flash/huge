# Contributing to pyhuge

## Local setup

```bash
cd python-package
pip install -e ".[dev]"
```

## Run tests

```bash
pytest
```

To run the complete native suite without an installed R package:

```bash
python -m pytest -ra --ignore=tests/test_native_vs_r_parity.py \
  --ignore=tests/test_strict_r_parity.py \
  --ignore=tests/test_streaming_r_parity.py
```

CI discovers all non-R tests on each supported Python version and with
NumPy/SciPy 1.23.0/1.9.0 and 1.23.5/1.9.3. Keep new regression tests under
`tests/` so these jobs collect them. Plotting tests require the `viz` extra; R parity tests
require the current R package installed locally.

## Run docs locally

```bash
mkdocs serve
```

## Validate docs build

```bash
mkdocs build --strict
```

## Build wheel/sdist

```bash
bash scripts/build_dist.sh
```

Validate release artifacts after installing them in a fresh environment,
from a directory outside this checkout. Use `python -I` for smoke checks,
verify `pyhuge.__file__` and `pyhuge._native_core.__file__` point to the
installation, and run `pyhuge-doctor`. When testing an installed wheel,
copy the runtime tests outside the checkout: the repository's `conftest.py`
intentionally adds the source package to `sys.path`.

## Bump version and prepare release

```bash
python scripts/bump_version.py 2.0.2
bash scripts/release.sh 2.0.2
```

## Code principles

- Keep public API aligned with huge-style semantics.
- Preserve backward-compatible dataclass fields when possible.
- Prefer sparse graph outputs (`scipy.sparse`) for path objects.
- Add tests for every externally visible behavior change.
