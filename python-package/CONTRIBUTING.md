# Contributing (Python Wrapper)

## Local setup

```bash
cd python-package
pip install -e ".[dev]"
```

## Run tests

```bash
pytest
```

## Run end-to-end tests with local R runtime

```bash
export R_LIBS_USER=/Users/tourzhao/Desktop/huge-master/.Rlib
R CMD INSTALL /Users/tourzhao/Desktop/huge-master
pytest tests/test_e2e_optional.py -rA
```

## Run docs locally

```bash
mkdocs serve
```

## Validate docs build (CI-equivalent)

```bash
mkdocs build --strict
```

## Build wheel/sdist for release

```bash
bash scripts/build_dist.sh
```

## Bump version and prepare release

```bash
python scripts/bump_version.py 0.2.0
bash scripts/release.sh 0.2.0
```

## CI workflows

- Tests: `.github/workflows/python-wrapper-tests.yml`
- Docs website: `.github/workflows/python-package-docs.yml`
- Package release: `.github/workflows/python-package-release.yml`

## Code principles

- Keep public API aligned with R package semantics.
- Preserve backward compatibility in result fields.
- Convert graph outputs to SciPy sparse matrices when possible.
- Keep `raw` R objects available for advanced users.
