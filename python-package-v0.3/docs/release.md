# Release Process

## 1. Bump version

From `python-package-v0.3`:

```bash
python scripts/bump_version.py 0.3.1
```

This updates:

- `pyproject.toml`
- `pyhuge/__init__.py`
- `CHANGELOG.md` (adds heading if missing)

## 2. Build and validate wheel/sdist

```bash
bash scripts/build_dist.sh
```

This runs:

- `python -m build`
- `python -m twine check dist/*`

## 3. Prepare git tag

```bash
git add pyproject.toml pyhuge/__init__.py CHANGELOG.md
git commit -m "pyhuge: release 0.3.1"
git tag pyhuge-v0.3.1
git push origin <branch> --tags
```

## 4. Publish via CI

Publishing workflow tag pattern:

- `pyhuge-v*`

Recommended dedicated workflow file for this package directory:

- `.github/workflows/python-package-v03-release.yml`

## 5. Docs website

Docs source:

- `python-package-v0.3/mkdocs.yml`

Recommended dedicated workflow file:

- `.github/workflows/python-package-v03-docs.yml`
