# Release Process

## 1. Bump version

From `python-package`:

```bash
python scripts/bump_version.py 0.2.0
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
git commit -m "pyhuge: release 0.2.0"
git tag pyhuge-v0.2.0
git push origin <branch> --tags
```

## 4. Publish via CI

Publishing workflow is triggered by tag pattern:

- `pyhuge-v*`

Workflow file:
- `.github/workflows/python-package-release.yml`

It builds wheel/sdist and publishes using PyPI trusted publishing.

## 5. Documentation website

Docs build/deploy workflow:

- `.github/workflows/python-package-docs.yml`
- local docs build script: `scripts/build_docs.sh`

On push to `master`, it builds `python-package/mkdocs.yml` and deploys to GitHub Pages.
