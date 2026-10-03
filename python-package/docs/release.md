# Release Process

## 1. Bump version

From `python-package`:

```bash
python scripts/bump_version.py 2.0.2
```

This updates:

- `pyproject.toml`
- `pyhuge/__init__.py`
- `CHANGELOG.md` (adds heading if missing)
- The shared CMake version and installed-consumer fixture
- Release examples and the version-alignment test fixture

For a coordinated R/Python release, also update the R `DESCRIPTION`,
`configure.ac`, generated `configure`, and `NEWS.md`. The Python script
leaves those R files to the R release step. Move the current development
entries into the chosen release in `NEWS.md` and `CHANGELOG.md`, preserving
older entries. Do not assign a publication date before publication.

## 2. Build and validate wheel/sdist

```bash
bash scripts/build_dist.sh
```

This runs:

- `python -m build`
- `python -m twine check dist/*`

## 3. Prepare git tag

```bash
git add -p ..  # Review and stage all intended release changes.
git commit -m "pyhuge: release 2.0.2"
git tag pyhuge-v2.0.2
git push origin <branch> --tags
```

## 4. Publish via CI

Publishing workflow tag pattern:

- `pyhuge-v*`

Recommended dedicated workflow file for this package directory:

- `.github/workflows/python-package-release.yml`

## 5. Docs website

Docs source:

- `python-package/mkdocs.yml`

Recommended dedicated workflow file:

- `.github/workflows/python-package-docs.yml`
