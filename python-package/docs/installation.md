# Installation

## Recommended path (most users)

```bash
pip install pyhuge
```

Then run:

```bash
python -c "import pyhuge; print(pyhuge.test())"
```

If you see `runtime=True`, installation is ready.

## What must exist locally

`pyhuge` needs all of the following:

1. Python `>=3.9`
2. R installed and callable (`R --version`)
3. R package `huge` installed in the library path visible to your Python process
4. Matching architectures between Python and R (`arm64` with `arm64`, `x86_64` with `x86_64`)

## Install R package `huge`

```bash
R -q -e 'install.packages("huge", repos="https://cloud.r-project.org")'
R -q -e 'packageVersion("huge")'
```

If you use a custom R library path:

```bash
R_LIBS_USER=/path/to/Rlib R -q -e 'install.packages("huge", repos="https://cloud.r-project.org")'
R_LIBS_USER=/path/to/Rlib R -q -e 'print(.libPaths()); packageVersion("huge")'
```

## Architecture check (important)

```bash
python -c 'import platform; print(platform.machine())'
R -q -e 'cat(R.version$arch, "\n")'
```

These must match.

## Source install (for contributors)

```bash
git clone https://github.com/Gatech-Flash/huge.git
cd huge/python-package
pip install -e .
python -c "import pyhuge; print(pyhuge.test())"
```

Optional extras:

```bash
pip install -e ".[viz]"      # matplotlib + networkx
pip install -e ".[test]"     # pytest
pip install -e ".[docs]"     # mkdocs tooling
pip install -e ".[dev]"      # common contributor setup
```

## First command to run when something fails

```bash
python -c "import pyhuge; print(pyhuge.test())"
```

Then use [Troubleshooting](troubleshooting.md).
