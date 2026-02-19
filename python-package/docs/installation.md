# Installation

## Prerequisites

1. Python `>=3.9`
2. R installed and callable from shell (`R --version`)
3. R package `huge` installed in the R library path visible to your environment
4. Python architecture must match R architecture (for `rpy2`)

Check R package visibility:

```bash
R -q -e 'packageVersion("huge")'
```

Check Python / R architecture compatibility:

```bash
python -c 'import platform; print(platform.machine())'
R -q -e 'cat(R.version$arch, "\n")'
```

Both should be the same architecture (for example both `arm64`).

## Install pyhuge

From repository root:

```bash
cd python-package
pip install -e .
```

Quick verification:

```bash
python -c "import pyhuge; print(pyhuge.test())"
```

Install optional visualization dependencies:

```bash
cd python-package
pip install -e ".[viz]"
```

Install optional release tooling:

```bash
cd python-package
pip install -e ".[release]"
```

## Optional docs tooling

```bash
pip install mkdocs mkdocs-material
```

Then run:

```bash
cd python-package
mkdocs serve
```
