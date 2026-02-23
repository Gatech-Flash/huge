# Installation

## TL;DR (recommended for beginners)

```bash
pip install pyhuge
pip install "pyhuge[runtime]"
R -q -e 'install.packages(c("huge","Rcpp","RcppEigen","igraph"), repos="https://cloud.r-project.org")'
pyhuge-doctor --require-runtime
```

If the last command prints all `OK`, you are ready.

## What is required

1. Python `>=3.9`
2. R installed and callable from shell (`R --version`)
3. R package `huge` visible to the same runtime used by Python
4. Matching Python/R architecture (`arm64` with `arm64`, or `x86_64` with `x86_64`)

## Install from PyPI

Base install (always safe):

```bash
pip install pyhuge
```

Install runtime bridge (`rpy2`) for actual model execution:

```bash
pip install "pyhuge[runtime]"
```

Optional visualization dependencies:

```bash
pip install "pyhuge[viz]"
```

## Install from source

```bash
git clone https://github.com/Gatech-Flash/huge.git
cd huge/python-package
pip install -e .
pip install -e ".[runtime]"
```

## Runtime diagnostics

Use the built-in doctor command:

```bash
pyhuge-doctor
```

Machine-readable output:

```bash
python -m pyhuge.doctor --json
```

Fail-fast mode for scripts/CI:

```bash
pyhuge-doctor --require-runtime
```

## Verify R package visibility

```bash
R -q -e 'print(.libPaths()); packageVersion("huge")'
```

If using a custom R library path, set it consistently for both install and runtime:

```bash
export R_LIBS_USER=/path/to/Rlib
R -q -e 'install.packages(c("huge","Rcpp","RcppEigen","igraph"), repos="https://cloud.r-project.org")'
pyhuge-doctor --require-runtime
```

## Architecture compatibility check

```bash
python -c 'import platform; print(platform.machine())'
R -q -e 'cat(R.version$arch, "\n")'
```

Both values should match.

## Optional docs tooling

```bash
pip install mkdocs mkdocs-material
cd python-package
mkdocs serve
```
