# Troubleshooting

## First diagnostic step (always run this first)

Run:

```bash
python -c "import pyhuge; print(pyhuge.test())"
```

If this already prints `runtime=True`, your base runtime is healthy.

## Fast environment snapshot

```bash
python -c 'import sys,platform; print("python", sys.executable); print("machine", platform.machine())'
R -q -e 'cat("R arch:", R.version$arch, "\n"); print(.libPaths()); suppressWarnings(print(packageVersion("huge")))'
```

This captures the most common mismatch causes in one pass.

## Symptom -> likely cause -> fix

## `PyHugeError: rpy2 is required`

Cause: `rpy2` is not installed in the active Python environment.

Fix:

```bash
pip install rpy2
```

## `R package huge is not installed in current R library paths`

Cause: R package `huge` is missing from the R library path visible to the
Python process.

Fix:

```bash
R -q -e 'install.packages("huge", repos="https://cloud.r-project.org")'
```

If you use a custom R library path:

```bash
R_LIBS_USER=/path/to/Rlib R -q -e 'install.packages("huge", repos="https://cloud.r-project.org")'
R_LIBS_USER=/path/to/Rlib python your_script.py
```

## Python imports `pyhuge`, but runtime calls still fail

Cause: `python` command and `pip` command point to different environments.

Fix:

```bash
which python
python -m pip show pyhuge
python -c "import pyhuge; print(pyhuge.__file__)"
```

Use the same `python -m pip ...` pair consistently.

## `rpy2` or `libR` architecture mismatch

Typical error text:

```bash
incompatible architecture (have 'arm64', need 'x86_64')
```

Cause: Python and R architectures differ.

Fix:

```bash
python -c 'import platform; print(platform.machine())'
R -q -e 'cat(R.version$arch, "\n")'
```

Use a Python interpreter matching your R build.
On Apple Silicon with arm64 R, prefer `/opt/homebrew/bin/python3`.

## Plotting warnings in restricted environments

Set writable cache + headless backend:

```bash
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/mplconfig
export XDG_CACHE_HOME=/tmp/xdg-cache
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
```

## `networkx is required for network plotting`

Cause: visualization extra is missing.

Fix:

```bash
pip install networkx matplotlib
```

or (source install):

```bash
pip install -e ".[viz]"
```

## `R_LIBS_USER` not visible in Python run

Run Python with the same variable:

```bash
R_LIBS_USER=/path/to/Rlib python your_script.py
```

## Support checklist for issue reports

Include:

- output of `python -c "import pyhuge; print(pyhuge.test())"`
- Python executable path
- `R.version$arch`
- `.libPaths()`
- full traceback
