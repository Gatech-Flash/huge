# Troubleshooting

## `PyHugeError: rpy2 is required`

Install dependency:

```bash
pip install rpy2
```

## `R package huge is not installed`

Install in R:

```bash
R -q -e 'install.packages("huge", repos="https://cloud.r-project.org")'
```

If using a custom R library path:

```bash
R_LIBS_USER=/path/to/Rlib R -q -e 'install.packages("huge", repos="https://cloud.r-project.org")'
```

## Python can import `pyhuge` but call fails to find R package

Your Python process may be linked to a different R library environment.
Check:

```bash
R -q -e 'print(.libPaths()); packageVersion("huge")'
```

Then ensure the same environment variables are visible to Python process
(e.g. `R_LIBS_USER`).

## `rpy2` fails with incompatible architecture (arm64/x86_64)

If you see errors like `incompatible architecture (have 'arm64', need 'x86_64')`,
your Python and R binaries are mismatched.

Check:

```bash
python -c 'import platform; print(platform.machine())'
R -q -e 'cat(R.version$arch, "\n")'
```

Use a Python interpreter matching your R build (for Apple Silicon R, use arm64 Python, e.g. `/opt/homebrew/bin/python3`).

## Sparse outputs are not in expected format

`pyhuge` converts graph matrices to `scipy.sparse.csc_matrix` when possible.
For dense numeric objects (`icov`, `cov`), output is `numpy.ndarray`.

## Matplotlib cache / fontconfig warnings

In restricted environments, matplotlib may warn about cache directory permissions.
Set a writable config directory:

```bash
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/mplconfig
mkdir -p "$MPLCONFIGDIR"
```

## `networkx is required for network plotting`

Install optional visualization dependency:

```bash
pip install networkx
```

or with extras:

```bash
pip install -e ".[viz]"
```

## Performance expectation

This wrapper runs full computation in R backend.
For very frequent small calls, bridge overhead may be non-negligible.
Batch calls where possible.
