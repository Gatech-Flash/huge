# Performance Notes

Measure the complete workflow on representative inputs. Native solver time,
correlation preprocessing, screening, sparse-output conversion, and repeated
model-selection fits can each dominate different workloads. A faster inner
solver does not necessarily produce the same improvement in the public API.

## Practical guidance

- Use `ct` for fast threshold-style path baselines.
- Use `mb` or `glasso` when selection quality matters more than raw speed.
- `stars` is slower than `ric`/`ebic` because it resamples repeatedly.
  `n_jobs > 1` runs subsample fits concurrently, but each native fit may
  already use OpenMP and threaded BLAS. Benchmark on the target runtime;
  `n_jobs=1` is usually the safest choice for an OpenMP-enabled build and
  avoids nested oversubscription.
- `ric` uses BLAS products over panels of up to 128 columns for large rotated
  cross-products. Ordinary inputs use approximately `8 * d * 128` scratch bytes
  per active worker. Small dimensions and ambiguous numerical-zero cases use
  the original dense calculation; budget up to `8 * d * d` bytes per worker
  for that fallback. Frontend preprocessing also retains quadratic buffers.
- Reuse transformed data from `huge_npn(...)` when running multiple methods.

## Multicore builds

The core parallelizes per-column solvers with OpenMP when the extension is
built with it (automatic when a toolkit is present; on macOS install
Homebrew's `libomp` first — see [Installation](installation.md)). Compare serial
and parallel execution with the same BLAS thread budget. Speedup depends on
dimension, active-set sizes, memory bandwidth, and how much of a TIGER path can
be certified; it is not a fixed multiplier.

## Memory and path length

Glasso and TIGER return dense precision matrices, requiring
`8 * p * d * d` bytes for a retained path of `p = len(fit.lambda_path)` points. Glasso
`cov_output=True` adds an equally large covariance path. Each returned precision
or covariance matrix now owns a separate native allocation, avoiding a copy into
a shared NumPy path cube. Retaining one `opt_icov`, `opt_cov`, or path matrix after
discarding the fit retains only that matrix's buffer.

TIGER allocates dense precision matrices after determining the common certified
path prefix. Automatic paths can return fewer points than requested, so discarded
points do not allocate dense precision matrices. During fitting, TIGER retains
inverse-variance scalars for the requested path and sparse coefficients; these,
the correlation matrix, solver scratch, and graph buffers still contribute to
peak memory.

These matrices are writable, column-major NumPy arrays. Use
`np.ascontiguousarray(matrix)` if a downstream consumer specifically requires
row-major storage. The result remains a list of arrays; the default private
native bindings retain their existing dense cube output. Use an appropriate
path length and measure resident memory separately from elapsed time.

Both Python and R nonparanormal inference stream their variance calculations using
`O(n*d + d*d)` working memory. Arithmetic remains expensive,
`O(n*n*d*d + n*d*d*d)`. This avoids constructing a `d*d`-by-`d*d`
variance matrix, but large sample sizes and dimensions still require substantial
computation.

ROC evaluation preserves sparse graph storage. Its work then depends on stored
edges rather than scanning every absent edge; mixed paths also avoid densifying
their sparse members. Dense inputs still require scanning their entries. R and
Python retain their existing counting conventions: R counts both directions,
while Python counts the strict upper triangle.

## The native core

`mb`, `glasso`, and `tiger` require the native extension
(`pyhuge._native_core`); it is not optional for the estimators. Check:

```python
import pyhuge
print(pyhuge.test()["native_extension"])
```

## Benchmark pattern

```python
import time
import numpy as np
from pyhuge import huge

x = np.random.default_rng(0).normal(size=(300, 100))
t0 = time.perf_counter()
fit = huge(x, method="mb", nlambda=10, verbose=False)
print("sec", time.perf_counter() - t0, "path", len(fit.path))
```

## Native vs R parity report

A dedicated script produces reproducible parity metrics against local R `huge`
(when available):

```bash
cd python-package
python scripts/r_parity_report.py --out parity_report.json
```

Current behavior:

- `ct + stars` parity is evaluated by default.
- `glasso + ebic` parity is evaluated via the native C++ backend.

Use the JSON output to track drift after solver or selection changes.
