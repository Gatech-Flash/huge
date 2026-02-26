# Performance Notes

`pyhuge` 0.3 runtime is dominated by numerical solver cost.

## Practical guidance

- Use `ct` for fast threshold-style path baselines.
- Use `mb` or `glasso` when selection quality matters more than raw speed.
- `stars` is slower than `ric`/`ebic` because it resamples repeatedly.
- Reuse transformed data from `huge_npn(...)` when running multiple methods.

## Optional acceleration

`pyhuge._native_core` accelerates selected kernels (e.g., threshold path/sparsity).

Check availability:

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
cd python-package-v0.3
python scripts/r_parity_report.py --out parity_report.json
```

Current behavior:

- `ct + stars` parity is evaluated by default.
- `glasso + ebic` parity is included when `scikit-learn` is installed.

Use the JSON output to track drift after solver or selection changes.
