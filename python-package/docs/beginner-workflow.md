# Beginner Workflow

This is a complete beginner pipeline:

1. generate data
2. estimate graph path
3. select one graph
4. inspect summary
5. visualize result

## End-to-end example

```python
import numpy as np
import matplotlib.pyplot as plt

from pyhuge import (
    huge_generator,
    huge_npn,
    huge_glasso,
    huge_select,
    huge_summary,
    huge_select_summary,
    huge_plot_sparsity,
    huge_plot_network,
)

# 1) data generation (or replace with your own matrix X: shape n x d)
sim = huge_generator(n=150, d=40, graph="hub", g=4, verbose=False)
x = sim.data

# 2) optional nonparanormal transform
x_npn = huge_npn(x, npn_func="shrinkage", verbose=False)

# 3) graph path estimation
fit = huge_glasso(x_npn, nlambda=12, lambda_min_ratio=0.05, verbose=False)

# 4) model selection
sel = huge_select(fit, criterion="ebic", ebic_gamma=0.5, verbose=False)

# 5) summaries
print(huge_summary(fit))
print(huge_select_summary(sel))

# 6) plots
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
huge_plot_sparsity(fit, ax=axes[0])
huge_plot_network(fit, index=-1, ax=axes[1], layout="spring")
fig.tight_layout()
plt.show()
```

## What to change for your own data

- Replace `x = sim.data` with your matrix.
- Ensure your matrix is numeric, 2D, and finite.
- Shape must be `(n_samples, n_features)`.
- Start with `method="mb"` or `method="glasso"` first.

## Suggested defaults for new users

- Estimation:
  - `huge(..., method="mb", nlambda=8~12)`
  - or `huge_glasso(..., nlambda=8~12)`
- Selection:
  - `criterion="ric"` for fast checks
  - `criterion="ebic"` for glasso-based workflows
- Plot:
  - `huge_plot_sparsity` to inspect path trend
  - `huge_plot_network(..., index=-1)` for one network view

## If your run fails

Run:

```bash
python -c "import pyhuge; print(pyhuge.test())"
```

Then follow [Troubleshooting](troubleshooting.md).
