# Quick Start

## Runtime check

```python
import pyhuge
print(pyhuge.test())
```

If you want strict enforcement:

```python
import pyhuge
pyhuge.test(require_runtime=True)
```

## Basic graph path estimation

```python
import numpy as np
from pyhuge import huge

rng = np.random.default_rng(123)
x = rng.normal(size=(120, 30))

fit = huge(
    x,
    method="mb",          # one of: mb, glasso, ct, tiger
    nlambda=8,
    lambda_min_ratio=0.1,
    verbose=False,
)

print(fit.method)
print(fit.lambda_path)
print(len(fit.path), fit.path[0].shape)
```

Equivalent method-specific shortcut:

```python
from pyhuge import huge_mb
fit = huge_mb(x, nlambda=8, lambda_min_ratio=0.1, verbose=False)
```

## Nonparanormal preprocessing + selection

```python
from pyhuge import huge_npn, huge_select

x_npn = huge_npn(x, npn_func="shrinkage", verbose=False)
fit = huge(x_npn, method="glasso", nlambda=10, verbose=False)
sel = huge_select(fit, criterion="ebic", ebic_gamma=0.5, verbose=False)

print(sel.opt_lambda, sel.opt_sparsity)
print(sel.refit.shape)
```

## Summary and plotting helpers

```python
from pyhuge import (
    huge_summary, huge_select_summary,
    huge_plot_sparsity, huge_plot_graph_matrix, huge_plot_network,
)
import matplotlib.pyplot as plt

s1 = huge_summary(fit)
s2 = huge_select_summary(sel)
print(s1)
print(s2)

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
huge_plot_sparsity(fit, ax=axes[0])
huge_plot_graph_matrix(fit, index=-1, ax=axes[1])
fig.tight_layout()

# network visualization
fig2, ax2 = plt.subplots(figsize=(4.5, 4.5))
huge_plot_network(fit, index=-1, ax=ax2, layout="spring")
```

## Using custom lambda path

```python
lam = np.logspace(0, -2, num=6)  # descending positive sequence
fit = huge(x, method="mb", lambda_=lam, verbose=False)
```

## Simulation + ROC + Inference

```python
from pyhuge import huge_generator, huge_roc, huge_inference

sim = huge_generator(n=120, d=20, graph="hub", g=4, verbose=False)
fit = huge(sim.data, method="glasso", nlambda=6, verbose=False)

roc = huge_roc(fit.path, sim.theta, verbose=False, plot=False)
print("AUC:", roc.auc)

inf = huge_inference(
    data=sim.data,
    t=fit.icov[-1],        # last precision estimate
    adj=sim.theta,
    alpha=0.05,
    type_="Gaussian",
)
print("Type I error:", inf.error)
```

For script-style walkthroughs, see `tutorials.md`.
