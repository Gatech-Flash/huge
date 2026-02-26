# Quick Start

## 0) Runtime check

```python
import pyhuge
print(pyhuge.test())
```

If you want strict enforcement:

```python
import pyhuge
pyhuge.test(require_runtime=True)
```

## 1) Basic graph path estimation

```python
import numpy as np
from pyhuge import huge, huge_summary

rng = np.random.default_rng(123)
x = rng.normal(size=(120, 30))

fit = huge(
    x,
    method="mb",          # mb, glasso, ct, tiger
    nlambda=8,
    lambda_min_ratio=0.1,
    verbose=False,
)

print(fit.method)
print(fit.lambda_path)
print(len(fit.path), fit.path[0].shape)
print(huge_summary(fit))
```

Shortcut wrapper:

```python
from pyhuge import huge_mb
fit = huge_mb(x, nlambda=8, lambda_min_ratio=0.1, verbose=False)
```

## 2) NPN preprocessing + selection

```python
from pyhuge import huge_npn, huge_select, huge_select_summary

x_npn = huge_npn(x, npn_func="shrinkage", verbose=False)
fit = huge(x_npn, method="glasso", nlambda=10, verbose=False)
sel = huge_select(fit, criterion="ebic", ebic_gamma=0.5, verbose=False)

print(sel.opt_lambda, sel.opt_sparsity)
print(sel.refit.shape)
print(huge_select_summary(sel))
```

## 3) Plot helpers

```python
from pyhuge import huge_plot_sparsity, huge_plot_graph_matrix, huge_plot_network, huge_plot
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
huge_plot_sparsity(fit, ax=axes[0])
huge_plot_graph_matrix(fit, index=-1, ax=axes[1])
fig.tight_layout()

fig2, ax2 = plt.subplots(figsize=(4.5, 4.5))
huge_plot_network(fit, index=-1, ax=ax2, layout="spring")

huge_plot(sel.refit, epsflag=False)
```

## 4) Custom lambda path

```python
import numpy as np
from pyhuge import huge

lam = np.logspace(0, -2, num=6)  # descending positive sequence
fit = huge(x, method="mb", lambda_=lam, verbose=False)
```

## 5) Simulation + ROC + Inference

```python
from pyhuge import huge_generator, huge_roc, huge_inference

sim = huge_generator(n=120, d=20, graph="hub", g=4, verbose=False)
fit = huge(sim.data, method="glasso", nlambda=6, verbose=False)

roc = huge_roc(fit.path, sim.theta, verbose=False, plot=False)
print("AUC:", roc.auc)

inf = huge_inference(
    data=sim.data,
    t=fit.icov[-1],
    adj=sim.theta,
    alpha=0.05,
    type_="Gaussian",
)
print("Type I error:", inf.error)
```

## 6) Built-in stock dataset

```python
from pyhuge import huge_stockdata

stock = huge_stockdata()
print(stock.data.shape)
print(stock.info.shape)
```
