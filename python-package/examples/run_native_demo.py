from __future__ import annotations

import numpy as np

from pyhuge import huge, huge_select

rng = np.random.default_rng(42)
x = rng.normal(size=(100, 20))

fit = huge(x, method="mb", nlambda=6, backend="native")
sel = huge_select(fit, criterion="ric", backend="native")

print("method:", fit.method)
print("path length:", len(fit.path))
print("selected lambda:", sel.opt_lambda)
print("selected sparsity:", sel.opt_sparsity)
