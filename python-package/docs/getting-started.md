# Start Here (5 Minutes)

This page is for first-time users.

Goal: run one `pyhuge` graph estimation end-to-end and know what to do when it
fails.

## 1. What pyhuge is

`pyhuge` is not a pure-Python reimplementation.
It calls the R package `huge` through `rpy2`.

So your runtime has 3 required layers:

1. Python
2. R
3. R package `huge`

If any layer is missing, function calls fail.

## 2. Install in the simplest way

```bash
pip install pyhuge
```

Then verify import and runtime:

```bash
python -c "import pyhuge; print(pyhuge.test())"
```

Expected:

- `python_import: True`
- `rpy2: True`
- `runtime: True`

If `runtime` is `False`, go to [Troubleshooting](troubleshooting.md).

## 3. Run your first example

```python
import numpy as np
from pyhuge import huge, huge_select

rng = np.random.default_rng(42)
x = rng.normal(size=(120, 30))

fit = huge(x, method="mb", nlambda=8, verbose=False)
sel = huge_select(fit, criterion="ric", rep_num=10, verbose=False)

print("method:", fit.method)
print("path length:", len(fit.path))
print("selected lambda:", sel.opt_lambda)
print("selected sparsity:", sel.opt_sparsity)
```

## 4. Interpret output quickly

- `fit.path`: list of adjacency matrices across regularization path
- `fit.lambda_path`: regularization values aligned with `fit.path`
- `sel.refit`: selected graph from model selection
- `sel.opt_lambda`: selected regularization level

## 5. Next pages

- [Installation](installation.md): architecture checks and install modes
- [Quick Start](quickstart.md): more API examples
- [Beginner Workflow](beginner-workflow.md): a full practical workflow
- [FAQ](faq.md): common beginner questions
