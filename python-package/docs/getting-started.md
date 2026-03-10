# Start Here (5 Minutes)

Goal: run one full graph-estimation flow and verify your local environment.

## 1. Install

```bash
pip install pyhuge
```

Or from source:

```bash
cd python-package
pip install -e ".[runtime]"
```

## 2. Verify runtime

```bash
python -c "import pyhuge; print(pyhuge.test())"
```

Expected minimal status:

- `python_import: True`
- `numpy: True`
- `scipy: True`
- `native_extension: True`
- `runtime: True`

## 3. First estimation

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

## 4. Read outputs quickly

- `fit.path`: adjacency matrices along regularization path
- `fit.lambda_path`: regularization sequence aligned with `fit.path`
- `sel.refit`: selected graph matrix
- `sel.opt_lambda`: selected regularization level

## 5. Next

- [Installation](installation.md)
- [Quick Start](quickstart.md)
- [Beginner Workflow](beginner-workflow.md)
- [Troubleshooting](troubleshooting.md)
