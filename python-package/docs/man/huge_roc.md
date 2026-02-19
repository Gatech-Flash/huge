# `huge_roc`

## Usage

```python
huge_roc(path, theta, verbose=True, plot=False) -> HugeRocResult
```

## Description

ROC computation mapped to R `huge.roc()`.

## Key arguments

- `path`: sequence of square adjacency matrices
- `theta`: true square adjacency matrix
- `plot`: if `False`, plotting is redirected to temporary PDF backend for headless safety

## Returns

`HugeRocResult` with `f1`, `tp`, `fp`, and `auc`.
