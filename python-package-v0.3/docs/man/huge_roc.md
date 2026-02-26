# `huge_roc`

## Usage

```python
huge_roc(path, theta, verbose=True, plot=False) -> HugeRocResult
```

## Description

Native ROC computation across graph path estimates.

## Key arguments

- `path`: sequence of adjacency matrices
- `theta`: ground-truth adjacency matrix
- `plot`: if `True`, draw ROC curve via matplotlib helper

## Returns

`HugeRocResult` with:

- `f1`: F1-score array
- `tp`: true-positive-rate array
- `fp`: false-positive-rate array
- `auc`: area under ROC curve
