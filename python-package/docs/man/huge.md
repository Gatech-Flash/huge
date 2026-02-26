# `huge`

## Usage

```python
huge(
    x,
    lambda_=None,
    nlambda=None,
    lambda_min_ratio=None,
    method="mb",
    scr=None,
    scr_num=None,
    cov_output=False,
    sym="or",
    verbose=True,
    backend="native",
) -> HugeResult
```

## Description

Native graph path estimation entry.

## Key arguments

- `x`: 2D numeric matrix (`n_samples x n_features`) or square covariance input
- `method`: one of `"mb"`, `"glasso"`, `"ct"`, `"tiger"`
- `lambda_`: strictly decreasing positive path (optional)
- `nlambda`: path length when `lambda_` is not provided
- `lambda_min_ratio`: minimum lambda ratio in `(0, 1]`
- `scr`, `scr_num`: currently argument-validated for compatibility
- `cov_output`: only valid for `method="glasso"`
- `sym`: only applicable to `mb` and `tiger`
- `backend`: currently only `"native"`

## Returns

`HugeResult` with `method`, `lambda_path`, `sparsity`, `path`, `data`, and
optional `icov/cov/df/loglik` depending on method.
