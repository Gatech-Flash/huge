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
) -> HugeResult
```

## Description

Main graph path estimation entry, mapped to R `huge()`.

## Key arguments

- `x`: 2D numeric matrix (`n_samples x n_features`)
- `method`: one of `"mb"`, `"glasso"`, `"ct"`, `"tiger"`
- `lambda_`: strictly decreasing positive lambda path (optional)
- `nlambda`: number of lambda values if `lambda_` is not provided
- `lambda_min_ratio`: minimum lambda ratio in `(0, 1]`
- `scr`, `scr_num`: screening options (`mb` / `glasso` only)
- `cov_output`: only valid for `method="glasso"`
- `sym`: only applicable to `mb` and `tiger`

## Returns

`HugeResult` with `method`, `lambda_path`, `sparsity`, `path`, `data`, optional `icov/cov`, and `raw`.

## Example

```python
fit = huge(x, method="mb", nlambda=8, verbose=False)
```
