# `huge_glasso`

## Usage

```python
huge_glasso(
    x,
    lambda_=None,
    nlambda=None,
    lambda_min_ratio=None,
    scr=None,
    cov_output=False,
    verbose=True,
) -> HugeResult
```

## Description

Convenience wrapper for `huge(..., method="glasso")`.

## Notes

- `cov_output=True` is supported for glasso only.
