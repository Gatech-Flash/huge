# `huge_select`

## Usage

```python
huge_select(
    est,
    criterion=None,
    ebic_gamma=0.5,
    stars_thresh=0.1,
    stars_subsample_ratio=None,
    rep_num=20,
    verbose=True,
) -> HugeSelectResult
```

## Description

Model selection for an estimated graph path, mapped to R `huge.select()`.

## Key arguments

- `est`: `HugeResult` or raw R `huge` object
- `criterion`: `"ric"`, `"stars"`, or `"ebic"`
- `ebic_gamma`: EBIC tuning parameter
- `stars_thresh`: threshold in `(0, 1]`
- `stars_subsample_ratio`: optional subsample ratio
- `rep_num`: repetition number for stochastic criteria

## Returns

`HugeSelectResult` with `opt_lambda`, `opt_sparsity`, `refit`, and optional fields (`opt_icov`, `opt_cov`, etc.).

## Notes

- `criterion="ebic"` requires a glasso fit when `est` is `HugeResult`.
