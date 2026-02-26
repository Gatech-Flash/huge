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
    backend="native",
) -> HugeSelectResult
```

## Description

Native model selection for an estimated graph path.

## Key arguments

- `est`: `HugeResult`
- `criterion`: `"ric"`, `"stars"`, or `"ebic"`
- `ebic_gamma`: EBIC tuning parameter
- `stars_thresh`: threshold in `(0, 1]`
- `stars_subsample_ratio`: optional subsample ratio
- `rep_num`: repetition count for stochastic criteria
- `backend`: currently only `"native"`

## Returns

`HugeSelectResult` with `opt_lambda`, `opt_sparsity`, `refit`, and optional
fields (`opt_icov`, `opt_cov`, `variability`, `ebic_score`).

## Notes

- `criterion="ebic"` requires a glasso fit.
- `opt_index` is 1-based for compatibility with prior wrapper behavior.
