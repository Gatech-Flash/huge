# API Reference

## `pyhuge.huge`

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

Main wrapper over R `huge()`.

### Parameters

- `x`: 2D `numpy.ndarray`, input data matrix.
- `lambda_`: optional descending positive sequence.
- `nlambda`: optional path length (if `lambda_` is not provided).
- `lambda_min_ratio`: optional path end ratio.
- `method`: `"mb" | "glasso" | "ct" | "tiger"`.
- `scr`, `scr_num`, `cov_output`, `sym`, `verbose`: same semantics as R package.

### Returns: `HugeResult`

- `method`: method string.
- `lambda_path`: `numpy.ndarray`.
- `sparsity`: `numpy.ndarray`.
- `path`: list of sparse graph matrices (`scipy.sparse`).
- `cov_input`: bool.
- `data`: input matrix as `numpy.ndarray`.
- Optional fields: `df`, `loglik`, `icov`, `cov`, `idx_mat`.
- `raw`: original R object.

## `pyhuge.test`

```python
test(require_runtime=False) -> dict
```

Environment probe helper (similar to `pycasso.test()` style):

- checks whether `rpy2` is importable;
- checks whether runtime bridge (`R` + R package `huge`) is ready.

Returns a status dictionary:

- `python_import`: always `True` once module is imported;
- `rpy2`: whether `rpy2` can be resolved;
- `runtime`: whether `_r_env()` successfully initializes.

If `require_runtime=True`, it raises `PyHugeError` when runtime requirements are not met.

## Method-specific wrappers

These are convenience wrappers around `huge(...)` with fixed `method`:

```python
huge_mb(...)
huge_glasso(...)
huge_ct(...)
huge_tiger(...)
```

They return the same `HugeResult` object.

- `huge_mb`: equivalent to `huge(..., method="mb")`
- `huge_glasso`: equivalent to `huge(..., method="glasso")`
- `huge_ct`: equivalent to `huge(..., method="ct")`
- `huge_tiger`: equivalent to `huge(..., method="tiger")`

## `pyhuge.huge_select`

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

Wrapper over R `huge.select()`.

### Parameters

- `est`: `HugeResult` or raw R `huge` object.
- `criterion`: optional `"ric" | "stars" | "ebic"`.
- `ebic_gamma`, `stars_thresh`, `stars_subsample_ratio`, `rep_num`, `verbose`: aligned with R interface.

### Returns: `HugeSelectResult`

- `criterion`: selection criterion used.
- `opt_lambda`: selected lambda.
- `opt_sparsity`: selected sparsity.
- `refit`: selected graph (`scipy.sparse`).
- Optional fields: `opt_index`, `variability`, `ebic_score`, `opt_icov`, `opt_cov`.
- `raw`: original R object.

## `pyhuge.huge_npn`

```python
huge_npn(x, npn_func="shrinkage", verbose=True) -> numpy.ndarray
```

Wrapper over R `huge.npn()`.

### Parameters

- `x`: 2D `numpy.ndarray`.
- `npn_func`: `"shrinkage" | "truncation" | "skeptic"` (as supported by R package).
- `verbose`: bool.

### Returns

- Transformed data matrix as `numpy.ndarray`.

## `pyhuge.huge_generator`

```python
huge_generator(
    n=200,
    d=50,
    graph="random",
    v=None,
    u=None,
    g=None,
    prob=None,
    vis=False,
    verbose=True,
) -> HugeGeneratorResult
```

Wrapper over R `huge.generator()`.

### Returns: `HugeGeneratorResult`

- `data`, `sigma`, `omega`, `sigmahat`: dense `numpy.ndarray`.
- `theta`: graph adjacency as `scipy.sparse` matrix.
- `sparsity`: float.
- `graph_type`: string.
- `raw`: original R object.

## `pyhuge.huge_inference`

```python
huge_inference(
    data,
    t,
    adj,
    alpha=0.05,
    type_="Gaussian",
    method="score",
) -> HugeInferenceResult
```

Wrapper over R `huge.inference()`.

### Returns: `HugeInferenceResult`

- `data`: input matrix (dense `numpy.ndarray`).
- `p`: p-value matrix (`numpy.ndarray`).
- `error`: type I error estimate (float).
- `raw`: original R object.

## `pyhuge.huge_roc`

```python
huge_roc(path, theta, verbose=True, plot=False) -> HugeRocResult
```

Wrapper over R `huge.roc()`.

### Parameters

- `path`: sequence of adjacency matrices.
- `theta`: true adjacency matrix.
- `plot`: if `False` (default), plotting is redirected to a temporary PDF device.

### Returns: `HugeRocResult`

- `f1`, `tp`, `fp`: `numpy.ndarray`.
- `auc`: float.
- `raw`: original R object.

## Summaries

```python
huge_summary(fit: HugeResult) -> HugeSummary
huge_select_summary(sel: HugeSelectResult) -> HugeSelectSummary
```

These return compact dataclasses for logging/reporting.

## Plot helpers

```python
huge_plot_sparsity(fit, ax=None, show_points=True)
huge_plot_roc(roc, ax=None)
huge_plot_graph_matrix(fit, index=-1, ax=None)
huge_plot_network(fit, index=-1, ax=None, layout="spring", ...)
```

- `huge_plot_sparsity`: lambda vs sparsity
- `huge_plot_roc`: ROC curve from `HugeRocResult`
- `huge_plot_graph_matrix`: adjacency matrix heatmap for one path index
- `huge_plot_network`: node-edge network visualization for one path index

Plotting dependencies:
- `matplotlib` for all plots
- `networkx` additionally for `huge_plot_network`

## Exceptions

- `PyHugeError`: raised for wrapper/runtime issues, e.g. missing `rpy2` or missing R package `huge`.

## Validation rules (high level)

- `huge`:
  - `method` in `{"mb", "glasso", "ct", "tiger"}`
  - `lambda_` must be strictly decreasing positive sequence
  - `nlambda` positive integer
  - `lambda_min_ratio` in `(0, 1]`
  - `cov_output=True` only for `method="glasso"`
- `huge_select`:
  - `criterion` in `{"ric", "stars", "ebic"}`
  - `stars_thresh` in `(0, 1]`
  - `rep_num` positive integer
  - `criterion="ebic"` requires glasso fit when `est` is `HugeResult`
- `huge_inference`:
  - `data` is 2D
  - `t` and `adj` are square and match data dimension
- `huge_roc`:
  - non-empty `path`
  - every matrix in `path` has same shape as square `theta`
