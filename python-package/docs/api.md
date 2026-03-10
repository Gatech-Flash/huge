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
    backend="native",
) -> HugeResult
```

Native entry for graph-path estimation.

## `pyhuge.test`

```python
test(require_runtime=False) -> dict
```

Environment probe for native runtime.

Returned keys include:

- `python_import`
- `numpy`
- `scipy`
- `sklearn` (compatibility field; not required for runtime)
- `native_extension`
- `runtime`
- `rpy2` (compatibility field)

## Wrapper shortcuts

```python
huge_mb(...)
huge_glasso(...)
huge_ct(...)
huge_tiger(...)
```

These call `huge(...)` with fixed method.

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
    backend="native",
) -> HugeSelectResult
```

Model selection on `HugeResult`.

## `pyhuge.huge_npn`

```python
huge_npn(x, npn_func="shrinkage", verbose=True) -> numpy.ndarray
```

Native nonparanormal transformation.

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
    random_state=None,
) -> HugeGeneratorResult
```

Native synthetic data generator.

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

Native edge-wise inference approximation.

## `pyhuge.huge_roc`

```python
huge_roc(path, theta, verbose=True, plot=False) -> HugeRocResult
```

Native ROC metrics over graph path.

## `pyhuge.huge_stockdata`

```python
huge_stockdata() -> HugeStockDataResult
```

Loads packaged stock dataset (`1258 x 452` matrix + `452 x 3` info table).

## Summaries

```python
huge_summary(fit: HugeResult) -> HugeSummary
huge_select_summary(sel: HugeSelectResult) -> HugeSelectSummary
```

## Plot helpers

```python
huge_plot_sparsity(fit, ax=None, show_points=True)
huge_plot_roc(roc, ax=None)
huge_plot_graph_matrix(fit, index=-1, ax=None)
huge_plot_network(fit, index=-1, ax=None, layout="spring", ...)
huge_plot(g, epsflag=False, graph_name="default", cur_num=1, location=None)
```

## Dataclasses

- `HugeResult`
- `HugeSelectResult`
- `HugeGeneratorResult`
- `HugeInferenceResult`
- `HugeRocResult`
- `HugeStockDataResult`
- `HugeSummary`
- `HugeSelectSummary`

## Exception

- `PyHugeError`: raised for validation failures or missing native dependencies.
