# FAQ

## Is `pyhuge` 0.3 pure Python?

Yes. `pyhuge` 0.3 is native Python and does not require `rpy2`.

## `runtime=False` in `pyhuge.test()` means what?

Usually at least one core dependency is missing:

- `numpy`
- `scipy`
- `scikit-learn`

Install with:

```bash
pip install "pyhuge[runtime]"
```

## Which method should I start with?

Use `method="mb"` first. Then compare with `method="glasso"`.

## Difference between `fit.path` and `sel.refit`?

- `fit.path`: full path of estimated graphs
- `sel.refit`: single selected graph under criterion (`ric`, `stars`, `ebic`)

## Which selection criterion should I use?

- `ric`: fast and simple
- `stars`: stability-focused, slower
- `ebic`: common for glasso

## Why is plotting failing?

Install visualization deps:

```bash
pip install "pyhuge[viz]"
```

In headless environments:

```bash
export MPLBACKEND=Agg
```

## Can input be covariance/correlation matrix?

Yes. For `ct` and `glasso`, `huge(...)` accepts square covariance/correlation input.
For `mb` and `tiger`, use raw data matrix (`n x d`).

## Is there a built-in dataset?

Yes:

```python
from pyhuge import huge_stockdata
stock = huge_stockdata()
print(stock.data.shape, stock.info.shape)
```

## Where are full function docs?

- API overview: [api.md](api.md)
- One-page function manual: [man/index.md](man/index.md)
