# FAQ

## Is `pyhuge` pure Python?

No. `pyhuge` calls the R package `huge` using `rpy2`.
You must have both Python and R set up.

## I installed `pyhuge` but `runtime=False` in `pyhuge.test()`

Usually one of these is missing:

- R is not installed or not in `PATH`
- R package `huge` is not installed in the active R library path
- Python architecture and R architecture mismatch (`arm64` vs `x86_64`)

See [Troubleshooting](troubleshooting.md).

## Which method should beginners use first?

Use `method="mb"` first for a stable starting point.
Then compare against `method="glasso"` if needed.

## What is the difference between `fit.path` and `sel.refit`?

- `fit.path`: all graphs across regularization levels
- `sel.refit`: the single graph selected by a criterion (`ric`, `stars`, `ebic`)

## Which selection criterion should I use?

- `ric`: fast and simple
- `stars`: more stability-focused but slower
- `ebic`: common with glasso workflows

Use `ebic` only when estimator method is `glasso`.

## Why is plotting failing?

- For `huge_plot_network`, install `networkx` and `matplotlib`
- In headless environments, set:

```bash
export MPLBACKEND=Agg
```

## Can I use covariance/correlation matrix as input?

Yes. `huge(...)` accepts either data matrix (`n x d`) or covariance/correlation
matrix (`d x d`). The wrapper follows R `huge` behavior.

## Is there a stock dataset I can try quickly?

Yes:

```python
from pyhuge import huge_stockdata
stock = huge_stockdata()
print(stock.data.shape, stock.info.shape)
```

## Where are full function docs?

- API overview: [api.md](api.md)
- One-page-per-function manual: [man/index.md](man/index.md)
