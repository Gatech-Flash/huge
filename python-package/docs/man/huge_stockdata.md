# `huge_stockdata`

## Usage

```python
huge_stockdata() -> HugeStockDataResult
```

## Description

Loads built-in dataset `stockdata` from R package `huge`.

## Returns

`HugeStockDataResult` containing:

- `data`: stock price matrix (`1258 x 452`)
- `info`: stock metadata matrix (`452 x 3`)
- `raw`: original R object
