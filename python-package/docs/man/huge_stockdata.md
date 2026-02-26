# `huge_stockdata`

## Usage

```python
huge_stockdata() -> HugeStockDataResult
```

## Description

Loads packaged stock dataset distributed with `pyhuge`.

## Returns

`HugeStockDataResult` containing:

- `data`: stock price matrix (`1258 x 452`)
- `info`: metadata matrix (`452 x 3`)
- `raw`: metadata dict with source reference
