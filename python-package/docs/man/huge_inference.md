# `huge_inference`

## Usage

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

## Description

Native edge-wise inference helper using a partial-correlation z-test
approximation.

## Key arguments

- `data`: sample matrix (`n x d`)
- `t`: precision-like matrix (`d x d`)
- `adj`: reference adjacency (`d x d`)
- `type_`: `"Gaussian"` or `"Nonparanormal"`
- `method`: `"score"` or `"wald"` (API compatibility)

## Returns

`HugeInferenceResult` with transformed `data`, p-value matrix `p`, and `error`.
