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

Edge inference mapped to R `huge.inference()`.

## Key arguments

- `data`: 2D matrix
- `t`: estimated precision matrix (square)
- `adj`: target adjacency matrix (square)
- `alpha`: significance level in `(0, 1]`
- `type_`: one of `"Gaussian"`, `"nonparanormal"`
- `method`: one of `"score"`, `"wald"`

## Returns

`HugeInferenceResult` with p-value matrix `p` and estimated type-I error.
