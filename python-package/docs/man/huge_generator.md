# `huge_generator`

## Usage

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

## Description

Synthetic data generator mapped to R `huge.generator()`.

## Key arguments

- `graph`: one of `"random"`, `"hub"`, `"cluster"`, `"band"`, `"scale-free"`
- `n`, `d`: sample size and feature count
- `v`, `u`, `g`, `prob`: graph-specific options

## Returns

`HugeGeneratorResult` containing `data`, covariance/precision matrices, graph `theta`, and `sparsity`.
