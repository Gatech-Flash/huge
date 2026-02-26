# `huge_npn`

## Usage

```python
huge_npn(x, npn_func="shrinkage", verbose=True) -> numpy.ndarray
```

## Description

Native nonparanormal transformation.

## Key arguments

- `npn_func`: `"shrinkage"`, `"truncation"`, `"skeptic"`

## Returns

- transformed data matrix (`shrinkage`/`truncation`)
- rank-based correlation matrix (`skeptic`)
