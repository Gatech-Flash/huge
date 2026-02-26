# `test`

## Usage

```python
test(require_runtime=False) -> dict
```

## Description

Native runtime readiness probe for `pyhuge`.

## Returns

Dictionary with keys including:

- `python_import`
- `numpy`
- `scipy`
- `sklearn`
- `native_extension`
- `runtime`
- `rpy2` (compatibility field)

## Notes

- If `require_runtime=True`, raises `PyHugeError` when required deps are missing.
