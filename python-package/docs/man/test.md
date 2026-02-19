# `test`

## Usage

```python
test(require_runtime=False) -> dict
```

## Description

Runtime readiness probe for `pyhuge`.

## Returns

Dictionary with keys:

- `python_import`
- `rpy2`
- `runtime`

## Notes

- If `require_runtime=True`, raises `PyHugeError` when runtime is unavailable.
- Useful as a first diagnostic step after installation.
