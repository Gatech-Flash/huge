# `doctor`

## Usage

```python
doctor(require_runtime=False) -> dict
format_doctor_report(report: dict) -> str
```

CLI:

```bash
pyhuge-doctor
python -m pyhuge.doctor --json
python -m pyhuge.doctor --require-runtime
```

## Description

Runtime diagnostics helper for `pyhuge`. It checks:

- Python interpreter and architecture
- `rpy2` importability
- R executable visibility (`R` in `PATH`)
- R package `huge` availability
- end-to-end `pyhuge` runtime readiness

## Returns

Dictionary containing sections:

- `python`
- `rpy2`
- `r`
- `r_packages`
- `runtime`
- `suggestions`

## Notes

- Use `--require-runtime` in CI/scripts to fail fast when runtime is unavailable.
- Use `--json` for machine-readable diagnostics.
