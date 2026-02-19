# Performance Notes

`pyhuge` is an R-bridge wrapper (`Python -> rpy2 -> R huge`), so total runtime is:

- R backend solve time (dominant for medium/large problems)
- Python/R bridge overhead (visible for very small repeated calls)

## Recommended benchmarking method

1. Benchmark `huge` directly in R for backend-only performance.
2. Benchmark `pyhuge` end-to-end for user-facing Python workflow time.
3. Compare both to estimate bridge overhead.

## Practical tips

- Batch work when possible rather than many tiny calls.
- Reuse transformed data (`huge_npn`) if running multiple methods.
- Use method wrappers (`huge_mb`, `huge_glasso`, `huge_ct`, `huge_tiger`) to keep scripts simple.
- In CI/headless plotting, set `MPLBACKEND=Agg`.

## Smoke benchmark command pattern

```bash
cd python-package
python -m pytest -q tests/test_e2e_optional.py -rA
```
