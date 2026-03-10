# Active-Set Baseline Runner

This folder stores frozen data, baseline snapshots, and run artifacts for
low-risk active-set refactors.

## Files

- `data/*.rds`: fixed input matrices (generated once, then reused)
- `baseline/*.rds`: frozen baseline results for each dataset/method case
- `runs/<run_id>/results/*.rds`: current run outputs
- `runs/<run_id>/logs/*.log`: per-case logs
- `runs/<run_id>/summary_record.csv`: summary in record mode
- `runs/<run_id>/summary_compare.csv`: summary in compare mode

## Usage

From repo root:

```bash
Rscript benchmark/active_set_baseline.R mode=record
Rscript benchmark/active_set_baseline.R mode=compare
```

Useful knobs:

```bash
Rscript benchmark/active_set_baseline.R mode=compare hard_timeout=300 soft_timeout=240 nlambda=8 lambda_min_ratio=0.2
```

## Hang-Safety Design

- Each case runs in an isolated `Rscript` process.
- `system2(..., timeout=...)` enforces hard process timeout.
- Case script uses `setTimeLimit(...)` for soft timeout.
- One case failure does not stop other cases.
