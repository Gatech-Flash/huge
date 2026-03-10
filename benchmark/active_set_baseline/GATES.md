# Active-Set Optimization Gates

## Baseline Window

- Runs: `step1-baseline-1` ... `step1-baseline-5`
- Summary files:
  - `benchmark/active_set_baseline/runs/step1-baseline-1/summary_compare.csv`
  - `benchmark/active_set_baseline/runs/step1-baseline-2/summary_compare.csv`
  - `benchmark/active_set_baseline/runs/step1-baseline-3/summary_compare.csv`
  - `benchmark/active_set_baseline/runs/step1-baseline-4/summary_compare.csv`
  - `benchmark/active_set_baseline/runs/step1-baseline-5/summary_compare.csv`

## Baseline Metrics (elapsed_runner_sec)

- `mb`: mean `0.9690`, median `0.9420`, sd `0.0625`
- `tiger`: mean `0.9685`, median `0.9570`, sd `0.0638`
- `glasso`: mean `0.9044`, median `0.8860`, sd `0.0549`
- `ct`: mean `1.0531`, median `1.0220`, sd `0.0749`
- overall mean `0.9738`

## Pass/Fail Gates

1. Correctness (hard gate)
- all rows `case_status == "ok"`
- `lambda_max_abs_diff == 0` (or NA where not applicable)
- `path_mismatch_count == 0` (or NA where not applicable)
- `icov_max_abs_diff <= 1e-10` (or NA where not applicable)

2. Performance (soft-no-regression gate)
- `mb` and `tiger`: regression must be `<= +1.0%` versus baseline mean
- overall regression must be `<= +1.0%` versus baseline mean

3. Promotion requirement (for final keep)
- at least one of `mb` or `tiger` must show a stable speedup over baseline mean
- no correctness gate violations
