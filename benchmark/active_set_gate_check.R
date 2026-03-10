#!/usr/bin/env Rscript

# Usage:
#   Rscript benchmark/active_set_gate_check.R candidate=runA,runB baseline=run1,run2,run3
#
# Enforces:
# 1) all cases are OK
# 2) numeric parity: lambda/path/icov deltas within tolerance
# 3) performance gates for mb/tiger and overall elapsed time

args <- commandArgs(trailingOnly = TRUE)

parse_args <- function(argv) {
  out <- list()
  for (item in argv) {
    kv <- strsplit(item, "=", fixed = TRUE)[[1]]
    if (length(kv) != 2) {
      stop(sprintf("Invalid argument: %s. Expected key=value.", item))
    }
    out[[kv[[1]]]] <- kv[[2]]
  }
  out
}

read_summary <- function(run_id) {
  path <- file.path("benchmark", "active_set_baseline", "runs", run_id, "summary_compare.csv")
  if (!file.exists(path)) {
    stop(sprintf("summary file not found for run_id=%s (%s)", run_id, path))
  }
  df <- read.csv(path, stringsAsFactors = FALSE)
  df$run_id <- run_id
  df
}

opt <- parse_args(args)
candidate <- opt$candidate
candidate_runs <- if (is.null(candidate)) character(0) else strsplit(candidate, ",", fixed = TRUE)[[1]]
baseline_runs <- if (is.null(opt$baseline)) character(0) else strsplit(opt$baseline, ",", fixed = TRUE)[[1]]

if (length(candidate_runs) == 0) {
  stop("Missing required arg: candidate=runA,runB,...")
}
if (length(baseline_runs) == 0) {
  stop("Missing required arg: baseline=run1,run2,...")
}

lambda_tol <- as.numeric(ifelse(is.null(opt$lambda_tol), "0", opt$lambda_tol))
path_tol <- as.numeric(ifelse(is.null(opt$path_tol), "0", opt$path_tol))
icov_tol <- as.numeric(ifelse(is.null(opt$icov_tol), "1e-10", opt$icov_tol))
method_regress_tol_pct <- as.numeric(ifelse(is.null(opt$method_regress_tol_pct), "1.0", opt$method_regress_tol_pct))
overall_regress_tol_pct <- as.numeric(ifelse(is.null(opt$overall_regress_tol_pct), "1.0", opt$overall_regress_tol_pct))

baseline_df <- do.call(rbind, lapply(baseline_runs, read_summary))
cand_df <- do.call(rbind, lapply(candidate_runs, read_summary))

fail <- FALSE

check_or_fail <- function(cond, msg) {
  if (!isTRUE(cond)) {
    cat(sprintf("[FAIL] %s\n", msg))
    fail <<- TRUE
  } else {
    cat(sprintf("[ OK ] %s\n", msg))
  }
}

# Hard correctness checks on candidate run
check_or_fail(all(cand_df$case_status == "ok"), "candidate all case_status == ok")
check_or_fail(all(is.na(cand_df$lambda_max_abs_diff) | cand_df$lambda_max_abs_diff <= lambda_tol),
              sprintf("candidate lambda_max_abs_diff <= %.3g", lambda_tol))
check_or_fail(all(is.na(cand_df$path_mismatch_count) | cand_df$path_mismatch_count <= path_tol),
              sprintf("candidate path_mismatch_count <= %.3g", path_tol))
check_or_fail(all(is.na(cand_df$icov_max_abs_diff) | cand_df$icov_max_abs_diff <= icov_tol),
              sprintf("candidate icov_max_abs_diff <= %.3g", icov_tol))

# Performance checks
base_m <- aggregate(elapsed_runner_sec ~ method, baseline_df, mean)
cand_m <- aggregate(elapsed_runner_sec ~ method, cand_df, mean)
perf <- merge(base_m, cand_m, by = "method", suffixes = c("_base", "_cand"))
perf$delta_pct <- 100 * (perf$elapsed_runner_sec_cand - perf$elapsed_runner_sec_base) / perf$elapsed_runner_sec_base

for (m in c("mb", "tiger")) {
  row <- perf[perf$method == m, , drop = FALSE]
  if (nrow(row) == 1) {
    check_or_fail(row$delta_pct <= method_regress_tol_pct,
                  sprintf("%s regression <= %.2f%% (actual %.3f%%)", m, method_regress_tol_pct, row$delta_pct))
  } else {
    check_or_fail(FALSE, sprintf("missing method in perf table: %s", m))
  }
}

overall_base <- mean(baseline_df$elapsed_runner_sec)
overall_cand <- mean(cand_df$elapsed_runner_sec)
overall_delta_pct <- 100 * (overall_cand - overall_base) / overall_base
check_or_fail(overall_delta_pct <= overall_regress_tol_pct,
              sprintf("overall regression <= %.2f%% (actual %.3f%%)", overall_regress_tol_pct, overall_delta_pct))

cat("\nPerformance summary (candidate vs baseline mean):\n")
print(perf[order(perf$method), ], row.names = FALSE)
cat(sprintf("\nOverall base=%.6f cand=%.6f delta_pct=%.3f%%\n", overall_base, overall_cand, overall_delta_pct))

if (fail) {
  quit(status = 2L)
}
