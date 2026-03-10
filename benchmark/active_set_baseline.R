#!/usr/bin/env Rscript

# Baseline runner for active-set related refactors.
# - record mode: run cases and save baseline snapshots
# - compare mode: run cases and compare against stored baseline

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

as_bool <- function(x, default = FALSE) {
  if (is.null(x) || !nzchar(x)) {
    return(default)
  }
  tolower(x) %in% c("1", "true", "t", "yes", "y")
}

now_stamp <- function() {
  format(Sys.time(), "%Y%m%d-%H%M%S")
}

safe_max_abs_diff <- function(a, b) {
  if (is.null(a) || is.null(b)) {
    return(NA_real_)
  }
  if (!identical(dim(a), dim(b))) {
    return(Inf)
  }
  max(abs(as.numeric(a) - as.numeric(b)))
}

safe_count_diff <- function(a, b) {
  if (is.null(a) || is.null(b)) {
    return(NA_real_)
  }
  if (!identical(dim(a), dim(b))) {
    return(Inf)
  }
  sum(as.integer(a) != as.integer(b))
}

opt <- parse_args(args)
mode <- ifelse(is.null(opt$mode), "record", opt$mode)
if (!mode %in% c("record", "compare")) {
  stop("mode must be one of: record, compare")
}

hard_timeout <- as.integer(ifelse(is.null(opt$hard_timeout), "240", opt$hard_timeout))
soft_timeout <- as.integer(ifelse(is.null(opt$soft_timeout), "180", opt$soft_timeout))
nlambda <- as.integer(ifelse(is.null(opt$nlambda), "8", opt$nlambda))
lambda_min_ratio <- as.numeric(ifelse(is.null(opt$lambda_min_ratio), "0.2", opt$lambda_min_ratio))
run_id <- ifelse(is.null(opt$run_id), now_stamp(), opt$run_id)
keep_logs <- as_bool(opt$keep_logs, default = TRUE)

repo_root <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
base_dir <- file.path(repo_root, "benchmark", "active_set_baseline")
data_dir <- file.path(base_dir, "data")
baseline_dir <- file.path(base_dir, "baseline")
run_dir <- file.path(base_dir, "runs", run_id)
result_dir <- file.path(run_dir, "results")
log_dir <- file.path(run_dir, "logs")

dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(baseline_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(result_dir, recursive = TRUE, showWarnings = FALSE)
if (keep_logs) {
  dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)
}

suppressPackageStartupMessages(library(huge))

datasets <- list(
  list(id = "hub_n120_d80_seed20260305", n = 120L, d = 80L, graph = "hub", g = 6L, seed = 20260305L),
  list(id = "band_n180_d100_seed20260305", n = 180L, d = 100L, graph = "band", g = 2L, seed = 20260305L),
  list(id = "sf_n160_d90_seed20260305", n = 160L, d = 90L, graph = "scale-free", g = NA_integer_, seed = 20260305L)
)

methods <- list(
  list(method = "mb", scr = FALSE, sym = "or"),
  list(method = "tiger", scr = FALSE, sym = "or"),
  list(method = "glasso", scr = FALSE, sym = "or"),
  list(method = "ct", scr = FALSE, sym = "or")
)

ensure_data <- function(cfg) {
  out_file <- file.path(data_dir, paste0(cfg$id, ".rds"))
  if (!file.exists(out_file)) {
    set.seed(cfg$seed)
    gen_args <- list(n = cfg$n, d = cfg$d, graph = cfg$graph, verbose = FALSE)
    if (!is.na(cfg$g)) {
      gen_args$g <- cfg$g
    }
    sim <- do.call(huge.generator, gen_args)
    saveRDS(sim$data, out_file)
  }
  out_file
}

case_script <- file.path(repo_root, "benchmark", "active_set_case.R")
if (!file.exists(case_script)) {
  stop(sprintf("Missing case script: %s", case_script))
}

rscript_bin <- file.path(R.home("bin"), "Rscript")

rows <- list()
k <- 0L

for (ds in datasets) {
  data_file <- ensure_data(ds)

  for (mcfg in methods) {
    case_id <- paste(ds$id, mcfg$method, paste0("scr", as.integer(mcfg$scr)), sep = "__")
    out_file <- file.path(result_dir, paste0(case_id, ".rds"))
    log_file <- file.path(log_dir, paste0(case_id, ".log"))
    base_file <- file.path(baseline_dir, paste0(case_id, ".rds"))

    cmd_args <- c(
      case_script,
      sprintf("data_file=%s", data_file),
      sprintf("method=%s", mcfg$method),
      sprintf("out_file=%s", out_file),
      sprintf("nlambda=%d", nlambda),
      sprintf("lambda_min_ratio=%s", format(lambda_min_ratio, scientific = FALSE)),
      sprintf("soft_timeout=%d", soft_timeout),
      sprintf("scr=%s", ifelse(mcfg$scr, "true", "false")),
      sprintf("sym=%s", mcfg$sym)
    )

    start_time <- proc.time()[["elapsed"]]
    exit_status <- suppressWarnings(
      system2(
        command = rscript_bin,
        args = cmd_args,
        stdout = if (keep_logs) log_file else TRUE,
        stderr = if (keep_logs) log_file else TRUE,
        timeout = hard_timeout
      )
    )
    wall <- proc.time()[["elapsed"]] - start_time

    case_status <- "missing"
    err_msg <- NA_character_
    lambda_diff <- NA_real_
    path_diff_count <- NA_real_
    icov_max_diff <- NA_real_

    if (file.exists(out_file)) {
      cur <- readRDS(out_file)
      case_status <- cur$status
      if (!identical(case_status, "ok")) {
        err_msg <- ifelse(is.null(cur$message), "unknown error", cur$message)
      }

      if (identical(mode, "record") && identical(case_status, "ok")) {
        saveRDS(cur, base_file)
      }

      if (identical(mode, "compare")) {
        if (!file.exists(base_file)) {
          case_status <- "no_baseline"
          err_msg <- "baseline file missing"
        } else if (identical(case_status, "ok")) {
          base <- readRDS(base_file)
          if (!identical(base$status, "ok")) {
            case_status <- "bad_baseline"
            err_msg <- "baseline status is not ok"
          } else {
            if (length(cur$lambda) != length(base$lambda)) {
              lambda_diff <- Inf
            } else {
              lambda_diff <- max(abs(cur$lambda - base$lambda))
            }
            path_diff_count <- safe_count_diff(cur$path, base$path)
            icov_max_diff <- safe_max_abs_diff(cur$icov, base$icov)
          }
        }
      }
    } else if (isTRUE(exit_status == 124L)) {
      case_status <- "timeout"
      err_msg <- sprintf("hard timeout reached: %ds", hard_timeout)
    } else {
      case_status <- "missing_output"
      err_msg <- "output file not found"
    }

    k <- k + 1L
    rows[[k]] <- data.frame(
      run_id = run_id,
      mode = mode,
      dataset = ds$id,
      method = mcfg$method,
      exit_status = as.integer(exit_status),
      case_status = case_status,
      elapsed_runner_sec = as.numeric(wall),
      out_file = out_file,
      log_file = if (keep_logs) log_file else "",
      lambda_max_abs_diff = lambda_diff,
      path_mismatch_count = path_diff_count,
      icov_max_abs_diff = icov_max_diff,
      error = ifelse(is.na(err_msg), "", err_msg),
      stringsAsFactors = FALSE
    )
  }
}

summary_df <- do.call(rbind, rows)
summary_file <- file.path(run_dir, ifelse(mode == "record", "summary_record.csv", "summary_compare.csv"))
write.csv(summary_df, summary_file, row.names = FALSE)

cat(sprintf("Mode: %s\n", mode))
cat(sprintf("Run ID: %s\n", run_id))
cat(sprintf("Summary: %s\n", summary_file))
cat("Case status counts:\n")
print(table(summary_df$case_status))

if (any(summary_df$case_status %in% c("error", "timeout", "missing_output", "no_baseline", "bad_baseline"))) {
  quit(status = 2L)
}
