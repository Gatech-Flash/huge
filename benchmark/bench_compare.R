#!/usr/bin/env Rscript
# =================================================================
# Comprehensive benchmark: huge 1.5 (local) vs huge 1.4 (CRAN)
# Compares computational accuracy and efficiency across all methods,
# graph types, and problem sizes.
# =================================================================

CRAN_LIB <- "/tmp/huge_cran_lib"
LOCAL_LIB <- .libPaths()[1]
SEED <- 2024
REPS <- 5  # timing repetitions

# ── helpers ──────────────────────────────────────────────────────
run_one_version <- function(lib_path, label) {
  # Load the specific version
  if (label == "cran") {
    # prepend CRAN lib so it takes priority
    library(huge, lib.loc = lib_path)
  } else {
    library(huge)
  }
  ver <- as.character(packageVersion("huge"))
  cat(sprintf("\n========== %s  (huge %s) ==========\n", label, ver))

  methods  <- c("glasso", "mb", "ct", "tiger")
  graphs   <- c("hub", "cluster", "band", "random", "scale-free")
  sizes    <- list(
    small  = list(n = 200, d = 50),
    medium = list(n = 200, d = 200)
  )

  results <- list()

  for (sz_name in names(sizes)) {
    n <- sizes[[sz_name]]$n
    d <- sizes[[sz_name]]$d

    for (gr in graphs) {
      set.seed(SEED)
      sim <- huge.generator(n = n, d = d, graph = gr, verbose = FALSE)
      theta_true <- as.matrix(sim$theta)
      omega_true <- sim$omega

      for (meth in methods) {
        tag <- paste(sz_name, gr, meth, sep = "|")
        cat(sprintf("  %-40s", tag))

        # --- timing (average of REPS runs) ---
        times <- numeric(REPS)
        est <- NULL
        for (r in seq_len(REPS)) {
          t0 <- proc.time()[3]
          est <- tryCatch(
            huge(sim$data, method = meth, verbose = FALSE),
            error = function(e) NULL
          )
          times[r] <- proc.time()[3] - t0
        }
        if (is.null(est)) {
          cat("  FAILED\n")
          next
        }
        avg_time <- mean(times)

        # --- accuracy: pick the path element closest to true sparsity ---
        true_sparsity <- sum(theta_true != 0) / (d * (d - 1))
        best_idx <- which.min(abs(est$sparsity - true_sparsity))
        adj_est  <- as.matrix(est$path[[best_idx]])

        # confusion matrix (off-diagonal only)
        diag(theta_true) <- 0
        diag(adj_est)    <- 0
        P <- (theta_true != 0)
        Q <- (adj_est != 0)

        tp <- sum(P & Q)
        fp <- sum(!P & Q)
        fn <- sum(P & !Q)
        tn <- sum(!P & !Q)
        tpr <- if ((tp + fn) > 0) tp / (tp + fn) else 0
        fpr <- if ((fp + tn) > 0) fp / (fp + tn) else 0
        precision <- if ((tp + fp) > 0) tp / (tp + fp) else 0
        recall    <- tpr
        f1  <- if ((precision + recall) > 0) 2 * precision * recall / (precision + recall) else 0

        # Frobenius norm of precision matrix error (glasso & tiger only)
        frob <- NA
        if (meth %in% c("glasso", "tiger") && !is.null(est$icov)) {
          icov_est <- as.matrix(est$icov[[best_idx]])
          frob <- norm(icov_est - omega_true, "F") / norm(omega_true, "F")
        }

        rec <- data.frame(
          version   = label,
          size      = sz_name,
          graph     = gr,
          method    = meth,
          n         = n,
          d         = d,
          time_mean = round(avg_time, 4),
          f1        = round(f1, 4),
          tpr       = round(tpr, 4),
          fpr       = round(fpr, 4),
          precision = round(precision, 4),
          frob_rel  = round(frob, 4),
          best_idx  = best_idx,
          stringsAsFactors = FALSE
        )
        results[[tag]] <- rec
        cat(sprintf("  time=%.4fs  F1=%.4f  TPR=%.4f  FPR=%.4f  Frob=%.4s\n",
                    avg_time, f1, tpr, fpr,
                    if (is.na(frob)) "  NA" else sprintf("%.4f", frob)))
      }
    }
  }

  # --- model selection benchmark ---
  cat("\n  --- model selection ---\n")
  for (sz_name in names(sizes)) {
    n <- sizes[[sz_name]]$n
    d <- sizes[[sz_name]]$d

    for (gr in c("hub", "cluster", "band")) {
      set.seed(SEED)
      sim <- huge.generator(n = n, d = d, graph = gr, verbose = FALSE)
      theta_true <- as.matrix(sim$theta)

      # glasso + ebic
      est_gl <- tryCatch(huge(sim$data, method = "glasso", verbose = FALSE), error = function(e) NULL)
      if (!is.null(est_gl)) {
        t0 <- proc.time()[3]
        sel <- tryCatch(huge.select(est_gl, criterion = "ebic", verbose = FALSE), error = function(e) NULL)
        sel_time <- proc.time()[3] - t0
        if (!is.null(sel)) {
          adj_sel <- as.matrix(sel$refit)
          diag(adj_sel) <- 0; diag(theta_true) <- 0
          P <- (theta_true != 0); Q <- (adj_sel != 0)
          tp <- sum(P & Q); fp <- sum(!P & Q); fn <- sum(P & !Q)
          prec <- if ((tp+fp)>0) tp/(tp+fp) else 0
          rec  <- if ((tp+fn)>0) tp/(tp+fn) else 0
          f1s  <- if ((prec+rec)>0) 2*prec*rec/(prec+rec) else 0
          tag <- paste(sz_name, gr, "glasso+ebic", sep = "|")
          cat(sprintf("  %-40s  time=%.4fs  F1=%.4f\n", tag, sel_time, f1s))
          results[[tag]] <- data.frame(
            version=label, size=sz_name, graph=gr, method="glasso+ebic",
            n=n, d=d, time_mean=round(sel_time,4),
            f1=round(f1s,4), tpr=round(rec,4), fpr=NA,
            precision=round(prec,4), frob_rel=NA, best_idx=NA,
            stringsAsFactors=FALSE
          )
        }
      }

      # mb + stars
      est_mb <- tryCatch(huge(sim$data, method = "mb", verbose = FALSE), error = function(e) NULL)
      if (!is.null(est_mb)) {
        t0 <- proc.time()[3]
        sel2 <- tryCatch(huge.select(est_mb, criterion = "stars", rep.num = 10, verbose = FALSE), error = function(e) NULL)
        sel_time2 <- proc.time()[3] - t0
        if (!is.null(sel2)) {
          adj_sel2 <- as.matrix(sel2$refit)
          diag(adj_sel2) <- 0; diag(theta_true) <- 0
          P <- (theta_true != 0); Q <- (adj_sel2 != 0)
          tp <- sum(P & Q); fp <- sum(!P & Q); fn <- sum(P & !Q)
          prec <- if ((tp+fp)>0) tp/(tp+fp) else 0
          rec  <- if ((tp+fn)>0) tp/(tp+fn) else 0
          f1s  <- if ((prec+rec)>0) 2*prec*rec/(prec+rec) else 0
          tag <- paste(sz_name, gr, "mb+stars", sep = "|")
          cat(sprintf("  %-40s  time=%.4fs  F1=%.4f\n", tag, sel_time2, f1s))
          results[[tag]] <- data.frame(
            version=label, size=sz_name, graph=gr, method="mb+stars",
            n=n, d=d, time_mean=round(sel_time2,4),
            f1=round(f1s,4), tpr=round(rec,4), fpr=NA,
            precision=round(prec,4), frob_rel=NA, best_idx=NA,
            stringsAsFactors=FALSE
          )
        }
      }
    }
  }

  do.call(rbind, results)
}

# ── run CRAN version in a subprocess ─────────────────────────────
cat("Installing/checking CRAN version...\n")
cran_script <- tempfile(fileext = ".R")
cran_out    <- tempfile(fileext = ".rds")

# Write the helpers + run_one_version into a self-contained child script
writeLines(c(
  sprintf("CRAN_LIB <- '%s'", CRAN_LIB),
  sprintf("SEED <- %d", SEED),
  sprintf("REPS <- %d", REPS),
  # inline the function
  deparse(run_one_version, control = "all"),
  sprintf('.libPaths(c("%s", .libPaths()))', CRAN_LIB),
  sprintf('res <- run_one_version("%s", "cran")', CRAN_LIB),
  sprintf('saveRDS(res, "%s")', cran_out)
), cran_script)

cat("Running CRAN 1.4 benchmarks...\n")
system2("Rscript", cran_script, stdout = "", stderr = "")
cran_res <- readRDS(cran_out)

# ── run local version ────────────────────────────────────────────
cat("Running local 1.5 benchmarks...\n")
local_res <- run_one_version(LOCAL_LIB, "local")

# ── merge and compare ────────────────────────────────────────────
all_res <- rbind(cran_res, local_res)

cat("\n\n")
cat("================================================================\n")
cat("              COMPARISON: CRAN 1.4 vs LOCAL 1.5\n")
cat("================================================================\n\n")

# Pivot for side-by-side comparison
cran_df  <- cran_res
local_df <- local_res

key_cols <- c("size", "graph", "method")
merged <- merge(cran_df, local_df, by = key_cols, suffixes = c(".cran", ".local"))

# Compute speedup and accuracy delta
merged$speedup    <- round(merged$time_mean.cran / merged$time_mean.local, 2)
merged$f1_delta   <- round(merged$f1.local - merged$f1.cran, 4)
merged$frob_delta <- round(merged$frob_rel.local - merged$frob_rel.cran, 4)

cat("─── Timing & F1 Comparison ───\n\n")
cat(sprintf("%-12s %-12s %-14s %10s %10s %8s %8s %8s %10s\n",
            "Size", "Graph", "Method",
            "Time.CRAN", "Time.1.5", "Speedup",
            "F1.CRAN", "F1.1.5", "F1.delta"))
cat(paste(rep("-", 104), collapse = ""), "\n")

for (i in seq_len(nrow(merged))) {
  r <- merged[i, ]
  cat(sprintf("%-12s %-12s %-14s %10.4f %10.4f %7.2fx %8.4f %8.4f %+10.4f\n",
              r$size, r$graph, r$method,
              r$time_mean.cran, r$time_mean.local, r$speedup,
              r$f1.cran, r$f1.local, r$f1_delta))
}

cat("\n─── Precision Matrix Frobenius Error (glasso & tiger only) ───\n\n")
frob_rows <- merged[!is.na(merged$frob_rel.cran) & !is.na(merged$frob_rel.local), ]
if (nrow(frob_rows) > 0) {
  cat(sprintf("%-12s %-12s %-14s %12s %12s %12s\n",
              "Size", "Graph", "Method", "Frob.CRAN", "Frob.1.5", "Delta"))
  cat(paste(rep("-", 78), collapse = ""), "\n")
  for (i in seq_len(nrow(frob_rows))) {
    r <- frob_rows[i, ]
    cat(sprintf("%-12s %-12s %-14s %12.4f %12.4f %+12.4f\n",
                r$size, r$graph, r$method,
                r$frob_rel.cran, r$frob_rel.local, r$frob_delta))
  }
}

cat("\n─── Summary Statistics ───\n\n")
cat(sprintf("Average speedup (1.5 vs CRAN):     %.2fx\n", mean(merged$speedup, na.rm = TRUE)))
cat(sprintf("Median  speedup:                    %.2fx\n", median(merged$speedup, na.rm = TRUE)))
cat(sprintf("Average F1 delta (1.5 - CRAN):     %+.4f\n", mean(merged$f1_delta, na.rm = TRUE)))
frob_d <- merged$frob_delta[!is.na(merged$frob_delta)]
if (length(frob_d) > 0)
  cat(sprintf("Average Frob delta (1.5 - CRAN):   %+.4f  (negative = better)\n", mean(frob_d)))

# Save full results
out_path <- file.path(dirname(sys.frame(1)$ofile %||% "."), "bench_results.rds")
tryCatch({
  saveRDS(all_res, "/Users/tourzhao/Desktop/huge-master/benchmark/bench_results.rds")
  cat(sprintf("\nFull results saved to benchmark/bench_results.rds\n"))
}, error = function(e) NULL)
