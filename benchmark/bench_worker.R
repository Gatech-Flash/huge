#!/usr/bin/env Rscript
# Worker script: run benchmarks for one version of huge
# Usage: Rscript bench_worker.R <label> <output.rds> [cran_lib_path]

args <- commandArgs(trailingOnly = TRUE)
label    <- args[1]                          # "cran" or "local"
out_file <- args[2]                          # output RDS path
cran_lib <- if (length(args) >= 3) args[3] else NULL

if (label == "cran" && !is.null(cran_lib)) {
  .libPaths(c(cran_lib, .libPaths()))
}
library(huge)
ver <- as.character(packageVersion("huge"))
cat(sprintf("\n========== %s  (huge %s) ==========\n", label, ver))

SEED <- 2024
REPS <- 5

methods <- c("glasso", "mb", "ct", "tiger")
graphs  <- c("hub", "cluster", "band", "random", "scale-free")
sizes   <- list(
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
      if (is.null(est)) { cat("  FAILED\n"); next }
      avg_time <- mean(times)

      true_sp  <- sum(theta_true != 0) / (d * (d - 1))
      best_idx <- which.min(abs(est$sparsity - true_sp))
      adj_est  <- as.matrix(est$path[[best_idx]])
      diag(theta_true) <- 0; diag(adj_est) <- 0
      P <- (theta_true != 0); Q <- (adj_est != 0)
      tp <- sum(P & Q); fp <- sum(!P & Q); fn <- sum(P & !Q); tn <- sum(!P & !Q)
      tpr <- if ((tp+fn)>0) tp/(tp+fn) else 0
      fpr <- if ((fp+tn)>0) fp/(fp+tn) else 0
      prec_v <- if ((tp+fp)>0) tp/(tp+fp) else 0
      f1 <- if ((prec_v+tpr)>0) 2*prec_v*tpr/(prec_v+tpr) else 0

      frob <- NA_real_
      if (meth %in% c("glasso","tiger") && !is.null(est$icov)) {
        icov_est <- as.matrix(est$icov[[best_idx]])
        frob <- norm(icov_est - omega_true, "F") / norm(omega_true, "F")
      }

      results[[tag]] <- data.frame(
        version=label, size=sz_name, graph=gr, method=meth,
        n=n, d=d, time_mean=round(avg_time,4),
        f1=round(f1,4), tpr=round(tpr,4), fpr=round(fpr,4),
        precision=round(prec_v,4), frob_rel=round(frob,4), best_idx=best_idx,
        stringsAsFactors=FALSE)
      cat(sprintf("  time=%.4fs  F1=%.4f  TPR=%.4f  FPR=%.4f  Frob=%s\n",
                  avg_time, f1, tpr, fpr,
                  if (is.na(frob)) "  NA" else sprintf("%.4f", frob)))
    }
  }
}

# model selection
cat("\n  --- model selection ---\n")
for (sz_name in names(sizes)) {
  n <- sizes[[sz_name]]$n; d <- sizes[[sz_name]]$d
  for (gr in c("hub", "cluster", "band")) {
    set.seed(SEED)
    sim <- huge.generator(n=n, d=d, graph=gr, verbose=FALSE)
    theta_true <- as.matrix(sim$theta)

    # glasso + ebic
    est_gl <- tryCatch(huge(sim$data, method="glasso", verbose=FALSE), error=function(e) NULL)
    if (!is.null(est_gl)) {
      t0 <- proc.time()[3]
      sel <- tryCatch(huge.select(est_gl, criterion="ebic", verbose=FALSE), error=function(e) NULL)
      sel_time <- proc.time()[3] - t0
      if (!is.null(sel)) {
        adj_sel <- as.matrix(sel$refit); diag(adj_sel) <- 0; diag(theta_true) <- 0
        P <- (theta_true!=0); Q <- (adj_sel!=0)
        tp <- sum(P&Q); fp <- sum(!P&Q); fn <- sum(P&!Q)
        prec_v <- if((tp+fp)>0) tp/(tp+fp) else 0
        rec_v  <- if((tp+fn)>0) tp/(tp+fn) else 0
        f1s <- if((prec_v+rec_v)>0) 2*prec_v*rec_v/(prec_v+rec_v) else 0
        tag <- paste(sz_name, gr, "glasso+ebic", sep="|")
        cat(sprintf("  %-40s  time=%.4fs  F1=%.4f\n", tag, sel_time, f1s))
        results[[tag]] <- data.frame(version=label, size=sz_name, graph=gr,
          method="glasso+ebic", n=n, d=d, time_mean=round(sel_time,4),
          f1=round(f1s,4), tpr=round(rec_v,4), fpr=NA_real_,
          precision=round(prec_v,4), frob_rel=NA_real_, best_idx=NA_integer_,
          stringsAsFactors=FALSE)
      }
    }

    # mb + stars
    est_mb <- tryCatch(huge(sim$data, method="mb", verbose=FALSE), error=function(e) NULL)
    if (!is.null(est_mb)) {
      t0 <- proc.time()[3]
      sel2 <- tryCatch(huge.select(est_mb, criterion="stars", rep.num=10, verbose=FALSE), error=function(e) NULL)
      sel_time2 <- proc.time()[3] - t0
      if (!is.null(sel2)) {
        adj_sel2 <- as.matrix(sel2$refit); diag(adj_sel2) <- 0; diag(theta_true) <- 0
        P <- (theta_true!=0); Q <- (adj_sel2!=0)
        tp <- sum(P&Q); fp <- sum(!P&Q); fn <- sum(P&!Q)
        prec_v <- if((tp+fp)>0) tp/(tp+fp) else 0
        rec_v  <- if((tp+fn)>0) tp/(tp+fn) else 0
        f1s <- if((prec_v+rec_v)>0) 2*prec_v*rec_v/(prec_v+rec_v) else 0
        tag <- paste(sz_name, gr, "mb+stars", sep="|")
        cat(sprintf("  %-40s  time=%.4fs  F1=%.4f\n", tag, sel_time2, f1s))
        results[[tag]] <- data.frame(version=label, size=sz_name, graph=gr,
          method="mb+stars", n=n, d=d, time_mean=round(sel_time2,4),
          f1=round(f1s,4), tpr=round(rec_v,4), fpr=NA_real_,
          precision=round(prec_v,4), frob_rel=NA_real_, best_idx=NA_integer_,
          stringsAsFactors=FALSE)
      }
    }
  }
}

res <- do.call(rbind, results)
saveRDS(res, out_file)
cat(sprintf("\nResults saved to %s (%d rows)\n", out_file, nrow(res)))
