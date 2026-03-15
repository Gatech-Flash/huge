#!/usr/bin/env Rscript
# =============================================================
# Comprehensive benchmark: huge 1.5 (local) vs huge 1.3.5 (CRAN)
# Metrics: objective value gap, computation speed, F1/TPR/FPR, Frobenius error
# Grid: {n} x {d} x {graph} x {method}
# =============================================================

CRAN_LIB <- "/tmp/huge_cran_lib"
SEED     <- 2025
REPS     <- 5     # timing repetitions per cell
OUT_RDS  <- "/Users/tourzhao/Desktop/huge-master/benchmark/bench_vs_cran.rds"

# ── glasso objective: -log det(Omega) + tr(S Omega) + lambda ||Omega||_1 (off-diag)
glasso_obj <- function(Omega, S, lambda) {
  sign_det <- determinant(Omega, logarithm = TRUE)
  if (sign_det$sign <= 0) return(NA_real_)
  logdet     <- as.numeric(sign_det$modulus)
  tr_SOmega  <- sum(S * Omega)
  l1_off     <- sum(abs(Omega[upper.tri(Omega)]))
  -logdet + tr_SOmega + lambda * 2 * l1_off
}

# ── confusion matrix metrics
graph_metrics <- function(theta_true, adj_est) {
  diag_t <- theta_true; diag(diag_t) <- 0
  diag_e <- adj_est;    diag(diag_e) <- 0
  P <- (diag_t != 0); Q <- (diag_e != 0)
  tp <- sum(P & Q); fp <- sum(!P & Q)
  fn <- sum(P & !Q); tn <- sum(!P & !Q)
  tpr  <- if ((tp+fn)>0) tp/(tp+fn) else 0
  fpr  <- if ((fp+tn)>0) fp/(fp+tn) else 0
  prec <- if ((tp+fp)>0) tp/(tp+fp) else 0
  f1   <- if ((prec+tpr)>0) 2*prec*tpr/(prec+tpr) else 0
  c(f1=f1, tpr=tpr, fpr=fpr, precision=prec)
}

# ── benchmark grid ────────────────────────────────────────────
sizes <- list(
  "n100_d50"  = list(n=100,  d=50),
  "n200_d100" = list(n=200,  d=100),
  "n200_d200" = list(n=200,  d=200),
  "n500_d200" = list(n=500,  d=200),
  "n200_d500" = list(n=200,  d=500)
)
graphs  <- c("hub", "cluster", "band", "random", "scale-free")
methods <- c("glasso", "mb", "ct", "tiger")
# g sweeps for hub/cluster to vary effective sparsity; NULL = package default
g_values <- list(hub=c(2,4,8), cluster=c(2,4,8),
                 band=list(NULL), random=list(NULL), "scale-free"=list(NULL))

# ── worker function ───────────────────────────────────────────
run_version <- function(ver_label) {
  library(huge)
  ver <- as.character(packageVersion("huge"))
  cat(sprintf("\n\n========== %s  (huge %s) ==========\n", ver_label, ver))

  results <- list()
  for (sz_name in names(sizes)) {
    n <- sizes[[sz_name]]$n
    d <- sizes[[sz_name]]$d
    for (gr in graphs) {
      g_list <- g_values[[gr]]
      for (gv in g_list) {
        set.seed(SEED)
        sim <- if (!is.null(gv))
          huge.generator(n=n, d=d, graph=gr, g=gv, verbose=FALSE)
        else
          huge.generator(n=n, d=d, graph=gr, verbose=FALSE)
        g_tag      <- if (!is.null(gv)) as.character(gv) else "def"
        theta_true <- as.matrix(sim$theta)
        omega_true <- sim$omega
        S          <- cor(sim$data)
        true_sp    <- sum(theta_true != 0) / (d*(d-1))

        for (meth in methods) {
          tag <- paste(sz_name, gr, g_tag, meth, sep="|")
          cat(sprintf("  %-52s", tag))

          times <- numeric(REPS); est <- NULL
          for (r in seq_len(REPS)) {
            t0  <- proc.time()[3]
            est <- tryCatch(huge(sim$data, method=meth, nlambda=20, verbose=FALSE),
                            error=function(e) NULL)
            times[r] <- proc.time()[3] - t0
          }
          if (is.null(est)) { cat("  FAILED\n"); next }
          avg_time <- mean(times)

          best_idx <- which.min(abs(est$sparsity - true_sp))
          adj_est  <- as.matrix(est$path[[best_idx]])
          gm <- graph_metrics(theta_true, adj_est)

          frob <- NA_real_
          if (meth %in% c("glasso","tiger") && !is.null(est$icov) && !is.null(omega_true)) {
            icov_est <- as.matrix(est$icov[[best_idx]])
            frob <- norm(icov_est - omega_true, "F") / norm(omega_true, "F")
          }

          obj_val <- NA_real_
          if (meth == "glasso" && !is.null(est$icov)) {
            lam_best <- est$lambda[best_idx]
            icov_est <- as.matrix(est$icov[[best_idx]])
            obj_val  <- glasso_obj(icov_est, S, lam_best)
          }

          cat(sprintf("  t=%.3fs  F1=%.4f  Frob=%s  Obj=%s\n",
                      avg_time, gm["f1"],
                      if (is.na(frob))    "    NA" else sprintf("%.4f", frob),
                      if (is.na(obj_val)) "    NA" else sprintf("%.4f", obj_val)))

          results[[tag]] <- data.frame(
            version=ver_label, size=sz_name, graph=gr, g=g_tag, method=meth,
            n=n, d=d, true_sparsity=round(true_sp,4),
            time_mean=round(avg_time,5),
            f1=round(gm["f1"],4), tpr=round(gm["tpr"],4),
            fpr=round(gm["fpr"],4), precision=round(gm["precision"],4),
            frob_rel=round(frob,6), obj_val=round(obj_val,6),
            best_idx=best_idx, best_lambda=round(est$lambda[best_idx],6),
            stringsAsFactors=FALSE, row.names=NULL
          )
        }
      }
    }
  }
  do.call(rbind, results)
}

# ── run CRAN 1.3.5 in subprocess ──────────────────────────────
cat("Running CRAN 1.3.5 benchmarks in subprocess...\n")
env_rds    <- tempfile(fileext=".rds")
cran_rds   <- tempfile(fileext=".rds")
cran_script <- tempfile(fileext=".R")

# Save the shared environment (functions + data) to an RDS
saveRDS(list(
  glasso_obj   = glasso_obj,
  graph_metrics = graph_metrics,
  run_version  = run_version,
  sizes        = sizes,
  graphs       = graphs,
  methods      = methods,
  g_values     = g_values,
  SEED         = SEED,
  REPS         = REPS
), env_rds)

# Minimal child script: load env, prepend CRAN lib, run
writeLines(c(
  sprintf("env <- readRDS('%s')", env_rds),
  "list2env(env, globalenv())",
  sprintf(".libPaths(c('%s', .libPaths()))", CRAN_LIB),
  sprintf("res <- run_version('cran-1.3.5')"),
  sprintf("saveRDS(res, '%s')", cran_rds)
), cran_script)

status <- system2("Rscript", cran_script, stdout="", stderr="")
if (status != 0) stop("CRAN subprocess failed with status ", status)
cran_res <- readRDS(cran_rds)

# ── run local 1.5 ─────────────────────────────────────────────
local_res <- run_version("local-1.5")

# ── merge and report ──────────────────────────────────────────
all_res <- rbind(cran_res, local_res)
saveRDS(all_res, OUT_RDS)

cat("\n\n")
cat(strrep("=", 115), "\n")
cat("                   COMPARISON: huge 1.3.5 (CRAN) vs 1.5 (local)\n")
cat(strrep("=", 115), "\n\n")

keys   <- c("size","graph","g","method")
merged <- merge(cran_res, local_res, by=keys, suffixes=c(".cran",".local"))
merged$speedup    <- round(merged$time_mean.cran / merged$time_mean.local, 2)
merged$f1_delta   <- round(merged$f1.local       - merged$f1.cran,         4)
merged$frob_delta <- round(merged$frob_rel.local - merged$frob_rel.cran,   6)
merged$obj_delta  <- round(merged$obj_val.local  - merged$obj_val.cran,    6)

# Section 1: Speed & accuracy
cat("─── Speed & Graph-Recovery Accuracy ───\n\n")
cat(sprintf("%-18s %-12s %4s %-8s %9s %9s %8s %8s %9s %9s\n",
            "Size","Graph","g","Method","t.cran","t.local","Speedup",
            "F1.cran","F1.local","F1.delta"))
cat(strrep("-", 115), "\n")
for (i in seq_len(nrow(merged))) {
  r <- merged[i,]
  cat(sprintf("%-18s %-12s %4s %-8s %9.4f %9.4f %7.2fx %8.4f %9.4f %+9.4f\n",
              r$size, r$graph, r$g, r$method,
              r$time_mean.cran, r$time_mean.local, r$speedup,
              r$f1.cran, r$f1.local, r$f1_delta))
}

# Section 2: glasso objective value gap
cat("\n─── Glasso Objective Value (lower = better) ───\n\n")
gl_rows <- merged[merged$method=="glasso" & !is.na(merged$obj_val.cran) & !is.na(merged$obj_val.local),]
if (nrow(gl_rows) > 0) {
  cat(sprintf("%-18s %-12s %4s  %14s %14s %15s %14s\n",
              "Size","Graph","g","Obj.cran","Obj.local","Delta(loc-cran)","Frob.delta"))
  cat(strrep("-", 98), "\n")
  for (i in seq_len(nrow(gl_rows))) {
    r <- gl_rows[i,]
    cat(sprintf("%-18s %-12s %4s  %14.6f %14.6f %+15.6f %+14.6f\n",
                r$size, r$graph, r$g,
                r$obj_val.cran, r$obj_val.local, r$obj_delta,
                if (is.na(r$frob_delta)) 0 else r$frob_delta))
  }
}

# Section 3: Summary
cat("\n─── Summary Statistics ───\n\n")
cat(sprintf("  Average speedup (local vs CRAN):      %.2fx\n",  mean(merged$speedup, na.rm=TRUE)))
cat(sprintf("  Median  speedup:                      %.2fx\n",  median(merged$speedup, na.rm=TRUE)))
cat(sprintf("  Max     speedup:                      %.2fx\n",  max(merged$speedup, na.rm=TRUE)))
cat(sprintf("  Average F1 delta (local - CRAN):      %+.4f\n", mean(merged$f1_delta, na.rm=TRUE)))
frob_d <- merged$frob_delta[!is.na(merged$frob_delta)]
if (length(frob_d)>0)
  cat(sprintf("  Average Frob delta (local - CRAN):    %+.6f  (negative = better Omega estimate)\n", mean(frob_d)))
obj_d <- merged$obj_delta[!is.na(merged$obj_delta)]
if (length(obj_d)>0)
  cat(sprintf("  Average Obj  delta (local - CRAN):    %+.6f  (negative = lower objective)\n", mean(obj_d)))

cat("\n─── Per-Method Speedup ───\n\n")
for (m in methods) {
  sub <- merged[merged$method==m,]
  if (nrow(sub)==0) next
  cat(sprintf("  %-8s  mean=%.2fx  median=%.2fx  range=[%.2f, %.2f]\n",
              m, mean(sub$speedup,na.rm=TRUE), median(sub$speedup,na.rm=TRUE),
              min(sub$speedup,na.rm=TRUE), max(sub$speedup,na.rm=TRUE)))
}

cat("\n─── Per-Graph Speedup & F1 delta ───\n\n")
for (gr in graphs) {
  sub <- merged[merged$graph==gr,]
  if (nrow(sub)==0) next
  cat(sprintf("  %-12s  speedup mean=%.2fx  F1 delta mean=%+.4f\n",
              gr, mean(sub$speedup,na.rm=TRUE), mean(sub$f1_delta,na.rm=TRUE)))
}

cat(sprintf("\nFull results saved to: %s  (%d rows)\n", OUT_RDS, nrow(all_res)))
