#!/usr/bin/env Rscript
# Merge two worker RDS outputs and print comparison report

cran_res  <- readRDS("benchmark/bench_cran.rds")
local_res <- readRDS("benchmark/bench_local.rds")

key_cols <- c("size", "graph", "method")
m <- merge(cran_res, local_res, by = key_cols, suffixes = c(".cran", ".local"))
m$speedup  <- round(m$time_mean.cran / m$time_mean.local, 2)
m$f1_delta <- round(m$f1.local - m$f1.cran, 4)
m$frob_delta <- round(m$frob_rel.local - m$frob_rel.cran, 4)

# sort
m <- m[order(m$size, m$graph, m$method), ]

cat("\n")
cat("================================================================\n")
cat("         COMPARISON: CRAN 1.4  vs  LOCAL 1.5\n")
cat("================================================================\n\n")

# ── 1. Timing & F1 ──
cat("--- Timing & F1 Score ---\n\n")
cat(sprintf("%-8s %-12s %-14s %9s %9s %8s %7s %7s %9s\n",
            "Size", "Graph", "Method",
            "T(CRAN)", "T(1.5)", "Speedup",
            "F1.CRAN", "F1.1.5", "F1.delta"))
cat(paste(rep("-", 97), collapse=""), "\n")
for (i in seq_len(nrow(m))) {
  r <- m[i,]
  cat(sprintf("%-8s %-12s %-14s %9.4f %9.4f %7.2fx %7.4f %7.4f %+9.4f\n",
              r$size, r$graph, r$method,
              r$time_mean.cran, r$time_mean.local, r$speedup,
              r$f1.cran, r$f1.local, r$f1_delta))
}

# ── 2. Frobenius norm (glasso & tiger) ──
frob_rows <- m[!is.na(m$frob_rel.cran) & !is.na(m$frob_rel.local), ]
if (nrow(frob_rows) > 0) {
  cat("\n--- Precision Matrix Relative Frobenius Error (glasso & tiger) ---\n\n")
  cat(sprintf("%-8s %-12s %-14s %11s %11s %11s\n",
              "Size", "Graph", "Method", "Frob.CRAN", "Frob.1.5", "Delta"))
  cat(paste(rep("-", 71), collapse=""), "\n")
  for (i in seq_len(nrow(frob_rows))) {
    r <- frob_rows[i,]
    cat(sprintf("%-8s %-12s %-14s %11.4f %11.4f %+11.4f\n",
                r$size, r$graph, r$method,
                r$frob_rel.cran, r$frob_rel.local, r$frob_delta))
  }
}

# ── 3. TPR / FPR detail ──
detail <- m[!is.na(m$tpr.cran) & !is.na(m$fpr.cran), ]
cat("\n--- TPR / FPR Detail ---\n\n")
cat(sprintf("%-8s %-12s %-14s %7s %7s %7s %7s\n",
            "Size", "Graph", "Method", "TPR.CR", "TPR.15", "FPR.CR", "FPR.15"))
cat(paste(rep("-", 70), collapse=""), "\n")
for (i in seq_len(nrow(detail))) {
  r <- detail[i,]
  cat(sprintf("%-8s %-12s %-14s %7.4f %7.4f %7.4f %7.4f\n",
              r$size, r$graph, r$method,
              r$tpr.cran, r$tpr.local, r$fpr.cran, r$fpr.local))
}

# ── 4. Summary ──
cat("\n--- Summary ---\n\n")

# Estimation only (exclude model selection methods)
est_rows <- m[!grepl("\\+", m$method), ]
sel_rows <- m[grepl("\\+", m$method), ]

cat(sprintf("  Graph estimation (4 methods x 5 graphs x 2 sizes = %d configs):\n", nrow(est_rows)))
cat(sprintf("    Mean speedup :  %.2fx\n", mean(est_rows$speedup, na.rm=TRUE)))
cat(sprintf("    Median speedup: %.2fx\n", median(est_rows$speedup, na.rm=TRUE)))
cat(sprintf("    Mean F1 delta:  %+.4f\n", mean(est_rows$f1_delta, na.rm=TRUE)))
frob_d <- est_rows$frob_delta[!is.na(est_rows$frob_delta)]
if (length(frob_d) > 0)
  cat(sprintf("    Mean Frob delta: %+.4f  (negative = 1.5 better)\n", mean(frob_d)))

if (nrow(sel_rows) > 0) {
  cat(sprintf("\n  Model selection (%d configs):\n", nrow(sel_rows)))
  cat(sprintf("    Mean speedup:   %.2fx\n", mean(sel_rows$speedup, na.rm=TRUE)))
  cat(sprintf("    Mean F1 delta:  %+.4f\n", mean(sel_rows$f1_delta, na.rm=TRUE)))
}

# Accuracy equivalence check
cat("\n  Accuracy equivalence:\n")
if (all(abs(m$f1_delta) < 0.02, na.rm = TRUE)) {
  cat("    All F1 deltas within +/-0.02 => EQUIVALENT\n")
} else {
  diffs <- m[abs(m$f1_delta) >= 0.02, c("size","graph","method","f1.cran","f1.local","f1_delta")]
  cat("    Some F1 deltas exceed 0.02:\n")
  print(diffs, row.names = FALSE)
}

cat("\n")
