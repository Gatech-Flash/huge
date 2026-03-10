#!/usr/bin/env Rscript

# Run one huge() case and serialize enough information for regression checks.

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

matrix_list_to_array <- function(lst, integer_mode = FALSE) {
  if (is.null(lst) || length(lst) == 0) {
    return(NULL)
  }
  d <- nrow(as.matrix(lst[[1]]))
  k <- length(lst)
  mode_type <- if (integer_mode) "integer" else "double"
  arr <- array(0, dim = c(d, d, k))
  storage.mode(arr) <- mode_type
  for (i in seq_len(k)) {
    mat <- as.matrix(lst[[i]])
    if (integer_mode) {
      arr[, , i] <- ifelse(mat != 0, 1L, 0L)
    } else {
      arr[, , i] <- mat
    }
  }
  arr
}

opt <- parse_args(args)

data_file <- opt$data_file
method <- opt$method
out_file <- opt$out_file

if (is.null(data_file) || is.null(method) || is.null(out_file)) {
  stop("Missing required args: data_file, method, out_file")
}

nlambda <- as.integer(ifelse(is.null(opt$nlambda), "8", opt$nlambda))
lambda_min_ratio <- as.numeric(ifelse(is.null(opt$lambda_min_ratio), "0.2", opt$lambda_min_ratio))
soft_timeout <- as.numeric(ifelse(is.null(opt$soft_timeout), "0", opt$soft_timeout))
scr <- as_bool(opt$scr, default = FALSE)
sym <- ifelse(is.null(opt$sym), "or", opt$sym)

suppressPackageStartupMessages(library(huge))

dir.create(dirname(out_file), recursive = TRUE, showWarnings = FALSE)

if (!file.exists(data_file)) {
  stop(sprintf("data_file not found: %s", data_file))
}

x <- readRDS(data_file)
if (!is.matrix(x)) {
  stop(sprintf("Expected a matrix in %s", data_file))
}

if (!is.finite(soft_timeout) || soft_timeout < 0) {
  soft_timeout <- 0
}

build_call <- function() {
  if (method == "mb") {
    return(list(
      x = x,
      method = "mb",
      nlambda = nlambda,
      lambda.min.ratio = lambda_min_ratio,
      sym = sym,
      verbose = FALSE
    ))
  }
  if (method == "glasso") {
    return(list(
      x = x,
      method = "glasso",
      nlambda = nlambda,
      lambda.min.ratio = lambda_min_ratio,
      scr = scr,
      verbose = FALSE
    ))
  }
  if (method == "tiger") {
    return(list(
      x = x,
      method = "tiger",
      nlambda = nlambda,
      lambda.min.ratio = lambda_min_ratio,
      sym = sym,
      verbose = FALSE
    ))
  }
  if (method == "ct") {
    return(list(
      x = x,
      method = "ct",
      nlambda = nlambda,
      lambda.min.ratio = min(lambda_min_ratio, 0.2),
      verbose = FALSE
    ))
  }
  stop(sprintf("Unsupported method: %s", method))
}

result <- tryCatch({
  if (soft_timeout > 0) {
    setTimeLimit(elapsed = soft_timeout, transient = TRUE)
  }
  t0 <- proc.time()[["elapsed"]]
  est <- do.call(huge, build_call())
  elapsed <- proc.time()[["elapsed"]] - t0
  setTimeLimit(cpu = Inf, elapsed = Inf, transient = TRUE)

  path_arr <- matrix_list_to_array(est$path, integer_mode = TRUE)
  icov_arr <- matrix_list_to_array(est$icov, integer_mode = FALSE)
  path_edges <- if (!is.null(est$path)) {
    vapply(est$path, function(m) sum(as.matrix(m)[upper.tri(as.matrix(m))] != 0), numeric(1))
  } else {
    numeric(0)
  }

  list(
    status = "ok",
    method = method,
    n = nrow(x),
    d = ncol(x),
    elapsed = elapsed,
    lambda = est$lambda,
    sparsity = est$sparsity,
    path_edges = path_edges,
    path = path_arr,
    icov = icov_arr
  )
}, error = function(e) {
  list(
    status = "error",
    method = method,
    message = conditionMessage(e)
  )
})

saveRDS(result, out_file)

if (!identical(result$status, "ok")) {
  quit(status = 2L)
}
