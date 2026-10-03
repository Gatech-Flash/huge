# Freeze the original dense Nonparanormal equations as an independent oracle.
# Keep dimensions small: its covariance and Kronecker matrices each contain d^4
# entries, and it deliberately does not call any production inference helper.
dense_nonparanormal_reference <- function(x, T, method) {
  n <- nrow(x)
  d <- ncol(x)
  U <- matrix(0, d, d)
  G <- replicate(n, matrix(0, d, d), simplify = FALSE)
  concordance <- matrix(0, n, n)
  for (j in seq_len(d)) {
    for (k in seq_len(d)) {
      if (j == k) {
        U[j, k] <- 1
        next
      }
      for (i in seq_len(n)) {
        for (other in seq_len(n)) {
          concordance[i, other] <- sign(x[i, j] - x[other, j]) *
            sign(x[i, k] - x[other, k])
          G[[i]][j, k] <- G[[i]][j, k] - pi / 2 * concordance[i, other]
        }
      }
      U[j, k] <- sin(pi / 2 * sum(concordance) / ((n - 1) * n))
      for (i in seq_len(n)) {
        G[[i]][j, k] <- G[[i]][j, k] / (n - 1) + asin(U[j, k])
      }
    }
  }
  F <- apply(U, 1, function(column) sqrt(1 - column^2))
  R <- matrix(0, d^2, d^2)
  for (i in seq_len(n)) {
    R <- R + as.matrix(as.vector(F * G[[i]])) %*% as.vector(F * G[[i]])
  }
  R <- R / n
  T.kron <- kronecker(T, T)
  statistic <- matrix(0, d, d)
  for (j in seq_len(d)) {
    for (k in seq_len(d)) {
      index <- (k - 1) * d + j
      w <- as.matrix(-T.kron[, index][-index] / T.kron[index, index])
      covariance <- as.matrix(R[index, ][-index])
      sigma <- sqrt(R[index, index] - 2 * t(covariance) %*% w +
        t(w) %*% R[-index, -index] %*% w)
      ej <- matrix(0, d, 1)
      ek <- matrix(0, d, 1)
      ej[j] <- 1
      ek[k] <- 1
      if (method == "score") {
        T.h <- T
        T.h[j, k] <- 0
        score <- t(ej) %*% t(T.h) %*% U %*% T.h %*% ek /
          (T[j, j] * T[k, k])
        statistic[j, k] <- score * sqrt(n) / (2 * sigma)
      } else {
        TU <- T %*% U
        UT <- U %*% T
        wald <- (T[j, k] * t(ej) %*% TU %*% ek +
          T[j, k] * t(ej) %*% UT %*% ek -
          t(ej) %*% t(T) %*% UT %*% ek) /
          (t(ej) %*% TU %*% ek + t(ej) %*% UT %*% ek - 1)
        statistic[j, k] <- wald * sqrt(n) / (2 * sigma * T[j, j] * T[k, k])
      }
    }
  }
  2 * (1 - pnorm(abs(statistic)))
}

test_that("streamed Nonparanormal inference agrees with dense equations", {
  set.seed(813)
  for (ties in c(FALSE, TRUE)) {
    x <- matrix(rnorm(12 * 5), 12, 5)
    if (ties) x <- round(x)
    for (asymmetric in c(FALSE, TRUE)) {
      T <- diag(5) + matrix(rnorm(25, sd = .08), 5, 5)
      if (!asymmetric) T <- (T + t(T)) / 2
      dimnames(T) <- list(letters[1:5], LETTERS[1:5])
      dimnames(x) <- list(paste0("sample", 1:12), letters[1:5])
      adj <- matrix(seq_len(25) %% 3 == 0, 5, 5)
      for (method in c("score", "wald")) {
        reference <- dense_nonparanormal_reference(x, T, method)
        result <- huge.inference(
          x, T, adj, alpha = .5, type = "Nonparanormal", method = method
        )
        info <- paste(ties, asymmetric, method)
        expect_identical(names(result), c("data", "p", "error"), info = info)
        expect_identical(result$data, x, info = info)
        expect_null(dimnames(result$p), info = info)
        expect_identical(is.na(result$p), is.na(reference), info = info)
        expect_equal(result$p, reference, tolerance = 1e-11, info = info)
        expected.error <- sum(reference < .5 & adj == 0 & row(adj) != col(adj)) / 25
        expect_equal(result$error, expected.error, tolerance = 0, info = info)
      }
    }
  }
})

test_that("streamed inference preserves rank limits and extreme finite data", {
  cases <- list(
    matrix(c(0, 1, 0, 1), 2, 2),
    matrix(c(0, 1, 1, 0), 2, 2),
    cbind(c(1e308, -1e308, 0), c(0, 0, 1)),
    cbind(c(-2, -1, 0, 1, 2, 3), c(3, -1, 2, -2, 1, 0)) * 1e-300
  )
  for (x in cases) {
    for (method in c("score", "wald")) {
      reference <- dense_nonparanormal_reference(x, diag(2), method)
      result <- huge.inference(
        x, diag(2), matrix(0, 2, 2), type = "Nonparanormal", method = method
      )
      expect_identical(is.nan(result$p), is.nan(reference))
      expect_equal(result$p, reference, tolerance = 1e-12)
      expect_true(all(is.finite(result$p[row(result$p) != col(result$p)])))
    }
  }
})

test_that("rank comparisons accept the full finite integer range", {
  bound <- .Machine$integer.max
  x <- cbind(c(-bound, 0L, bound), c(0L, 1L, 0L))
  expect_type(x, "integer")
  for (method in c("score", "wald")) {
    reference <- dense_nonparanormal_reference(x * 1.0, diag(2), method)
    expect_no_warning(result <- huge.inference(
      x, diag(2), matrix(0, 2, 2), type = "Nonparanormal", method = method
    ))
    expect_identical(result$data, x)
    expect_equal(result$p, reference, tolerance = 1e-13)
  }
})

test_that("score diagonals preserve the direct quadratic form near diagonal T", {
  x <- cbind(c(-2, -1, 0, 1, 2, 3), c(3, -1, 2, -2, 1, 0))
  for (epsilon in c(0, 1e-4, 1e-8, 1e-12)) {
    T <- matrix(c(1, epsilon, 2 * epsilon, 1), 2, 2)
    reference <- dense_nonparanormal_reference(x, T, "score")
    result <- huge.inference(x, T, matrix(0, 2, 2), type = "Nonparanormal")
    expect_identical(is.nan(result$p), is.nan(reference))
    expect_equal(result$p, reference, tolerance = 1e-13)
  }
})

test_that("streamed inference agrees with dense equations across T column scales", {
  set.seed(815)
  x <- matrix(rnorm(10 * 3), 10, 3)
  T <- diag(3) + matrix(rnorm(9, sd = .08), 3, 3)
  T <- sweep(T, 2, c(1e-60, 1, 1e60), "*")
  for (method in c("score", "wald")) {
    reference <- dense_nonparanormal_reference(x, T, method)
    result <- huge.inference(x, T, matrix(0, 3, 3),
      type = "Nonparanormal", method = method)
    expect_identical(is.na(result$p), is.na(reference))
    expect_equal(result$p, reference, tolerance = 1e-11)
  }
})

test_that("streamed inference retains invalid variance scale checks", {
  x <- cbind(c(-2, -1, 0, 1, 2, 3), c(3, -1, 2, -2, 1, 0))
  for (T in list(diag(2) * 1e-200, diag(2) * 1e200)) {
    for (method in c("score", "wald")) {
      expect_error(
        huge.inference(x, T, matrix(0, 2, 2), type = "Nonparanormal", method = method),
        "Products of T diagonal entries must remain finite and positive"
      )
    }
  }
  extreme.ratio <- matrix(c(1e-100, 1e100, 5e99, 1e-100), 2, 2)
  for (method in c("score", "wald")) {
    reference <- suppressWarnings(dense_nonparanormal_reference(x, extreme.ratio, method))
    expect_true(any(!is.finite(reference[row(reference) != col(reference)])))
    expect_error(
      huge.inference(x, extreme.ratio, matrix(0, 2, 2),
        type = "Nonparanormal", method = method),
      "Inference produced non-finite edge p-values;.*numerically degenerate"
    )
  }
})

test_that("streamed inference avoids quartic memory allocations", {
  skip_if_not(capabilities("profmem"))
  set.seed(814)
  n <- 12L
  d <- 64L
  x <- matrix(rnorm(n * d), n, d)
  T <- diag(d) + matrix(rnorm(d * d, sd = .005), d, d)
  adj <- matrix(0, d, d)
  profile <- tempfile("huge-inference-memory-", fileext = ".log")
  run.profile <- function() {
    on.exit(Rprofmem(NULL), add = TRUE)
    Rprofmem(profile)
    huge.inference(x, T, adj, type = "Nonparanormal")
  }
  result <- run.profile()
  allocations <- readLines(profile, warn = FALSE)
  unlink(profile)
  sizes <- as.numeric(sub(" .*", "", allocations[grepl("^[0-9]+", allocations)]))
  # A single old d^2-by-d^2 matrix needs 128 MiB at d = 64.
  # Allow ample interpreter/BLAS overhead above the streamed d-by-d workspace.
  expect_lt(max(c(0, sizes)), 8 * 1024^2)
  expect_equal(dim(result$p), c(d, d))
  expect_true(all(is.finite(result$p)))
  expect_true(all(result$p >= 0 & result$p <= 1))
})
