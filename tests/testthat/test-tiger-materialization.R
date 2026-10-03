# Preserve precision bits separately from ordinary numerical comparisons:
# R's default identical() treats positive and negative zero as equal.
tiger_precision_bytes <- function(x) writeBin(as.double(x), raw(), size = 8L)

test_that("tiger precision preserves negative zeros on full empty paths", {
  for (d in c(1L, 7L)) {
    covariance <- diag(d)
    expected <- matrix(-0.0, d, d)
    diag(expected) <- 1
    fit <- huge.tiger(covariance, lambda = c(.8, .2, .2), verbose = FALSE)
    expect_length(fit$icov, 3L)
    for (precision in fit$icov) {
      expect_identical(dim(precision), c(d, d))
      expect_identical(tiger_precision_bytes(precision), tiger_precision_bytes(expected))
    }
    expect_equal(as.numeric(fit$df), rep(0, 3L * d))
    expect_true(all(vapply(fit$path, Matrix::nnzero, integer(1)) == 0L))
  }
})

test_that("tiger precision matches the independent two-variable solution", {
  lambda <- c(.6, .2, .2)
  for (rho in c(-.5, .5)) {
    covariance <- matrix(c(1, rho, rho, 1), 2, 2)
    original <- serialize(covariance, NULL)
    fit <- huge.tiger(covariance, lambda = lambda, verbose = FALSE)
    expect_identical(fit$lambda, lambda)
    for (i in seq_along(lambda)) {
      if (lambda[i] >= abs(rho)) {
        expected <- matrix(-0.0, 2, 2)
        diag(expected) <- 1
        expect_identical(tiger_precision_bytes(fit$icov[[i]]), tiger_precision_bytes(expected))
      } else {
        tau <- sqrt((1-rho^2)/(1-lambda[i]^2))
        beta <- sign(rho) * (abs(rho)-lambda[i]*tau)
        expected <- matrix(c(1, -beta, -beta, 1), 2, 2)/(tau*tau)
        expect_equal(fit$icov[[i]], expected, tolerance = 2e-6)
        expect_equal(as.matrix(fit$beta[[i]]), matrix(c(0, beta, beta, 0), 2, 2),
                     tolerance = 2e-6)
      }
    }
    expect_identical(serialize(covariance, NULL), original)
  }
})

test_that("tiger long truncated paths retain an exact replayable precision prefix", {
  set.seed(722)
  x <- matrix(rnorm(24L * 48L), 24L, 48L)
  original <- serialize(x, NULL)
  expect_warning(
    fit <- huge.tiger(x, nlambda = 120, lambda.min.ratio = .02, verbose = FALSE),
    "certified prefix"
  )
  count <- length(fit$lambda)
  expect_gt(count, 1L)
  expect_lt(count, 120L)
  expect_length(fit$icov, count)
  expect_length(fit$beta, count)
  expect_length(fit$path, count)
  expect_length(fit$sparsity, count)
  expect_equal(ncol(fit$df), count)
  expect_silent(replay <- huge.tiger(x, lambda = fit$lambda, verbose = FALSE))
  expect_identical(replay$lambda, fit$lambda)
  for (i in seq_len(count)) {
    expect_identical(tiger_precision_bytes(replay$icov[[i]]), tiger_precision_bytes(fit$icov[[i]]))
    expect_identical(as.matrix(replay$path[[i]]), as.matrix(fit$path[[i]]))
  }
  expect_error(huge.tiger(x, lambda = c(fit$lambda, 1e-6), verbose = FALSE),
               "could not certify a supplied lambda")
  expect_identical(serialize(x, NULL), original)
})
