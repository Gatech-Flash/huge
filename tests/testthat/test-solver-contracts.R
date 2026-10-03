# Check optimality independently in the sample domain on singular Gram inputs.
solver_observations = function(seed) {
  set.seed(seed)
  x = matrix(rnorm(24 * 48), nrow = 24)
  x[, seq(2, 48, 2)] = x[, seq(2, 48, 2)] +
    0.7 * x[, seq(1, 48, 2)]
  x
}

nodewise_kkt_error = function(score, beta, penalty) {
  error = abs(score - penalty * sign(beta))
  inactive = beta == 0
  error[inactive] = pmax(abs(score[inactive]) - penalty, 0)
  diag(error) = 0
  max(error)
}

test_that("high-dimensional MB satisfies sample-domain KKT conditions", {
  for (seed in c(101, 303, 707)) {
    x = solver_observations(seed)
    original = x
    z = scale(x)
    fit = huge(x, method = "mb", lambda = c(.65, .32, .15, .15),
               input.type = "data", verbose = FALSE)
    expect_identical(x, original)
    expect_gt(sum(fit$path[[3]]), 0)
    for (index in seq_along(fit$lambda)) {
      beta = as.matrix(fit$beta[[index]])
      residual = z - z %*% beta
      score = crossprod(z, residual) / (nrow(x) - 1)
      expect_true(all(is.finite(beta)))
      expect_true(all(diag(beta) == 0))
      expect_lte(nodewise_kkt_error(score, beta, fit$lambda[[index]]), 1e-4)
      objective = .5 * colSums(residual^2) / (nrow(x) - 1) +
        fit$lambda[[index]] * colSums(abs(beta))
      expect_true(all(objective <= .5 + 1e-12))
    }
  }
})

test_that("high-dimensional TIGER returns a certified replayable prefix", {
  for (seed in c(101, 303, 707)) {
    x = solver_observations(seed)
    z = scale(x)
    messages = character()
    fit = withCallingHandlers(
      huge(x, method = "tiger", nlambda = 6, lambda.min.ratio = .2,
           input.type = "data", verbose = FALSE),
      warning = function(condition) {
        messages <<- c(messages, conditionMessage(condition))
        invokeRestart("muffleWarning")
      }
    )
    expect_gte(length(fit$lambda), 1)
    expect_lte(length(fit$lambda), 6)
    if (length(fit$lambda) < 6) {
      expect_length(messages, 1)
      expect_match(messages[[1]], "certified prefix")
    } else {
      expect_length(messages, 0)
    }
    for (index in seq_along(fit$lambda)) {
      beta = as.matrix(fit$beta[[index]])
      residual = z - z %*% beta
      tau = sqrt(colSums(residual^2) / (nrow(x) - 1))
      score = sweep(crossprod(z, residual) / (nrow(x) - 1), 2, tau, "/")
      expect_true(all(is.finite(beta)))
      expect_true(all(diag(beta) == 0))
      expect_true(all(tau > 0))
      expect_lte(nodewise_kkt_error(score, beta, fit$lambda[[index]]), 1e-6)
      expect_true(all(tau + fit$lambda[[index]] * colSums(abs(beta)) <= 1 + 1e-12))
    }
    replay = huge(x, method = "tiger", lambda = fit$lambda,
                  input.type = "data", verbose = FALSE)
    expect_equal(replay$beta, fit$beta, tolerance = 1e-12)
    expect_equal(replay$icov, fit$icov, tolerance = 1e-12)
  }
})

test_that("high-dimensional glasso satisfies primal-dual conditions", {
  for (seed in c(101, 303, 707)) {
    x = solver_observations(seed)
    correlation = cor(x)
    fit = huge(x, method = "glasso", lambda = c(.65, .32, .15, .15),
               cov.output = TRUE, input.type = "data", verbose = FALSE)
    for (index in seq_along(fit$lambda)) {
      precision = fit$icov[[index]]
      covariance = fit$cov[[index]]
      penalty = fit$lambda[[index]]
      expect_true(all(is.finite(precision)))
      expect_identical(precision, t(precision))
      factor = chol(precision)
      expect_lte(max(rowSums(abs(covariance %*% precision - diag(ncol(x))))), 1e-3)
      dual = solve(precision) - correlation
      error = abs(dual - penalty * sign(precision))
      inactive = precision == 0
      error[inactive] = pmax(abs(dual[inactive]) - penalty, 0)
      expect_lte(max(error), 1e-4)
      loglik = 2 * sum(log(diag(factor))) - sum(correlation * precision)
      expect_equal(fit$loglik[[index]], loglik, tolerance = 1e-10)
      objective = -loglik + penalty * sum(abs(precision))
      expect_lte(objective, ncol(x) * (log1p(penalty) + 1) + 1e-10)
    }
  }
})
