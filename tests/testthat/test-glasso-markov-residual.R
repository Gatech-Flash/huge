test_that("glasso preserves the independent Markov completion across residual dispatch", {
  cases <- list(c(511, 1), c(512, 1), c(512, 1e-100), c(512, 1e100))
  for (case in cases) {
    d <- as.integer(case[1])
    scale <- case[2]
    rho <- .7
    penalty <- .55
    distance <- abs(outer(seq_len(d), seq_len(d), "-"))
    covariance <- scale * rho^distance
    original <- serialize(covariance, NULL)
    a <- 1 + penalty
    b <- rho - penalty
    r <- b / a
    denominator <- a * (1 - r^2)
    expected.precision <- diag((1 + r^2) / denominator, d)
    expected.precision[1, 1] <- expected.precision[d, d] <- 1 / denominator
    adjacent <- seq_len(d - 1L)
    expected.precision[cbind(adjacent, adjacent + 1L)] <- -r / denominator
    expected.precision[cbind(adjacent + 1L, adjacent)] <- -r / denominator
    expected.covariance <- a * r^distance
    # Independent KKT justification: all inactive distances have slack.
    expect_lt(rho^2 - b^2 / a, penalty)
    expect_lt(rho^3, penalty)

    result <- huge.glasso(covariance, lambda = scale * penalty,
                          input.type = "covariance", cov.output = TRUE,
                          verbose = FALSE)

    expect_identical(serialize(covariance, NULL), original)
    expect_equal(as.matrix(result$path[[1]]) != 0, distance == 1)
    expect_equal(result$icov[[1]] != 0, distance <= 1)
    expect_lt(max(abs(result$icov[[1]] * scale - expected.precision)), 2e-4)
    expect_lt(max(abs(result$cov[[1]] / scale - expected.covariance)), 2e-4)
    trace <- (2 + (d - 2) * (1 + r^2) - 2 * (d - 1) * rho * r) / denominator
    expected.loglik <- -d * log(scale) - d * log(a) -
      (d - 1) * log1p(-r^2) - trace
    expect_lt(abs(result$loglik[1] - expected.loglik), 2e-3)
    expect_equal(result$df[1], d - 1)
  }
})
