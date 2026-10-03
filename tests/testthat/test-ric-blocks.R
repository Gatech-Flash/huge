test_that("RIC panel edges preserve rotations and input data", {
  set.seed(839)
  for (d in c(1L, 127L, 128L, 129L, 130L, 257L, 258L)) {
    x <- matrix(rnorm(17 * d), 17, d)
    before <- serialize(x, NULL)
    rotations <- c(0L, 8L, 17L)
    expected <- vapply(rotations, function(rotation) {
      rows <- (seq_len(nrow(x)) - 1L + rotation) %% nrow(x) + 1L
      product <- crossprod(x[rows, , drop = FALSE], x)
      if (d == 1L) 0 else max(abs(product[upper.tri(product)]))
    }, numeric(1))
    actual <- vapply(rotations, function(rotation) {
      huge:::RIC(x, d, nrow(x), rotation, 1L)
    }, numeric(1))
    expect_equal(actual, expected, tolerance = 2e-13)
    expect_equal(huge:::RIC(x, d, nrow(x), rotations, length(rotations)),
      min(expected), tolerance = 2e-13)
    expect_identical(serialize(x, NULL), before)
    expect_identical(rotations, c(0L, 8L, 17L))
  }
})

test_that("RIC high-dimensional selection preserves empty and weak refits", {
  n <- 256L
  d <- 255L
  walsh <- outer(0:(n - 1L), seq_len(d), Vectorize(function(i, j) {
    1 - 2 * (sum(as.integer(intToBits(bitwAnd(i, j)))) %% 2)
  }))
  for (strength in c(0, 1e-8)) {
    x <- walsh
    x[, d] <- x[, d] + strength * x[, d - 1L]
    for (scale in c(1, 1e100)) {
      fit <- huge(x * scale, method = "ct", lambda = 1, verbose = FALSE)
      selected <- huge.select(fit, criterion = "ric", rep.num = n, verbose = FALSE)
      if (strength == 0) {
        expect_identical(selected$opt.lambda, 0)
        expect_equal(sum(selected$refit), 0)
      } else {
        expect_gt(selected$opt.lambda, 0)
        expected <- (n - 1) * strength / (n * sqrt(1 + strength^2))
        expect_lte(abs(selected$opt.lambda - expected), 2 * n * .Machine$double.eps)
        expect_equal(sum(selected$refit), 2)
        expect_equal(as.numeric(selected$refit[d - 1L, d]), 1)
        expect_equal(as.numeric(selected$refit[d, d - 1L]), 1)
        refit <- huge(x * scale, method = "ct", lambda = selected$opt.lambda, verbose = FALSE)
        expect_equal(selected$refit, refit$path[[1]], tolerance = 0)
      }
    }
  }
})
