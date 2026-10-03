test_that("covariance normalization preserves mixed scales, names, and input", {
  normalize = getFromNamespace(".huge_validate_estimation_input", "huge")
  variances = c(.Machine$double.xmax, 1e-300, 1e-20, 1, 4, 1e300)
  positions = (seq_along(variances) - 1)^2
  expected = .25 ^ abs(outer(positions, positions, "-"))
  sd = sqrt(variances)
  covariance = sweep(expected * sd, 2, sd, "*")
  diag(covariance) = variances
  covariance[lower.tri(covariance)] = t(covariance)[lower.tri(covariance)]
  dimnames(covariance) = list(letters[1:6], letters[1:6])

  for (order in list(seq_along(variances), rev(seq_along(variances)))) {
    current = covariance[order, order]
    original = current
    actual = normalize(current, input.type = "covariance")$correlation
    expect_identical(current, original)
    expect_identical(dimnames(actual), dimnames(current))
    expect_identical(actual, t(actual))
    # Relative error matters for tiny representable off-diagonal values.
    target = expected[order, order]
    expect_lte(max(abs(unname(actual) / target - 1)), 5e-15)
  }
})

test_that("covariance normalization retains weak extreme-scale correlations", {
  normalize = getFromNamespace(".huge_validate_estimation_input", "huge")
  variances = c(.Machine$double.xmax, 1e-300, 2)
  rho = 1e-200
  weak = (rho * sqrt(variances[[1]])) * sqrt(variances[[2]])
  covariance = diag(variances)
  covariance[1, 2] = covariance[2, 1] = weak
  for (order in list(1:3, 3:1)) {
    actual = normalize(covariance[order, order], input.type = "covariance")$correlation
    location = match(c(1, 2), order)
    expect_lte(abs(actual[location[[1]], location[[2]]] / rho - 1), 1e-14)
    expect_identical(actual, t(actual))
  }
})

test_that("covariance normalization keeps first invalid entry diagnostics", {
  normalize = getFromNamespace(".huge_validate_estimation_input", "huge")
  # All inputs are finite and symmetric. The second variance scale makes a
  # finite off-diagonal overflow only during normalization, so validation
  # must still report the earlier Cauchy-Schwarz violation first.
  for (finite.first in c(TRUE, FALSE)) {
    covariance = diag(c(1, 1e-300, 1, 1e-300))
    if (finite.first) {
      covariance[1, 4] = covariance[4, 1] = 1.1e-150
      covariance[2, 4] = covariance[4, 2] = .Machine$double.xmax
      message = "not a valid covariance"
    } else {
      covariance[1, 4] = covariance[4, 1] = .Machine$double.xmax
      covariance[2, 4] = covariance[4, 2] = 1.1e-300
      message = "finite correlation"
    }
    expect_error(normalize(covariance, input.type = "covariance", require.psd = FALSE),
                 message)
  }
})

test_that("covariance normalization preserves clipping and PSD rejection", {
  normalize = getFromNamespace(".huge_validate_estimation_input", "huge")
  roundoff = matrix(c(1, 1 + 1e-9, 1 + 1e-9, 1), 2)
  expect_identical(normalize(roundoff, input.type = "covariance")$correlation,
                   matrix(1, 2, 2))
  violation = matrix(c(1, 1 + 2e-8, 1 + 2e-8, 1), 2)
  expect_error(normalize(violation, input.type = "covariance"), "valid covariance")
  indefinite = matrix(c(1, .9, .9, .9, 1, 0, .9, 0, 1), 3)
  expect_error(normalize(indefinite, input.type = "covariance"), "positive semidefinite")
  expect_identical(normalize(indefinite, input.type = "covariance", require.psd = FALSE)$correlation,
                   indefinite)
})
