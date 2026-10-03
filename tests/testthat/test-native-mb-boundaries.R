test_that("R MB adapters preserve sparse coefficients across active-cache boundaries", {
  # For response 1, c = G beta + lambda sign(beta) specifies a known unique
  # Lasso solution. Active sizes bracket both limits of the packed Gram cache.
  for (active.size in c(15L, 16L, 256L, 257L)) {
    d = active.size + 1L
    positions = seq_len(active.size) - 1L
    gram = .2 ^ abs(outer(positions, positions, "-"))
    expected = ifelse(positions %% 2L == 0L, 1, -1) *
      (.005 + .01 * (positions %% 7L) / 6)
    penalty = .005
    cross = as.vector(gram %*% expected) + penalty * sign(expected)
    correlation = diag(d)
    correlation[-1, -1] = gram
    correlation[1, -1] = correlation[-1, 1] = cross
    lambda = c(.25, penalty, penalty)

    for (screened in c(FALSE, TRUE)) {
      label = paste("active", active.size, "screened", screened)
      if (screened) {
        # Keep every predictor, but reverse its order to exercise the native
        # zero-based screen indices independently of sparse output ordering.
        indices = vapply(seq_len(d), function(response) {
          rev(seq_len(d)[-response]) - 1L
        }, integer(active.size))
        out = .Call("_huge_SPMBscr", correlation, lambda, length(lambda), d,
                    indices, active.size, PACKAGE = "huge")
      } else {
        out = .Call("_huge_SPMBgraph", correlation, lambda, length(lambda), d,
                    PACKAGE = "huge")
      }

      expect_false(out$hit_max_iter, info = label)
      expect_length(out$col_cnz, d + 1L)
      expect_identical(out$col_cnz[[1]], 0L)
      expect_true(all(diff(out$col_cnz) >= 0), info = label)
      expect_equal(tail(out$col_cnz, 1), length(out$x), info = label)
      expect_length(out$row_idx, length(out$x))
      expect_true(all(out$row_idx >= 0 & out$row_idx < length(lambda) * d),
                  info = label)

      # The adapter packs lambda blocks down rows and responses in columns.
      stacked = methods::new(
        "dgCMatrix", Dim = as.integer(c(length(lambda) * d, d)),
        x = out$x, p = out$col_cnz, i = out$row_idx
      )
      expect_true(methods::validObject(stacked), info = label)
      expect_equal(Matrix::nnzero(stacked[seq_len(d), , drop = FALSE]), 0,
                   info = label)
      for (path.index in 2:3) {
        rows = (path.index - 1L) * d + seq_len(d)
        coefficients = as.matrix(stacked[rows, , drop = FALSE])
        expect_true(all(diag(coefficients) == 0), info = label)
        expect_identical(sign(coefficients[-1, 1]), sign(expected), info = label)
        expect_true(max(abs(coefficients[-1, 1] - expected)) <= 1e-6,
                    info = label)
      }
    }
  }
})
