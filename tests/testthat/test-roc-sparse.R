roc_pair_reference <- function(path, theta) {
  # Enumerate ordered vertex pairs; R ROC counts both directions, including
  # asymmetric graphs. This oracle does not use sparse intersection algebra.
  theta = as.matrix(theta)
  pairs = expand.grid(i = seq_len(nrow(theta)), j = seq_len(ncol(theta)))
  pairs = pairs[pairs$i != pairs$j, , drop = FALSE]
  coordinates = as.matrix(pairs)
  truth = which(theta[coordinates] != 0)
  rates = vapply(path, function(graph) {
    prediction = which(as.matrix(graph)[coordinates] != 0)
    hits = length(intersect(prediction, truth))
    c(tp = hits / length(truth),
      fp = length(setdiff(prediction, truth)) / (nrow(pairs) - length(truth)),
      F1 = 2 * hits / (length(prediction) + length(truth)))
  }, numeric(3))
  points = rates[, order(rates["fp", ], rates["tp", ]), drop = FALSE]
  auc = 0
  if(ncol(points) > 1) {
    for(i in seq_len(ncol(points) - 1)) {
      auc = auc + (points["fp", i + 1] - points["fp", i]) *
        (points["tp", i] + points["tp", i + 1]) / 2
    }
  }
  list(tp = unname(rates["tp", ]), fp = unname(rates["fp", ]),
       F1 = unname(rates["F1", ]), AUC = unname(auc))
}

expect_roc_pair_reference <- function(path, theta) {
  expected = roc_pair_reference(path, theta)
  before = serialize(list(path, theta), NULL)
  actual = huge.roc(path, theta, verbose = FALSE)
  expect_identical(serialize(list(path, theta), NULL), before)
  expect_s3_class(actual, "roc")
  expect_named(actual, c("tp", "fp", "F1", "AUC"))
  expect_equal(unclass(actual), expected, tolerance = 1e-15)
  invisible(actual)
}

test_that("ROC sparse and mixed graphs match directed-pair enumeration", {
  grDevices::pdf(NULL)
  on.exit(grDevices::dev.off(), add = TRUE)
  for(seed in c(7, 19, 83)) {
    set.seed(seed)
    theta = matrix(sample(c(0, 0, 0, -2, 3), 81, TRUE), 9)
    theta[1, 2] = -2
    theta[1, 3] = 0
    path = replicate(4, matrix(sample(c(0, 0, -3, 2), 81, TRUE), 9), simplify = FALSE)
    sparse.theta = Matrix::Matrix(theta, sparse = TRUE)
    sparse.path = lapply(path, Matrix::Matrix, sparse = TRUE)
    expect_roc_pair_reference(sparse.path, sparse.theta)
    expect_roc_pair_reference(list(sparse.path[[1]], path[[2]],
                                   sparse.path[[3]], path[[4]]), sparse.theta)
    expect_roc_pair_reference(sparse.path, theta)
    expect_roc_pair_reference(lapply(path, as.data.frame), as.data.frame(theta))
  }
})

test_that("ROC sums triplet duplicates before counting support", {
  grDevices::pdf(NULL)
  on.exit(grDevices::dev.off(), add = TRUE)
  for(values in list(c(3, -3, 0), c(1e16, -1e16, 1), c(1e16, 1, -1e16))) {
    graph = methods::new("dgTMatrix", Dim = c(4L, 4L),
      i = as.integer(c(0, 0, 0, 0, 0, 2, 3, 3)),
      j = as.integer(c(1, 3, 1, 2, 1, 3, 0, 3)),
      x = c(values[1], 2, values[2], 0, values[3], -3, 5, 1))
    expect_roc_pair_reference(list(graph, Matrix::Diagonal(4), matrix(1, 4, 4)), graph)
  }
  # Duplicate logical and pattern entries represent one edge, not two.
  logical.graph = methods::new("lgTMatrix", Dim = c(4L, 4L),
    i = c(0L, 0L, 2L), j = c(1L, 1L, 0L), x = c(TRUE, FALSE, TRUE))
  pattern.graph = methods::new("ngTMatrix", Dim = c(4L, 4L),
    i = c(0L, 0L, 2L), j = c(1L, 1L, 0L))
  expect_roc_pair_reference(list(logical.graph, pattern.graph), logical.graph)
  expect_roc_pair_reference(list(logical.graph, pattern.graph), pattern.graph)
})

test_that("ROC honors symmetric, unit triangular, diagonal, and permutation storage", {
  grDevices::pdf(NULL)
  on.exit(grDevices::dev.off(), add = TRUE)
  theta = matrix(0, 4, 4)
  theta[1, 2] = theta[2, 1] = -2
  theta[1, 3] = theta[3, 1] = 3
  symmetric = Matrix::forceSymmetric(Matrix::Matrix(theta, sparse = TRUE))
  unit = methods::new("dtCMatrix", Dim = c(4L, 4L),
    i = c(0L, 0L), p = c(0L, 0L, 1L, 2L, 2L), x = c(2, -3), diag = "U")
  permutation = methods::as(diag(4)[c(2, 1, 3, 4), ], "pMatrix")
  path = list(symmetric, unit, Matrix::Diagonal(4), permutation)
  expect_roc_pair_reference(path, symmetric)
  expect_roc_pair_reference(path, unit)
  expect_roc_pair_reference(path, permutation)
  # This lower-only edge contributes to R's directed count.
  lower = matrix(0, 4, 4)
  lower[4, 1] = -5
  actual = expect_roc_pair_reference(list(lower), lower)
  expect_identical(actual$tp, 1)
  expect_identical(actual$fp, 0)
})

test_that("ROC handles large sparse dimensions without dense conversion or integer overflow", {
  grDevices::pdf(NULL)
  on.exit(grDevices::dev.off(), add = TRUE)
  d = 70000L
  theta = Matrix::sparseMatrix(i = c(1, 8), j = c(2, 13), x = c(2, -1), dims = c(d, d))
  pred = Matrix::sparseMatrix(i = c(1, 3, 5, 9), j = c(2, 4, 5, 10),
                              x = c(4, -2, 9, 0), dims = c(d, d))
  before = serialize(list(theta, pred), NULL)
  original = getS3method("as.matrix", "Matrix")
  registerS3method("as.matrix", "Matrix", function(x, ...) {
    stop("Sparse ROC must not convert Matrix objects to dense matrices")
  }, envir = asNamespace("base"))
  on.exit(registerS3method("as.matrix", "Matrix", original,
                          envir = asNamespace("base")), add = TRUE)
  actual = huge.roc(list(pred), theta, verbose = FALSE)
  expect_identical(actual$tp, .5)
  expect_identical(actual$fp, 1 / (as.double(d) * (d - 1) - 2))
  expect_identical(actual$F1, .5)
  expect_identical(actual$AUC, 0)
  expect_identical(serialize(list(theta, pred), NULL), before)
})

test_that("ROC validates all entries, dimensions, and one-class truth", {
  grDevices::pdf(NULL)
  on.exit(grDevices::dev.off(), add = TRUE)
  theta = matrix(0, 3, 3)
  theta[1, 2] = 1
  for(bad in c(NA_real_, NaN, Inf, -Inf)) {
    for(position in list(c(1, 1), c(3, 1), c(1, 3))) {
      invalid = theta
      invalid[position[1], position[2]] = bad
      for(input in list(invalid, Matrix::Matrix(invalid, sparse = TRUE))) {
        before = serialize(input, NULL)
        expect_error(huge.roc(list(theta), input, verbose = FALSE), "theta.*non-finite")
        expect_error(huge.roc(list(input), theta, verbose = FALSE), "path.*non-finite")
        expect_identical(serialize(input, NULL), before)
      }
    }
  }
  expect_error(huge.roc(list(), theta, verbose = FALSE), "at least one adjacency")
  for(input in list(matrix(0, 2, 3), Matrix::Matrix(matrix(0, 2, 3), sparse = TRUE))) {
    expect_error(huge.roc(list(theta), input, verbose = FALSE), "theta.*square")
    expect_error(huge.roc(list(input), theta, verbose = FALSE), "path.*dimensions")
  }
  expect_error(huge.roc(list(matrix("bad", 3, 3)), theta, verbose = FALSE), "numeric 2D")
  for(input in list(Matrix::Diagonal(3), Matrix::Matrix(matrix(0, 3, 3), sparse = TRUE),
                   Matrix::Matrix(matrix(1, 3, 3), sparse = TRUE))) {
    expect_error(huge.roc(list(theta), input, verbose = FALSE), "ROC/AUC.*one-class")
  }
})

test_that("ROC AUC tie ordering preserves returned input order and graphics state", {
  grDevices::pdf(NULL)
  on.exit(grDevices::dev.off(), add = TRUE)
  theta = matrix(0, 4, 4)
  theta[1, 2] = theta[2, 1] = theta[1, 3] = theta[3, 1] = 1
  low = matrix(0, 4, 4)
  low[3, 4] = low[4, 3] = 1
  high = theta + low
  complete = matrix(1, 4, 4)
  for(path in list(list(low, high, complete), list(high, low, complete),
                  list(low, high, high, complete))) {
    graphics::par(mfrow = c(2, 2), mar = c(2, 2, 1, 1))
    before = graphics::par(c("mfrow", "mar", "cex", "mex"))
    actual = expect_roc_pair_reference(lapply(path, Matrix::Matrix, sparse = TRUE),
                                       Matrix::Matrix(theta, sparse = TRUE))
    expect_identical(actual$AUC, .75)
    expect_equal(graphics::par(names(before)), before, tolerance = 1e-15)
  }
})

test_that("ROC preserves named dense inputs on success and dimension errors", {
  grDevices::pdf(NULL)
  on.exit(grDevices::dev.off(), add = TRUE)
  theta = matrix(0, 4, 4, dimnames = list(letters[1:4], LETTERS[1:4]))
  theta[1, 2] = -2
  theta[3, 1] = 4
  diag(theta) = 7
  path = list(theta * 2, t(theta), theta * 0)
  expect_roc_pair_reference(path, theta)
  expect_roc_pair_reference(lapply(path, as.data.frame), as.data.frame(theta))
  rectangular = theta[, 1:3, drop = FALSE]
  before = serialize(rectangular, NULL)
  expect_error(huge.roc(list(theta), rectangular, verbose = FALSE), "theta.*square")
  expect_error(huge.roc(list(rectangular), theta, verbose = FALSE), "path.*dimensions")
  expect_identical(serialize(rectangular, NULL), before)
})
