"""Full-result parity through both public APIs on identical fixed inputs."""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from pyhuge import huge, huge_select
from pyhuge.parity import has_r_huge


@pytest.mark.skipif(not has_r_huge(), reason="requires local R with package huge")
@pytest.mark.parametrize(
    "method,input_type,sym,screened",
    (
        ("mb", "data", "or", False),
        ("mb", "data", "and", False),
        ("mb", "data", "or", True),
        ("glasso", "data", "or", False),
        ("glasso", "covariance", "or", False),
        ("tiger", "data", "or", False),
        ("tiger", "covariance", "and", False),
    ),
)
def test_fixed_high_dimensional_results_match_r(
    tmp_path, method, input_type, sym, screened
):
    # Both languages read these exact observations, independent of their RNGs.
    # n < d exercises singular sample correlations; the path includes a tie
    # after a nonzero fit so this cannot pass by comparing only empty graphs.
    x = np.random.default_rng(101).normal(size=(32, 48))
    x[:, 1::2] += 0.7 * x[:, ::2]
    if input_type == "covariance":
        x = np.cov(x * np.geomspace(0.5, 2.0, x.shape[1]), rowvar=False)
    penalties = np.asarray([0.65, 0.5, 0.5])
    np.savetxt(tmp_path / "x.csv", x, delimiter=",")
    np.savetxt(tmp_path / "lambda.txt", penalties)
    script = tmp_path / "reference.R"
    script.write_text(
        """
        args <- commandArgs(trailingOnly = TRUE)
        directory <- args[[1]]
        method <- args[[2]]
        input.type <- args[[3]]
        sym <- args[[4]]
        screened <- args[[5]] == "true"
        suppressMessages(library(huge))
        x <- as.matrix(read.csv(file.path(directory, "x.csv"), header = FALSE))
        lambda <- scan(file.path(directory, "lambda.txt"), quiet = TRUE)
        arguments <- list(x = x, method = method, lambda = lambda,
                          sym = sym, input.type = input.type, verbose = FALSE)
        if (method == "glasso") arguments$cov.output <- TRUE
        if (screened) {
          arguments$scr <- TRUE
          arguments$scr.num <- 8L
        }
        fit <- do.call(huge, arguments)
        save_values <- function(values, name) {
          writeLines(sprintf("%.17g", as.numeric(values)),
                     file.path(directory, paste0(name, ".txt")))
        }
        save_path <- function(matrices, name) {
          save_values(unlist(lapply(matrices, as.matrix)), name)
        }
        save_values(fit$lambda, "lambda_out")
        save_values(fit$sparsity, "sparsity")
        save_values(fit$df, "df")
        save_path(fit$path, "path")
        if (!is.null(fit$icov)) save_path(fit$icov, "icov")
        if (method == "glasso") {
          save_path(fit$cov, "cov")
          save_values(fit$loglik, "loglik")
          if (input.type == "data") {
            selected <- huge.select(fit, criterion = "ebic", verbose = FALSE)
            save_values(selected$ebic.score, "ebic")
            save_values(selected$opt.index, "opt_index")
          }
        }
        """
    )
    result = subprocess.run(
        [
            "Rscript", str(script), str(tmp_path), method, input_type,
            sym, "true" if screened else "false",
        ],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    options = dict(
        method=method, input_type=input_type, sym=sym,
        lambda_=penalties, verbose=False,
    )
    if method == "glasso":
        options["cov_output"] = True
    if screened:
        options.update(scr=True, scr_num=8)
    actual = huge(x, **options)

    def load(name):
        return np.atleast_1d(np.loadtxt(tmp_path / (name + ".txt")))

    def load_path(name):
        # R writes each column-major matrix, then the next lambda's matrix.
        return load(name).reshape((penalties.size, x.shape[1], x.shape[1])).transpose(0, 2, 1)

    np.testing.assert_array_equal(actual.lambda_path, load("lambda_out"))
    np.testing.assert_array_equal(
        np.stack([matrix.toarray() for matrix in actual.path]), load_path("path")
    )
    assert actual.path[1].nnz > 0
    np.testing.assert_allclose(actual.sparsity, load("sparsity"), rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(np.asarray(actual.df).flatten(order="F"), load("df"))
    if actual.icov is not None:
        np.testing.assert_allclose(actual.icov, load_path("icov"), rtol=1e-9, atol=1e-9)
    if method == "glasso":
        np.testing.assert_allclose(actual.cov, load_path("cov"), rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(actual.loglik, load("loglik"), rtol=0.0, atol=1e-9)
        if input_type == "data":
            selected = huge_select(actual, criterion="ebic", verbose=False)
            np.testing.assert_allclose(selected.ebic_score, load("ebic"), rtol=0.0, atol=1e-7)
            assert selected.opt_index == int(load("opt_index")[0])
