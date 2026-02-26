"""R-reference parity helpers for pyhuge native backend."""

from __future__ import annotations

import subprocess
import tempfile
import textwrap
from pathlib import Path

import numpy as np


def has_r_huge() -> bool:
    """Return True when local R and package huge are available."""

    try:
        out = subprocess.run(
            ["R", "-q", "-e", 'cat(requireNamespace("huge", quietly=TRUE))'],
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception:
        return False
    return "TRUE" in out.stdout


def run_r_ct_reference(
    x: np.ndarray,
    lambda_ct: np.ndarray,
    *,
    rep_num: int = 6,
    stars_thresh: float = 0.1,
    seed: int = 123,
) -> dict[str, np.ndarray | float | int]:
    """Run R huge reference for ct + stars on fixed input."""

    with tempfile.TemporaryDirectory(prefix="pyhuge_rref_ct_") as td:
        out_dir = Path(td)
        x_path = out_dir / "x.csv"
        lam_path = out_dir / "lam.txt"
        np.savetxt(x_path, x, delimiter=",")
        np.savetxt(lam_path, lambda_ct)

        script = out_dir / "run_ref.R"
        script.write_text(
            textwrap.dedent(
                """
                args <- commandArgs(trailingOnly = TRUE)
                x <- as.matrix(read.csv(args[1], header=FALSE))
                lam <- as.numeric(scan(args[2], quiet=TRUE))
                out_dir <- args[3]
                rep_num <- as.integer(args[4])
                stars_thresh <- as.numeric(args[5])
                seed <- as.integer(args[6])

                suppressMessages(library(huge))
                fit <- huge(x, method='ct', lambda=lam, verbose=FALSE)
                set.seed(seed)
                sel <- huge.select(fit, criterion='stars', rep.num=rep_num, stars.thresh=stars_thresh, verbose=FALSE)

                write.table(fit$lambda, file.path(out_dir, 'lambda.txt'), row.names=FALSE, col.names=FALSE)
                write.table(fit$sparsity, file.path(out_dir, 'sparsity.txt'), row.names=FALSE, col.names=FALSE)
                edge_count <- function(mat) {
                  m <- as.matrix(mat)
                  diag(m) <- 0
                  sum(m != 0) / 2
                }
                write.table(sapply(fit$path, edge_count), file.path(out_dir, 'edges.txt'), row.names=FALSE, col.names=FALSE)
                write.table(c(sel$opt.lambda, sel$opt.sparsity, sel$opt.index), file.path(out_dir, 'sel.txt'), row.names=FALSE, col.names=FALSE)
                """
            )
        )

        subprocess.run(
            [
                "Rscript",
                str(script),
                str(x_path),
                str(lam_path),
                str(out_dir),
                str(rep_num),
                str(stars_thresh),
                str(seed),
            ],
            check=True,
            text=True,
            capture_output=True,
        )

        sel = np.loadtxt(out_dir / "sel.txt")
        return {
            "lambda": np.loadtxt(out_dir / "lambda.txt"),
            "sparsity": np.loadtxt(out_dir / "sparsity.txt"),
            "edges": np.loadtxt(out_dir / "edges.txt"),
            "opt_lambda": float(sel[0]),
            "opt_sparsity": float(sel[1]),
            "opt_index": int(sel[2]),
        }


def run_r_ct_default_reference(
    x: np.ndarray,
    *,
    nlambda: int = 20,
    lambda_min_ratio: float = 0.05,
) -> dict[str, np.ndarray]:
    """Run R huge reference for ct default rank-based path construction."""

    with tempfile.TemporaryDirectory(prefix="pyhuge_rref_ct_default_") as td:
        out_dir = Path(td)
        x_path = out_dir / "x.csv"
        np.savetxt(x_path, x, delimiter=",")

        script = out_dir / "run_ref.R"
        script.write_text(
            textwrap.dedent(
                """
                args <- commandArgs(trailingOnly = TRUE)
                x <- as.matrix(read.csv(args[1], header=FALSE))
                out_dir <- args[2]
                nlambda <- as.integer(args[3])
                lambda_min_ratio <- as.numeric(args[4])

                suppressMessages(library(huge))
                fit <- huge(
                  x,
                  method='ct',
                  nlambda=nlambda,
                  lambda.min.ratio=lambda_min_ratio,
                  verbose=FALSE
                )

                write.table(fit$lambda, file.path(out_dir, 'lambda.txt'), row.names=FALSE, col.names=FALSE)
                write.table(fit$sparsity, file.path(out_dir, 'sparsity.txt'), row.names=FALSE, col.names=FALSE)

                d <- ncol(x)
                nlam <- length(fit$path)
                arr <- array(0, dim=c(nlam, d, d))
                for (i in seq_len(nlam)) arr[i,,] <- as.matrix(fit$path[[i]])
                write.table(as.vector(arr), file.path(out_dir, 'path_flat.txt'), row.names=FALSE, col.names=FALSE)
                """
            )
        )

        subprocess.run(
            [
                "Rscript",
                str(script),
                str(x_path),
                str(out_dir),
                str(nlambda),
                str(lambda_min_ratio),
            ],
            check=True,
            text=True,
            capture_output=True,
        )

        lam = np.atleast_1d(np.loadtxt(out_dir / "lambda.txt")).astype(float)
        sparsity = np.atleast_1d(np.loadtxt(out_dir / "sparsity.txt")).astype(float)
        flat = np.atleast_1d(np.loadtxt(out_dir / "path_flat.txt")).astype(float)
        d = int(x.shape[1])
        nlam = int(lam.size)
        path = flat.reshape((nlam, d, d), order="F")

        return {"lambda": lam, "sparsity": sparsity, "path": path}


def run_r_glasso_reference(
    x: np.ndarray,
    lambda_gl: np.ndarray,
) -> dict[str, np.ndarray | float | int]:
    """Run R huge reference for glasso + ebic on fixed input."""

    with tempfile.TemporaryDirectory(prefix="pyhuge_rref_gl_") as td:
        out_dir = Path(td)
        x_path = out_dir / "x.csv"
        lam_path = out_dir / "lam.txt"
        np.savetxt(x_path, x, delimiter=",")
        np.savetxt(lam_path, lambda_gl)

        script = out_dir / "run_ref.R"
        script.write_text(
            textwrap.dedent(
                """
                args <- commandArgs(trailingOnly = TRUE)
                x <- as.matrix(read.csv(args[1], header=FALSE))
                lam <- as.numeric(scan(args[2], quiet=TRUE))
                out_dir <- args[3]

                suppressMessages(library(huge))
                fit <- huge(x, method='glasso', lambda=lam, verbose=FALSE)
                sel <- huge.select(fit, criterion='ebic', verbose=FALSE)

                write.table(fit$lambda, file.path(out_dir, 'lambda.txt'), row.names=FALSE, col.names=FALSE)
                write.table(fit$sparsity, file.path(out_dir, 'sparsity.txt'), row.names=FALSE, col.names=FALSE)
                edge_count <- function(mat) {
                  m <- as.matrix(mat)
                  diag(m) <- 0
                  sum(m != 0) / 2
                }
                write.table(sapply(fit$path, edge_count), file.path(out_dir, 'edges.txt'), row.names=FALSE, col.names=FALSE)
                write.table(c(sel$opt.lambda, sel$opt.sparsity, sel$opt.index), file.path(out_dir, 'sel.txt'), row.names=FALSE, col.names=FALSE)
                """
            )
        )

        subprocess.run(
            ["Rscript", str(script), str(x_path), str(lam_path), str(out_dir)],
            check=True,
            text=True,
            capture_output=True,
        )

        sel = np.loadtxt(out_dir / "sel.txt")
        return {
            "lambda": np.loadtxt(out_dir / "lambda.txt"),
            "sparsity": np.loadtxt(out_dir / "sparsity.txt"),
            "edges": np.loadtxt(out_dir / "edges.txt"),
            "opt_lambda": float(sel[0]),
            "opt_sparsity": float(sel[1]),
            "opt_index": int(sel[2]),
        }
