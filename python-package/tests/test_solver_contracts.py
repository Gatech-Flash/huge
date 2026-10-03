"""Independent numerical checks for high-dimensional solver paths."""

from __future__ import annotations

import numpy as np
import pytest

from pyhuge import huge

native = pytest.importorskip("pyhuge._native_core")


def _observations(seed, shape):
    x = np.random.default_rng(seed).normal(size=shape)
    x[:, 1::2] += 0.7 * x[:, ::2]
    return x


def _standardized(x):
    return (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)


def _residual_scores(z, beta):
    # Evaluate in the sample domain, independently of native Gram updates.
    residual = z - np.einsum("nj,mj->nm", z, beta)
    scores = np.einsum("nm,nj->mj", residual, z) / (z.shape[0] - 1)
    return residual, scores


def _max_nodewise_kkt_error(scores, beta, penalty):
    errors = np.where(
        beta != 0,
        np.abs(scores - penalty * np.sign(beta)),
        np.maximum(np.abs(scores) - penalty, 0.0),
    )
    np.fill_diagonal(errors, 0.0)
    return float(errors.max())


@pytest.mark.parametrize("seed", (101, 303, 707))
@pytest.mark.parametrize("shape", ((18, 40), (32, 80)))
def test_mb_high_dimensional_path_satisfies_lasso_kkt(seed, shape):
    x = _observations(seed, shape)
    z = _standardized(x)
    correlation = np.einsum("ni,nj->ij", z, z) / (shape[0] - 1)
    penalties = np.asarray([0.65, 0.32, 0.15, 0.15])
    out = native.spmb_graph(correlation, penalties)

    assert not out["hit_max_iter"]
    assert np.count_nonzero(out["beta"][-1]) > shape[1]
    for beta, penalty in zip(out["beta"], penalties):
        assert np.isfinite(beta).all()
        np.testing.assert_array_equal(np.diag(beta), np.zeros(shape[1]))
        residual, scores = _residual_scores(z, beta)
        assert _max_nodewise_kkt_error(scores, beta, penalty) <= 1e-4
        objective = (
            0.5 * np.sum(residual * residual, axis=0) / (shape[0] - 1)
            + penalty * np.abs(beta).sum(axis=1)
        )
        # Each response has unit sample variance; beta=0 has objective 1/2.
        assert np.all(objective <= 0.5 + 1e-12)


@pytest.mark.parametrize("seed", (101, 303, 707))
@pytest.mark.parametrize("shape", ((18, 40), (32, 80)))
def test_tiger_high_dimensional_prefix_is_certified_and_replayable(seed, shape):
    x = _observations(seed, shape)
    z = _standardized(x)
    out = native.spmb_graphsqrt(x, None, 6, 0.2)
    penalties = np.asarray(out["lambda"])

    assert 1 <= penalties.size <= 6
    assert out["path_truncated"] == (penalties.size < 6)
    correlation = np.einsum("ni,nj->ij", z, z) / (shape[0] - 1)
    np.fill_diagonal(correlation, 0.0)
    maximum = np.max(np.abs(correlation))
    expected = np.geomspace(maximum, 0.2 * maximum, 6)
    np.testing.assert_allclose(penalties, expected[:penalties.size], atol=1e-14)

    for beta, penalty in zip(out["beta"], penalties):
        assert np.isfinite(beta).all()
        np.testing.assert_array_equal(np.diag(beta), np.zeros(shape[1]))
        residual, scores = _residual_scores(z, beta)
        tau = np.sqrt(np.sum(residual * residual, axis=0) / (shape[0] - 1))
        assert np.all(tau > 0.0)
        assert _max_nodewise_kkt_error(scores / tau[:, None], beta, penalty) <= 1e-6
        # The square-root loss at the feasible zero vector is one.
        assert np.all(tau + penalty * np.abs(beta).sum(axis=1) <= 1.0 + 1e-12)

    replay = native.spmb_graphsqrt(x, penalties)
    assert not replay["path_truncated"]
    np.testing.assert_allclose(replay["beta"], out["beta"], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(replay["icov"], out["icov"], rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("seed", (101, 303, 707))
@pytest.mark.parametrize("shape", ((18, 40), (32, 80)))
def test_glasso_high_dimensional_path_satisfies_primal_dual_contract(seed, shape):
    x = _observations(seed, shape)
    z = _standardized(x)
    correlation = np.einsum("ni,nj->ij", z, z) / (shape[0] - 1)
    fit = huge(
        x, method="glasso", lambda_=[0.65, 0.32, 0.15, 0.15],
        cov_output=True, input_type="data", verbose=False,
    )
    assert any(matrix.nnz for matrix in fit.path)
    for index, (precision, covariance, penalty) in enumerate(
        zip(fit.icov, fit.cov, fit.lambda_path)
    ):
        assert np.isfinite(precision).all()
        np.testing.assert_array_equal(precision, precision.T)
        factor = np.linalg.cholesky(precision)
        product = np.einsum("ij,jk->ik", covariance, precision)
        assert np.linalg.norm(product - np.eye(shape[1]), ord=np.inf) <= 1e-3

        # The diagonal is penalized too: W-S = lambda*sign(Theta) on support.
        dual = np.linalg.inv(precision) - correlation
        errors = np.where(
            precision != 0,
            np.abs(dual - penalty * np.sign(precision)),
            np.maximum(np.abs(dual) - penalty, 0.0),
        )
        assert errors.max() <= 1e-4
        loglik = 2.0 * np.log(np.diag(factor)).sum() - np.sum(correlation * precision)
        assert fit.loglik[index] == pytest.approx(loglik, abs=1e-10)
        objective = -loglik + penalty * np.abs(precision).sum()
        diagonal_objective = shape[1] * (np.log1p(penalty) + 1.0)
        assert objective <= diagonal_objective + 1e-10
        expected_path = precision != 0
        np.fill_diagonal(expected_path, False)
        np.testing.assert_array_equal(fit.path[index].toarray() != 0, expected_path)


@pytest.mark.parametrize("method", ("mb", "glasso", "tiger"))
@pytest.mark.parametrize("seed", (101, 303))
@pytest.mark.parametrize("layout", ("fortran", "strided", "negative", "readonly"))
def test_estimators_preserve_layout_and_readonly_inputs(method, seed, layout):
    x = _observations(seed, (32, 48))
    if layout == "fortran":
        supplied = np.asfortranarray(x)
    elif layout == "strided":
        storage = np.zeros((2 * x.shape[0], 2 * x.shape[1]))
        storage[::2, ::2] = x
        supplied = storage[::2, ::2]
    elif layout == "negative":
        supplied = x[::-1, ::-1].copy()[::-1, ::-1]
    else:
        supplied = x.copy()
    supplied.setflags(write=False)
    lambda_storage = np.asarray([0.65, -1.0, 0.5, -1.0, 0.5, -1.0])
    penalties = lambda_storage[::2]
    penalties.setflags(write=False)
    original_penalties = lambda_storage.copy()
    kwargs = dict(method=method, lambda_=penalties, input_type="data", verbose=False)
    reference = huge(x, **kwargs)
    actual = huge(supplied, **kwargs)

    np.testing.assert_array_equal(supplied, x)
    np.testing.assert_array_equal(lambda_storage, original_penalties)
    np.testing.assert_array_equal(actual.lambda_path, penalties)
    np.testing.assert_array_equal(actual.df, reference.df)
    np.testing.assert_array_equal(actual.sparsity, reference.sparsity)
    for left, right in zip(actual.path, reference.path):
        np.testing.assert_array_equal(left.toarray(), right.toarray())
    # Repeated penalties exercise nonzero warm starts, not only empty graphs.
    assert actual.path[1].nnz > 0
    np.testing.assert_array_equal(actual.path[1].toarray(), actual.path[2].toarray())
    if actual.icov is not None:
        np.testing.assert_allclose(actual.icov, reference.icov, rtol=0.0, atol=1e-10)
        np.testing.assert_allclose(actual.icov[1], actual.icov[2], rtol=0.0, atol=1e-6)
    if actual.loglik is not None:
        np.testing.assert_allclose(actual.loglik, reference.loglik, rtol=0.0, atol=1e-10)
