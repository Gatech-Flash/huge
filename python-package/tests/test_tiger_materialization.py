"""Native result contracts for sparse TIGER precision materialization."""

import numpy as np
import pytest

from pyhuge import _native_core as native


@pytest.mark.parametrize("dimension", [1, 7])
@pytest.mark.parametrize("matrix_list", [False, True])
def test_tiger_empty_support_preserves_signed_precision_zeros(dimension, matrix_list):
    penalties = np.asarray([1.0, 0.5, 0.5])
    out = native.spmb_graphsqrt(
        np.eye(dimension), penalties, covariance_input=True,
        dense_output=False, matrix_list=matrix_list,
    )
    expected = np.full((dimension, dimension), -0.0)
    np.fill_diagonal(expected, 1.0)
    assert out["support_indices"].size == 0
    np.testing.assert_array_equal(out["lambda"], penalties)
    assert not out["path_truncated"]
    for matrix in out["icov"]:
        assert matrix.tobytes(order="C") == expected.tobytes(order="C")


@pytest.mark.parametrize("rho", [-0.8, 0.8])
def test_tiger_two_variable_solution_survives_empty_and_repeated_path_slots(rho):
    covariance = np.asarray([[1.0, rho], [rho, 1.0]])
    penalties = np.asarray([0.95, 0.5, 0.5, 0.2])
    out = native.spmb_graphsqrt(
        covariance, penalties, covariance_input=True, matrix_list=True,
    )
    # For one predictor, KKT gives beta = rho-sign(rho)*lambda*tau and
    # tau^2 = (1-rho^2)/(1-lambda^2), when abs(rho) > lambda.
    for penalty, precision in zip(penalties, out["icov"]):
        if penalty >= abs(rho):
            expected = np.full((2, 2), -0.0)
            np.fill_diagonal(expected, 1.0)
            assert precision.tobytes(order="C") == expected.tobytes(order="C")
        else:
            tau_square = (1.0 - rho * rho) / (1.0 - penalty * penalty)
            beta = rho - np.sign(rho) * penalty * np.sqrt(tau_square)
            expected = np.asarray([[1.0, -beta], [-beta, 1.0]]) / tau_square
            np.testing.assert_allclose(precision, expected, rtol=2e-6, atol=1e-9)
    assert len(out["icov"]) == penalties.size
    assert not out["path_truncated"]


@pytest.mark.parametrize("matrix_list", [False, True])
def test_tiger_long_generated_tail_returns_only_the_certified_identity(matrix_list):
    x = np.arange(1.0, 25.0)
    data = x[:, None] * np.asarray([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    out = native.spmb_graphsqrt(data, None, 64, 0.01, matrix_list=matrix_list)
    assert out["path_truncated"]
    assert out["hit_max_iter"]
    assert len(out["icov"]) == len(out["lambda"]) == 1
    expected = np.full((7, 7), -0.0)
    np.fill_diagonal(expected, 1.0)
    assert out["icov"][0].tobytes(order="C") == expected.tobytes(order="C")
    np.testing.assert_array_equal(out["beta"], np.zeros((1, 7, 7)))


def test_tiger_precision_tracks_coefficients_that_leave_and_change_sign():
    samples = np.random.default_rng(1717).normal(size=(12, 8))
    samples -= samples.mean(axis=0)
    samples /= np.sqrt(np.sum(samples * samples, axis=0))
    covariance = np.einsum("ni,nj->ij", samples, samples)
    np.fill_diagonal(covariance, 1.0)
    maximum = np.max(np.abs(covariance - np.eye(8)))
    penalties = np.geomspace(0.95 * maximum, 0.003, 24)
    out = native.spmb_graphsqrt(covariance, penalties, covariance_input=True)
    beta_path = out["beta"]
    assert beta_path[10, 7, 1] != 0.0
    assert beta_path[11, 7, 1] == 0.0
    assert np.any(beta_path[:, 7, 1] < 0.0)
    assert np.any(beta_path[:, 7, 1] > 0.0)

    for beta, precision in zip(beta_path, out["icov"]):
        # Calculate residual variances in the sample domain, independently of
        # the solver's correlation-gradient expression and saved scaling data.
        residual = samples - np.einsum("nj,mj->nm", samples, beta)
        inverse_variance = 1.0 / np.sum(residual * residual, axis=0)
        directed = -beta.T * inverse_variance[None, :]
        np.fill_diagonal(directed, inverse_variance)
        expected = 0.5 * directed + 0.5 * directed.T
        np.testing.assert_allclose(precision, expected, rtol=2e-10, atol=2e-12)
