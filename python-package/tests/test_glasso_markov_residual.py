"""An independent Markov completion oracle for a sparse glasso solution."""

import numpy as np
import pytest

from pyhuge import core


@pytest.mark.parametrize("d,scale", [(511, 1.0), (512, 1.0), (512, 1e-100), (512, 1e100)])
def test_glasso_markov_completion_across_residual_dispatch(d, scale):
    rho, penalty = 0.7, 0.55
    distance = np.abs(np.arange(d)[:, None] - np.arange(d)[None, :])
    covariance = scale * rho ** distance
    original = covariance.copy()
    a, b = 1.0 + penalty, rho - penalty
    r = b / a
    denominator = a * (1.0 - r * r)
    expected_precision = np.diag(np.full(d, (1.0 + r * r) / denominator))
    expected_precision[0, 0] = expected_precision[-1, -1] = 1.0 / denominator
    adjacent = np.arange(d - 1)
    expected_precision[adjacent, adjacent + 1] = -r / denominator
    expected_precision[adjacent + 1, adjacent] = -r / denominator
    expected_covariance = a * r ** distance
    # W-S = lambda*sign(Theta) on nonzeros, |W-S| < lambda elsewhere.
    # At distance two the largest inactive difference is rho**2-b**2/a;
    # at larger distances it is at most rho**3 < penalty.
    assert rho ** 2 - b * b / a < penalty
    assert rho ** 3 < penalty

    result = core.huge(covariance, method="glasso", lambda_=[scale * penalty],
                       input_type="covariance", cov_output=True, verbose=False)

    np.testing.assert_array_equal(covariance, original)
    np.testing.assert_array_equal(result.path[0].toarray() != 0, distance == 1)
    np.testing.assert_array_equal(result.icov[0] != 0, distance <= 1)
    np.testing.assert_allclose(result.icov[0] * scale, expected_precision,
                               rtol=2e-4, atol=2e-6)
    np.testing.assert_allclose(result.cov[0] / scale, expected_covariance,
                               rtol=2e-4, atol=2e-6)
    trace = (2 + (d - 2) * (1 + r * r) - 2 * (d - 1) * rho * r) / denominator
    expected_loglik = (-d * np.log(scale) - d * np.log(a)
                       - (d - 1) * np.log1p(-r * r) - trace)
    np.testing.assert_allclose(result.loglik[0], expected_loglik, rtol=0, atol=2e-3)
    assert result.df[0] == d - 1
