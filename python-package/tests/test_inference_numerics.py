"""Closed-form bivariate checks independent of matrix variance contraction."""

import math

import numpy as np
import pytest

from pyhuge import PyHugeError, huge_inference


def _data():
    return np.asarray([[-2, 3], [-1, -1], [0, 2], [1, -2], [2, 1], [3, 0]], dtype=float)


@pytest.mark.parametrize(
    "b,c",
    [(2 * epsilon, epsilon) for epsilon in (0.0, 1e-4, 1e-8, 1e-12, -1e-8)]
    + [(1e16, 1e-16), (1e-16, 1e16)],
)
def test_bivariate_score_avoids_subtractive_cancellation(b, c):
    x = _data()
    n = len(x)
    sums = [sum(np.sign(x[i, 0] - x[k, 0]) * np.sign(x[i, 1] - x[k, 1])
                for k in range(n)) for i in range(n)]
    u = math.sin(math.pi * sum(sums) / (2 * n * (n - 1)))
    h = [math.sqrt(1 - u * u) * (math.asin(u) - math.pi * s / (2 * (n - 1)))
         for s in sums]
    rms = math.sqrt(sum(value * value for value in h) / n)

    # For T=[[1,b],[c,1]], removing T[j,k] gives diagonal score c^2/b^2,
    # and off-diagonal score u+c/u+b. The variance contraction coefficients
    # are 2c, 2b, and 1+bc. No subtraction of order-one quantities is needed.
    score = np.asarray([[c * c, u + c], [u + b, b * b]])
    sigma = rms * np.asarray([[abs(2 * c), abs(1 + b * c)],
                              [abs(1 + b * c), abs(2 * b)]])
    with np.errstate(divide="ignore", invalid="ignore"):
        statistics = score * math.sqrt(n) / (2 * sigma)
    expected = np.asarray([[math.erfc(abs(z) / math.sqrt(2)) for z in row]
                           for row in statistics])
    actual = huge_inference(
        x, np.asarray([[1.0, b], [c, 1.0]]), np.zeros((2, 2)),
        type_="Nonparanormal", method="score",
    )
    np.testing.assert_allclose(actual.p, expected, atol=5e-15, rtol=0, equal_nan=True)


@pytest.mark.parametrize("method", ("score", "wald"))
def test_nonparanormal_rejects_overflowed_edge_variance(method):
    # Diagonal products are representable, but off-diagonal entries make
    # standardized influence values overflow. p=1 is not valid evidence here.
    t = np.asarray([[1e-100, 5e99], [1e100, 1e-100]])
    with pytest.raises(PyHugeError, match="numerically degenerate"):
        huge_inference(_data(), t, np.zeros((2, 2)),
                       type_="Nonparanormal", method=method)
