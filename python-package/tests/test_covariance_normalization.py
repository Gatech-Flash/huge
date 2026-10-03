"""Numerical contracts for covariance preprocessing."""

import numpy as np
import pytest

from pyhuge import PyHugeError, core


@pytest.mark.parametrize("layout", ("c", "fortran", "reversed"))
def test_mixed_scale_covariance_preserves_correlations(layout):
    variances = np.asarray(
        [np.finfo(float).max, 1e-300, 1e-20, 1.0, 4.0, 1e300]
    )
    # Uneven spacing makes this positive-definite correlation fixture change
    # under reversal, so layout checks also detect an index permutation.
    positions = np.arange(variances.size) ** 2
    expected = 0.25 ** np.abs(positions[:, None] - positions[None, :])
    sd = np.sqrt(variances)
    covariance = (expected * sd[:, None]) * sd[None, :]
    np.fill_diagonal(covariance, variances)
    if layout == "fortran":
        covariance = np.asfortranarray(covariance)
    elif layout == "reversed":
        covariance = covariance[::-1, ::-1]
        expected = expected[::-1, ::-1]
    original = covariance.copy()

    actual = core._cov_to_corr(covariance)

    np.testing.assert_allclose(actual, expected, rtol=5e-15, atol=0.0)
    np.testing.assert_array_equal(actual, actual.T)
    np.testing.assert_array_equal(covariance, original)


def test_covariance_roundoff_is_clipped_in_later_column():
    covariance = np.eye(4)
    covariance[:3, 3] = [1.0 + 1e-9, -1.0 - 1e-9, 0.25]
    covariance[3, :3] = covariance[:3, 3]

    actual = core._cov_to_corr(covariance, require_psd=False)

    np.testing.assert_array_equal(actual[:3, 3], [1.0, -1.0, 0.25])
    np.testing.assert_array_equal(actual, actual.T)


@pytest.mark.parametrize(
    "values,message",
    (
        ([1.1, np.nan], "not a valid covariance"),
        ([np.nan, 1.1], "finite correlation"),
        ([-1.1, np.inf], "not a valid covariance"),
        ([np.inf, -1.1], "finite correlation"),
    ),
)
def test_covariance_reports_first_invalid_entry_in_column(values, message):
    covariance = np.eye(4)
    covariance[:2, 3] = values
    covariance[3, :2] = values

    with pytest.raises(PyHugeError, match=message):
        core._cov_to_corr(covariance, require_psd=False)
