"""RIC panel boundaries and public selection/refit behavior."""

import numpy as np
import pytest

from pyhuge import core


@pytest.mark.parametrize("d", [1, 127, 128, 129, 130, 257, 258])
def test_native_ric_panel_edges_match_independent_rotation_products(d):
    x = np.random.default_rng(839).normal(size=(17, d))
    before = x.copy()
    rotations = np.array([0, 8, 17], dtype=np.int32)
    reference = []
    for rotation in rotations:
        # Long-double matrix arithmetic is independent of the double BLAS
        # kernels selected for the full matrix and the partial panels.
        high_precision = np.asarray(x, dtype=np.longdouble)
        product = np.roll(high_precision, -int(rotation), axis=0).T @ high_precision
        expected = float(np.max(np.abs(np.triu(product, 1)))) if d > 1 else 0.0
        actual = core._CPP.ric(x, np.array([rotation], dtype=np.int32))
        assert actual == pytest.approx(expected, rel=2e-13, abs=0)
        reference.append(expected)
    assert core._CPP.ric(x, rotations) == pytest.approx(min(reference), rel=2e-13, abs=0)
    np.testing.assert_array_equal(x, before)
    np.testing.assert_array_equal(rotations, [0, 8, 17])


@pytest.mark.parametrize("strength", [0.0, 1e-8])
@pytest.mark.parametrize("scale", [1.0, 1e100])
def test_ric_high_dimension_selection_keeps_empty_and_weak_refits(strength, scale):
    n, d = 256, 255
    # All nonconstant Walsh columns: every column is valid for standardization.
    x = np.array([[1.0 - 2.0 * (bin(i & j).count("1") % 2)
                   for j in range(1, d + 1)] for i in range(n)])
    x[:, -1] += strength * x[:, -2]
    fit = core.huge(x * scale, method="ct", lambda_=[1.0], verbose=False)
    selected = core.huge_select(fit, criterion="ric", rep_num=n, verbose=False)
    assert selected.raw["ric_fallback"] is False
    if strength == 0.0:
        assert selected.opt_lambda == 0.0
        assert selected.refit.nnz == 0
        assert selected.raw["ric_refit_lambda"] is None
    else:
        assert selected.opt_lambda > 0.0
        expected = (n - 1) * strength / (n * np.sqrt(1 + strength**2))
        assert abs(selected.opt_lambda - expected) <= 2 * n * np.finfo(float).eps
        assert selected.refit.nnz == 2
        assert selected.refit[d - 2, d - 1] == 1
        assert selected.refit[d - 1, d - 2] == 1
        refit = core.huge(x * scale, method="ct", lambda_=[selected.opt_lambda], verbose=False)
        np.testing.assert_array_equal(selected.refit.toarray(), refit.path[0].toarray())
