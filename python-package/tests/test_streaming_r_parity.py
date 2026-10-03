"""Inference parity at rank, tie, scale and asymmetric-input boundaries."""

import numpy as np
import pytest
from scipy import sparse

from pyhuge import huge_inference
from pyhuge.parity import has_r_huge, run_r_inference_reference


@pytest.mark.skipif(not has_r_huge(), reason="requires local R with package huge")
@pytest.mark.parametrize("method", ("score", "wald"))
@pytest.mark.parametrize(
    "case", ("symmetric", "asymmetric", "ties", "extreme_data", "scaled_t", "rank_limit", "near_diagonal")
)
def test_nonparanormal_streaming_matches_r_across_boundaries(case, method):
    rng = np.random.default_rng(29109)
    data = rng.normal(size=(19, 5))
    t = np.eye(5) + rng.normal(scale=0.04, size=(5, 5))
    if case == "symmetric":
        t = (t + t.T) / 2
    elif case == "ties":
        data = np.round(data)
    elif case == "extreme_data":
        # Every column varies; finite subtraction can overflow without
        # changing the signs needed by the rank statistic.
        data = rng.integers(-1, 2, size=(19, 5)).astype(float) * 1e308
    elif case == "scaled_t":
        t = t * np.geomspace(1e-60, 1e60, 5)
    elif case == "rank_limit":
        data = np.asarray([[0.0, 0.0], [1.0, 1.0]])
        t = np.eye(2)
    elif case == "near_diagonal":
        data = np.asarray([[-2, 3], [-1, -1], [0, 2], [1, -2], [2, 1], [3, 0]], dtype=float)
        t = np.asarray([[1.0, 2e-8], [1e-8, 1.0]])
    d = data.shape[1]
    adjacency = np.zeros((d, d))
    adjacency[0, 1] = -2  # A nonzero weight still denotes an existing edge.
    original = data.copy()
    reference = run_r_inference_reference(
        data, t, adjacency, alpha=0.2, type_="Nonparanormal", method=method,
    )
    actual = huge_inference(
        data, sparse.csc_matrix(t), sparse.csc_matrix(adjacency),
        alpha=0.2, type_="Nonparanormal", method=method,
    )

    np.testing.assert_array_equal(np.isnan(actual.p), np.isnan(reference["p"]))
    np.testing.assert_allclose(actual.p, reference["p"], atol=2e-12, rtol=1e-10,
                               equal_nan=True)
    np.testing.assert_allclose(actual.data, reference["data"], atol=0, rtol=1e-14)
    np.testing.assert_array_equal(data, original)
    assert actual.error == pytest.approx(reference["error"], abs=1e-15)
    offdiag = ~np.eye(d, dtype=bool)
    assert np.isfinite(actual.p[offdiag]).all()
    assert actual.error == np.count_nonzero(
        (actual.p < 0.2) & (adjacency == 0) & offdiag
    ) / float(d * d)
    if case == "rank_limit" and method == "score":
        assert np.isnan(np.diag(actual.p)).all()
