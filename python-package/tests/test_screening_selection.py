"""Screening must preserve score order and deterministic cutoff ties."""

import numpy as np
import pytest

from pyhuge import core, huge


def _ordered_neighborhoods(correlation, count):
    d = correlation.shape[0]
    # Define the contract directly: largest absolute score first; smallest
    # predictor index first on ties; the response never enters its own list.
    return np.asarray(
        [
            sorted(
                (row for row in range(d) if row != column),
                key=lambda row: (-abs(correlation[row, column]), row),
            )[:count]
            for column in range(d)
        ],
        dtype=np.int32,
    ).T


@pytest.mark.parametrize(
    "dimension,count",
    ((511, 7), (512, 1), (512, 128), (512, 129), (513, 128), (513, 512)),
)
@pytest.mark.parametrize("tied", (False, True))
@pytest.mark.parametrize("layout", ("c", "fortran", "negative"))
def test_screening_preserves_order_at_selection_boundaries(
    dimension, count, tied, layout
):
    random = np.random.default_rng(431).normal(size=(dimension, dimension))
    correlation = 0.5 * (random + random.T)
    if tied:
        correlation = np.round(correlation)
    np.fill_diagonal(correlation, 1.0)
    if layout == "fortran":
        correlation = np.asfortranarray(correlation)
    elif layout == "negative":
        correlation = correlation[::-1, ::-1].copy()[::-1, ::-1]
    original = correlation.copy()
    correlation.setflags(write=False)

    actual = core._build_screen_idx(correlation, count)

    np.testing.assert_array_equal(actual, _ordered_neighborhoods(correlation, count))
    np.testing.assert_array_equal(correlation, original)
    assert actual.dtype == np.int32
    assert np.all(actual != np.arange(dimension))


@pytest.mark.parametrize("weight", (0.0, 0.5))
def test_screening_equal_scores_choose_lowest_indices_without_self(weight):
    dimension, count = 512, 31
    correlation = np.full((dimension, dimension), weight)
    np.fill_diagonal(correlation, 1.0)

    actual = core._build_screen_idx(correlation, count)

    for column in range(dimension):
        expected = np.delete(np.arange(dimension), column)[:count]
        np.testing.assert_array_equal(actual[:, column], expected)


def test_partial_screening_matches_full_order_in_public_mb_fit(monkeypatch):
    x = np.random.default_rng(432).normal(size=(40, 600))
    kwargs = dict(
        method="mb", lambda_=[0.55, 0.4], scr=True,
        input_type="data", verbose=False,
    )
    actual = huge(x, **kwargs)
    monkeypatch.setattr(core, "_build_screen_idx", _ordered_neighborhoods)
    reference = huge(x, **kwargs)

    assert actual.raw["scr_num"] == x.shape[0] - 1
    np.testing.assert_array_equal(actual.df, reference.df)
    np.testing.assert_array_equal(actual.sparsity, reference.sparsity)
    for left, right in zip(actual.path, reference.path):
        np.testing.assert_array_equal(left.toarray(), right.toarray())
