"""Compatibility with original columnwise ranks and generic normal quantiles."""

import math
import warnings

import numpy as np
import pytest
from scipy import stats

from pyhuge import PyHugeError, huge_npn


def _legacy_npn(x, method):
    """Frozen columnwise ranks and generic inverse-normal formula; no production helpers."""
    n, d = x.shape
    if method == "skeptic" and d == 1:
        return np.ones((1, 1), dtype=float)
    ranks = np.apply_along_axis(stats.rankdata, 0, np.asarray(x, dtype=float))
    if method == "skeptic":
        # Preserve the old arithmetic and F layout, including reduction order.
        _, exponents = np.frexp(np.max(np.abs(ranks), axis=0))
        centered = np.ldexp(ranks, -exponents)
        centered -= centered[0:1, :].copy()
        centered -= np.mean(centered, axis=0)
        centered /= np.max(np.abs(centered), axis=0)
        centered /= np.std(centered, axis=0, ddof=1)
        rho = (centered.T @ centered) / float(n - 1)
        np.fill_diagonal(rho, 1.0)
        result = 2.0 * np.sin((np.pi / 6.0) * rho)
        np.fill_diagonal(result, 1.0)
        return result
    if method == "shrinkage":
        z = stats.norm.ppf(ranks / (n + 1.0))
    else:
        trunc = 1.0 / (4.0 * (n ** 0.25) * math.sqrt(np.pi * np.log(max(n, 2))))
        z = stats.norm.ppf(np.clip(ranks / n, trunc, 1.0 - trunc))
    sd = z.std(axis=0, ddof=1)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return z / sd


def _snapshot(x):
    """Include view metadata and the backing array outside a strided view."""
    snapshots = []
    while isinstance(x, np.ndarray):
        snapshots.append((
            x.dtype.str, x.shape, x.strides,
            x.flags.c_contiguous, x.flags.f_contiguous,
            x.flags.writeable, x.flags.owndata, x.tobytes(order="A"),
        ))
        x = x.base
    return snapshots


def _fixture(case):
    tied = np.asarray([
        [0.0, 3.0, 1.0, -2.0], [-0.0, 1.0, 2.0, -2.0],
        [2.0, 1.0, 2.0, 0.0], [2.0, -1.0, 1.0, -0.0],
        [-1.0, -1.0, 0.0, 1.0], [-1.0, 3.0, -0.0, 2.0],
        [1.0, 0.0, -1.0, 2.0], [1.0, -0.0, -1.0, 1.0],
    ])
    if case == "ties-c":
        return tied
    if case == "ties-f-readonly":
        x = np.array(tied, order="F")
    elif case == "ties-negative-stride-readonly":
        x = tied[::-1, ::-1]
    elif case == "finite-extremes-strided-readonly":
        maximum = np.finfo(float).max
        tiny = np.nextafter(0.0, 1.0)
        values = np.column_stack((
            [maximum, maximum, -maximum, -maximum, 1.0, -1.0, 0.0, -0.0],
            [tiny, -tiny, 0.0, -0.0, 2 * tiny, -2 * tiny, tiny, -tiny],
            [1.0, np.nextafter(1.0, 2.0), 1.0, np.nextafter(1.0, 0.0),
             -1.0, -1.0, 0.0, -0.0],
            tied[:, 3],
        ))
        storage = np.full((16, 8), 19.0)
        storage[::2, 1::2] = values
        x = storage[::2, 1::2]
    elif case == "one-row":
        return np.asarray([[1.0, -0.0, -3.0]])
    elif case == "two-rows":
        return np.asarray([[1.0, -1.0, 3.0], [2.0, -2.0, -3.0]])
    elif case == "one-column":
        return tied[:, :1].copy()
    elif case == "constant":
        return np.column_stack((tied[:, 0], np.full(8, 7.0), np.zeros(8)))
    elif case == "wide-layout":
        # Wide C-layout input repeats rank patterns across many columns.
        rows = np.arange(128.0)[:, None]
        multipliers = (np.arange(257.0) % 30.0 + 1.0)[None, :]
        return (rows * multipliers) % 31.0 - 15.0
    elif case == "tall-layout":
        # Tall C-layout input exercises reductions over many observations.
        rows = np.arange(20000.0)
        return np.column_stack((rows, -rows, rows % 31.0))
    else:
        raise AssertionError(case)
    x.setflags(write=False)
    return x


def _call_with_warnings(function):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = function()
    return value, [(item.category, str(item.message)) for item in caught]


@pytest.mark.parametrize("method", ["shrinkage", "truncation", "skeptic"])
@pytest.mark.parametrize("case", [
    "ties-c", "ties-f-readonly", "ties-negative-stride-readonly",
    "finite-extremes-strided-readonly", "one-row", "two-rows",
    "one-column", "constant", "wide-layout", "tall-layout",
])
def test_npn_complete_output_matches_columnwise_formula(case, method):
    x = _fixture(case)
    before = _snapshot(x)
    if method == "skeptic" and case in ("one-row", "constant"):
        message = "at least two observations" if case == "one-row" else "constant column"
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(PyHugeError, match=message):
                huge_npn(x, npn_func=method, verbose=False)
    else:
        expected, expected_warnings = _call_with_warnings(lambda: _legacy_npn(x, method))
        actual, actual_warnings = _call_with_warnings(
            lambda: huge_npn(x, npn_func=method, verbose=False)
        )
        assert actual.dtype == expected.dtype == np.dtype(float)
        assert actual.shape == expected.shape
        assert actual.strides == expected.strides
        assert tuple(actual.flags[key] for key in ("C", "F", "W", "O")) == tuple(
            expected.flags[key] for key in ("C", "F", "W", "O")
        )
        assert actual.tobytes(order="A") == expected.tobytes(order="A")
        assert actual_warnings == expected_warnings
        assert np.isfinite(actual).all()
        assert not np.shares_memory(actual, x)
    assert _snapshot(x) == before


@pytest.mark.parametrize("x, method, message", [
    (np.asarray([1.0, 2.0]), "shrinkage", "must be a 2D array"),
    (np.empty((0, 2)), "truncation", "must be non-empty"),
    (np.asarray([[1.0, np.nan], [2.0, 3.0]]), "shrinkage", "non-finite"),
    (np.asarray([[1.0, np.inf], [2.0, 3.0]]), "truncation", "non-finite"),
    (np.asarray([[1.0, -np.inf]]), "skeptic", "non-finite"),
    (np.asarray([[np.nan]]), "invalid", "`npn_func` must be one of"),
], ids=["ndim", "empty", "nan", "positive-inf", "finite-before-skeptic", "method-before-finite"])
def test_npn_validation_precedes_ranking(x, method, message):
    before = _snapshot(x)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(PyHugeError, match=message):
            huge_npn(x, npn_func=method, verbose=False)
    assert _snapshot(x) == before
