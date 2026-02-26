"""Runtime/validation error tests for pyhuge 0.3."""

from __future__ import annotations

import numpy as np
import pytest

from pyhuge import PyHugeError, huge, huge_npn, huge_plot, huge_select


def test_huge_invalid_method_raises():
    x = np.random.default_rng(0).normal(size=(20, 6))
    with pytest.raises(PyHugeError, match="`method` must be one of"):
        huge(x, method="bad", verbose=False)


def test_huge_npn_invalid_func_raises():
    x = np.random.default_rng(0).normal(size=(20, 6))
    with pytest.raises(PyHugeError, match="`npn_func` must be one of"):
        huge_npn(x, npn_func="bad", verbose=False)


def test_huge_plot_non_square_raises():
    g = np.ones((4, 3), dtype=float)
    with pytest.raises(PyHugeError, match="`g` must be square"):
        huge_plot(g)


def test_huge_select_ebic_requires_glasso():
    x = np.random.default_rng(1).normal(size=(30, 8))
    fit = huge(x, method="ct", nlambda=4, verbose=False)
    with pytest.raises(PyHugeError, match="requires a glasso fit"):
        huge_select(fit, criterion="ebic", verbose=False)
