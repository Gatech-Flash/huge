"""Core unit tests for pyhuge native implementation."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from scipy import sparse

from pyhuge import core


HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


def test_method_wrappers_route_method(monkeypatch):
    called = []

    def _fake_huge(**kwargs):
        called.append(kwargs["method"])
        return "ok"

    monkeypatch.setattr(core, "huge", _fake_huge)

    assert core.huge_mb(np.ones((5, 4)), verbose=False) == "ok"
    assert core.huge_glasso(np.ones((5, 4)), verbose=False) == "ok"
    assert core.huge_ct(np.ones((5, 4)), verbose=False) == "ok"
    assert core.huge_tiger(np.ones((5, 4)), verbose=False) == "ok"

    assert called == ["mb", "glasso", "ct", "tiger"]


def test_summary_helpers():
    fit = core.HugeResult(
        method="mb",
        lambda_path=np.array([0.3, 0.2, 0.1]),
        sparsity=np.array([0.05, 0.08, 0.12]),
        path=[sparse.csc_matrix(np.eye(4)) for _ in range(3)],
        cov_input=False,
        data=np.ones((20, 4)),
        raw=None,
    )
    sel = core.HugeSelectResult(
        criterion="ric",
        opt_lambda=0.2,
        opt_sparsity=0.08,
        refit=sparse.csc_matrix(np.eye(4)),
        raw=None,
    )

    s1 = core.huge_summary(fit)
    s2 = core.huge_select_summary(sel)

    assert s1.path_length == 3
    assert s1.n_samples == 20
    assert s2.criterion == "ric"
    assert s2.refit_n_features == 4


def test_stockdata_loader_shape():
    stock = core.huge_stockdata()
    assert stock.data.shape == (1258, 452)
    assert stock.info.shape == (452, 3)


@pytest.mark.skipif(not HAS_SKLEARN, reason="requires scikit-learn")
def test_ric_select_opt_index_is_one_based():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(80, 12))
    fit = core.huge(x, method="mb", nlambda=5, verbose=False)
    sel = core.huge_select(fit, criterion="ric", verbose=False)

    assert sel.opt_index is not None
    assert 1 <= sel.opt_index <= len(fit.path)


def test_roc_shapes():
    theta = sparse.csc_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float))
    path = [
        sparse.csc_matrix(np.zeros((3, 3), dtype=float)),
        sparse.csc_matrix(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)),
    ]
    roc = core.huge_roc(path, theta, plot=False)
    assert roc.f1.shape == (2,)
    assert roc.fp.shape == (2,)
    assert roc.tp.shape == (2,)
    assert 0.0 <= roc.auc <= 1.0
