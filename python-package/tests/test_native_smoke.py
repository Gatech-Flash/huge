from __future__ import annotations

import numpy as np
import pytest

from pyhuge import (
    huge,
    huge_generator,
    huge_glasso,
    huge_inference,
    huge_mb,
    huge_npn,
    huge_roc,
    huge_select,
)


def test_native_ct_runs():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 12))

    fit_ct = huge(x, method="ct", nlambda=5, backend="native")
    assert fit_ct.method == "ct"
    assert len(fit_ct.path) == 5


def test_native_mb_glasso_select_runs():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 12))

    fit_mb = huge_mb(x, nlambda=4, backend="native")
    sel_ric = huge_select(fit_mb, criterion="ric", backend="native")
    assert sel_ric.opt_lambda > 0
    assert 0.0 <= sel_ric.opt_sparsity <= 1.0

    fit_gl = huge_glasso(x, nlambda=4, cov_output=True, backend="native")
    sel_ebic = huge_select(fit_gl, criterion="ebic", backend="native")
    assert fit_gl.icov is not None and len(fit_gl.icov) == 4
    assert fit_gl.cov is not None and len(fit_gl.cov) == 4
    assert sel_ebic.opt_icov is not None


def test_native_generator_roc_inference_runs():
    sim = huge_generator(n=70, d=10, graph="hub", g=2, random_state=1)
    fit = huge(sim.data, method="ct", nlambda=4, backend="native")

    roc = huge_roc(fit.path, sim.theta)
    assert 0.0 <= roc.auc <= 1.0

    t = np.eye(sim.data.shape[1], dtype=float)
    out = huge_inference(sim.data, t, sim.theta, alpha=0.05, type_="Gaussian")
    assert out.p.shape == (10, 10)
    assert 0.0 <= out.error <= 1.0


def test_native_npn_modes():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(80, 8)) ** 3

    z1 = huge_npn(x, npn_func="shrinkage")
    z2 = huge_npn(x, npn_func="truncation")
    z3 = huge_npn(x, npn_func="skeptic")

    assert z1.shape == x.shape
    assert z2.shape == x.shape
    assert z3.shape == (8, 8)
