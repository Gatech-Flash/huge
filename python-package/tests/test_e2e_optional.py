"""Optional end-to-end tests (run only when local R runtime is available)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


def _runtime_ready() -> bool:
    repo_root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("R_LIBS_USER", str(repo_root / ".Rlib"))
    try:
        from pyhuge import core

        core._r_env()
        return True
    except Exception:
        return False


RUNTIME_READY = _runtime_ready()
REQUIRE_RUNTIME = os.environ.get("PYHUGE_REQUIRE_RUNTIME", "0") == "1"
SKIP_MARK = pytest.mark.skipif(not RUNTIME_READY, reason="Local R/rpy2/huge runtime is unavailable")


@pytest.fixture(scope="session", autouse=True)
def _enforce_runtime_if_required():
    if REQUIRE_RUNTIME and not RUNTIME_READY:
        pytest.fail("PYHUGE_REQUIRE_RUNTIME=1 but R runtime is unavailable.")


@SKIP_MARK
def test_e2e_mb_select_ric():
    from pyhuge import huge_mb, huge_select

    rng = np.random.default_rng(11)
    x = rng.normal(size=(60, 12))
    fit = huge_mb(x, nlambda=4, verbose=False)
    sel = huge_select(fit, criterion="ric", rep_num=5, verbose=False)

    assert fit.method == "mb"
    assert len(fit.path) == 4
    assert fit.lambda_path.shape == (4,)
    assert sel.opt_lambda > 0
    assert 0.0 <= sel.opt_sparsity <= 1.0


@SKIP_MARK
def test_e2e_ct_stars():
    from pyhuge import huge_ct, huge_select

    rng = np.random.default_rng(17)
    x = rng.normal(size=(50, 10))
    fit = huge_ct(x, nlambda=4, verbose=False)
    sel = huge_select(
        fit,
        criterion="stars",
        stars_thresh=0.1,
        stars_subsample_ratio=0.7,
        rep_num=3,
        verbose=False,
    )

    assert fit.method == "ct"
    assert sel.criterion == "stars"
    assert 0.0 <= sel.opt_sparsity <= 1.0


@SKIP_MARK
def test_e2e_tiger_runs():
    from pyhuge import huge_tiger

    rng = np.random.default_rng(23)
    x = rng.normal(size=(50, 10))
    fit = huge_tiger(x, nlambda=4, verbose=False)

    assert fit.method == "tiger"
    assert len(fit.path) == 4
    assert fit.icov is not None


@SKIP_MARK
def test_e2e_npn_glasso_ebic():
    from pyhuge import huge_glasso, huge_npn, huge_select

    rng = np.random.default_rng(31)
    x = rng.normal(size=(70, 12))
    x_npn = huge_npn(x, npn_func="shrinkage", verbose=False)
    fit = huge_glasso(x_npn, nlambda=4, verbose=False)
    sel = huge_select(fit, criterion="ebic", ebic_gamma=0.5, verbose=False)

    assert fit.method == "glasso"
    assert fit.loglik is not None
    assert sel.criterion == "ebic"
    assert sel.opt_icov is not None


@SKIP_MARK
def test_e2e_generator_roc_inference():
    from pyhuge import huge_generator, huge_glasso, huge_inference, huge_roc

    sim = huge_generator(n=80, d=10, graph="hub", g=2, verbose=False)
    fit = huge_glasso(sim.data, nlambda=4, verbose=False)
    roc = huge_roc(fit.path, sim.theta, verbose=False, plot=False)
    inf = huge_inference(
        data=sim.data,
        t=fit.icov[-1],
        adj=sim.theta,
        alpha=0.05,
        type_="Gaussian",
        method="score",
    )

    assert 0.0 <= roc.auc <= 1.0
    assert roc.fp.shape == roc.tp.shape
    assert 0.0 <= inf.error <= 1.0
    assert inf.p.shape[0] == inf.p.shape[1] == sim.data.shape[1]


@SKIP_MARK
def test_e2e_summary_and_network_plot():
    pytest.importorskip("networkx")
    pytest.importorskip("matplotlib")

    from pyhuge import huge_glasso, huge_plot_network, huge_select, huge_select_summary, huge_summary

    rng = np.random.default_rng(41)
    x = rng.normal(size=(60, 10))
    fit = huge_glasso(x, nlambda=4, verbose=False)
    sel = huge_select(fit, criterion="ebic", verbose=False)

    s1 = huge_summary(fit)
    s2 = huge_select_summary(sel)

    # Use non-interactive backend in CI/headless environments.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    huge_plot_network(fit, index=-1, ax=ax, layout="spring")
    plt.close(fig)

    assert s1.n_features == 10
    assert s2.refit_n_features == 10
