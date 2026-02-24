"""Parity tests: mirror examples from R Rd docs with pyhuge API."""

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
SKIP_MARK = pytest.mark.skipif(not RUNTIME_READY, reason="Local R/rpy2/huge runtime is unavailable")


@SKIP_MARK
def test_rd_huge_examples_main_methods_and_plotting(tmp_path):
    """From man/huge.Rd examples."""

    pytest.importorskip("matplotlib")
    pytest.importorskip("networkx")

    from pyhuge import (
        huge,
        huge_generator,
        huge_plot_graph_matrix,
        huge_plot_network,
    )

    sim = huge_generator(n=50, d=12, graph="hub", g=4, verbose=False)

    # out1 = huge(L$data)
    out_mb = huge(sim.data, verbose=False)
    assert out_mb.method == "mb"
    assert out_mb.data.shape == sim.data.shape
    assert len(out_mb.path) == out_mb.lambda_path.size

    # huge(cor(L$data), method = "glasso")
    sample_cor = np.corrcoef(sim.data, rowvar=False)
    out_cov = huge(sample_cor, method="glasso", nlambda=5, verbose=False)
    assert out_cov.method == "glasso"
    assert out_cov.cov_input is True
    assert out_cov.icov is not None

    # method variants in same Rd block.
    out_ct = huge(sim.data, method="ct", nlambda=5, verbose=False)
    out_glasso = huge(sim.data, method="glasso", nlambda=5, verbose=False)
    out_tiger = huge(sim.data, method="tiger", nlambda=5, verbose=False)
    assert out_ct.method == "ct"
    assert out_glasso.method == "glasso"
    assert out_tiger.method == "tiger"

    # huge.plot(out1$path[[3]]) equivalent in pyhuge.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    huge_plot_graph_matrix(out_mb, index=2, ax=axs[0])
    huge_plot_network(out_mb, index=2, ax=axs[1], layout="spring")
    out_png = tmp_path / "rd_huge_plot_equivalent.png"
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    assert out_png.exists() and out_png.stat().st_size > 0


@SKIP_MARK
def test_rd_huge_select_examples_ric_stars_ebic():
    """From man/huge.select.Rd examples."""

    from pyhuge import huge, huge_generator, huge_select

    sim = huge_generator(d=20, graph="hub", verbose=False)
    out_mb = huge(sim.data, nlambda=6, verbose=False)
    out_ct = huge(sim.data, method="ct", nlambda=6, verbose=False)
    out_glasso = huge(sim.data, method="glasso", nlambda=6, verbose=False)

    # out.select = huge.select(out.mb)
    sel_ric = huge_select(out_mb, criterion="ric", rep_num=10, verbose=False)
    assert sel_ric.criterion == "ric"
    assert sel_ric.opt_lambda > 0

    # huge.select(out.ct, criterion = "stars", stars.thresh = 0.05, rep.num=10)
    sel_stars = huge_select(
        out_ct,
        criterion="stars",
        stars_thresh=0.05,
        rep_num=10,
        verbose=False,
    )
    assert sel_stars.criterion == "stars"
    assert sel_stars.variability is not None

    # huge.select(out.glasso, criterion = "ebic")
    sel_ebic = huge_select(out_glasso, criterion="ebic", verbose=False)
    assert sel_ebic.criterion == "ebic"
    assert sel_ebic.opt_icov is not None


@SKIP_MARK
def test_rd_huge_npn_examples():
    """From man/huge.npn.Rd examples."""

    from pyhuge import huge_generator, huge_npn

    sim = huge_generator(graph="cluster", g=5, verbose=False)
    non_gaussian_data = sim.data**5

    q_shrinkage = huge_npn(non_gaussian_data, npn_func="shrinkage", verbose=False)
    q_truncation = huge_npn(non_gaussian_data, npn_func="truncation", verbose=False)
    q_skeptic = huge_npn(non_gaussian_data, npn_func="skeptic", verbose=False)

    n, d = non_gaussian_data.shape
    assert q_shrinkage.shape == (n, d)
    assert q_truncation.shape == (n, d)
    assert q_skeptic.shape == (d, d)


@SKIP_MARK
def test_rd_huge_generator_examples_graph_families():
    """From man/huge.generator.Rd examples."""

    from pyhuge import huge_generator

    # band graph with bandwidth 3
    band = huge_generator(graph="band", g=3, verbose=False)
    # random sparse graph
    random_sparse = huge_generator(verbose=False)
    # random dense graph
    random_dense = huge_generator(prob=0.5, verbose=False)
    # hub graph with 6 hubs
    hub = huge_generator(graph="hub", g=6, verbose=False)
    # cluster graph with 8 clusters
    cluster = huge_generator(graph="cluster", g=8, verbose=False)
    # scale-free graph
    scale_free = huge_generator(graph="scale-free", verbose=False)

    for sim in (band, random_sparse, random_dense, hub, cluster, scale_free):
        n, d = sim.data.shape
        assert sim.sigma.shape == (d, d)
        assert sim.omega.shape == (d, d)
        assert sim.sigmahat.shape == (d, d)
        assert sim.theta.shape == (d, d)
        assert n > 0 and d > 0


@SKIP_MARK
def test_rd_huge_roc_example():
    """From man/huge.roc.Rd examples."""

    from pyhuge import huge, huge_generator, huge_roc

    sim = huge_generator(d=200, graph="cluster", prob=0.3, verbose=False)
    fit = huge(sim.data, nlambda=6, verbose=False)
    roc = huge_roc(fit.path, sim.theta, verbose=False, plot=False)

    assert np.max(roc.f1) >= 0.0
    assert 0.0 <= roc.auc <= 1.0
    assert roc.fp.shape == roc.tp.shape


@SKIP_MARK
def test_rd_huge_inference_examples():
    """From man/huge.inference.Rd examples."""

    from pyhuge import huge, huge_generator, huge_inference

    sim = huge_generator(n=50, d=12, graph="hub", g=4, verbose=False)
    est = huge(sim.data, method="glasso", nlambda=6, verbose=False)
    assert est.icov is not None
    t_last = est.icov[-1]

    out_gaussian = huge_inference(sim.data, t_last, sim.theta, alpha=0.05, type_="Gaussian")
    out_npn_score = huge_inference(
        sim.data,
        t_last,
        sim.theta,
        alpha=0.05,
        type_="Nonparanormal",
        method="score",
    )
    out_npn_wald = huge_inference(
        sim.data,
        t_last,
        sim.theta,
        alpha=0.05,
        type_="Nonparanormal",
        method="wald",
    )
    out_npn_wald_01 = huge_inference(
        sim.data,
        t_last,
        sim.theta,
        alpha=0.1,
        type_="Nonparanormal",
        method="wald",
    )

    for out in (out_gaussian, out_npn_score, out_npn_wald, out_npn_wald_01):
        d = sim.data.shape[1]
        assert out.p.shape == (d, d)
        assert 0.0 <= out.error <= 1.0


@SKIP_MARK
def test_rd_huge_plot_examples(tmp_path):
    """From man/huge.plot.Rd examples."""

    from pyhuge import huge_generator, huge_plot

    # visualize the hub graph
    hub = huge_generator(graph="hub", verbose=False)
    assert huge_plot(hub.theta, epsflag=False) is None

    # visualize the band graph
    band = huge_generator(graph="band", g=5, verbose=False)
    assert huge_plot(band.theta, epsflag=False) is None

    # visualize the cluster graph
    cluster = huge_generator(graph="cluster", verbose=False)
    assert huge_plot(cluster.theta, epsflag=False) is None

    # save eps file
    eps_path = huge_plot(
        cluster.theta,
        epsflag=True,
        cur_num=5,
        location=str(tmp_path),
    )
    assert eps_path is not None
    assert eps_path.endswith("default5.eps")
    assert os.path.exists(eps_path) and os.path.getsize(eps_path) > 0


@SKIP_MARK
def test_rd_stockdata_examples(tmp_path):
    """From man/stockdata.Rd examples."""

    pytest.importorskip("matplotlib")
    from pyhuge import huge_stockdata

    stock = huge_stockdata()
    assert stock.data.ndim == 2
    assert stock.info.ndim == 2
    assert stock.data.shape[1] == stock.info.shape[0]
    assert stock.info.shape[1] == 3

    # image(stockdata$data)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.imshow(stock.data, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title("stockdata$data")
    ax.set_xlabel("Company")
    ax.set_ylabel("Trading Day")
    out_png = tmp_path / "stockdata_image.png"
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    assert out_png.exists() and out_png.stat().st_size > 0
