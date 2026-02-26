"""Parity tests mirroring R Rd-style examples using native pyhuge APIs."""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest


HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None
HAS_MPL = importlib.util.find_spec("matplotlib") is not None
HAS_NX = importlib.util.find_spec("networkx") is not None


@pytest.mark.skipif(not (HAS_SKLEARN and HAS_MPL and HAS_NX), reason="requires sklearn+matplotlib+networkx")
def test_rd_huge_examples_main_methods_and_plotting(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pyhuge import huge, huge_generator, huge_plot_graph_matrix, huge_plot_network

    sim = huge_generator(n=50, d=12, graph="hub", g=4, verbose=False)

    out_mb = huge(sim.data, verbose=False)
    assert out_mb.method == "mb"

    sample_cor = np.corrcoef(sim.data, rowvar=False)
    out_cov = huge(sample_cor, method="glasso", nlambda=5, verbose=False)
    assert out_cov.method == "glasso"
    assert out_cov.cov_input is True

    out_ct = huge(sim.data, method="ct", nlambda=5, verbose=False)
    out_glasso = huge(sim.data, method="glasso", nlambda=5, verbose=False)
    out_tiger = huge(sim.data, method="tiger", nlambda=5, verbose=False)
    assert out_ct.method == "ct"
    assert out_glasso.method == "glasso"
    assert out_tiger.method == "tiger"

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    huge_plot_graph_matrix(out_mb, index=2, ax=axs[0])
    huge_plot_network(out_mb, index=2, ax=axs[1], layout="spring")
    out_png = tmp_path / "rd_huge_plot_equivalent.png"
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    assert out_png.exists() and out_png.stat().st_size > 0


@pytest.mark.skipif(not HAS_SKLEARN, reason="requires scikit-learn")
def test_rd_huge_select_examples_ric_stars_ebic():
    from pyhuge import huge, huge_generator, huge_select

    sim = huge_generator(d=20, graph="hub", verbose=False)
    out_mb = huge(sim.data, nlambda=6, verbose=False)
    out_ct = huge(sim.data, method="ct", nlambda=6, verbose=False)
    out_glasso = huge(sim.data, method="glasso", nlambda=6, verbose=False)

    sel_ric = huge_select(out_mb, criterion="ric", rep_num=10, verbose=False)
    assert sel_ric.criterion == "ric"

    sel_stars = huge_select(out_ct, criterion="stars", stars_thresh=0.05, rep_num=6, verbose=False)
    assert sel_stars.criterion == "stars"

    sel_ebic = huge_select(out_glasso, criterion="ebic", verbose=False)
    assert sel_ebic.criterion == "ebic"


def test_rd_huge_npn_examples():
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


def test_rd_huge_generator_examples_graph_families():
    from pyhuge import huge_generator

    band = huge_generator(graph="band", g=3, verbose=False)
    random_sparse = huge_generator(verbose=False)
    random_dense = huge_generator(prob=0.5, verbose=False)
    hub = huge_generator(graph="hub", g=6, verbose=False)
    cluster = huge_generator(graph="cluster", g=8, verbose=False)
    scale_free = huge_generator(graph="scale-free", verbose=False)

    for sim in (band, random_sparse, random_dense, hub, cluster, scale_free):
        n, d = sim.data.shape
        assert sim.sigma.shape == (d, d)
        assert sim.omega.shape == (d, d)
        assert sim.sigmahat.shape == (d, d)
        assert sim.theta.shape == (d, d)
        assert n > 0 and d > 0


def test_rd_huge_roc_example():
    from pyhuge import huge, huge_generator, huge_roc

    sim = huge_generator(d=80, graph="cluster", prob=0.3, verbose=False)
    fit = huge(sim.data, method="ct", nlambda=6, verbose=False)
    roc = huge_roc(fit.path, sim.theta, verbose=False, plot=False)

    assert np.max(roc.f1) >= 0.0
    assert 0.0 <= roc.auc <= 1.0


@pytest.mark.skipif(not HAS_SKLEARN, reason="requires scikit-learn")
def test_rd_huge_inference_examples():
    from pyhuge import huge, huge_generator, huge_inference

    sim = huge_generator(n=50, d=12, graph="hub", g=4, verbose=False)
    est = huge(sim.data, method="glasso", nlambda=6, verbose=False)
    t_last = est.icov[-1]

    out_gaussian = huge_inference(sim.data, t_last, sim.theta, alpha=0.05, type_="Gaussian")
    out_npn_score = huge_inference(sim.data, t_last, sim.theta, alpha=0.05, type_="Nonparanormal", method="score")
    out_npn_wald = huge_inference(sim.data, t_last, sim.theta, alpha=0.05, type_="Nonparanormal", method="wald")

    for out in (out_gaussian, out_npn_score, out_npn_wald):
        d = sim.data.shape[1]
        assert out.p.shape == (d, d)
        assert 0.0 <= out.error <= 1.0


@pytest.mark.skipif(not (HAS_MPL and HAS_NX), reason="requires matplotlib+networkx")
def test_rd_huge_plot_examples(tmp_path):
    from pyhuge import huge_generator, huge_plot

    hub = huge_generator(graph="hub", verbose=False)
    assert huge_plot(hub.theta, epsflag=False) is None

    band = huge_generator(graph="band", g=5, verbose=False)
    assert huge_plot(band.theta, epsflag=False) is None

    cluster = huge_generator(graph="cluster", verbose=False)
    assert huge_plot(cluster.theta, epsflag=False) is None

    eps_path = huge_plot(cluster.theta, epsflag=True, cur_num=5, location=str(tmp_path))
    assert eps_path is not None
    assert eps_path.endswith("default5.eps")
    assert os.path.exists(eps_path) and os.path.getsize(eps_path) > 0


@pytest.mark.skipif(not HAS_MPL, reason="requires matplotlib")
def test_rd_stockdata_examples(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pyhuge import huge_stockdata

    stock = huge_stockdata()
    assert stock.data.shape[1] == stock.info.shape[0]
    assert stock.info.shape[1] == 3

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.imshow(stock.data, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title("stockdata$data")
    out_png = tmp_path / "stockdata_image.png"
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    assert out_png.exists() and out_png.stat().st_size > 0
