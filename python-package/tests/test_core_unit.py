"""Unit tests for pyhuge core parser logic."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from pyhuge import core


def test_huge_parses_result_with_mock_backend(monkeypatch):
    class _Pkg:
        @staticmethod
        def huge(**kwargs):
            return {"kwargs": kwargs}

    monkeypatch.setattr(core, "_r_env", lambda: {"huge_pkg": _Pkg()})
    monkeypatch.setattr(core, "_py2r", lambda x: x)
    monkeypatch.setattr(core, "_r2py", lambda x: x)
    monkeypatch.setattr(
        core,
        "_as_matrix_list",
        lambda _value, prefer_sparse=True: [sparse.csc_matrix(np.eye(3))],
    )
    monkeypatch.setattr(
        core,
        "_list_fields",
        lambda _value: {
            "method": np.array(["mb"]),
            "lambda": np.array([0.2, 0.1]),
            "sparsity": np.array([0.05, 0.1]),
            "path": object(),
            "cov.input": np.array([0]),
            "data": np.arange(12, dtype=float).reshape(4, 3),
            "df": np.array([[1, 2, 3]]),
        },
    )

    fit = core.huge(np.ones((4, 3)), method="mb", nlambda=2, verbose=False)

    assert fit.method == "mb"
    assert fit.lambda_path.shape == (2,)
    assert len(fit.path) == 1
    assert sparse.issparse(fit.path[0])
    assert fit.cov_input is False
    assert fit.df is not None


def test_huge_select_parses_result_with_mock_backend(monkeypatch):
    class _Pkg:
        @staticmethod
        def huge_select(**kwargs):
            return {"kwargs": kwargs}

    monkeypatch.setattr(core, "_r_env", lambda: {"huge_pkg": _Pkg()})
    monkeypatch.setattr(core, "_py2r", lambda x: x)
    monkeypatch.setattr(core, "_r2py", lambda x: x)
    monkeypatch.setattr(
        core,
        "_as_matrix",
        lambda _value, prefer_sparse=True: sparse.csc_matrix(np.eye(4))
        if prefer_sparse
        else np.eye(4),
    )
    monkeypatch.setattr(
        core,
        "_list_fields",
        lambda _value: {
            "criterion": np.array(["ric"]),
            "opt.lambda": np.array([0.12]),
            "opt.sparsity": np.array([0.08]),
            "refit": object(),
            "opt.index": np.array([3]),
        },
    )

    result = core.huge_select(object(), criterion="ric", verbose=False)

    assert result.criterion == "ric"
    assert isinstance(result.opt_lambda, float)
    assert isinstance(result.opt_sparsity, float)
    assert result.opt_index == 3
    assert sparse.issparse(result.refit)


def test_huge_generator_parses_result_with_mock_backend(monkeypatch):
    class _Pkg:
        @staticmethod
        def huge_generator(**kwargs):
            return {"kwargs": kwargs}

    monkeypatch.setattr(core, "_r_env", lambda: {"huge_pkg": _Pkg()})
    monkeypatch.setattr(core, "_r2py", lambda x: x)
    monkeypatch.setattr(core, "_scalar", lambda x: float(np.asarray(x).reshape(-1)[0]))
    monkeypatch.setattr(
        core,
        "_as_matrix",
        lambda _value, prefer_sparse=True: sparse.csc_matrix(np.eye(5))
        if prefer_sparse
        else np.eye(5),
    )
    monkeypatch.setattr(
        core,
        "_list_fields",
        lambda _value: {
            "data": np.arange(20, dtype=float).reshape(4, 5),
            "sigma": object(),
            "omega": object(),
            "sigmahat": object(),
            "theta": object(),
            "sparsity": np.array([0.2]),
            "graph.type": np.array(["hub"]),
        },
    )

    sim = core.huge_generator(n=4, d=5, graph="hub", verbose=False)

    assert sim.graph_type == "hub"
    assert isinstance(sim.sparsity, float)
    assert sim.data.shape == (4, 5)
    assert sparse.issparse(sim.theta)


def test_huge_inference_parses_result_with_mock_backend(monkeypatch):
    class _Pkg:
        @staticmethod
        def huge_inference(*args, **kwargs):
            return {"args": args, "kwargs": kwargs}

    monkeypatch.setattr(core, "_r_env", lambda: {"huge_pkg": _Pkg()})
    monkeypatch.setattr(core, "_py2r", lambda x: x)
    monkeypatch.setattr(core, "_r2py", lambda x: x)
    monkeypatch.setattr(core, "_scalar", lambda x: float(np.asarray(x).reshape(-1)[0]))
    monkeypatch.setattr(core, "_as_matrix", lambda _value, prefer_sparse=False: np.eye(3))
    monkeypatch.setattr(
        core,
        "_list_fields",
        lambda _value: {
            "data": np.arange(12, dtype=float).reshape(4, 3),
            "p": object(),
            "error": np.array([0.1]),
        },
    )

    out = core.huge_inference(np.ones((4, 3)), np.eye(3), np.eye(3), type_="Gaussian")
    assert out.data.shape == (4, 3)
    assert out.p.shape == (3, 3)
    assert isinstance(out.error, float)


def test_huge_roc_parses_result_with_mock_backend(monkeypatch):
    class _Pkg:
        @staticmethod
        def huge_roc(*args, **kwargs):
            return {"args": args, "kwargs": kwargs}

    class _RO:
        r = {
            "pdf": lambda _path: None,
            "dev.off": lambda: None,
        }

    monkeypatch.setattr(core, "_r_env", lambda: {"huge_pkg": _Pkg(), "ro": _RO()})
    monkeypatch.setattr(core, "_path_to_r", lambda path: path)
    monkeypatch.setattr(core, "_py2r", lambda x: x)
    monkeypatch.setattr(core, "_r2py", lambda x: x)
    monkeypatch.setattr(core, "_scalar", lambda x: float(np.asarray(x).reshape(-1)[0]))
    monkeypatch.setattr(
        core,
        "_list_fields",
        lambda _value: {
            "F1": np.array([0.2, 0.3]),
            "tp": np.array([0.4, 0.5]),
            "fp": np.array([0.1, 0.2]),
            "AUC": np.array([0.77]),
        },
    )

    roc = core.huge_roc([np.eye(3), np.eye(3)], np.eye(3), verbose=False, plot=False)
    assert roc.f1.shape == (2,)
    assert roc.tp.shape == (2,)
    assert roc.fp.shape == (2,)
    assert isinstance(roc.auc, float)


def test_method_specific_wrappers_route_method(monkeypatch):
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
        icov=None,
        cov=None,
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


def test_plot_helpers(monkeypatch):
    class _Ax:
        def plot(self, *args, **kwargs):
            return None

        def set_xscale(self, *args, **kwargs):
            return None

        def set_xlim(self, *args, **kwargs):
            return None

        def set_ylim(self, *args, **kwargs):
            return None

        def set_xlabel(self, *args, **kwargs):
            return None

        def set_ylabel(self, *args, **kwargs):
            return None

        def set_title(self, *args, **kwargs):
            return None

        def imshow(self, *args, **kwargs):
            return None

        def set_axis_off(self, *args, **kwargs):
            return None

    class _Plt:
        @staticmethod
        def subplots(*args, **kwargs):
            return object(), _Ax()

    monkeypatch.setattr(core, "_mpl_pyplot", lambda: _Plt())

    fit = core.HugeResult(
        method="mb",
        lambda_path=np.array([0.3, 0.2, 0.1]),
        sparsity=np.array([0.05, 0.08, 0.12]),
        path=[sparse.csc_matrix(np.eye(4)) for _ in range(3)],
        cov_input=False,
        data=np.ones((20, 4)),
        raw=None,
    )
    roc = core.HugeRocResult(
        f1=np.array([0.2, 0.3]),
        tp=np.array([0.4, 0.6]),
        fp=np.array([0.1, 0.2]),
        auc=0.7,
        raw=None,
    )

    ax1 = core.huge_plot_sparsity(fit)
    ax2 = core.huge_plot_roc(roc)
    ax3 = core.huge_plot_graph_matrix(fit, index=0)

    assert ax1 is not None
    assert ax2 is not None
    assert ax3 is not None


def test_plot_network_helper(monkeypatch):
    class _Ax:
        def set_title(self, *args, **kwargs):
            return None

        def set_axis_off(self, *args, **kwargs):
            return None

    class _Plt:
        @staticmethod
        def subplots(*args, **kwargs):
            return object(), _Ax()

    class _NX:
        @staticmethod
        def from_numpy_array(arr):
            return {"arr": arr}

        @staticmethod
        def spring_layout(_graph):
            return {0: (0.0, 0.0)}

        @staticmethod
        def kamada_kawai_layout(_graph):
            return {0: (0.0, 0.0)}

        @staticmethod
        def circular_layout(_graph):
            return {0: (0.0, 0.0)}

        @staticmethod
        def spectral_layout(_graph):
            return {0: (0.0, 0.0)}

        @staticmethod
        def shell_layout(_graph):
            return {0: (0.0, 0.0)}

        @staticmethod
        def draw_networkx(*args, **kwargs):
            return None

    monkeypatch.setattr(core, "_mpl_pyplot", lambda: _Plt())
    monkeypatch.setattr(core, "_networkx_pkg", lambda: _NX())

    fit = core.HugeResult(
        method="mb",
        lambda_path=np.array([0.2]),
        sparsity=np.array([0.1]),
        path=[sparse.csc_matrix(np.eye(4))],
        cov_input=False,
        data=np.ones((20, 4)),
        raw=None,
    )

    ax = core.huge_plot_network(fit, layout="spring")
    assert ax is not None

    with pytest.raises(core.PyHugeError, match="`layout`"):
        core.huge_plot_network(fit, layout="invalid")


def test_huge_invalid_method_raises():
    with pytest.raises(core.PyHugeError, match="`method`"):
        core.huge(np.ones((4, 3)), method="invalid")


def test_huge_lambda_must_be_decreasing():
    with pytest.raises(core.PyHugeError, match="strictly decreasing"):
        core.huge(np.ones((4, 3)), method="mb", lambda_=[0.2, 0.3])


def test_huge_scr_num_requires_scr_true():
    with pytest.raises(core.PyHugeError, match="requires `scr=True`"):
        core.huge(np.ones((4, 3)), method="mb", scr_num=5, scr=False)


def test_huge_select_invalid_criterion_raises():
    dummy = core.HugeResult(
        method="mb",
        lambda_path=np.array([0.2]),
        sparsity=np.array([0.1]),
        path=[sparse.csc_matrix(np.eye(3))],
        cov_input=False,
        data=np.ones((5, 3)),
    )
    with pytest.raises(core.PyHugeError, match="`criterion`"):
        core.huge_select(dummy, criterion="invalid")


def test_huge_select_ebic_requires_glasso():
    dummy = core.HugeResult(
        method="mb",
        lambda_path=np.array([0.2]),
        sparsity=np.array([0.1]),
        path=[sparse.csc_matrix(np.eye(3))],
        cov_input=False,
        data=np.ones((5, 3)),
    )
    with pytest.raises(core.PyHugeError, match="requires a glasso fit"):
        core.huge_select(dummy, criterion="ebic")


def test_huge_generator_invalid_graph_raises():
    with pytest.raises(core.PyHugeError, match="`graph`"):
        core.huge_generator(graph="invalid")


def test_huge_npn_invalid_func_raises():
    with pytest.raises(core.PyHugeError, match="`npn_func`"):
        core.huge_npn(np.ones((4, 3)), npn_func="invalid")


def test_huge_inference_shape_mismatch_raises():
    with pytest.raises(core.PyHugeError, match="`t` must have shape"):
        core.huge_inference(
            data=np.ones((5, 3)),
            t=np.eye(4),
            adj=np.eye(3),
        )


def test_huge_roc_path_shape_mismatch_raises():
    with pytest.raises(core.PyHugeError, match="must have shape"):
        core.huge_roc([np.eye(3), np.eye(4)], np.eye(3), plot=False)


def test_r_env_arch_mismatch_raises_early(monkeypatch):
    monkeypatch.setattr(core, "_R_ENV", None)
    monkeypatch.setattr(core, "_detect_arch_mismatch", lambda: ("x86_64", "arm64"))
    with pytest.raises(core.PyHugeError, match="architectures do not match"):
        core._r_env()


def test_detect_arch_mismatch_returns_none_when_same(monkeypatch):
    monkeypatch.setattr(core, "_python_arch", lambda: "arm64")
    monkeypatch.setattr(core, "_r_arch", lambda: "arm64")
    assert core._detect_arch_mismatch() is None


def test_detect_arch_mismatch_returns_tuple_when_different(monkeypatch):
    monkeypatch.setattr(core, "_python_arch", lambda: "x86_64")
    monkeypatch.setattr(core, "_r_arch", lambda: "arm64")
    assert core._detect_arch_mismatch() == ("x86_64", "arm64")
