"""Core API for Python-to-R wrapper of package huge."""

from __future__ import annotations

from dataclasses import dataclass
import os
import tempfile
from typing import Any, Optional, Sequence

import numpy as np
from scipy import sparse


class PyHugeError(RuntimeError):
    """Raised when pyhuge cannot call the underlying R package."""


@dataclass
class HugeResult:
    """Parsed output from R `huge()`."""

    method: str
    lambda_path: np.ndarray
    sparsity: np.ndarray
    path: list[sparse.spmatrix]
    cov_input: bool
    data: np.ndarray
    df: Optional[np.ndarray] = None
    loglik: Optional[np.ndarray] = None
    icov: Optional[list[np.ndarray]] = None
    cov: Optional[list[np.ndarray]] = None
    idx_mat: Optional[np.ndarray] = None
    raw: Any = None


@dataclass
class HugeSelectResult:
    """Parsed output from R `huge.select()`."""

    criterion: str
    opt_lambda: float
    opt_sparsity: float
    refit: sparse.spmatrix
    opt_index: Optional[int] = None
    variability: Optional[np.ndarray] = None
    ebic_score: Optional[np.ndarray] = None
    opt_icov: Optional[np.ndarray] = None
    opt_cov: Optional[np.ndarray] = None
    raw: Any = None


@dataclass
class HugeGeneratorResult:
    """Parsed output from R `huge.generator()`."""

    data: np.ndarray
    sigma: np.ndarray
    omega: np.ndarray
    sigmahat: np.ndarray
    theta: sparse.spmatrix
    sparsity: float
    graph_type: str
    raw: Any = None


@dataclass
class HugeInferenceResult:
    """Parsed output from R `huge.inference()`."""

    data: np.ndarray
    p: np.ndarray
    error: float
    raw: Any = None


@dataclass
class HugeRocResult:
    """Parsed output from R `huge.roc()`."""

    f1: np.ndarray
    tp: np.ndarray
    fp: np.ndarray
    auc: float
    raw: Any = None


@dataclass
class HugeSummary:
    """Compact summary for HugeResult."""

    method: str
    n_samples: int
    n_features: int
    path_length: int
    sparsity_min: float
    sparsity_max: float
    cov_input: bool
    has_icov: bool
    has_cov: bool


@dataclass
class HugeSelectSummary:
    """Compact summary for HugeSelectResult."""

    criterion: str
    opt_lambda: float
    opt_sparsity: float
    refit_n_features: int
    has_opt_icov: bool
    has_opt_cov: bool


_R_ENV: Optional[dict[str, Any]] = None
_ALLOWED_METHODS = {"mb", "glasso", "ct", "tiger"}
_ALLOWED_SYM = {"and", "or"}
_ALLOWED_CRITERIA = {"ric", "stars", "ebic"}
_ALLOWED_GRAPH_TYPES = {"random", "hub", "cluster", "band", "scale-free"}
_ALLOWED_NPN_FUNCS = {"shrinkage", "truncation", "skeptic"}
_ALLOWED_INFERENCE_TYPES = {"Gaussian", "Nonparanormal"}
_ALLOWED_INFERENCE_METHODS = {"score", "wald"}


def _r_env() -> dict[str, Any]:
    """Lazily import rpy2 and R package `huge`."""

    global _R_ENV
    if _R_ENV is not None:
        return _R_ENV

    try:
        import rpy2.robjects as ro
        from rpy2.robjects import default_converter, numpy2ri
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects.packages import PackageNotInstalledError, importr
    except ModuleNotFoundError as exc:
        raise PyHugeError(
            "rpy2 is required. Install with `pip install rpy2`."
        ) from exc

    try:
        huge_pkg = importr("huge")
    except PackageNotInstalledError as exc:
        raise PyHugeError(
            "R package `huge` is not installed in current R library paths."
        ) from exc

    _R_ENV = {
        "ro": ro,
        "default_converter": default_converter,
        "numpy2ri": numpy2ri,
        "localconverter": localconverter,
        "huge_pkg": huge_pkg,
    }
    return _R_ENV


def _py2r(value: Any) -> Any:
    env = _r_env()
    ro = env["ro"]
    if value is None:
        return ro.NULL
    with env["localconverter"](env["default_converter"] + env["numpy2ri"].converter):
        return ro.conversion.py2rpy(value)


def _r2py(value: Any) -> Any:
    env = _r_env()
    with env["localconverter"](env["default_converter"] + env["numpy2ri"].converter):
        return env["ro"].conversion.rpy2py(value)


def _scalar(value: Any) -> Optional[float]:
    arr = np.asarray(_r2py(value))
    if arr.size == 0:
        return None
    return float(arr.reshape(-1)[0])


def _r_class(value: Any) -> set[str]:
    env = _r_env()
    classes = env["ro"].r["class"](value)
    return {str(x) for x in classes}


def _rs4_to_sparse(value: Any) -> sparse.spmatrix:
    dim = np.asarray(_r2py(value.do_slot("Dim")), dtype=np.int32)
    i = np.asarray(_r2py(value.do_slot("i")), dtype=np.int32)
    p = np.asarray(_r2py(value.do_slot("p")), dtype=np.int32)
    x = np.asarray(_r2py(value.do_slot("x")), dtype=float)
    return sparse.csc_matrix((x, i, p), shape=(int(dim[0]), int(dim[1])))


def _as_matrix(value: Any, prefer_sparse: bool = True) -> sparse.spmatrix | np.ndarray:
    if prefer_sparse:
        classes = _r_class(value)
        if any(c.endswith("CMatrix") for c in classes):
            return _rs4_to_sparse(value)
    env = _r_env()
    dense = env["ro"].r["as.matrix"](value)
    array = np.asarray(_r2py(dense), dtype=float)
    return sparse.csc_matrix(array) if prefer_sparse else array


def _as_matrix_list(value: Any, prefer_sparse: bool = True) -> list[sparse.spmatrix | np.ndarray]:
    return [_as_matrix(value[i], prefer_sparse=prefer_sparse) for i in range(len(value))]


def _list_fields(r_list: Any) -> dict[str, Any]:
    names = list(r_list.names)
    return {str(n): r_list.rx2(str(n)) for n in names if n is not None}


def _to_dense_matrix(value: Any, name: str) -> np.ndarray:
    arr = value.toarray() if sparse.issparse(value) else np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise PyHugeError(f"`{name}` must be a 2D array-like matrix.")
    return np.asarray(arr, dtype=float)


def _path_to_r(path: Sequence[Any]) -> Any:
    env = _r_env()
    ro = env["ro"]
    items: list[tuple[str, Any]] = []
    for i, mat in enumerate(path, start=1):
        dense = _to_dense_matrix(mat, f"path[{i}]")
        items.append((str(i), _py2r(dense)))
    return ro.ListVector(items)


def _mpl_pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise PyHugeError(
            "matplotlib is required for plotting. Install with `pip install matplotlib`."
        ) from exc
    return plt


def _networkx_pkg() -> Any:
    try:
        import networkx as nx
    except ModuleNotFoundError as exc:
        raise PyHugeError(
            "networkx is required for network plotting. Install with `pip install networkx`."
        ) from exc
    return nx


def _ensure_2d_array(name: str, value: Any, finite: bool = True) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise PyHugeError(f"`{name}` must be a 2D array.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise PyHugeError(f"`{name}` must be non-empty.")
    if finite and not np.isfinite(arr).all():
        raise PyHugeError(f"`{name}` contains non-finite values.")
    return arr


def _ensure_positive_int(name: str, value: Any) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise PyHugeError(f"`{name}` must be a positive integer.")
    return ivalue


def _ensure_lambda_sequence(lambda_: Sequence[float]) -> np.ndarray:
    lam = np.asarray(lambda_, dtype=float).reshape(-1)
    if lam.size == 0:
        raise PyHugeError("`lambda_` must contain at least one value.")
    if not np.isfinite(lam).all():
        raise PyHugeError("`lambda_` contains non-finite values.")
    if np.any(lam <= 0):
        raise PyHugeError("`lambda_` must contain positive values.")
    if lam.size > 1 and np.any(np.diff(lam) >= 0):
        raise PyHugeError("`lambda_` must be strictly decreasing.")
    return lam


def _ensure_ratio(name: str, value: float, low_open: float = 0.0, high_closed: float = 1.0) -> float:
    fval = float(value)
    if not np.isfinite(fval):
        raise PyHugeError(f"`{name}` must be finite.")
    if not (fval > low_open and fval <= high_closed):
        raise PyHugeError(f"`{name}` must satisfy {low_open} < {name} <= {high_closed}.")
    return fval


def huge(
    x: np.ndarray,
    lambda_: Optional[Sequence[float]] = None,
    nlambda: Optional[int] = None,
    lambda_min_ratio: Optional[float] = None,
    method: str = "mb",
    scr: Optional[bool] = None,
    scr_num: Optional[int] = None,
    cov_output: bool = False,
    sym: str = "or",
    verbose: bool = True,
) -> HugeResult:
    """Run graph path estimation through R ``huge()``.

    Parameters follow R package semantics where possible.
    Returns a parsed :class:`HugeResult` and keeps the original R object
    in ``result.raw`` for advanced interoperability.
    """

    if method not in _ALLOWED_METHODS:
        raise PyHugeError(
            f"`method` must be one of {sorted(_ALLOWED_METHODS)}."
        )
    if sym not in _ALLOWED_SYM:
        raise PyHugeError(f"`sym` must be one of {sorted(_ALLOWED_SYM)}.")

    x = _ensure_2d_array("x", x, finite=True)

    if nlambda is not None:
        nlambda = _ensure_positive_int("nlambda", nlambda)
    if lambda_min_ratio is not None:
        lambda_min_ratio = _ensure_ratio("lambda_min_ratio", lambda_min_ratio)
    if lambda_ is not None:
        lambda_ = _ensure_lambda_sequence(lambda_)

    if method in {"ct", "tiger"} and scr is not None:
        raise PyHugeError("`scr` is only applicable for method `mb` and `glasso`.")
    if method != "mb" and scr_num is not None:
        raise PyHugeError("`scr_num` is only applicable for method `mb`.")
    if scr_num is not None:
        scr_num = _ensure_positive_int("scr_num", scr_num)
        if scr is not True:
            raise PyHugeError("`scr_num` requires `scr=True`.")
    if cov_output and method != "glasso":
        raise PyHugeError("`cov_output=True` is only valid for method `glasso`.")
    if method in {"ct", "glasso"} and sym != "or":
        raise PyHugeError("`sym` is only applicable to method `mb` and `tiger`.")

    env = _r_env()

    kwargs: dict[str, Any] = {
        "x": _py2r(x),
        "method": method,
        "cov.output": bool(cov_output),
        "sym": sym,
        "verbose": bool(verbose),
    }
    if lambda_ is not None:
        kwargs["lambda"] = _py2r(lambda_)
    if nlambda is not None:
        kwargs["nlambda"] = nlambda
    if lambda_min_ratio is not None:
        kwargs["lambda.min.ratio"] = lambda_min_ratio
    if scr is not None:
        kwargs["scr"] = bool(scr)
    if scr_num is not None:
        kwargs["scr.num"] = scr_num

    r_fit = env["huge_pkg"].huge(**kwargs)
    fields = _list_fields(r_fit)

    fit = HugeResult(
        method=str(np.asarray(_r2py(fields["method"])).reshape(-1)[0]),
        lambda_path=np.asarray(_r2py(fields["lambda"]), dtype=float),
        sparsity=np.asarray(_r2py(fields["sparsity"]), dtype=float),
        path=[m for m in _as_matrix_list(fields["path"], prefer_sparse=True)],
        cov_input=bool(_scalar(fields["cov.input"])),
        data=np.asarray(_r2py(fields["data"]), dtype=float),
        raw=r_fit,
    )

    if "df" in fields:
        fit.df = np.asarray(_r2py(fields["df"]))
    if "loglik" in fields:
        fit.loglik = np.asarray(_r2py(fields["loglik"]), dtype=float)
    if "idx_mat" in fields:
        fit.idx_mat = np.asarray(_r2py(fields["idx_mat"]))
    if "icov" in fields:
        fit.icov = [
            np.asarray(m, dtype=float) for m in _as_matrix_list(fields["icov"], prefer_sparse=False)
        ]
    if "cov" in fields:
        fit.cov = [
            np.asarray(m, dtype=float) for m in _as_matrix_list(fields["cov"], prefer_sparse=False)
        ]

    return fit


def huge_mb(
    x: np.ndarray,
    lambda_: Optional[Sequence[float]] = None,
    nlambda: Optional[int] = None,
    lambda_min_ratio: Optional[float] = None,
    scr: Optional[bool] = None,
    scr_num: Optional[int] = None,
    sym: str = "or",
    verbose: bool = True,
) -> HugeResult:
    """Convenience wrapper for ``huge(..., method="mb")``."""

    return huge(
        x=x,
        lambda_=lambda_,
        nlambda=nlambda,
        lambda_min_ratio=lambda_min_ratio,
        method="mb",
        scr=scr,
        scr_num=scr_num,
        cov_output=False,
        sym=sym,
        verbose=verbose,
    )


def huge_glasso(
    x: np.ndarray,
    lambda_: Optional[Sequence[float]] = None,
    nlambda: Optional[int] = None,
    lambda_min_ratio: Optional[float] = None,
    scr: Optional[bool] = None,
    cov_output: bool = False,
    verbose: bool = True,
) -> HugeResult:
    """Convenience wrapper for ``huge(..., method="glasso")``."""

    return huge(
        x=x,
        lambda_=lambda_,
        nlambda=nlambda,
        lambda_min_ratio=lambda_min_ratio,
        method="glasso",
        scr=scr,
        scr_num=None,
        cov_output=cov_output,
        sym="or",
        verbose=verbose,
    )


def huge_ct(
    x: np.ndarray,
    lambda_: Optional[Sequence[float]] = None,
    nlambda: Optional[int] = None,
    lambda_min_ratio: Optional[float] = None,
    verbose: bool = True,
) -> HugeResult:
    """Convenience wrapper for ``huge(..., method="ct")``."""

    return huge(
        x=x,
        lambda_=lambda_,
        nlambda=nlambda,
        lambda_min_ratio=lambda_min_ratio,
        method="ct",
        scr=None,
        scr_num=None,
        cov_output=False,
        sym="or",
        verbose=verbose,
    )


def huge_tiger(
    x: np.ndarray,
    lambda_: Optional[Sequence[float]] = None,
    nlambda: Optional[int] = None,
    lambda_min_ratio: Optional[float] = None,
    sym: str = "or",
    verbose: bool = True,
) -> HugeResult:
    """Convenience wrapper for ``huge(..., method="tiger")``."""

    return huge(
        x=x,
        lambda_=lambda_,
        nlambda=nlambda,
        lambda_min_ratio=lambda_min_ratio,
        method="tiger",
        scr=None,
        scr_num=None,
        cov_output=False,
        sym=sym,
        verbose=verbose,
    )


def huge_select(
    est: HugeResult | Any,
    criterion: Optional[str] = None,
    ebic_gamma: float = 0.5,
    stars_thresh: float = 0.1,
    stars_subsample_ratio: Optional[float] = None,
    rep_num: int = 20,
    verbose: bool = True,
) -> HugeSelectResult:
    """Run model selection through R ``huge.select()``.

    Parameters align with R ``huge.select``. ``est`` can be either
    :class:`HugeResult` or a raw R object returned by ``huge()``.
    """

    if criterion is not None and criterion not in _ALLOWED_CRITERIA:
        raise PyHugeError(f"`criterion` must be one of {sorted(_ALLOWED_CRITERIA)}.")
    if not np.isfinite(float(ebic_gamma)):
        raise PyHugeError("`ebic_gamma` must be finite.")
    stars_thresh = _ensure_ratio("stars_thresh", stars_thresh)
    rep_num = _ensure_positive_int("rep_num", rep_num)
    if stars_subsample_ratio is not None:
        stars_subsample_ratio = _ensure_ratio("stars_subsample_ratio", stars_subsample_ratio)
    if isinstance(est, HugeResult) and criterion == "ebic" and est.method != "glasso":
        raise PyHugeError("`criterion='ebic'` requires a glasso fit.")

    env = _r_env()
    r_est = est.raw if isinstance(est, HugeResult) else est

    kwargs: dict[str, Any] = {
        "est": r_est,
        "ebic.gamma": float(ebic_gamma),
        "stars.thresh": stars_thresh,
        "rep.num": rep_num,
        "verbose": bool(verbose),
    }
    if criterion is not None:
        kwargs["criterion"] = criterion
    if stars_subsample_ratio is not None:
        kwargs["stars.subsample.ratio"] = stars_subsample_ratio

    r_sel = env["huge_pkg"].huge_select(**kwargs)
    fields = _list_fields(r_sel)

    criterion_value = str(np.asarray(_r2py(fields.get("criterion", _py2r(["ric"])))).reshape(-1)[0])
    refit = _as_matrix(fields["refit"], prefer_sparse=True)

    result = HugeSelectResult(
        criterion=criterion_value,
        opt_lambda=float(_scalar(fields["opt.lambda"])),
        opt_sparsity=float(_scalar(fields["opt.sparsity"])),
        refit=refit if sparse.issparse(refit) else sparse.csc_matrix(refit),
        raw=r_sel,
    )

    if "opt.index" in fields:
        result.opt_index = int(_scalar(fields["opt.index"]))
    if "variability" in fields:
        result.variability = np.asarray(_r2py(fields["variability"]), dtype=float)
    if "ebic.score" in fields:
        result.ebic_score = np.asarray(_r2py(fields["ebic.score"]), dtype=float)
    if "opt.icov" in fields:
        result.opt_icov = np.asarray(_as_matrix(fields["opt.icov"], prefer_sparse=False), dtype=float)
    if "opt.cov" in fields:
        result.opt_cov = np.asarray(_as_matrix(fields["opt.cov"], prefer_sparse=False), dtype=float)

    return result


def huge_npn(x: np.ndarray, npn_func: str = "shrinkage", verbose: bool = True) -> np.ndarray:
    """Apply nonparanormal transformation via R ``huge.npn()``.

    Returns the transformed data matrix as ``numpy.ndarray``.
    """

    if npn_func not in _ALLOWED_NPN_FUNCS:
        raise PyHugeError(f"`npn_func` must be one of {sorted(_ALLOWED_NPN_FUNCS)}.")
    env = _r_env()
    x = _ensure_2d_array("x", x, finite=True)

    r_npn = env["huge_pkg"].huge_npn(
        _py2r(x),
        **{
            "npn.func": npn_func,
            "verbose": bool(verbose),
        },
    )
    return np.asarray(_r2py(r_npn), dtype=float)


def huge_generator(
    n: int = 200,
    d: int = 50,
    graph: str = "random",
    v: Optional[float] = None,
    u: Optional[float] = None,
    g: Optional[int] = None,
    prob: Optional[float] = None,
    vis: bool = False,
    verbose: bool = True,
) -> HugeGeneratorResult:
    """Generate synthetic data via R ``huge.generator()``."""

    n = _ensure_positive_int("n", n)
    d = _ensure_positive_int("d", d)
    if graph not in _ALLOWED_GRAPH_TYPES:
        raise PyHugeError(f"`graph` must be one of {sorted(_ALLOWED_GRAPH_TYPES)}.")
    if v is not None and float(v) <= 0:
        raise PyHugeError("`v` must be positive.")
    if u is not None and float(u) <= 0:
        raise PyHugeError("`u` must be positive.")
    if g is not None:
        g = _ensure_positive_int("g", g)
    if prob is not None:
        prob_val = float(prob)
        if not np.isfinite(prob_val) or not (0.0 <= prob_val <= 1.0):
            raise PyHugeError("`prob` must satisfy 0 <= prob <= 1.")
        prob = prob_val

    env = _r_env()
    kwargs: dict[str, Any] = {
        "n": n,
        "d": d,
        "graph": graph,
        "vis": bool(vis),
        "verbose": bool(verbose),
    }
    if v is not None:
        kwargs["v"] = float(v)
    if u is not None:
        kwargs["u"] = float(u)
    if g is not None:
        kwargs["g"] = int(g)
    if prob is not None:
        kwargs["prob"] = float(prob)

    r_sim = env["huge_pkg"].huge_generator(**kwargs)
    fields = _list_fields(r_sim)

    theta = _as_matrix(fields["theta"], prefer_sparse=True)
    sparsity_val = _scalar(fields["sparsity"])
    if sparsity_val is None:
        theta_dense = theta.toarray() if sparse.issparse(theta) else np.asarray(theta)
        d_theta = theta_dense.shape[1]
        sparsity_val = float(np.sum(theta_dense != 0) / (d_theta * (d_theta - 1)))

    graph_type = str(np.asarray(_r2py(fields["graph.type"])).reshape(-1)[0]) if "graph.type" in fields else graph

    return HugeGeneratorResult(
        data=np.asarray(_r2py(fields["data"]), dtype=float),
        sigma=np.asarray(_as_matrix(fields["sigma"], prefer_sparse=False), dtype=float),
        omega=np.asarray(_as_matrix(fields["omega"], prefer_sparse=False), dtype=float),
        sigmahat=np.asarray(_as_matrix(fields["sigmahat"], prefer_sparse=False), dtype=float),
        theta=theta if sparse.issparse(theta) else sparse.csc_matrix(theta),
        sparsity=float(sparsity_val),
        graph_type=graph_type,
        raw=r_sim,
    )


def huge_inference(
    data: np.ndarray,
    t: np.ndarray,
    adj: np.ndarray | sparse.spmatrix,
    alpha: float = 0.05,
    type_: str = "Gaussian",
    method: str = "score",
) -> HugeInferenceResult:
    """Run edge inference via R ``huge.inference()``."""

    if type_ not in _ALLOWED_INFERENCE_TYPES:
        raise PyHugeError(f"`type_` must be one of {sorted(_ALLOWED_INFERENCE_TYPES)}.")
    if method not in _ALLOWED_INFERENCE_METHODS:
        raise PyHugeError(f"`method` must be one of {sorted(_ALLOWED_INFERENCE_METHODS)}.")
    alpha = _ensure_ratio("alpha", alpha)

    data = _ensure_2d_array("data", data, finite=True)
    t = _to_dense_matrix(t, "t")
    adj = _to_dense_matrix(adj, "adj")
    d = data.shape[1]
    if t.shape != (d, d):
        raise PyHugeError(f"`t` must have shape ({d}, {d}).")
    if adj.shape != (d, d):
        raise PyHugeError(f"`adj` must have shape ({d}, {d}).")
    if not np.isfinite(t).all() or not np.isfinite(adj).all():
        raise PyHugeError("`t` and `adj` must contain finite values.")
    env = _r_env()

    r_inf = env["huge_pkg"].huge_inference(
        _py2r(data),
        _py2r(t),
        _py2r(adj),
        alpha=alpha,
        type=type_,
        method=method,
    )
    fields = _list_fields(r_inf)

    return HugeInferenceResult(
        data=np.asarray(_r2py(fields["data"]), dtype=float),
        p=np.asarray(_as_matrix(fields["p"], prefer_sparse=False), dtype=float),
        error=float(_scalar(fields["error"])),
        raw=r_inf,
    )


def huge_roc(
    path: Sequence[np.ndarray | sparse.spmatrix],
    theta: np.ndarray | sparse.spmatrix,
    verbose: bool = True,
    plot: bool = False,
) -> HugeRocResult:
    """Compute ROC metrics via R ``huge.roc()``.

    By default ``plot=False`` routes plotting to a temporary PDF device to avoid
    GUI-device requirements in headless environments.
    """

    if len(path) == 0:
        raise PyHugeError("`path` must contain at least one adjacency matrix.")
    theta_dense = _to_dense_matrix(theta, "theta")
    if theta_dense.shape[0] != theta_dense.shape[1]:
        raise PyHugeError("`theta` must be square.")
    d = theta_dense.shape[0]
    for i, mat in enumerate(path, start=1):
        dense = _to_dense_matrix(mat, f"path[{i}]")
        if dense.shape != (d, d):
            raise PyHugeError(f"`path[{i}]` must have shape ({d}, {d}).")

    env = _r_env()
    ro = env["ro"]
    r_path = _path_to_r(path)

    pdf_fd: Optional[int] = None
    pdf_path: Optional[str] = None
    try:
        if not plot:
            pdf_fd, pdf_path = tempfile.mkstemp(prefix="pyhuge_roc_", suffix=".pdf")
            os.close(pdf_fd)
            pdf_fd = None
            ro.r["pdf"](pdf_path)

        r_roc = env["huge_pkg"].huge_roc(r_path, _py2r(theta_dense), verbose=bool(verbose))
    finally:
        if not plot:
            try:
                ro.r["dev.off"]()
            except Exception:
                pass
            if pdf_path and os.path.exists(pdf_path):
                try:
                    os.remove(pdf_path)
                except OSError:
                    pass
            if pdf_fd is not None:
                try:
                    os.close(pdf_fd)
                except OSError:
                    pass

    fields = _list_fields(r_roc)
    return HugeRocResult(
        f1=np.asarray(_r2py(fields["F1"]), dtype=float),
        tp=np.asarray(_r2py(fields["tp"]), dtype=float),
        fp=np.asarray(_r2py(fields["fp"]), dtype=float),
        auc=float(_scalar(fields["AUC"])),
        raw=r_roc,
    )


def huge_summary(fit: HugeResult) -> HugeSummary:
    """Return a concise, Python-native summary of ``HugeResult``."""

    n_samples, n_features = fit.data.shape
    return HugeSummary(
        method=fit.method,
        n_samples=int(n_samples),
        n_features=int(n_features),
        path_length=int(len(fit.lambda_path)),
        sparsity_min=float(np.min(fit.sparsity)),
        sparsity_max=float(np.max(fit.sparsity)),
        cov_input=bool(fit.cov_input),
        has_icov=fit.icov is not None,
        has_cov=fit.cov is not None,
    )


def huge_select_summary(sel: HugeSelectResult) -> HugeSelectSummary:
    """Return a concise, Python-native summary of ``HugeSelectResult``."""

    refit_shape = sel.refit.shape
    return HugeSelectSummary(
        criterion=sel.criterion,
        opt_lambda=float(sel.opt_lambda),
        opt_sparsity=float(sel.opt_sparsity),
        refit_n_features=int(refit_shape[1]),
        has_opt_icov=sel.opt_icov is not None,
        has_opt_cov=sel.opt_cov is not None,
    )


def huge_plot_sparsity(
    fit: HugeResult,
    ax: Optional[Any] = None,
    show_points: bool = True,
) -> Any:
    """Plot sparsity level versus regularization path."""

    if fit.lambda_path.size == 0:
        raise PyHugeError("`fit.lambda_path` is empty.")
    if fit.sparsity.size != fit.lambda_path.size:
        raise PyHugeError("`fit.sparsity` length must match `fit.lambda_path` length.")

    plt = _mpl_pyplot()
    if ax is None:
        _, ax = plt.subplots(1, 1)

    ax.plot(fit.lambda_path, fit.sparsity, "-", lw=1.6)
    if show_points:
        ax.plot(fit.lambda_path, fit.sparsity, "o", ms=3)
    if np.all(fit.lambda_path > 0):
        ax.set_xscale("log")
        ax.set_xlim(np.max(fit.lambda_path), np.min(fit.lambda_path))
    ax.set_xlabel("Regularization Parameter")
    ax.set_ylabel("Sparsity Level")
    ax.set_title(f"Sparsity Path ({fit.method})")
    return ax


def huge_plot_roc(roc: HugeRocResult, ax: Optional[Any] = None) -> Any:
    """Plot ROC curve from ``HugeRocResult``."""

    plt = _mpl_pyplot()
    if ax is None:
        _, ax = plt.subplots(1, 1)

    order = np.argsort(roc.fp)
    fp = roc.fp[order]
    tp = roc.tp[order]
    ax.plot(fp, tp, "-o", ms=3, lw=1.6)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve (AUC={roc.auc:.4f})")
    return ax


def huge_plot_graph_matrix(
    fit: HugeResult,
    index: int = -1,
    ax: Optional[Any] = None,
) -> Any:
    """Visualize one adjacency matrix on the path as a heatmap."""

    if len(fit.path) == 0:
        raise PyHugeError("`fit.path` is empty.")

    plt = _mpl_pyplot()
    if ax is None:
        _, ax = plt.subplots(1, 1)

    if index < 0:
        index = len(fit.path) + index
    if index < 0 or index >= len(fit.path):
        raise PyHugeError(f"`index` out of range: got {index}, path length={len(fit.path)}")

    mat = fit.path[index]
    dense = mat.toarray() if sparse.issparse(mat) else np.asarray(mat)
    ax.imshow(dense, cmap="Greys", interpolation="nearest")
    if index < fit.lambda_path.size:
        title = f"Graph Matrix (idx={index}, lambda={fit.lambda_path[index]:.4g})"
    else:
        title = f"Graph Matrix (idx={index})"
    ax.set_title(title)
    ax.set_xlabel("Node")
    ax.set_ylabel("Node")
    return ax


def huge_plot_network(
    fit: HugeResult,
    index: int = -1,
    ax: Optional[Any] = None,
    layout: str = "spring",
    with_labels: bool = False,
    node_size: float = 120.0,
    node_color: str = "#c44e52",
    edge_color: str = "#4d4d4d",
    min_abs_weight: float = 0.0,
) -> Any:
    """Plot one estimated graph as a node-edge network.

    Parameters
    ----------
    fit:
        `HugeResult` containing graph path.
    index:
        Path index to visualize; supports negative indexing.
    layout:
        One of ``spring``, ``kamada_kawai``, ``circular``, ``spectral``, ``shell``.
    with_labels:
        Whether to draw node labels.
    node_size, node_color, edge_color:
        Visual styling parameters.
    min_abs_weight:
        Keep only edges with ``abs(weight) >= min_abs_weight``.
    """

    if len(fit.path) == 0:
        raise PyHugeError("`fit.path` is empty.")
    if min_abs_weight < 0:
        raise PyHugeError("`min_abs_weight` must be non-negative.")

    if index < 0:
        index = len(fit.path) + index
    if index < 0 or index >= len(fit.path):
        raise PyHugeError(f"`index` out of range: got {index}, path length={len(fit.path)}")

    mat = fit.path[index]
    dense = mat.toarray() if sparse.issparse(mat) else np.asarray(mat, dtype=float)
    dense = np.asarray(dense, dtype=float)
    if dense.ndim != 2 or dense.shape[0] != dense.shape[1]:
        raise PyHugeError("Selected graph matrix must be square.")

    # Keep symmetric adjacency and drop weak edges.
    dense = 0.5 * (dense + dense.T)
    if min_abs_weight > 0:
        dense = dense.copy()
        dense[np.abs(dense) < min_abs_weight] = 0.0

    nx = _networkx_pkg()
    plt = _mpl_pyplot()
    if ax is None:
        _, ax = plt.subplots(1, 1)

    graph = nx.from_numpy_array(dense)
    layout_map = {
        "spring": nx.spring_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
        "circular": nx.circular_layout,
        "spectral": nx.spectral_layout,
        "shell": nx.shell_layout,
    }
    if layout not in layout_map:
        raise PyHugeError(f"`layout` must be one of {sorted(layout_map)}.")
    pos = layout_map[layout](graph)

    nx.draw_networkx(
        graph,
        pos=pos,
        ax=ax,
        with_labels=with_labels,
        node_size=float(node_size),
        node_color=node_color,
        edge_color=edge_color,
        width=1.2,
        alpha=0.9,
    )

    if index < fit.lambda_path.size:
        title = f"Network (idx={index}, lambda={fit.lambda_path[index]:.4g})"
    else:
        title = f"Network (idx={index})"
    ax.set_title(title)
    ax.set_axis_off()
    return ax
