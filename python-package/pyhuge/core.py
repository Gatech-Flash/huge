"""Native pyhuge 0.3 core implementation (no rpy2 dependency)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Optional, Sequence
import warnings

from importlib import resources

import numpy as np
from scipy import sparse, stats


class PyHugeError(RuntimeError):
    """Raised when pyhuge encounters invalid inputs or backend failures."""


@dataclass
class HugeResult:
    """Result from native ``huge()``."""

    method: str
    lambda_path: np.ndarray
    sparsity: np.ndarray
    path: list[sparse.csc_matrix]
    cov_input: bool
    data: np.ndarray
    sym: str = "or"
    df: Optional[np.ndarray] = None
    loglik: Optional[np.ndarray] = None
    icov: Optional[list[np.ndarray]] = None
    cov: Optional[list[np.ndarray]] = None
    idx_mat: Optional[np.ndarray] = None
    raw: Any = None


@dataclass
class HugeSelectResult:
    """Result from native ``huge_select()``."""

    criterion: str
    opt_lambda: float
    opt_sparsity: float
    refit: sparse.csc_matrix
    opt_index: Optional[int] = None
    variability: Optional[np.ndarray] = None
    ebic_score: Optional[np.ndarray] = None
    opt_icov: Optional[np.ndarray] = None
    opt_cov: Optional[np.ndarray] = None
    raw: Any = None


@dataclass
class HugeGeneratorResult:
    """Result from native ``huge_generator()``."""

    data: np.ndarray
    sigma: np.ndarray
    omega: np.ndarray
    sigmahat: np.ndarray
    theta: sparse.csc_matrix
    sparsity: float
    graph_type: str
    raw: Any = None


@dataclass
class HugeInferenceResult:
    """Result from native ``huge_inference()``."""

    data: np.ndarray
    p: np.ndarray
    error: float
    raw: Any = None


@dataclass
class HugeRocResult:
    """Result from native ``huge_roc()``."""

    f1: np.ndarray
    tp: np.ndarray
    fp: np.ndarray
    auc: float
    raw: Any = None


@dataclass
class HugeStockDataResult:
    """Built-in stock dataset result."""

    data: np.ndarray
    info: np.ndarray
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


_ALLOWED_METHODS = {"mb", "glasso", "ct", "tiger"}
_ALLOWED_SYM = {"and", "or"}
_ALLOWED_CRITERIA = {"ric", "stars", "ebic"}
_ALLOWED_GRAPH_TYPES = {"random", "hub", "cluster", "band", "scale-free"}
_ALLOWED_NPN_FUNCS = {"shrinkage", "truncation", "skeptic"}
_ALLOWED_INFERENCE_TYPES = {"Gaussian", "Nonparanormal"}
_ALLOWED_INFERENCE_METHODS = {"score", "wald"}


try:  # optional acceleration
    from . import _native_core as _CPP
except Exception:  # pragma: no cover - extension optional
    _CPP = None


def _ensure_backend_native(backend: str) -> None:
    if backend != "native":
        raise PyHugeError("pyhuge 0.3 supports only `backend=\"native\"`.")


def _ensure_2d_array(name: str, value: Any, finite: bool = True) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise PyHugeError(f"`{name}` must be a 2D array.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise PyHugeError(f"`{name}` must be non-empty.")
    if finite and not np.isfinite(arr).all():
        raise PyHugeError(f"`{name}` contains non-finite values.")
    return arr


def _to_dense_matrix(value: Any, name: str) -> np.ndarray:
    arr = value.toarray() if sparse.issparse(value) else np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise PyHugeError(f"`{name}` must be a 2D array-like matrix.")
    return np.asarray(arr, dtype=float)


def _ensure_positive_int(name: str, value: Any) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise PyHugeError(f"`{name}` must be a positive integer.")
    return ivalue


def _ensure_ratio(name: str, value: float, low_open: float = 0.0, high_closed: float = 1.0) -> float:
    fval = float(value)
    if not np.isfinite(fval):
        raise PyHugeError(f"`{name}` must be finite.")
    if not (fval > low_open and fval <= high_closed):
        raise PyHugeError(f"`{name}` must satisfy {low_open} < {name} <= {high_closed}.")
    return fval


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


def _ensure_ct_lambda_sequence(lambda_: Sequence[float]) -> np.ndarray:
    lam = np.asarray(lambda_, dtype=float).reshape(-1)
    if lam.size == 0:
        raise PyHugeError("`lambda_` must contain at least one value.")
    if not np.isfinite(lam).all():
        raise PyHugeError("`lambda_` contains non-finite values.")
    if np.any(lam <= 0):
        raise PyHugeError("`lambda_` must contain positive values.")
    if lam.size > 1 and np.any(np.diff(lam) > 0):
        raise PyHugeError("`lambda_` must be decreasing for method `ct` (ties are allowed).")
    return lam


def _is_covariance_input(x: np.ndarray) -> bool:
    return x.shape[0] == x.shape[1] and np.allclose(x, x.T, rtol=1e-5, atol=1e-8)


def _standardize(x: np.ndarray) -> np.ndarray:
    mu = np.mean(x, axis=0)
    xc = x - mu
    sd = np.std(xc, axis=0, ddof=1)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return xc / sd


def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    sd = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    corr = cov / np.outer(sd, sd)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def _offdiag_abs_max(mat: np.ndarray) -> float:
    d = mat.shape[0]
    if d <= 1:
        return 1e-3
    m = np.max(np.abs(mat[~np.eye(d, dtype=bool)]))
    return float(max(m, 1e-3))


def _default_nlambda(method: str) -> int:
    return 20 if method == "ct" else 10


def _default_lambda_min_ratio(method: str) -> float:
    return 0.05 if method == "ct" else 0.1


def _build_lambda_path(
    *,
    base_matrix: np.ndarray,
    method: str,
    lambda_: Optional[Sequence[float]],
    nlambda: Optional[int],
    lambda_min_ratio: Optional[float],
) -> np.ndarray:
    if lambda_ is not None:
        return _ensure_lambda_sequence(lambda_)

    nlam = _default_nlambda(method) if nlambda is None else _ensure_positive_int("nlambda", nlambda)
    ratio = (
        _default_lambda_min_ratio(method)
        if lambda_min_ratio is None
        else _ensure_ratio("lambda_min_ratio", lambda_min_ratio)
    )
    lam_max = _offdiag_abs_max(base_matrix)
    lam_min = lam_max * ratio

    if method == "ct":
        lam = np.linspace(lam_max, lam_min, nlam)
    else:
        lam = np.geomspace(lam_max, max(lam_min, lam_max * 1e-6), nlam)
    return _ensure_lambda_sequence(lam)


def _adj_sparsity(adj: np.ndarray) -> float:
    d = adj.shape[0]
    if d <= 1:
        return 0.0
    edges = float(np.count_nonzero(np.triu(adj, 1)))
    return (2.0 * edges) / (d * (d - 1))


def _path_sparsity(path: list[sparse.csc_matrix]) -> np.ndarray:
    if _CPP is not None:
        try:
            dense_path = [np.asarray(p.toarray() != 0, dtype=np.uint8) for p in path]
            out = np.asarray(_CPP.sparsity_path(dense_path), dtype=float)
            if out.shape == (len(path),):
                return out
        except Exception:
            pass
    return np.asarray([_adj_sparsity(p.toarray() != 0) for p in path], dtype=float)


def _symmetrize(directed: np.ndarray, sym: str) -> np.ndarray:
    if sym == "or":
        adj = np.logical_or(directed, directed.T)
    else:
        adj = np.logical_and(directed, directed.T)
    np.fill_diagonal(adj, False)
    return adj


def _run_ct(corr: np.ndarray, lambda_path: np.ndarray) -> list[sparse.csc_matrix]:
    if _CPP is not None:
        try:
            dense_path = _CPP.threshold_path(np.asarray(corr, dtype=float), np.asarray(lambda_path, dtype=float))
            out: list[sparse.csc_matrix] = []
            for m in dense_path:
                a = np.asarray(m, dtype=bool)
                np.fill_diagonal(a, False)
                out.append(sparse.csc_matrix(a.astype(float)))
            return out
        except Exception:
            pass

    out: list[sparse.csc_matrix] = []
    abs_corr = np.abs(corr)
    for lam in lambda_path:
        thr = float(lam) + 64.0 * np.finfo(float).eps * max(1.0, abs(float(lam)))
        adj = abs_corr > thr
        np.fill_diagonal(adj, False)
        out.append(sparse.csc_matrix(adj.astype(float)))
    return out


def _run_ct_default_rank(
    corr: np.ndarray,
    nlambda: int,
    lambda_min_ratio: float,
) -> tuple[np.ndarray, list[sparse.csc_matrix], np.ndarray]:
    # Match R huge.ct default path construction: rank-based density schedule.
    d = int(corr.shape[0])
    if d <= 1:
        lam = np.repeat(_offdiag_abs_max(corr), nlambda).astype(float)
        path = [sparse.csc_matrix((d, d), dtype=float) for _ in range(nlambda)]
        sparsity = np.zeros(nlambda, dtype=float)
        return lam, path, sparsity

    s = np.abs(np.asarray(corr, dtype=float))
    np.fill_diagonal(s, 0.0)

    density_max = float(lambda_min_ratio) * d * (d - 1) / 2.0
    density_min = 1.0
    density_all = np.ceil(np.linspace(density_min, density_max, num=nlambda)).astype(np.int64) * 2

    flat_f = s.reshape(-1, order="F")
    tie_order = np.arange(flat_f.size, dtype=np.int64)
    s_rank = np.lexsort((tie_order, -flat_f))

    lambda_path = flat_f[s_rank[density_all - 1]].astype(float)
    sparsity = density_all.astype(float) / float(d * (d - 1))

    path: list[sparse.csc_matrix] = []
    for k in density_all:
        adj_flat = np.zeros(flat_f.size, dtype=bool)
        if k > 0:
            adj_flat[s_rank[: int(k)]] = True
        adj = adj_flat.reshape((d, d), order="F")
        np.fill_diagonal(adj, False)
        path.append(sparse.csc_matrix(adj.astype(float)))

    return lambda_path, path, sparsity


def _ct_path_sparsity(path: list[sparse.csc_matrix]) -> np.ndarray:
    if len(path) == 0:
        return np.asarray([], dtype=float)
    d = int(path[0].shape[0])
    denom = float(d * (d - 1))
    if denom <= 0:
        return np.zeros(len(path), dtype=float)
    return np.asarray([float(p.nnz) / denom for p in path], dtype=float)


def _require_sklearn(component: str) -> None:
    try:
        import sklearn  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise PyHugeError(
            f"scikit-learn is required for native `{component}`. Install with `pip install scikit-learn`."
        ) from exc


def _run_glasso(
    s_mat: np.ndarray,
    lambda_path: np.ndarray,
    scr: bool,
    cov_output: bool,
) -> tuple[list[sparse.csc_matrix], list[np.ndarray], Optional[list[np.ndarray]], np.ndarray, np.ndarray]:
    if _CPP is not None:
        try:
            out = _CPP.hugeglasso(
                np.asarray(s_mat, dtype=float),
                np.asarray(lambda_path, dtype=float),
                bool(scr),
                bool(cov_output),
            )
            path_cube = np.asarray(out["path"], dtype=np.uint8)
            icov_cube = np.asarray(out["icov"], dtype=float)
            loglik = np.asarray(out["loglik"], dtype=float).reshape(-1)
            df = np.asarray(out["df"], dtype=float).reshape(-1)

            path: list[sparse.csc_matrix] = []
            icov: list[np.ndarray] = []
            cov_list: list[np.ndarray] = []
            for i in range(path_cube.shape[0]):
                adj = path_cube[i] != 0
                np.fill_diagonal(adj, False)
                path.append(sparse.csc_matrix(adj.astype(float)))
                icov.append(np.asarray(icov_cube[i], dtype=float))

            cov_raw = out.get("cov", None)
            if cov_output and cov_raw is not None:
                cov_cube = np.asarray(cov_raw, dtype=float)
                for i in range(cov_cube.shape[0]):
                    cov_list.append(np.asarray(cov_cube[i], dtype=float))
                cov_ret: Optional[list[np.ndarray]] = cov_list
            else:
                cov_ret = None

            return path, icov, cov_ret, df, loglik
        except Exception:
            pass

    _require_sklearn("glasso")
    from sklearn.covariance import graphical_lasso
    from sklearn.exceptions import ConvergenceWarning

    path: list[sparse.csc_matrix] = []
    icov: list[np.ndarray] = []
    cov_list: list[np.ndarray] = []
    loglik = np.zeros(lambda_path.size, dtype=float)
    df = np.zeros(lambda_path.size, dtype=float)

    d = s_mat.shape[0]
    stable_cov = (s_mat + s_mat.T) / 2.0

    def _solve(cov_input: np.ndarray, lam: float, max_iter: int) -> tuple[np.ndarray, np.ndarray, bool]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            cov_est_, prec_est_ = graphical_lasso(cov_input, alpha=float(lam), max_iter=max_iter, tol=1e-4)
        has_conv_warn = any(isinstance(w.message, ConvergenceWarning) for w in caught)
        return cov_est_, prec_est_, has_conv_warn

    for i, lam in enumerate(lambda_path):
        jitter = 1e-4 * np.eye(d)
        try:
            cov_est, prec_est, has_warn = _solve(stable_cov, float(lam), 500)
            if has_warn:
                cov_est, prec_est, _ = _solve(stable_cov + jitter, float(lam), 1200)
        except Exception:
            cov_est, prec_est, _ = _solve(stable_cov + jitter, float(lam), 1200)

        prec_est = (prec_est + prec_est.T) / 2.0
        cov_est = (cov_est + cov_est.T) / 2.0

        adj = np.abs(prec_est) > 1e-8
        np.fill_diagonal(adj, False)

        path.append(sparse.csc_matrix(adj.astype(float)))
        icov.append(prec_est)
        cov_list.append(cov_est)
        df[i] = np.count_nonzero(np.triu(adj, 1)) * 2.0

        sign, logdet = np.linalg.slogdet(prec_est)
        if sign <= 0:
            loglik[i] = -np.inf
        else:
            loglik[i] = float(logdet - np.trace(stable_cov @ prec_est))

    return path, icov, (cov_list if cov_output else None), df, loglik


def _build_screen_idx(corr: np.ndarray, scr_num: int) -> np.ndarray:
    d = corr.shape[0]
    if scr_num <= 0 or scr_num >= d:
        raise PyHugeError("`scr_num` must satisfy 1 <= scr_num < d.")
    order = np.argsort(-np.abs(corr), axis=0)
    return np.asarray(order[1 : scr_num + 1, :], dtype=np.int32)


def _run_mb(
    x_data: np.ndarray,
    corr: np.ndarray,
    lambda_path: np.ndarray,
    sym: str,
    scr: bool,
    scr_num: Optional[int],
) -> tuple[list[sparse.csc_matrix], np.ndarray]:
    if _CPP is not None:
        try:
            if scr:
                if scr_num is None:
                    raise PyHugeError("`scr=True` requires `scr_num` in native MB C++ core.")
                idx_mat = _build_screen_idx(corr, scr_num)
                out = _CPP.spmb_scr(np.asarray(corr, dtype=float), np.asarray(lambda_path, dtype=float), idx_mat)
            else:
                out = _CPP.spmb_graph(np.asarray(corr, dtype=float), np.asarray(lambda_path, dtype=float))

            beta = np.asarray(out["beta"], dtype=float)
            df = np.asarray(out["df"], dtype=float)
            nlam = beta.shape[0]

            path: list[sparse.csc_matrix] = []
            for li in range(nlam):
                directed = np.abs(beta[li]) > 0
                np.fill_diagonal(directed, False)
                adj = _symmetrize(directed, sym)
                path.append(sparse.csc_matrix(adj.astype(float)))
            return path, df
        except Exception:
            pass

    _require_sklearn("mb")
    from sklearn.linear_model import lasso_path

    x_std = _standardize(x_data)
    _, d = x_std.shape
    nlam = lambda_path.size
    directed = np.zeros((nlam, d, d), dtype=bool)
    df = np.zeros((d, nlam), dtype=float)

    for j in range(d):
        mask = np.ones(d, dtype=bool)
        mask[j] = False
        X = x_std[:, mask]
        y = x_std[:, j]

        alphas_out, coefs, _ = lasso_path(X, y, alphas=lambda_path, max_iter=4000)
        alphas_out = np.asarray(alphas_out, dtype=float)
        coefs = np.asarray(coefs, dtype=float)

        if coefs.ndim == 1:
            coefs = coefs[:, None]

        if coefs.shape[1] != nlam or not np.allclose(alphas_out, lambda_path, rtol=1e-6, atol=1e-8):
            idx = [int(np.argmin(np.abs(alphas_out - a))) for a in lambda_path]
            coefs = coefs[:, idx]

        for li in range(nlam):
            nz = np.abs(coefs[:, li]) > 1e-8
            directed[li, j, mask] = nz
            df[j, li] = float(np.count_nonzero(nz))

    path: list[sparse.csc_matrix] = []
    for li in range(nlam):
        adj = _symmetrize(directed[li], sym)
        path.append(sparse.csc_matrix(adj.astype(float)))
    return path, df


def _run_tiger(
    x_data: np.ndarray,
    lambda_path: np.ndarray,
    sym: str,
) -> tuple[list[sparse.csc_matrix], np.ndarray, Optional[list[np.ndarray]]]:
    x_std = _standardize(x_data)
    if _CPP is not None:
        try:
            out = _CPP.spmb_graphsqrt(np.asarray(x_std, dtype=float), np.asarray(lambda_path, dtype=float))
            beta = np.asarray(out["beta"], dtype=float)
            df = np.asarray(out["df"], dtype=float)
            icov_cube = np.asarray(out["icov"], dtype=float)

            path: list[sparse.csc_matrix] = []
            for li in range(beta.shape[0]):
                directed = np.abs(beta[li]) > 0
                np.fill_diagonal(directed, False)
                adj = _symmetrize(directed, sym)
                path.append(sparse.csc_matrix(adj.astype(float)))
            icov = [np.asarray(icov_cube[i], dtype=float) for i in range(icov_cube.shape[0])]
            return path, df, icov
        except Exception:
            pass

    _require_sklearn("tiger")
    from sklearn.linear_model import lasso_path

    _, d = x_std.shape
    nlam = lambda_path.size
    directed = np.zeros((nlam, d, d), dtype=bool)
    df = np.zeros((d, nlam), dtype=float)

    for j in range(d):
        mask = np.ones(d, dtype=bool)
        mask[j] = False
        X = x_std[:, mask]
        y = x_std[:, j]

        alphas_out, coefs, _ = lasso_path(X, y, alphas=lambda_path, max_iter=4000)
        alphas_out = np.asarray(alphas_out, dtype=float)
        coefs = np.asarray(coefs, dtype=float)

        if coefs.ndim == 1:
            coefs = coefs[:, None]

        if coefs.shape[1] != nlam or not np.allclose(alphas_out, lambda_path, rtol=1e-6, atol=1e-8):
            idx = [int(np.argmin(np.abs(alphas_out - a))) for a in lambda_path]
            coefs = coefs[:, idx]

        for li in range(nlam):
            nz = np.abs(coefs[:, li]) > 1e-8
            directed[li, j, mask] = nz
            df[j, li] = float(np.count_nonzero(nz))

    path: list[sparse.csc_matrix] = []
    for li in range(nlam):
        adj = _symmetrize(directed[li], sym)
        path.append(sparse.csc_matrix(adj.astype(float)))

    cov_mat = np.asarray(np.cov(x_data, rowvar=False), dtype=float)
    return path, df, _precision_path_from_cov(cov_mat, lambda_path)


def _precision_path_from_cov(cov_mat: np.ndarray, lambda_path: np.ndarray) -> list[np.ndarray]:
    d = cov_mat.shape[0]
    eye = np.eye(d, dtype=float)
    stable_cov = (cov_mat + cov_mat.T) / 2.0

    out: list[np.ndarray] = []
    for lam in lambda_path:
        reg = stable_cov + float(lam) * eye
        reg = (reg + reg.T) / 2.0
        try:
            prec = np.linalg.inv(reg)
        except np.linalg.LinAlgError:
            prec = np.linalg.pinv(reg, rcond=1e-7)
        out.append((prec + prec.T) / 2.0)
    return out


def _edge_count(path: Sequence[sparse.csc_matrix]) -> np.ndarray:
    return np.asarray([np.count_nonzero(np.triu(p.toarray() != 0, 1)) for p in path], dtype=float)


def _selected_index_ric(sparsity: np.ndarray) -> int:
    if sparsity.size <= 2:
        return int(sparsity.size - 1)
    curv = np.zeros_like(sparsity)
    curv[1:-1] = np.abs(np.diff(sparsity, n=2))
    return int(np.argmax(curv))


def _ebic_from_fit(est: HugeResult, ebic_gamma: float) -> np.ndarray:
    if est.method != "glasso":
        raise PyHugeError("`criterion='ebic'` requires a glasso fit.")

    d = est.path[0].shape[0]
    n = est.data.shape[0] if not est.cov_input else d
    edge_k = _edge_count(est.path)

    loglik = est.loglik
    if loglik is None:
        if est.icov is None:
            raise PyHugeError("`criterion='ebic'` requires glasso fit with log-likelihood information.")
        cov_mat = np.asarray(np.cov(est.data, rowvar=False), dtype=float) if not est.cov_input else est.data
        vals = np.zeros(len(est.icov), dtype=float)
        for i, prec in enumerate(est.icov):
            sign, logdet = np.linalg.slogdet(np.asarray(prec, dtype=float))
            vals[i] = -np.inf if sign <= 0 else float(logdet - np.trace(cov_mat @ prec))
        loglik = vals

    l = np.asarray(loglik, dtype=float) * (n / 2.0)
    return -2.0 * l + edge_k * np.log(max(n, 2)) + 4.0 * ebic_gamma * edge_k * np.log(max(d, 2))


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
    backend: str = "native",
) -> HugeResult:
    """Native graph path estimation.

    ``backend`` is kept for explicitness and future extension; only ``native``
    is supported in 0.3.x.
    """

    del verbose

    _ensure_backend_native(backend)

    if method not in _ALLOWED_METHODS:
        raise PyHugeError(f"`method` must be one of {sorted(_ALLOWED_METHODS)}.")
    if sym not in _ALLOWED_SYM:
        raise PyHugeError(f"`sym` must be one of {sorted(_ALLOWED_SYM)}.")

    x = _ensure_2d_array("x", x, finite=True)

    if nlambda is not None:
        nlambda = _ensure_positive_int("nlambda", nlambda)
    if lambda_min_ratio is not None:
        lambda_min_ratio = _ensure_ratio("lambda_min_ratio", lambda_min_ratio)
    if lambda_ is not None:
        lambda_ = _ensure_ct_lambda_sequence(lambda_) if method == "ct" else _ensure_lambda_sequence(lambda_)

    if method in {"ct", "tiger"} and scr is not None:
        raise PyHugeError("`scr` is only applicable for method `mb` and `glasso`.")
    if method != "mb" and scr_num is not None:
        raise PyHugeError("`scr_num` is only applicable for method `mb`.")

    if method == "mb":
        scr = False if scr is None else bool(scr)
        if scr_num is not None:
            scr_num = _ensure_positive_int("scr_num", scr_num)
            if not scr:
                raise PyHugeError("`scr_num` requires `scr=True`.")
    elif method == "glasso":
        scr = False if scr is None else bool(scr)
    else:
        scr = None

    if cov_output and method != "glasso":
        raise PyHugeError("`cov_output=True` is only valid for method `glasso`.")
    if method in {"ct", "glasso"} and sym != "or":
        raise PyHugeError("`sym` is only applicable to method `mb` and `tiger`.")

    cov_input = _is_covariance_input(x)
    if cov_input and method in {"mb", "tiger"}:
        raise PyHugeError(f"`method={method}` requires raw data matrix (n x d), not covariance matrix.")

    if cov_input:
        cov_mat = (x + x.T) / 2.0
        corr = _cov_to_corr(cov_mat)
    else:
        x_std = _standardize(x)
        cov_mat = np.asarray(np.cov(x, rowvar=False), dtype=float)
        corr = np.asarray(np.corrcoef(x_std, rowvar=False), dtype=float)

    if method == "mb" and bool(scr) and scr_num is None:
        n, d = x.shape
        if n < d:
            scr_num = n - 1
        else:
            # Match huge: without explicit scr_num in n>=d, lossy screening is skipped.
            scr = False
    if method == "mb" and bool(scr) and scr_num is not None:
        d = corr.shape[0]
        scr_num = min(int(scr_num), d - 1)
        if scr_num <= 0:
            scr = False

    if method == "ct":
        if lambda_ is None:
            nlam = _default_nlambda("ct") if nlambda is None else _ensure_positive_int("nlambda", nlambda)
            ratio = _default_lambda_min_ratio("ct") if lambda_min_ratio is None else float(lambda_min_ratio)
            lambda_path, path, sparsity = _run_ct_default_rank(corr, nlam, ratio)
        else:
            lambda_path = _ensure_ct_lambda_sequence(lambda_)
            path = _run_ct(corr, lambda_path)
            sparsity = _ct_path_sparsity(path)
        return HugeResult(
            method=method,
            lambda_path=lambda_path,
            sparsity=sparsity,
            path=path,
            cov_input=cov_input,
            data=np.asarray(x, dtype=float),
            sym="or",
            raw={"backend": "native"},
        )

    s_glasso = cov_mat if cov_input else corr
    base = corr if method in {"mb", "tiger"} else s_glasso
    lambda_path = _build_lambda_path(
        base_matrix=base,
        method=method,
        lambda_=lambda_,
        nlambda=nlambda,
        lambda_min_ratio=lambda_min_ratio,
    )

    if method == "glasso":
        path, icov, cov_list, df, loglik = _run_glasso(
            s_glasso,
            lambda_path,
            scr=bool(scr),
            cov_output=cov_output,
        )
        return HugeResult(
            method=method,
            lambda_path=lambda_path,
            sparsity=_path_sparsity(path),
            path=path,
            cov_input=cov_input,
            data=np.asarray(x, dtype=float),
            sym="or",
            df=df,
            loglik=loglik,
            icov=icov,
            cov=cov_list,
            raw={"backend": "native", "scr": bool(scr), "cov_output": bool(cov_output)},
        )

    if method == "mb":
        path, df = _run_mb(
            x_data=np.asarray(x, dtype=float),
            corr=np.asarray(corr, dtype=float),
            lambda_path=lambda_path,
            sym=sym,
            scr=bool(scr),
            scr_num=scr_num,
        )
        icov: Optional[list[np.ndarray]] = None
        raw = {"backend": "native", "scr": bool(scr), "scr_num": scr_num}
    else:
        path, df, icov = _run_tiger(np.asarray(x, dtype=float), lambda_path, sym=sym)
        raw = {"backend": "native"}

    return HugeResult(
        method=method,
        lambda_path=lambda_path,
        sparsity=_path_sparsity(path),
        path=path,
        cov_input=False,
        data=np.asarray(x, dtype=float),
        sym=sym,
        df=df,
        icov=icov,
        raw=raw,
    )


def huge_mb(
    x: np.ndarray,
    lambda_: Optional[Sequence[float]] = None,
    nlambda: Optional[int] = None,
    lambda_min_ratio: Optional[float] = None,
    scr: Optional[bool] = None,
    scr_num: Optional[int] = None,
    sym: str = "or",
    verbose: bool = True,
    backend: str = "native",
) -> HugeResult:
    """Convenience wrapper for ``huge(..., method='mb')``."""

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
        backend=backend,
    )


def huge_glasso(
    x: np.ndarray,
    lambda_: Optional[Sequence[float]] = None,
    nlambda: Optional[int] = None,
    lambda_min_ratio: Optional[float] = None,
    scr: Optional[bool] = None,
    cov_output: bool = False,
    verbose: bool = True,
    backend: str = "native",
) -> HugeResult:
    """Convenience wrapper for ``huge(..., method='glasso')``."""

    return huge(
        x=x,
        lambda_=lambda_,
        nlambda=nlambda,
        lambda_min_ratio=lambda_min_ratio,
        method="glasso",
        scr=scr,
        cov_output=cov_output,
        sym="or",
        verbose=verbose,
        backend=backend,
    )


def huge_ct(
    x: np.ndarray,
    lambda_: Optional[Sequence[float]] = None,
    nlambda: Optional[int] = None,
    lambda_min_ratio: Optional[float] = None,
    verbose: bool = True,
    backend: str = "native",
) -> HugeResult:
    """Convenience wrapper for ``huge(..., method='ct')``."""

    return huge(
        x=x,
        lambda_=lambda_,
        nlambda=nlambda,
        lambda_min_ratio=lambda_min_ratio,
        method="ct",
        sym="or",
        verbose=verbose,
        backend=backend,
    )


def huge_tiger(
    x: np.ndarray,
    lambda_: Optional[Sequence[float]] = None,
    nlambda: Optional[int] = None,
    lambda_min_ratio: Optional[float] = None,
    sym: str = "or",
    verbose: bool = True,
    backend: str = "native",
) -> HugeResult:
    """Convenience wrapper for ``huge(..., method='tiger')``."""

    return huge(
        x=x,
        lambda_=lambda_,
        nlambda=nlambda,
        lambda_min_ratio=lambda_min_ratio,
        method="tiger",
        sym=sym,
        verbose=verbose,
        backend=backend,
    )


def huge_select(
    est: HugeResult,
    criterion: Optional[str] = None,
    ebic_gamma: float = 0.5,
    stars_thresh: float = 0.1,
    stars_subsample_ratio: Optional[float] = None,
    rep_num: int = 20,
    verbose: bool = True,
    backend: str = "native",
) -> HugeSelectResult:
    """Native model selection for ``HugeResult``."""

    del verbose
    _ensure_backend_native(backend)

    if not isinstance(est, HugeResult):
        raise PyHugeError("`est` must be HugeResult in native backend.")
    if est.cov_input:
        raise PyHugeError("Model selection is not available when using covariance matrix as input.")

    crit = "ric" if criterion is None else str(criterion)
    if crit not in _ALLOWED_CRITERIA:
        raise PyHugeError(f"`criterion` must be one of {sorted(_ALLOWED_CRITERIA)}.")

    if not np.isfinite(float(ebic_gamma)):
        raise PyHugeError("`ebic_gamma` must be finite.")
    stars_thresh = _ensure_ratio("stars_thresh", stars_thresh)
    rep_num = _ensure_positive_int("rep_num", rep_num)
    if stars_subsample_ratio is not None:
        stars_subsample_ratio = _ensure_ratio("stars_subsample_ratio", stars_subsample_ratio)

    nlam = est.lambda_path.size
    if nlam == 0 or len(est.path) == 0:
        raise PyHugeError("`est` has an empty path.")
    if len(est.path) != nlam:
        raise PyHugeError("`est.path` length must match `est.lambda_path` length.")

    scr_meta = False
    scr_num_meta: Optional[int] = None
    if isinstance(est.raw, dict):
        scr_meta = bool(est.raw.get("scr", False))
        if est.raw.get("scr_num") is not None:
            try:
                scr_num_meta = int(est.raw.get("scr_num"))
            except Exception:
                scr_num_meta = None

    opt_idx = 0
    variability: Optional[np.ndarray] = None
    ebic_score: Optional[np.ndarray] = None

    if crit == "ric":
        x = _ensure_2d_array("est.data", est.data, finite=True)
        n = x.shape[0]
        if _CPP is not None:
            if n > rep_num:
                rng = np.random.default_rng(0)
                r = np.asarray(rng.choice(n, size=rep_num, replace=False), dtype=np.int32)
            else:
                r = np.arange(n, dtype=np.int32)

            opt_lambda = float(_CPP.ric(np.asarray(x, dtype=float), r)) / float(max(n, 1))
            nearest_idx = int(np.argmin(np.abs(est.lambda_path - opt_lambda)))
            refit_fit = huge(
                x=x,
                lambda_=[opt_lambda],
                method=est.method,
                scr=scr_meta if est.method in {"mb", "glasso"} else None,
                scr_num=scr_num_meta if est.method == "mb" else None,
                cov_output=(est.method == "glasso" and est.cov is not None),
                sym=est.sym,
                verbose=False,
                backend="native",
            )

            out = HugeSelectResult(
                criterion=crit,
                opt_lambda=float(opt_lambda),
                opt_sparsity=float(refit_fit.sparsity[0]),
                refit=refit_fit.path[0],
                opt_index=int(nearest_idx + 1),
                variability=None,
                ebic_score=None,
                raw={"backend": "native", "criterion": crit},
            )
            if refit_fit.icov is not None:
                out.opt_icov = np.asarray(refit_fit.icov[0], dtype=float)
            if refit_fit.cov is not None:
                out.opt_cov = np.asarray(refit_fit.cov[0], dtype=float)
            return out

        # Fallback when C++ core is unavailable.
        opt_idx = _selected_index_ric(est.sparsity)

    elif crit == "ebic":
        ebic_score = _ebic_from_fit(est, float(ebic_gamma))
        opt_idx = int(np.argmin(ebic_score))

    else:  # stars
        x = _ensure_2d_array("est.data", est.data, finite=True)
        n = x.shape[0]
        ratio = stars_subsample_ratio
        if ratio is None:
            ratio = 0.8 if n <= 144 else min(0.99, 10.0 * math.sqrt(n) / n)
        m = max(2, int(n * ratio))

        rng = np.random.default_rng(0)
        d = x.shape[1]
        freq = np.zeros((nlam, d, d), dtype=float)

        for _ in range(rep_num):
            idx = rng.choice(n, size=m, replace=False)
            subfit = huge(
                x[idx],
                lambda_=est.lambda_path,
                method=est.method,
                sym=est.sym,
                scr=scr_meta if est.method in {"mb", "glasso"} else None,
                scr_num=scr_num_meta if est.method == "mb" else None,
                backend="native",
            )
            for li, p in enumerate(subfit.path):
                freq[li] += (p.toarray() != 0).astype(float)

        freq /= float(rep_num)
        variability = np.zeros(nlam, dtype=float)
        for li in range(nlam):
            p_mat = 0.5 * (freq[li] + freq[li].T)
            np.fill_diagonal(p_mat, 0.0)
            variability[li] = float(4.0 * np.sum(p_mat * (1.0 - p_mat)) / (d * (d - 1)))

        stars_cross = np.where(variability >= stars_thresh)[0]
        if stars_cross.size == 0:
            opt_idx = int(nlam - 1)
        else:
            opt_idx = max(int(stars_cross[0]) - 1, 0)

    refit = est.path[opt_idx]
    out = HugeSelectResult(
        criterion=crit,
        opt_lambda=float(est.lambda_path[opt_idx]),
        opt_sparsity=float(est.sparsity[opt_idx]),
        refit=refit,
        opt_index=int(opt_idx + 1),
        variability=variability,
        ebic_score=ebic_score,
        raw={"backend": "native", "criterion": crit},
    )

    if est.icov is not None:
        out.opt_icov = np.asarray(est.icov[opt_idx], dtype=float)
    if est.cov is not None:
        out.opt_cov = np.asarray(est.cov[opt_idx], dtype=float)
    return out


def huge_npn(
    x: np.ndarray,
    npn_func: str = "shrinkage",
    verbose: bool = True,
) -> np.ndarray:
    """Native nonparanormal transformation."""

    del verbose

    if npn_func not in _ALLOWED_NPN_FUNCS:
        raise PyHugeError(f"`npn_func` must be one of {sorted(_ALLOWED_NPN_FUNCS)}.")

    x = _ensure_2d_array("x", x, finite=True)
    n, d = x.shape

    if npn_func == "skeptic":
        rho, _ = stats.spearmanr(x, axis=0)
        rho = np.asarray(rho, dtype=float)
        if rho.ndim == 0:
            rho = np.eye(d, dtype=float)
        if rho.shape[0] != d:
            rho = rho[:d, :d]
        out = 2.0 * np.sin((np.pi / 6.0) * rho)
        np.fill_diagonal(out, 1.0)
        return out

    z = np.zeros_like(x, dtype=float)
    trunc = 1.0 / (4.0 * (n ** 0.25) * math.sqrt(np.pi * np.log(max(n, 2))))

    for j in range(d):
        col = x[:, j]
        rank = stats.rankdata(col, method="average")
        u = rank / (n + 1.0)
        if npn_func == "shrinkage":
            eps = 1.0 / (2.0 * n)
            u = np.clip(u, eps, 1.0 - eps)
        else:
            u = np.clip(u, trunc, 1.0 - trunc)
        z[:, j] = stats.norm.ppf(u)

    return z


def _group_partitions(d: int, g: int) -> list[np.ndarray]:
    idx = np.arange(d)
    return [arr for arr in np.array_split(idx, g) if arr.size > 0]


def _theta_random(d: int, prob: float, rng: np.random.Generator) -> np.ndarray:
    a = np.zeros((d, d), dtype=float)
    tri = np.triu(rng.random((d, d)) < prob, 1)
    a[tri] = 1.0
    a = a + a.T
    return a


def _theta_hub(d: int, g: int) -> np.ndarray:
    a = np.zeros((d, d), dtype=float)
    for grp in _group_partitions(d, g):
        if grp.size <= 1:
            continue
        c = int(grp[0])
        for j in grp[1:]:
            a[c, j] = 1.0
            a[j, c] = 1.0
    return a


def _theta_cluster(d: int, g: int, prob: float, rng: np.random.Generator) -> np.ndarray:
    a = np.zeros((d, d), dtype=float)
    for grp in _group_partitions(d, g):
        m = grp.size
        if m <= 1:
            continue
        mask = np.triu(rng.random((m, m)) < prob, 1)
        for i in range(m):
            for j in range(i + 1, m):
                if mask[i, j]:
                    u = int(grp[i])
                    v = int(grp[j])
                    a[u, v] = 1.0
                    a[v, u] = 1.0
    return a


def _theta_band(d: int, g: int) -> np.ndarray:
    a = np.zeros((d, d), dtype=float)
    for i in range(d):
        for j in range(max(0, i - g), min(d, i + g + 1)):
            if i != j:
                a[i, j] = 1.0
    a = np.maximum(a, a.T)
    np.fill_diagonal(a, 0.0)
    return a


def _theta_scale_free(d: int, rng: np.random.Generator) -> np.ndarray:
    a = np.zeros((d, d), dtype=float)
    if d <= 1:
        return a
    a[0, 1] = 1.0
    a[1, 0] = 1.0
    degrees = np.sum(a, axis=0)
    for new_node in range(2, d):
        total_deg = float(np.sum(degrees[:new_node]))
        if total_deg <= 0:
            target = int(rng.integers(0, new_node))
        else:
            prob = degrees[:new_node] / total_deg
            target = int(rng.choice(np.arange(new_node), p=prob))
        a[new_node, target] = 1.0
        a[target, new_node] = 1.0
        degrees = np.sum(a, axis=0)
    return a


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
    random_state: Optional[int] = None,
) -> HugeGeneratorResult:
    """Native data generator."""

    del vis, verbose

    n = _ensure_positive_int("n", n)
    d = _ensure_positive_int("d", d)
    if graph not in _ALLOWED_GRAPH_TYPES:
        raise PyHugeError(f"`graph` must be one of {sorted(_ALLOWED_GRAPH_TYPES)}.")

    rng = np.random.default_rng(random_state)
    v_val = 0.3 if v is None else float(v)
    u_val = 0.1 if u is None else float(u)

    if v_val <= 0 or u_val <= 0:
        raise PyHugeError("`v` and `u` must be positive.")

    if graph in {"hub", "cluster"}:
        g_val = 2 if d < 40 else max(2, d // 20)
        if g is not None:
            g_val = _ensure_positive_int("g", g)
    elif graph == "band":
        g_val = 1 if g is None else _ensure_positive_int("g", g)
    else:
        g_val = 1

    if graph == "random":
        p_val = (3.0 / d) if prob is None else float(prob)
    elif graph == "cluster":
        p_val = (6.0 * g_val / d) if (d / max(g_val, 1)) <= 30 else 0.3
        if prob is not None:
            p_val = float(prob)
    else:
        p_val = 0.0

    if p_val < 0.0 or p_val > 1.0:
        raise PyHugeError("`prob` must satisfy 0 <= prob <= 1.")

    if graph == "random":
        theta = _theta_random(d, p_val, rng)
    elif graph == "hub":
        theta = _theta_hub(d, g_val)
    elif graph == "cluster":
        theta = _theta_cluster(d, g_val, p_val, rng)
    elif graph == "band":
        theta = _theta_band(d, g_val)
    else:
        if _CPP is not None:
            try:
                seed = None if random_state is None else int(random_state)
                theta = np.asarray(_CPP.sfgen(2, int(d), seed), dtype=float)
            except Exception:
                theta = _theta_scale_free(d, rng)
        else:
            theta = _theta_scale_free(d, rng)

    base = v_val * theta
    min_eig = float(np.min(np.linalg.eigvalsh(base)))
    shift = abs(min_eig) + 0.1 + u_val
    omega = base + np.eye(d) * shift
    sigma = np.linalg.inv(omega)
    sigma = (sigma + sigma.T) / 2.0

    data = rng.multivariate_normal(mean=np.zeros(d), cov=sigma, size=n)
    sigmahat = np.asarray(np.cov(data, rowvar=False), dtype=float)

    return HugeGeneratorResult(
        data=np.asarray(data, dtype=float),
        sigma=np.asarray(sigma, dtype=float),
        omega=np.asarray(omega, dtype=float),
        sigmahat=sigmahat,
        theta=sparse.csc_matrix(theta),
        sparsity=_adj_sparsity(theta != 0),
        graph_type=graph,
        raw={"backend": "native"},
    )


def huge_roc(
    path: Sequence[np.ndarray | sparse.spmatrix],
    theta: np.ndarray | sparse.spmatrix,
    verbose: bool = True,
    plot: bool = False,
) -> HugeRocResult:
    """Native ROC metrics for graph path."""

    del verbose

    if len(path) == 0:
        raise PyHugeError("`path` must contain at least one adjacency matrix.")

    theta_dense = _to_dense_matrix(theta, "theta")
    if theta_dense.shape[0] != theta_dense.shape[1]:
        raise PyHugeError("`theta` must be square.")

    d = theta_dense.shape[0]
    truth = np.asarray(theta_dense != 0, dtype=bool)
    np.fill_diagonal(truth, False)
    truth_u = np.triu(truth, 1)

    total_pos = max(int(np.count_nonzero(truth_u)), 1)
    total_pairs = d * (d - 1) // 2
    total_neg = max(total_pairs - total_pos, 1)

    tp = np.zeros(len(path), dtype=float)
    fp = np.zeros(len(path), dtype=float)
    f1 = np.zeros(len(path), dtype=float)

    for i, p in enumerate(path):
        pred = _to_dense_matrix(p, f"path[{i + 1}]")
        if pred.shape != (d, d):
            raise PyHugeError(f"`path[{i + 1}]` must have shape ({d}, {d}).")

        pred_u = np.triu(pred != 0, 1)
        tp_count = int(np.count_nonzero(pred_u & truth_u))
        fp_count = int(np.count_nonzero(pred_u & (~truth_u)))
        pred_count = int(np.count_nonzero(pred_u))

        tp[i] = tp_count / total_pos
        fp[i] = fp_count / total_neg

        precision = tp_count / max(pred_count, 1)
        recall = tp[i]
        denom = precision + recall
        f1[i] = 0.0 if denom <= 0 else (2.0 * precision * recall / denom)

    order = np.argsort(fp)
    auc = float(np.trapezoid(tp[order], fp[order]))
    out = HugeRocResult(f1=f1, tp=tp, fp=fp, auc=auc, raw={"backend": "native"})

    if plot:
        huge_plot_roc(out)

    return out


def huge_inference(
    data: np.ndarray,
    t: np.ndarray,
    adj: np.ndarray | sparse.spmatrix,
    alpha: float = 0.05,
    type_: str = "Gaussian",
    method: str = "score",
) -> HugeInferenceResult:
    """Native edge-wise inference via partial-correlation z test approximation."""

    if type_ not in _ALLOWED_INFERENCE_TYPES:
        raise PyHugeError(f"`type_` must be one of {sorted(_ALLOWED_INFERENCE_TYPES)}.")
    if method not in _ALLOWED_INFERENCE_METHODS:
        raise PyHugeError(f"`method` must be one of {sorted(_ALLOWED_INFERENCE_METHODS)}.")
    alpha = _ensure_ratio("alpha", alpha)

    x = _ensure_2d_array("data", data, finite=True)
    n, d = x.shape

    t_mat = _to_dense_matrix(t, "t")
    if t_mat.shape != (d, d):
        raise PyHugeError(f"`t` must have shape ({d}, {d}).")

    adj_mat = _to_dense_matrix(adj, "adj")
    if adj_mat.shape != (d, d):
        raise PyHugeError(f"`adj` must have shape ({d}, {d}).")

    if type_ == "Nonparanormal":
        x = huge_npn(x, npn_func="shrinkage")

    diag = np.clip(np.diag(t_mat), 1e-12, None)
    denom = np.sqrt(np.outer(diag, diag))
    rho = -t_mat / denom
    rho = np.clip(rho, -0.999999, 0.999999)

    z = 0.5 * np.log((1.0 + rho) / (1.0 - rho)) * np.sqrt(max(n - 3, 1))
    p = 2.0 * (1.0 - stats.norm.cdf(np.abs(z)))
    np.fill_diagonal(p, 0.0)

    offdiag = ~np.eye(d, dtype=bool)
    null_mask = (adj_mat == 0) & offdiag
    error = float(np.mean(p[null_mask] <= alpha)) if np.any(null_mask) else 0.0

    return HugeInferenceResult(data=np.asarray(x, dtype=float), p=np.asarray(p, dtype=float), error=error)


def huge_stockdata() -> HugeStockDataResult:
    """Load packaged stock dataset (converted from R ``stockdata``)."""

    try:
        stock_path = resources.files("pyhuge").joinpath("data/stockdata.npz")
        with resources.as_file(stock_path) as resolved:
            payload = np.load(resolved, allow_pickle=True)
            data = np.asarray(payload["data"], dtype=float)
            info = np.asarray(payload["info"])
    except FileNotFoundError as exc:
        raise PyHugeError("Built-in stock dataset is missing from the package installation.") from exc
    except Exception as exc:
        raise PyHugeError("Failed to load built-in stock dataset.") from exc

    return HugeStockDataResult(data=data, info=info, raw={"source": str(stock_path)})


def huge_summary(fit: HugeResult) -> HugeSummary:
    """Return a concise summary of ``HugeResult``."""

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
    """Return a concise summary of ``HugeSelectResult``."""

    return HugeSelectSummary(
        criterion=sel.criterion,
        opt_lambda=float(sel.opt_lambda),
        opt_sparsity=float(sel.opt_sparsity),
        refit_n_features=int(sel.refit.shape[1]),
        has_opt_icov=sel.opt_icov is not None,
        has_opt_cov=sel.opt_cov is not None,
    )


def _mpl_pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise PyHugeError("matplotlib is required for plotting. Install with `pip install matplotlib`.") from exc
    return plt


def _networkx_pkg() -> Any:
    try:
        import networkx as nx
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise PyHugeError("networkx is required for network plotting. Install with `pip install networkx`.") from exc
    return nx


def huge_plot_sparsity(fit: HugeResult, ax: Optional[Any] = None, show_points: bool = True) -> Any:
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
    ax.plot(roc.fp[order], roc.tp[order], "-o", ms=3, lw=1.6)
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

    if index < 0:
        index = len(fit.path) + index
    if index < 0 or index >= len(fit.path):
        raise PyHugeError(f"`index` out of range: got {index}, path length={len(fit.path)}")

    plt = _mpl_pyplot()
    if ax is None:
        _, ax = plt.subplots(1, 1)

    mat = fit.path[index].toarray()
    ax.imshow(mat, cmap="Greys", interpolation="nearest")
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
    """Plot one estimated graph as a node-edge network."""

    if len(fit.path) == 0:
        raise PyHugeError("`fit.path` is empty.")
    if min_abs_weight < 0:
        raise PyHugeError("`min_abs_weight` must be non-negative.")

    if index < 0:
        index = len(fit.path) + index
    if index < 0 or index >= len(fit.path):
        raise PyHugeError(f"`index` out of range: got {index}, path length={len(fit.path)}")

    dense = fit.path[index].toarray().astype(float)
    dense = 0.5 * (dense + dense.T)
    if min_abs_weight > 0:
        dense[np.abs(dense) < min_abs_weight] = 0.0

    nx = _networkx_pkg()
    plt = _mpl_pyplot()
    if ax is None:
        _, ax = plt.subplots(1, 1)

    g = nx.from_numpy_array(dense)
    layout_map = {
        "spring": nx.spring_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
        "circular": nx.circular_layout,
        "spectral": nx.spectral_layout,
        "shell": nx.shell_layout,
    }
    if layout not in layout_map:
        raise PyHugeError(f"`layout` must be one of {sorted(layout_map)}.")

    pos = layout_map[layout](g)
    nx.draw_networkx(
        g,
        pos=pos,
        ax=ax,
        with_labels=with_labels,
        node_size=float(node_size),
        node_color=node_color,
        edge_color=edge_color,
        width=1.2,
        alpha=1.0,
    )
    if index < fit.lambda_path.size:
        title = f"Network (idx={index}, lambda={fit.lambda_path[index]:.4g})"
    else:
        title = f"Network (idx={index})"
    ax.set_title(title)
    ax.set_axis_off()
    return ax


def huge_plot(
    g: np.ndarray | sparse.spmatrix,
    epsflag: bool = False,
    graph_name: str = "default",
    cur_num: int = 1,
    location: Optional[str] = None,
) -> Optional[str]:
    """R ``huge.plot``-style visualization in native Python."""

    adj = _to_dense_matrix(g, "g")
    if adj.shape[0] != adj.shape[1]:
        raise PyHugeError("`g` must be square.")

    cur_num = _ensure_positive_int("cur_num", cur_num)
    if not graph_name:
        raise PyHugeError("`graph_name` must be a non-empty string.")

    fit = HugeResult(
        method="plot",
        lambda_path=np.asarray([1.0]),
        sparsity=np.asarray([_adj_sparsity(adj != 0)]),
        path=[sparse.csc_matrix((adj != 0).astype(float))],
        cov_input=False,
        data=np.asarray(adj, dtype=float),
    )

    ax = huge_plot_network(fit, index=0)
    plt = _mpl_pyplot()

    if not epsflag:
        plt.close(ax.figure)
        return None

    if location is None:
        out_dir = Path.cwd()
    else:
        out_dir = Path(location)
        if not out_dir.is_dir():
            raise PyHugeError("`location` must be an existing directory.")

    out_path = out_dir / f"{graph_name}{int(cur_num)}.eps"
    ax.figure.savefig(out_path, format="eps", dpi=150, bbox_inches="tight")
    plt.close(ax.figure)
    return str(out_path)
