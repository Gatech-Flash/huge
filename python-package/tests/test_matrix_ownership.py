"""Independent precision/covariance owners at the C++/NumPy boundary."""

from concurrent.futures import ThreadPoolExecutor
import gc
import weakref

import numpy as np
import pytest

from pyhuge import core


native = pytest.importorskip("pyhuge._native_core")


def _covariance(d):
    return .45 ** np.abs(np.arange(d)[:, None] - np.arange(d)[None, :])


def _assert_independent_matrices(matrices, cube):
    assert isinstance(matrices, list)
    assert len(matrices) == cube.shape[0]
    assert len({id(matrix.base) for matrix in matrices}) == len(matrices)
    for matrix, reference in zip(matrices, cube):
        np.testing.assert_array_equal(matrix, reference)
        assert matrix.dtype == np.float64
        assert matrix.flags.f_contiguous
        assert matrix.flags.writeable
        assert not matrix.flags.owndata
        assert matrix.strides == (8, 8 * matrix.shape[0])
        assert type(matrix.base).__name__ == "PyCapsule"
    for i, first in enumerate(matrices):
        for second in matrices[i + 1:]:
            assert not np.shares_memory(first, second)


@pytest.mark.parametrize("d", [1, 5])
@pytest.mark.parametrize("screening", [False, True])
@pytest.mark.parametrize("cov_output", [False, True])
def test_glasso_owned_matrices_match_default_private_cube(d, screening, cov_output):
    covariance = _covariance(d)
    original = covariance.copy()
    lambdas = np.asarray([.6, .3, .12])
    legacy = native.hugeglasso(covariance, lambdas, screening, cov_output)
    owned = native.hugeglasso(
        covariance, lambdas, screening, cov_output, matrix_list=True
    )
    explicit_legacy = native.hugeglasso(
        covariance, lambdas, screening, cov_output, matrix_list=False
    )
    assert legacy.keys() == owned.keys() == explicit_legacy.keys()
    assert isinstance(legacy["icov"], np.ndarray)
    assert legacy["icov"].shape == (3, d, d)
    assert legacy["icov"].flags.c_contiguous
    _assert_independent_matrices(owned["icov"], legacy["icov"])
    for field in ("path", "loglik", "sparsity", "df", "hit_max_iter"):
        np.testing.assert_array_equal(owned[field], legacy[field])
        np.testing.assert_array_equal(explicit_legacy[field], legacy[field])
    np.testing.assert_array_equal(explicit_legacy["icov"], legacy["icov"])
    if cov_output:
        assert legacy["cov"].flags.c_contiguous
        _assert_independent_matrices(owned["cov"], legacy["cov"])
        assert all(not np.shares_memory(precision, covariance)
                   for precision in owned["icov"] for covariance in owned["cov"])
    else:
        assert owned["cov"] is legacy["cov"] is None
    np.testing.assert_array_equal(covariance, original)


@pytest.mark.parametrize("d", [1, 5])
@pytest.mark.parametrize("dense_output", [False, True])
def test_tiger_owned_matrices_preserve_both_support_contracts(d, dense_output):
    covariance = _covariance(d)
    lambdas = np.asarray([.6, .3, .12])
    legacy = native.spmb_graphsqrt(
        covariance, lambdas, covariance_input=True, dense_output=dense_output
    )
    owned = native.spmb_graphsqrt(
        covariance, lambdas, covariance_input=True,
        dense_output=dense_output, matrix_list=True,
    )
    assert owned.keys() == legacy.keys()
    assert isinstance(legacy["icov"], np.ndarray)
    assert legacy["icov"].flags.c_contiguous
    _assert_independent_matrices(owned["icov"], legacy["icov"])
    for field in legacy.keys() - {"icov"}:
        if legacy[field] is None:
            assert owned[field] is None
        else:
            np.testing.assert_array_equal(owned[field], legacy[field])


def test_tiger_truncated_path_moves_only_certified_matrices():
    # Collinear data reaches the same certified prefix in either output mode.
    x = np.arange(1., 25.)
    data = np.column_stack((x, x, -x))
    legacy = native.spmb_graphsqrt(data, None, 8, .01, False, False)
    owned = native.spmb_graphsqrt(data, None, 8, .01, False, False, True)
    assert legacy["path_truncated"]
    assert 0 < len(legacy["lambda"]) < 8
    assert owned["path_truncated"] == legacy["path_truncated"]
    assert len(owned["icov"]) == len(owned["lambda"])
    _assert_independent_matrices(owned["icov"], legacy["icov"])
    for field in ("lambda", "support_indptr", "support_indices", "df"):
        np.testing.assert_array_equal(owned[field], legacy[field])


@pytest.mark.parametrize("method", ["glasso", "tiger"])
@pytest.mark.parametrize("transpose", [False, True])
def test_retained_matrix_view_survives_result_deletion_and_native_reuse(method, transpose):
    covariance = _covariance(5)

    def fit():
        if method == "glasso":
            return native.hugeglasso(covariance, [.6, .3, .12], cov_output=True,
                                     matrix_list=True)
        return native.spmb_graphsqrt(covariance, [.6, .3, .12],
                                     covariance_input=True, matrix_list=True)

    result = fit()
    references = [weakref.ref(matrix) for matrix in result["icov"]]
    view = (result["icov"][1].T if transpose else result["icov"][1])[::-1, ::2]
    expected = view.copy()
    del result
    gc.collect()
    assert references[0]() is references[2]() is None
    assert references[1]() is not None
    # New fits would quickly reuse dangling native memory if ownership ended
    # when the result dictionary or C++ result went out of scope.
    for _ in range(3):
        current = fit()
        current["icov"][1][:] = -13
        del current
    np.testing.assert_array_equal(view, expected)
    view[0, 0] = 123.5
    row, column = (0, 4) if transpose else (4, 0)
    assert references[1]()[row, column] == 123.5
    del view
    gc.collect()
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("method", ["glasso", "tiger"])
def test_public_paths_keep_owned_arrays_and_allow_independent_mutation(method):
    covariance = _covariance(5)
    result = core.huge(covariance, lambda_=[.6, .3, .12], method=method,
                       input_type="covariance", cov_output=(method == "glasso"),
                       verbose=False)
    before = [matrix.copy() for matrix in result.icov]
    saved_path = [graph.copy() for graph in result.path]
    _assert_independent_matrices(result.icov, np.stack(before))
    result.icov[0][0, 1] = 17
    np.testing.assert_array_equal(result.icov[1], before[1])
    np.testing.assert_array_equal(result.icov[2], before[2])
    assert all((actual != expected).nnz == 0
               for actual, expected in zip(result.path, saved_path))
    if result.cov is not None:
        cov_before = [matrix.copy() for matrix in result.cov]
        _assert_independent_matrices(result.cov, np.stack(cov_before))
        result.cov[0][0, 1] = -11
        np.testing.assert_array_equal(result.cov[1], cov_before[1])
        assert result.icov[0][0, 1] == 17


def test_selected_glasso_matrix_does_not_keep_other_path_owners_alive():
    data = np.random.default_rng(51).normal(size=(60, 5))
    data[:, 1] += .6 * data[:, 0]
    result = core.huge(data, method="glasso", lambda_=[.6, .3, .12],
                       cov_output=True, verbose=False)
    selection = core.huge_select(result, criterion="ebic", verbose=False)
    chosen = selection.opt_index - 1
    references = [weakref.ref(matrix) for matrix in result.icov]
    cov_references = [weakref.ref(matrix) for matrix in result.cov]
    expected = result.icov[chosen].copy()
    expected_cov = result.cov[chosen].copy()
    assert selection.opt_icov is result.icov[chosen]
    assert selection.opt_cov is result.cov[chosen]
    assert type(selection.opt_icov.base).__name__ == "PyCapsule"
    assert type(selection.opt_cov.base).__name__ == "PyCapsule"
    del result
    gc.collect()
    assert [reference() is not None for reference in references] == [i == chosen for i in range(3)]
    assert [reference() is not None for reference in cov_references] == [i == chosen for i in range(3)]
    np.testing.assert_array_equal(selection.opt_icov, expected)
    np.testing.assert_array_equal(selection.opt_cov, expected_cov)
    del selection
    gc.collect()
    assert all(reference() is None for reference in references + cov_references)


def test_frontend_exception_releases_returned_native_matrix_owners(monkeypatch):
    original = native.hugeglasso
    references = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        references.extend(weakref.ref(matrix) for matrix in result["icov"])
        references.extend(weakref.ref(matrix) for matrix in result["cov"])
        return result

    def fail_after_native(*args):
        raise RuntimeError("post-native failure")

    monkeypatch.setattr(native, "hugeglasso", capture)
    monkeypatch.setattr(core, "_warn_if_not_converged", fail_after_native)
    with pytest.raises(RuntimeError, match="post-native failure"):
        core._run_glasso(_covariance(4), np.asarray([.6, .3]), False, True)
    gc.collect()
    assert len(references) == 4
    assert all(reference() is None for reference in references)


def test_concurrent_native_results_have_separate_owners():
    covariance = _covariance(4)

    def fit(_):
        result = native.hugeglasso(covariance, [.5, .2], matrix_list=True)
        return result["icov"][1].T[:, ::-1]

    with ThreadPoolExecutor(max_workers=3) as pool:
        matrices = list(pool.map(fit, range(3)))
    expected = matrices[1].copy()
    assert len({id(matrix.base.base) for matrix in matrices}) == 3
    assert all(type(matrix.base.base).__name__ == "PyCapsule" for matrix in matrices)
    gc.collect()
    for _ in range(3):
        fit(0)[:] = -3
    matrices[0][:] = 8
    for matrix in matrices[1:]:
        np.testing.assert_array_equal(matrix, expected)
