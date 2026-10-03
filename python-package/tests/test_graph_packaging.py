"""Graph packaging preserves public CSC paths and general sparse edge counts."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from pyhuge import core


LAMBDAS = np.asarray([.95, .45, .45])
CONFIGURATIONS = [
    pytest.param("mb", "or", id="mb-or"),
    pytest.param("mb", "and", id="mb-and"),
    pytest.param("tiger", "or", id="tiger-or"),
    pytest.param("tiger", "and", id="tiger-and"),
    pytest.param("glasso", "or", id="glasso"),
]


def _block_data(d):
    # Orthogonal, centered harmonics produce two independent blocks, each
    # with correlation .8. Lambda .95 removes both edges; .45 retains them.
    angles = 2 * np.pi * np.arange(64) / 64
    basis = np.column_stack([np.sin(k * angles) for k in range(1, 5)])
    data = np.column_stack((basis[:, 0], .8 * basis[:, 0] + .6 * basis[:, 1],
                            basis[:, 2], .8 * basis[:, 2] + .6 * basis[:, 3]))
    return data[:, :d]


def _assert_csc_equal(actual, expected):
    assert sparse.isspmatrix_csc(actual)
    assert actual.shape == expected.shape
    assert actual.dtype == np.float64
    assert actual.has_canonical_format
    assert actual.has_sorted_indices
    np.testing.assert_array_equal(actual.indptr, expected.indptr)
    np.testing.assert_array_equal(actual.indices, expected.indices)
    np.testing.assert_array_equal(actual.data, expected.data)


def _upper_density(graph):
    d = graph.shape[0]
    if d <= 1:
        return 0.0
    return np.count_nonzero(np.triu(graph.toarray(), k=1)) / (d * (d - 1) / 2)


@pytest.mark.parametrize("method,sym", CONFIGURATIONS)
@pytest.mark.parametrize("d", [1, 4])
def test_public_graph_paths_keep_binary_csc_buffers_and_density(method, sym, d):
    data = _block_data(d)
    original = data.copy()
    expected_dense = np.zeros((d, d))
    if d == 4:
        expected_dense[0, 1] = expected_dense[1, 0] = 1.0
        expected_dense[2, 3] = expected_dense[3, 2] = 1.0
    expected_path = [sparse.csc_matrix((d, d)),
                     sparse.csc_matrix(expected_dense),
                     sparse.csc_matrix(expected_dense)]

    def fit():
        return core.huge(data, method=method, sym=sym, lambda_=LAMBDAS,
                         input_type="data", verbose=False)

    # Repeated lambdas and repeated fits must preserve graph values and CSC
    # structure, including empty singleton paths and nonempty block paths.
    first, repeated = fit(), fit()
    for result in (first, repeated):
        np.testing.assert_array_equal(result.lambda_path, LAMBDAS)
        assert len(result.path) == 3
        for actual, expected in zip(result.path, expected_path):
            _assert_csc_equal(actual, expected)
            assert (actual != actual.T).nnz == 0
            assert np.all(actual.data == 1.0)
            assert not actual.diagonal().any()
        expected_density = np.asarray([_upper_density(p) for p in expected_path])
        np.testing.assert_array_equal(result.sparsity, expected_density)
        np.testing.assert_array_equal(result.sparsity,
                                      [_upper_density(p) for p in result.path])
    np.testing.assert_array_equal(data, original)
    if d == 4:
        first.path[1].data[:] = 7.0
        _assert_csc_equal(first.path[2], expected_path[2])
        _assert_csc_equal(repeated.path[1], expected_path[1])


def _native_cube(mask, layout):
    cube = np.zeros((3, 4, 4), dtype=np.uint8)
    for i, frame in enumerate(cube):
        if mask == "sparse":
            j, k = i, (i + 1) % 4
            frame[j, k] = 2
            frame[k, j] = 255
        elif mask == "dense":
            frame[:] = np.arange(1, 17, dtype=np.uint8).reshape(4, 4)
        # Diagonal values must never become stored graph entries, including
        # when the off-diagonal graph is empty.
        np.fill_diagonal(frame, i + 3)
    if layout == "C":
        return np.ascontiguousarray(cube)
    if layout == "F":
        return np.asfortranarray(cube)
    return cube[::-1, ::-1, ::-1]


@pytest.mark.parametrize("mask", ["empty", "sparse", "dense"])
@pytest.mark.parametrize("layout", ["C", "F", "negative-stride"])
def test_glasso_wrap_preserves_original_float_csc_conversion(monkeypatch, mask, layout):
    cube = _native_cube(mask, layout)
    original = cube.copy()
    expected_path = []
    for frame in cube:
        adjacency = frame != 0
        np.fill_diagonal(adjacency, False)
        expected_path.append(sparse.csc_matrix(adjacency.astype(float)))
    precision = [np.eye(4) * (i + 1) for i in range(3)]
    covariance = [np.eye(4) / (i + 1) for i in range(3)]
    expected_df = np.asarray([p.nnz / 2 for p in expected_path])
    expected_loglik = np.asarray([-8., -7., -6.])

    def native_glasso(*args, **kwargs):
        return {"path": cube, "icov": precision, "cov": covariance,
                "df": expected_df, "loglik": expected_loglik}

    monkeypatch.setattr(core, "_CPP", SimpleNamespace(hugeglasso=native_glasso))
    path, icov, cov, df, loglik = core._run_glasso(np.eye(4), LAMBDAS, False, True)
    assert len(path) == 3
    for actual, expected in zip(path, expected_path):
        _assert_csc_equal(actual, expected)
        assert not actual.diagonal().any()
        assert np.all(actual.data == 1.0)
    for actual, expected in zip(icov, precision):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(cov, covariance):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(df, expected_df)
    np.testing.assert_array_equal(loglik, expected_loglik)
    np.testing.assert_array_equal(cube, original)


@pytest.mark.parametrize("transpose", [False, True])
def test_custom_asymmetric_csc_counts_logical_upper_edges(transpose):
    # Unsorted duplicate entries include a canceling upper edge, a surviving
    # duplicate upper edge, stored zeros, lower-only edges, and nonzero diagonal.
    graph = sparse.csc_matrix((
        np.asarray([5., 0., 2., 9., -2., 8., 0., 3., 11., 4., -6., 0., 10.]),
        np.asarray([2, 3, 0, 1, 0, 3, 0, 1, 3, 1, 0, 2, 3]),
        np.asarray([0, 2, 6, 10, 13]),
    ), shape=(4, 4))
    if transpose:
        graph = graph.T.tocsc()
    original = (graph.data.copy(), graph.indices.copy(), graph.indptr.copy())
    assert not graph.has_canonical_format
    assert np.any(graph.data == 0)
    expected_edges = 3.0 if transpose else 2.0
    assert np.count_nonzero(np.triu(graph.toarray(), k=1)) == expected_edges
    path = [graph, sparse.csc_matrix((4, 4)), graph.copy()]
    np.testing.assert_array_equal(core._edge_count(path),
                                  [expected_edges, 0., expected_edges])
    np.testing.assert_array_equal(core._path_sparsity(path),
                                  [expected_edges / 6, 0., expected_edges / 6])
    for actual, expected in zip((graph.data, graph.indices, graph.indptr), original):
        np.testing.assert_array_equal(actual, expected)
