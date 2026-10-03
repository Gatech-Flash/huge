"""ROC support counting without dense expansion of sparse graphs."""

import pickle
import warnings

import numpy as np
import pytest
from scipy import sparse

from pyhuge import core


def _reference(path, theta):
    """Enumerate unordered vertex pairs independently of sparse algebra."""
    truth = np.asarray(theta, dtype=float)
    pairs = [(i, j) for i in range(len(truth)) for j in range(i + 1, len(truth))]
    edges = {pair for pair in pairs if truth[pair] != 0}
    rates = []
    for graph in path:
        prediction = {pair for pair in pairs if graph[pair] != 0}
        hits = len(prediction.intersection(edges))
        tp = hits / len(edges)
        fp = len(prediction.difference(edges)) / (len(pairs) - len(edges))
        f1 = 2 * hits / (len(prediction) + len(edges))
        rates.append((tp, fp, f1))
    ordered = sorted((fp, tp) for tp, fp, _ in rates)
    auc = sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(ordered, ordered[1:]))
    return np.asarray(rates), auc


def _assert_reference(path, theta):
    dense_theta = theta.toarray() if sparse.issparse(theta) else theta
    dense_path = [p.toarray() if sparse.issparse(p) else p for p in path]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.ComplexWarning if hasattr(np, "exceptions") else np.ComplexWarning)
        dense_path = [np.asarray(p, dtype=float) for p in dense_path]
        expected, auc = _reference(dense_path, dense_theta)
        original = pickle.dumps((theta, path))
        actual = core.huge_roc(path, theta, verbose=False, plot=False)
    assert pickle.dumps((theta, path)) == original
    np.testing.assert_allclose(actual.tp, expected[:, 0], rtol=0, atol=1e-15)
    np.testing.assert_allclose(actual.fp, expected[:, 1], rtol=0, atol=1e-15)
    np.testing.assert_allclose(actual.f1, expected[:, 2], rtol=0, atol=1e-15)
    assert actual.auc == pytest.approx(auc, rel=0, abs=1e-15)
    assert actual.raw == {"backend": "native"}


@pytest.mark.parametrize("array", [False, True], ids=["matrix", "array"])
def test_empty_noncanonical_coo_prediction(array):
    constructor = getattr(sparse, "coo_array", None) if array else sparse.coo_matrix
    if constructor is None:
        pytest.skip("Sparse arrays unavailable")
    prediction = constructor(
        (np.array([], dtype=float),
         (np.array([], dtype=np.intp), np.array([], dtype=np.intp))),
        shape=(3, 3),
    )
    assert not prediction.has_canonical_format
    truth = sparse.csr_matrix(([1.], ([0], [1])), shape=(3, 3))
    _assert_reference([prediction], truth)


@pytest.mark.parametrize("values", [
    np.array([1e16, -1e16, 1.]),
    np.array([1e16, 1., -1e16]),
    np.array([-128, -128, 0], dtype=np.int8),
], ids=["retains-edge", "cancels-edge", "int8-overflow"])
def test_duplicate_bsr_blocks_preserve_dense_accumulation(values):
    blocks = np.zeros((3, 2, 2), dtype=values.dtype)
    blocks[:, 0, 0] = values
    prediction = sparse.bsr_matrix(
        (blocks, np.array([1, 1, 1]), np.array([0, 3, 3])), shape=(4, 4)
    )
    truth = np.zeros((4, 4))
    truth[0, 1] = truth[0, 2] = 1
    _assert_reference([prediction], truth)


@pytest.mark.parametrize("seed", [7, 19, 83])
@pytest.mark.parametrize("format_", ["csr", "csc", "coo", "lil", "dok", "bsr", "dia"])
def test_sparse_and_mixed_inputs_match_pair_enumeration(seed, format_):
    rng = np.random.default_rng(seed)
    truth = rng.choice([0.0, 0.0, 0.0, -2.0, 3.0], size=(9, 9))
    truth[0, 1], truth[0, 2] = -2, 0
    path = [rng.choice([0.0, 0.0, -3.0, 2.0], size=(9, 9)) for _ in range(4)]
    constructor = getattr(sparse, f"{format_}_matrix")
    sparse_truth = constructor(truth)
    sparse_path = [constructor(graph) for graph in path]
    _assert_reference(sparse_path, sparse_truth)
    _assert_reference([sparse_path[0], path[1], sparse_path[2], path[3]], sparse_truth)
    _assert_reference(sparse_path, truth)


@pytest.mark.parametrize("layout", ["c", "f", "reversed"])
def test_dense_layouts_and_asymmetry_preserve_upper_triangle_semantics(layout):
    theta = np.zeros((4, 4))
    theta[0, 1], theta[0, 3], theta[2, 0], theta[2, 2] = -2, 3, -7, 5
    path = [np.zeros((4, 4)) for _ in range(4)]
    path[0][3, 0] = 8  # Lower triangle is never an edge for Python ROC.
    np.fill_diagonal(path[0], 10)
    path[1][0, 1], path[1][1, 2] = 1, -1
    path[2][0, 1], path[2][0, 3] = -1, 1
    path[3][:] = 1
    if layout == "f":
        theta = np.asfortranarray(theta)
        path = [np.asfortranarray(graph) for graph in path]
    elif layout == "reversed":
        theta = theta[::-1, ::-1]
        path = [graph[::-1, ::-1] for graph in path]
    _assert_reference(path, theta)
    if layout != "reversed":
        actual = core.huge_roc(path, theta, verbose=False)
        np.testing.assert_array_equal(actual.tp, [0, .5, 1, 1])
        np.testing.assert_array_equal(actual.fp, [0, .25, 0, 1])
        assert actual.auc == .75


def _duplicate_matrix(format_, values):
    # Nonadjacent duplicates and deliberately unsorted minor indices expose
    # changes in both aggregation order and source-dtype integer arithmetic.
    rows = np.array([0, 0, 0, 0, 0, 2, 3, 3])
    cols = np.array([1, 3, 1, 2, 1, 3, 0, 3])
    edge_weight = 3 if values.dtype.kind == "u" else -3
    data = np.asarray([values[0], 2, values[1], 0, values[2], edge_weight, 5, 1], dtype=values.dtype)
    if format_ == "coo":
        return sparse.coo_matrix((data, (rows, cols)), shape=(4, 4))
    major, minor = (rows, cols) if format_ == "csr" else (cols, rows)
    order = np.argsort(major, kind="stable")
    indptr = np.r_[0, np.cumsum(np.bincount(major, minlength=4))]
    return getattr(sparse, f"{format_}_matrix")((data[order], minor[order], indptr), shape=(4, 4))


@pytest.mark.parametrize("format_", ["coo", "csr", "csc"])
@pytest.mark.parametrize("values", [
    np.array([1e16, -1e16, 1.]),
    np.array([1e16, 1., -1e16]),
    np.array([3., -3., 0.]),
    np.array([-128, -128, 0], dtype=np.int8),
    np.array([128, 128, 0], dtype=np.uint8),
    np.array([True, True, False]),
    np.array([1j, -1j, complex(0, np.inf)]),
])
def test_duplicates_sum_before_boolean_in_original_dtype_and_order(format_, values):
    graph = _duplicate_matrix(format_, values)
    _assert_reference([graph, sparse.csr_matrix((4, 4)), np.ones((4, 4))], graph)


@pytest.mark.parametrize("values", [
    [1., -1.], [1., 0.], [np.nan, 1.], [np.inf, 0.], [-np.inf, -1.],
])
def test_lil_duplicates_keep_last_value_including_overwritten_nonfinite(values):
    # LIL normally avoids duplicates, but its public row/data storage permits
    # them and toarray() keeps the last assignment, unlike COO/CSR/CSC.
    graph = sparse.lil_matrix((4, 4))
    graph.rows[0] = [1, 1, 3]
    graph.data[0] = [values[0], values[1], 2.]
    graph.rows[2] = [3]
    graph.data[2] = [-3.]
    assert graph.toarray()[0, 1] == values[-1]
    _assert_reference([graph, sparse.csr_matrix((4, 4)), np.ones((4, 4))], graph)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_lil_duplicates_reject_a_nonfinite_last_value(bad):
    theta = sparse.csr_matrix(([1.], ([0], [1])), shape=(3, 3))
    graph = sparse.lil_matrix((3, 3))
    graph.rows[0], graph.data[0] = [1, 1], [1., bad]
    original = pickle.dumps(graph)
    with pytest.raises(core.PyHugeError, match="theta.*non-finite"):
        core.huge_roc([theta], graph)
    with pytest.raises(core.PyHugeError, match=r"path\[1\].*non-finite"):
        core.huge_roc([graph], theta)
    assert pickle.dumps(graph) == original


@pytest.mark.parametrize("format_", ["csr", "csc", "coo"])
def test_sparse_arrays_match_sparse_matrices(format_):
    constructor = getattr(sparse, f"{format_}_array", None)
    if constructor is None:
        pytest.skip("SciPy does not provide this sparse array type")
    graph = _duplicate_matrix(format_, np.array([4., -4., 1.]))
    _assert_reference([constructor(graph)], constructor(graph))


def test_sparse_roc_never_materializes_dense_graphs(monkeypatch):
    d = 70000
    theta = sparse.csc_matrix(([2., -1.], ([0, 7], [1, 12])), shape=(d, d))
    pred = sparse.coo_matrix(([4., -2., 9., 0.], ([0, 2, 4, 8], [1, 3, 4, 9])), shape=(d, d))
    original = pickle.dumps((theta, pred))

    def forbid_dense(*args, **kwargs):
        raise AssertionError("Sparse ROC must not call toarray() or todense()")

    for class_ in (sparse.csr_matrix, sparse.csc_matrix, sparse.coo_matrix):
        monkeypatch.setattr(class_, "toarray", forbid_dense)
        monkeypatch.setattr(class_, "todense", forbid_dense)
    monkeypatch.setattr(core, "huge_plot_roc", lambda *args: pytest.fail("plot=False must not plot"))
    actual = core.huge_roc([pred], theta, verbose=False, plot=False)
    np.testing.assert_array_equal(actual.tp, [.5])
    np.testing.assert_array_equal(actual.f1, [.5])
    np.testing.assert_array_equal(actual.fp, [1 / (d * (d - 1) // 2 - 2)])
    assert actual.auc == 0
    assert pickle.dumps((theta, pred)) == original


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("position", [(0, 0), (2, 0), (0, 2)])
@pytest.mark.parametrize("sparse_input", [False, True])
def test_nonfinite_entries_are_rejected_even_outside_counted_triangle(bad, position, sparse_input):
    theta = np.zeros((3, 3))
    theta[0, 1] = 1
    invalid = theta.copy()
    invalid[position] = bad
    if sparse_input:
        invalid = sparse.coo_matrix(invalid)
    with pytest.raises(core.PyHugeError, match="theta.*non-finite"):
        core.huge_roc([theta], invalid)
    with pytest.raises(core.PyHugeError, match=r"path\[1\].*non-finite"):
        core.huge_roc([invalid], theta)


def test_validation_and_one_class_truth_are_unchanged():
    theta = sparse.csr_matrix(([1.], ([0], [1])), shape=(3, 3))
    with pytest.raises(core.PyHugeError, match="at least one adjacency"):
        core.huge_roc([], theta)
    for invalid in (np.ones((2, 3)), sparse.csr_matrix((2, 3))):
        with pytest.raises(core.PyHugeError, match="theta.*square"):
            core.huge_roc([theta], invalid)
        with pytest.raises(core.PyHugeError, match=r"path\[1\].*shape"):
            core.huge_roc([invalid], theta)
    with pytest.raises(core.PyHugeError, match="2D"):
        core.huge_roc([theta], np.ones(3))
    with pytest.raises(core.PyHugeError, match="numeric"):
        core.huge_roc([[['bad']]], theta)
    for invalid in (sparse.eye(3), sparse.csr_matrix(np.tril(np.ones((3, 3)))),
                    sparse.csc_matrix(np.triu(np.ones((3, 3)), 1))):
        with pytest.raises(core.PyHugeError, match="ROC/AUC.*one-class"):
            core.huge_roc([theta], invalid)


def test_auc_ties_duplicates_and_output_order_for_sparse_paths():
    truth = sparse.csr_matrix(([1, 1], ([0, 0], [1, 2])), shape=(4, 4))
    low = sparse.csc_matrix(([1], ([2], [3])), shape=(4, 4))
    high = truth + low
    complete = sparse.csr_matrix(np.ones((4, 4)))
    for path in ([low, high, complete], [high, low, complete], [low, high, high, complete]):
        _assert_reference(path, truth)
        assert core.huge_roc(path, truth).auc == .75
