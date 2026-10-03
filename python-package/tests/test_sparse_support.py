"""Sparse native supports must stay loop-free without inserting stored zeros."""

import warnings

import numpy as np
import pytest
from scipy import sparse

from pyhuge.core import _column_support_to_path


@pytest.mark.parametrize("sym", ["or", "and"])
@pytest.mark.parametrize("self_loops", [False, True])
def test_sparse_support_handles_empty_columns_duplicates_and_self_loops(sym, self_loops):
    # The first two nodes are reciprocal; 2 -> 3 is a one-way edge.
    # The second lambda contains only self-loops (or no stored entries).
    columns = [[1, 1], [0], [3], []]
    if self_loops:
        columns = [[j] + rows for j, rows in enumerate(columns)]
    paths = [columns, [[j] if self_loops else [] for j in range(4)]]
    indptr = np.array([
        np.r_[0, np.cumsum([len(rows) for rows in path])]
        for path in paths
    ], dtype=np.int64)
    indices = np.array([
        row for path in paths for rows in path for row in rows
    ], dtype=np.int32)
    support = {"support_indptr": indptr, "support_indices": indices}
    original_indptr, original_indices = indptr.copy(), indices.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("error", sparse.SparseEfficiencyWarning)
        actual = _column_support_to_path(support, 2, 4, sym)

    expected = np.zeros((4, 4))
    expected[0, 1] = expected[1, 0] = 1.0
    if sym == "or":
        expected[2, 3] = expected[3, 2] = 1.0
    np.testing.assert_array_equal(actual[0].toarray(), expected)
    assert actual[1].nnz == 0
    for graph in actual:
        assert sparse.isspmatrix_csc(graph)
        assert graph.has_canonical_format
        assert np.all(graph.data == 1.0)
        assert not graph.diagonal().any()
    np.testing.assert_array_equal(indptr, original_indptr)
    np.testing.assert_array_equal(indices, original_indices)
