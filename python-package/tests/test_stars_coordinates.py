"""StARS counts stored support without depending on coordinate ordering."""

from collections import Counter
from contextlib import nullcontext
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from pyhuge import core


def _replica_path(replica, canonical):
    path = []
    for point in range(3):
        entries = [(1, 1, 2.0), (3, 1, 0.0), (1, 3, -0.0)]
        if point >= 1:
            for row, column, replicas, value in (
                (0, 1, (0, 1), 1.0),
                (1, 0, (2,), -2.0),
                (2, 3, (0, 2, 4), 1.0),
                (3, 2, (1, 3, 4), -1.0),
            ):
                if replica in replicas:
                    entries.append((row, column, value))
        if point == 2:
            if replica in (0, 3):
                entries.append((0, 2, 1.0))
            if replica == 2:
                entries.append((2, 0, 1.0))
            if replica in (1, 4):
                entries.append((0, 3, 1.0))
                if not canonical:
                    # Stored nonzero duplicates contribute once to the old
                    # indexed update, even when their numeric values cancel.
                    entries.append((0, 3, -1.0))
        data, rows, indptr = [], [], [0]
        for column in range(4):
            current = sorted(
                ((row, value) for row, col, value in entries if col == column),
                key=lambda item: item[0], reverse=not canonical,
            )
            rows.extend(row for row, _ in current)
            data.extend(value for _, value in current)
            indptr.append(len(rows))
        matrix = sparse.csc_matrix(
            (np.asarray(data), np.asarray(rows), np.asarray(indptr)), shape=(4, 4)
        )
        assert matrix.has_canonical_format == canonical
        for buffer in (matrix.data, matrix.indices, matrix.indptr):
            buffer.setflags(write=False)
        path.append(matrix)
    return path


def _snapshot(matrix):
    return (
        matrix.shape, matrix.has_sorted_indices, matrix.has_canonical_format,
        tuple(
            (buffer.dtype.str, buffer.shape, buffer.strides, buffer.flags.writeable,
             hashlib.sha256(buffer.tobytes()).hexdigest())
            for buffer in (matrix.data, matrix.indices, matrix.indptr)
        ),
    )


def _stored_support(matrix):
    support = set()
    for column in range(matrix.shape[1]):
        for offset in range(matrix.indptr[column], matrix.indptr[column + 1]):
            if matrix.data[offset] != 0:
                support.add((int(matrix.indices[offset]), column))
    return support


@pytest.mark.parametrize("method", ("ct", "mb", "glasso"))
@pytest.mark.parametrize("canonical", (True, False), ids=("canonical", "unsorted-duplicates"))
@pytest.mark.parametrize("n_jobs", (1, 2))
def test_stars_stored_support_preserves_frequency_and_threshold_boundaries(
    monkeypatch, method, canonical, n_jobs
):
    identifiers = np.arange(12.0)
    x = np.column_stack((identifiers, identifiers**2, identifiers % 3, -identifiers))
    rng = np.random.default_rng(0)
    sample_keys = [tuple(x[rng.choice(12, size=6, replace=False), 0]) for _ in range(5)]
    assert len(set(sample_keys)) == 5
    replica_paths = [_replica_path(replica, canonical) for replica in range(5)]
    lookup = dict(zip(sample_keys, replica_paths))
    snapshots = [[_snapshot(matrix) for matrix in path] for path in replica_paths]
    calls = []

    def subfit(sample, **kwargs):
        key = tuple(sample[:, 0])
        calls.append(key)
        return SimpleNamespace(path=lookup[key])

    monkeypatch.setattr(core, "huge", subfit)
    estimate = core.HugeResult(
        method=method, lambda_path=np.asarray([0.7, 0.4, 0.1]),
        sparsity=np.asarray([0.0, 0.2, 0.4]),
        path=[sparse.csc_matrix((4, 4)) for _ in range(3)],
        cov_input=False, data=x,
    )

    counts = np.zeros((3, 4, 4), dtype=int)
    for path in replica_paths:
        for point, matrix in enumerate(path):
            for row, column in _stored_support(matrix):
                if method == "glasso":
                    counts[point, row, column] += 1
                elif row < column:
                    counts[point, row, column] += 1
                    counts[point, column, row] += 1
    assert np.any(counts == 2) and np.any(counts == 3)
    expected = np.zeros(3)
    for point in range(3):
        probability = counts[point].astype(float) / 5.0
        if method == "glasso":
            probability = 0.5 * (probability + probability.T)
        np.fill_diagonal(probability, 0.0)
        expected[point] = 4.0 * np.sum(probability * (1.0 - probability)) / 12
    assert expected[0] == 0 < expected[1] < expected[2] < 1

    thresholds = (np.nextafter(expected[1], 0.0), expected[1],
                  np.nextafter(expected[1], 1.0))
    for threshold, expected_index in zip(thresholds, (1, 1, 2)):
        warning = (pytest.warns(RuntimeWarning, match="OpenMP or BLAS")
                   if n_jobs > 1 else nullcontext())
        with warning:
            selected = core.huge_select(
                estimate, criterion="stars", rep_num=5, n_jobs=n_jobs,
                stars_subsample_ratio=0.5, stars_thresh=threshold, verbose=False,
            )
        np.testing.assert_array_equal(selected.variability, expected)
        assert selected.opt_index == expected_index
        assert selected.refit is estimate.path[expected_index - 1]
        assert selected.opt_lambda == estimate.lambda_path[expected_index - 1]

    assert Counter(calls) == Counter({key: 3 for key in sample_keys})
    assert [[_snapshot(matrix) for matrix in path] for path in replica_paths] == snapshots
