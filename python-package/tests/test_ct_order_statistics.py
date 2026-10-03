"""Automatic CT thresholds retain exact order statistics and input values."""

import numpy as np
import pytest

from pyhuge import huge


_SMALLEST = float.fromhex("0x0.0000000000001p-1022")


def _correlation(scale):
    # Seven variables give 21 undirected edges. Strict diagonal dominance
    # makes both scales positive definite; ties and signed zero are deliberate.
    weights = [
        7, -6, 5, -5, -0.0, 0.0, 4, -4, 3, -3, 2,
        -2, 1, -1, 6, -5, 4, -3, 2, -1, 0.0,
    ]
    matrix = np.eye(7)
    index = 0
    for row in range(7):
        for column in range(row + 1, 7):
            matrix[row, column] = matrix[column, row] = weights[index] * scale
            index += 1
    return matrix


@pytest.mark.parametrize("scale", (0.01, _SMALLEST), ids=("ordinary", "subnormal"))
@pytest.mark.parametrize("layout", ("c", "fortran", "negative-strides"))
@pytest.mark.parametrize(
    "nlambda,ratio,targets",
    (
        # One requested point uses the schedule's start, even at ratio=1.
        pytest.param(1, 1.0, [1], id="single-point"),
        # 1, 1.5, ..., 21 rounds up to the following exact integer schedule.
        pytest.param(
            41, 1.0, [1] + [rank for rank in range(2, 22) for _ in range(2)],
            id="repeated-targets",
        ),
        # All five positive schedule values are at most one.
        pytest.param(5, _SMALLEST, [1] * 5, id="smallest-positive-ratio"),
    ),
)
def test_ct_automatic_order_statistics_match_full_sort(
    scale, layout, nlambda, ratio, targets
):
    supplied = _correlation(scale)
    if layout == "fortran":
        supplied = np.asfortranarray(supplied)
    elif layout == "negative-strides":
        supplied = supplied[::-1, ::-1]
    supplied.setflags(write=False)
    original_bytes = supplied.tobytes()
    original_strides = supplied.strides

    dimension = supplied.shape[0]
    ordered = sorted(
        (
            abs(float(supplied[row, column]))
            for row in range(dimension)
            for column in range(row + 1, dimension)
        ),
        reverse=True,
    )
    expected_lambda = np.asarray(
        [ordered[target] if target < len(ordered) else 0.0 for target in targets]
    )

    fit = huge(
        supplied, method="ct", nlambda=nlambda, lambda_min_ratio=ratio,
        input_type="covariance", verbose=False,
    )

    # Bit comparison also checks the sign of zero and every subnormal bit.
    np.testing.assert_array_equal(
        fit.lambda_path.view(np.uint64), expected_lambda.view(np.uint64)
    )
    assert len(fit.path) == nlambda
    for index, threshold in enumerate(expected_lambda):
        expected_graph = np.asarray(
            [
                [
                    row != column and abs(float(supplied[row, column])) > threshold
                    for column in range(dimension)
                ]
                for row in range(dimension)
            ],
            dtype=float,
        )
        np.testing.assert_array_equal(fit.path[index].toarray(), expected_graph)
        assert fit.sparsity[index] == np.count_nonzero(expected_graph) / (
            dimension * (dimension - 1)
        )

    assert supplied.tobytes() == original_bytes
    assert supplied.strides == original_strides
    assert not supplied.flags.writeable
