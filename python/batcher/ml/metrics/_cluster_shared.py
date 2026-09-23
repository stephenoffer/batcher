"""Shared contingency-table math for the clustering metrics.

The clustering scores and the diagnostic tables all reduce to one object — the contingency table
of two labelings and the entropy/mutual-information quantities derived from it. Those live here so
both `clustering` (the scores) and the table functions can share one implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher.ml.stats._shared import require_columns
from batcher.plan.expr_ir.constructors import col

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset


def _contingency(ds: Dataset, labels_true: str, labels_pred: str):
    """The contingency counts, row totals, column totals, and n, as numpy arrays."""
    import numpy as np

    require_columns(ds, labels_true, labels_pred)
    table = ds.group_by(labels_true, labels_pred).agg(__bt_n=col(labels_true).count()).collect()
    true_values = table.column(labels_true).to_pylist()
    pred_values = table.column(labels_pred).to_pylist()
    counts = table.column("__bt_n").to_pylist()
    row_index = {value: i for i, value in enumerate(dict.fromkeys(true_values))}
    col_index = {value: j for j, value in enumerate(dict.fromkeys(pred_values))}
    matrix = np.zeros((len(row_index), len(col_index)))
    for t, p, c in zip(true_values, pred_values, counts, strict=True):
        matrix[row_index[t], col_index[p]] += c
    return matrix, matrix.sum(1), matrix.sum(0), matrix.sum()


def _entropy(counts, n: float) -> float:
    """The Shannon entropy (natural log) of a count vector."""
    import numpy as np

    positive = counts[counts > 0]
    fractions = positive / n
    return float(-np.sum(fractions * np.log(fractions)))


def _mutual_info(matrix, row_totals, col_totals, n: float) -> float:
    """The mutual information (natural log) between the two labelings."""
    import numpy as np

    i, j = np.nonzero(matrix)
    cell = matrix[i, j]
    return float(np.sum(cell / n * np.log(cell * n / (row_totals[i] * col_totals[j]))))


def _expected_mutual_info(row_totals, col_totals, n: float) -> float:
    """The mutual information expected by chance for the given margins (Vinh et al., 2010).

    ``EMI = sum over (a_i, b_j) of sum over n_ij of (n_ij / n) log(n n_ij / (a_i b_j)) P(n_ij)``,
    with ``P`` the hypergeometric probability of the cell count given the margins. The sum is
    the same one scikit-learn's Cython ``expected_mutual_information`` takes; it used to be a
    Python triple loop, 56 s against scikit-learn's 1.4 s for 200 x 200 clusters over 100,000
    rows. Here it is vectorized with numpy: margins that repeat are folded together (the
    terms depend only on the margin *values*), and for each distinct row margin every
    ``(b_j, n_ij)`` term is evaluated in one ragged array, so memory stays O(n) per step.
    """
    import numpy as np

    total = int(n)
    if total == 0:
        return 0.0
    # log(k!) for k in 0..n, so each hypergeometric weight is a handful of table lookups.
    log_factorial = np.concatenate(([0.0], np.cumsum(np.log(np.arange(1, total + 1)))))
    a_values, a_weights = np.unique(row_totals.astype(np.int64), return_counts=True)
    b_values, b_weights = np.unique(col_totals.astype(np.int64), return_counts=True)
    expected = 0.0
    for a, a_weight in zip(a_values, a_weights, strict=True):
        start = np.maximum(1, a + b_values - total)
        stop = np.minimum(a, b_values)
        lengths = np.maximum(stop - start + 1, 0)
        if not lengths.any():
            continue
        offsets = np.repeat(start - np.cumsum(lengths) + lengths, lengths)
        nij = offsets + np.arange(lengths.sum())
        b = np.repeat(b_values, lengths)
        weight = np.repeat(b_weights, lengths)
        log_p = (
            log_factorial[a]
            + log_factorial[b]
            + log_factorial[total - a]
            + log_factorial[total - b]
            - log_factorial[total]
            - log_factorial[nij]
            - log_factorial[a - nij]
            - log_factorial[b - nij]
            - log_factorial[total - a - b + nij]
        )
        term = nij / n * np.log(n * nij / (a * b))
        expected += float(a_weight) * float(np.sum(weight * term * np.exp(log_p)))
    return expected
