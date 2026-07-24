"""Shared contingency-table math for the clustering metrics.

The clustering scores and the diagnostic tables all reduce to one object — the contingency table
of two labelings and the entropy/mutual-information quantities derived from it. Those live here so
both `clustering` (the scores) and the table functions can share one implementation.
"""

from __future__ import annotations

import math
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

    total = 0.0
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] > 0:
                total += (
                    matrix[i, j] / n * np.log(matrix[i, j] * n / (row_totals[i] * col_totals[j]))
                )
    return float(total)


def _expected_mutual_info(matrix, row_totals, col_totals, n: float) -> float:
    """The mutual information expected by chance for the given margins (Vinh et al., 2010)."""
    total = 0.0
    rows, cols = matrix.shape
    for i in range(rows):
        ai = int(row_totals[i])
        for j in range(cols):
            bj = int(col_totals[j])
            start = max(1, int(ai + bj - n))
            end = min(ai, bj)
            for nij in range(start, end + 1):
                term = nij / n * math.log(n * nij / (ai * bj))
                log_weight = (
                    math.lgamma(ai + 1)
                    + math.lgamma(bj + 1)
                    + math.lgamma(n - ai + 1)
                    + math.lgamma(n - bj + 1)
                    - math.lgamma(n + 1)
                    - math.lgamma(nij + 1)
                    - math.lgamma(ai - nij + 1)
                    - math.lgamma(bj - nij + 1)
                    - math.lgamma(n - ai - bj + nij + 1)
                )
                total += term * math.exp(log_weight)
    return total
