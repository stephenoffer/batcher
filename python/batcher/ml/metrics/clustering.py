"""Clustering-quality metrics — scoring a labeling against a reference, from a contingency table.

An unsupervised clustering has no accuracy, because its cluster ids are arbitrary. What it has
is *agreement* with a reference labeling — the true classes, or another clustering — and every
standard measure of that agreement is a function of one small object: the contingency table of
how many rows fall in each ``(reference, cluster)`` pair. Batcher builds that table with a single
``group_by`` over the two label columns, then the score is closed-form arithmetic on the driver.

The metrics differ in what they reward. The Rand-family (`adjusted_rand_score`,
`fowlkes_mallows_score`) count agreeing *pairs* of points; the information-theoretic family
(`normalized_mutual_info_score`, `homogeneity_score`, `completeness_score`, `v_measure_score`)
measure shared *entropy*. All are checked against scikit-learn.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from batcher.ml.metrics._cluster_shared import (
    _contingency,
    _entropy,
    _expected_mutual_info,
    _mutual_info,
)
from batcher.plan.expr_ir.constructors import col

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = [
    "adjusted_mutual_info_score",
    "adjusted_rand_score",
    "completeness_score",
    "contingency_matrix",
    "fowlkes_mallows_score",
    "homogeneity_score",
    "mutual_info_score",
    "normalized_mutual_info_score",
    "pair_confusion_matrix",
    "rand_score",
    "v_measure_score",
]


def adjusted_rand_score(ds: Dataset, labels_true: str, labels_pred: str) -> float:
    """The Rand index of two labelings, corrected for chance — 1 is identical, 0 is random.

    Counts the pairs of points the two labelings agree on (together or apart), then subtracts the
    agreement expected by chance, so a random clustering scores about 0 rather than the inflated
    value the raw Rand index gives. The standard headline score when a reference labeling exists.

    Args:
        ds: The dataset holding both label columns.
        labels_true: The reference labeling.
        labels_pred: The clustering to score.

    Returns:
        The adjusted Rand index, at most 1 (can be slightly negative).

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import adjusted_rand_score
            >>> ds = bt.from_pydict({"t": [0, 0, 1, 1], "p": [1, 1, 0, 0]})
            >>> adjusted_rand_score(ds, "t", "p")
            1.0
    """
    matrix, a, b, n = _contingency(ds, labels_true, labels_pred)

    def choose2(values):
        import numpy as np

        return np.sum(values * (values - 1) / 2)

    sum_ij = choose2(matrix.ravel())
    sum_a, sum_b = choose2(a), choose2(b)
    total_pairs = n * (n - 1) / 2
    expected = sum_a * sum_b / total_pairs if total_pairs else 0.0
    maximum = (sum_a + sum_b) / 2
    if maximum == expected:
        return 1.0
    return float((sum_ij - expected) / (maximum - expected))


def normalized_mutual_info_score(ds: Dataset, labels_true: str, labels_pred: str) -> float:
    """The mutual information between two labelings, normalized to ``[0, 1]``.

    The information-theoretic agreement: how many bits knowing the cluster tells you about the
    true label, divided by the arithmetic mean of the two labelings' entropies (scikit-learn's
    default normalization). 1 means the labelings determine each other; 0 means independent.

    Args:
        ds: The dataset holding both label columns.
        labels_true: The reference labeling.
        labels_pred: The clustering to score.

    Returns:
        The normalized mutual information in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import normalized_mutual_info_score
            >>> ds = bt.from_pydict({"t": [0, 0, 1, 1], "p": [5, 5, 9, 9]})
            >>> round(normalized_mutual_info_score(ds, "t", "p"), 6)
            1.0
    """
    matrix, a, b, n = _contingency(ds, labels_true, labels_pred)
    entropy_true, entropy_pred = _entropy(a, n), _entropy(b, n)
    if entropy_true == 0.0 and entropy_pred == 0.0:
        return 1.0
    denominator = (entropy_true + entropy_pred) / 2
    if denominator == 0.0:
        return 0.0
    return float(_mutual_info(matrix, a, b, n) / denominator)


def homogeneity_score(ds: Dataset, labels_true: str, labels_pred: str) -> float:
    """How much each cluster contains only members of a single true class, in ``[0, 1]``.

    1 means every cluster is pure — no cluster mixes two true classes — regardless of whether a
    class was split across clusters. It is `completeness_score`'s complement, and the two together
    make `v_measure_score`.

    Args:
        ds: The dataset holding both label columns.
        labels_true: The reference labeling.
        labels_pred: The clustering to score.

    Returns:
        The homogeneity in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import homogeneity_score
            >>> ds = bt.from_pydict({"t": [0, 0, 1, 1], "p": [0, 0, 1, 1]})
            >>> homogeneity_score(ds, "t", "p")
            1.0
    """
    matrix, a, b, n = _contingency(ds, labels_true, labels_pred)
    entropy_true = _entropy(a, n)
    if entropy_true == 0.0:
        return 1.0
    return float(_mutual_info(matrix, a, b, n) / entropy_true)


def completeness_score(ds: Dataset, labels_true: str, labels_pred: str) -> float:
    """How much all members of each true class are assigned to the same cluster, in ``[0, 1]``.

    1 means no true class is split across clusters, regardless of whether a cluster mixes classes.
    It is the symmetric counterpart of `homogeneity_score` (swap the two labelings and the two
    scores swap), and the two combine into `v_measure_score`.

    Args:
        ds: The dataset holding both label columns.
        labels_true: The reference labeling.
        labels_pred: The clustering to score.

    Returns:
        The completeness in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import completeness_score
            >>> ds = bt.from_pydict({"t": [0, 0, 1, 1], "p": [0, 0, 1, 1]})
            >>> completeness_score(ds, "t", "p")
            1.0
    """
    matrix, a, b, n = _contingency(ds, labels_true, labels_pred)
    entropy_pred = _entropy(b, n)
    if entropy_pred == 0.0:
        return 1.0
    return float(_mutual_info(matrix, a, b, n) / entropy_pred)


def v_measure_score(ds: Dataset, labels_true: str, labels_pred: str) -> float:
    """The harmonic mean of homogeneity and completeness, in ``[0, 1]``.

    The single balanced score of a clustering against a reference: a clustering must be both pure
    (`homogeneity_score`) and unfragmented (`completeness_score`) to score high, and either failing
    drags it down. Symmetric in the two labelings, and equal to `normalized_mutual_info_score` with
    the arithmetic normalization.

    Args:
        ds: The dataset holding both label columns.
        labels_true: The reference labeling.
        labels_pred: The clustering to score.

    Returns:
        The V-measure in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import v_measure_score
            >>> ds = bt.from_pydict({"t": [0, 0, 1, 1], "p": [3, 3, 8, 8]})
            >>> v_measure_score(ds, "t", "p")
            1.0
    """
    homogeneity = homogeneity_score(ds, labels_true, labels_pred)
    completeness = completeness_score(ds, labels_true, labels_pred)
    if homogeneity + completeness == 0.0:
        return 0.0
    return 2.0 * homogeneity * completeness / (homogeneity + completeness)


def fowlkes_mallows_score(ds: Dataset, labels_true: str, labels_pred: str) -> float:
    """The geometric mean of the pairwise precision and recall of two labelings, in ``[0, 1]``.

    A Rand-family score built on point *pairs*: of the pairs the clustering groups together, how
    many the reference agrees on (precision), and of the pairs the reference groups together, how
    many the clustering recovers (recall), combined as their geometric mean. Unlike
    `adjusted_rand_score` it is not chance-corrected, so it stays comparable across different
    cluster counts.

    Args:
        ds: The dataset holding both label columns.
        labels_true: The reference labeling.
        labels_pred: The clustering to score.

    Returns:
        The Fowlkes-Mallows index in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import fowlkes_mallows_score
            >>> ds = bt.from_pydict({"t": [0, 0, 1, 1], "p": [0, 0, 1, 1]})
            >>> fowlkes_mallows_score(ds, "t", "p")
            1.0
    """
    import numpy as np

    matrix, a, b, n = _contingency(ds, labels_true, labels_pred)
    tk = float(np.sum(matrix**2) - n)
    pk = float(np.sum(a**2) - n)
    qk = float(np.sum(b**2) - n)
    if pk <= 0.0 or qk <= 0.0:
        return 0.0
    return tk / math.sqrt(pk * qk)


def rand_score(ds: Dataset, labels_true: str, labels_pred: str) -> float:
    """The Rand index — the fraction of point pairs the two labelings agree on, in ``[0, 1]``.

    Of all pairs of points, the share the two labelings treat the same way (both together or both
    apart). It is the raw, un-chance-corrected version of `adjusted_rand_score`: intuitive and
    always in ``[0, 1]``, but inflated toward 1 when there are many clusters, because most random
    pairs land in different clusters and trivially agree. Use `adjusted_rand_score` to compare
    across cluster counts; use this when a plain agreement fraction is what you want to report.

    Args:
        ds: The dataset holding both label columns.
        labels_true: The reference labeling.
        labels_pred: The clustering to score.

    Returns:
        The Rand index in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import rand_score
            >>> ds = bt.from_pydict({"t": [0, 0, 1, 1], "p": [1, 1, 0, 0]})
            >>> rand_score(ds, "t", "p")
            1.0
    """
    import numpy as np

    matrix, a, b, n = _contingency(ds, labels_true, labels_pred)

    def choose2(values):
        return np.sum(values * (values - 1) / 2)

    total_pairs = n * (n - 1) / 2
    if total_pairs == 0:
        return 1.0
    agreeing = total_pairs + 2 * choose2(matrix.ravel()) - choose2(a) - choose2(b)
    return float(agreeing / total_pairs)


def mutual_info_score(ds: Dataset, labels_true: str, labels_pred: str) -> float:
    """The mutual information between two labelings, in nats — unnormalized.

    How much knowing the cluster reduces uncertainty about the true label, in natural-log units.
    It is the raw quantity that `normalized_mutual_info_score` rescales to ``[0, 1]``; use it when
    you want the information content itself rather than a normalized comparison, and note it has no
    fixed upper bound (it rises with the number of clusters).

    Args:
        ds: The dataset holding both label columns.
        labels_true: The reference labeling.
        labels_pred: The clustering to score.

    Returns:
        The mutual information in nats, at least 0.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import mutual_info_score
            >>> ds = bt.from_pydict({"t": [0, 0, 1, 1], "p": [0, 0, 1, 1]})
            >>> round(mutual_info_score(ds, "t", "p"), 6)
            0.693147
    """
    matrix, a, b, n = _contingency(ds, labels_true, labels_pred)
    return _mutual_info(matrix, a, b, n)


def adjusted_mutual_info_score(ds: Dataset, labels_true: str, labels_pred: str) -> float:
    """The mutual information between two labelings, corrected for chance — 1 identical, 0 random.

    `normalized_mutual_info_score` still rewards a labeling with many clusters, because splitting
    the data finer raises the mutual information even at random. This subtracts the mutual
    information expected by chance for the given cluster sizes, then normalizes, so a random
    clustering scores about 0 whatever its cluster count. It is the information-theoretic
    counterpart of `adjusted_rand_score` and the safest of the mutual-information scores to compare
    across different numbers of clusters.

    Args:
        ds: The dataset holding both label columns.
        labels_true: The reference labeling.
        labels_pred: The clustering to score.

    Returns:
        The adjusted mutual information, at most 1 (can be slightly negative).

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import adjusted_mutual_info_score
            >>> ds = bt.from_pydict({"t": [0, 0, 1, 1], "p": [2, 2, 8, 8]})
            >>> round(adjusted_mutual_info_score(ds, "t", "p"), 6)
            1.0
    """
    matrix, a, b, n = _contingency(ds, labels_true, labels_pred)
    entropy_true, entropy_pred = _entropy(a, n), _entropy(b, n)
    if entropy_true == 0.0 and entropy_pred == 0.0:
        return 1.0
    mutual = _mutual_info(matrix, a, b, n)
    expected = _expected_mutual_info(matrix, a, b, n)
    normalizer = (entropy_true + entropy_pred) / 2
    denominator = normalizer - expected
    if abs(denominator) < 1e-15:
        return 0.0
    return float((mutual - expected) / denominator)


def contingency_matrix(ds: Dataset, labels_true: str, labels_pred: str) -> Dataset:
    """The co-occurrence table of two labelings, as a labeled matrix.

    Cell ``[i, j]`` counts the rows that the reference put in class `i` and the clustering put in
    cluster `j`. It is the raw object every other clustering metric is a function of, and reading
    it directly shows *how* two labelings disagree — which true class a cluster splits, which
    clusters a class is scattered across. Built with one `group_by` over the two label columns.

    Args:
        ds: The dataset holding both label columns.
        labels_true: The reference labeling (the rows of the table).
        labels_pred: The clustering (the columns of the table).

    Returns:
        A `Dataset` with a leading ``labels_true`` label column and one integer column per
        distinct `labels_pred` value.

    Raises:
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import contingency_matrix
            >>> ds = bt.from_pydict({"t": [0, 0, 1, 1], "p": [0, 0, 0, 1]})
            >>> table = contingency_matrix(ds, "t", "p").to_pydict()
            >>> table["t"], table["0"], table["1"]
            ([0, 1], [2, 1], [0, 1])
    """
    import batcher as bt

    true_labels = [
        v.as_py()
        for v in ds.select(labels_true).distinct().sort(labels_true).collect().column(labels_true)
    ]
    pred_labels = [
        v.as_py()
        for v in ds.select(labels_pred).distinct().sort(labels_pred).collect().column(labels_pred)
    ]
    # Rebuild in sorted label order so the table reads deterministically.
    counts = _contingency_counts(ds, labels_true, labels_pred, true_labels, pred_labels)
    table: dict[str, list] = {labels_true: true_labels}
    for j, pred in enumerate(pred_labels):
        table[str(pred)] = [int(counts[i][j]) for i in range(len(true_labels))]
    return bt.from_pydict(table)


def _contingency_counts(ds, labels_true, labels_pred, true_labels, pred_labels):
    """The contingency counts as a nested list indexed by the given sorted label orders."""
    table = ds.group_by(labels_true, labels_pred).agg(__bt_n=col(labels_true).count()).collect()
    row_of = {label: i for i, label in enumerate(true_labels)}
    col_of = {label: j for j, label in enumerate(pred_labels)}
    counts = [[0] * len(pred_labels) for _ in true_labels]
    for r in range(table.num_rows):
        t = table.column(labels_true)[r].as_py()
        p = table.column(labels_pred)[r].as_py()
        counts[row_of[t]][col_of[p]] = int(table.column("__bt_n")[r].as_py())
    return counts


def pair_confusion_matrix(ds: Dataset, labels_true: str, labels_pred: str) -> dict[str, int]:
    """The pair-counting confusion of two labelings — the basis of the Rand-family scores.

    Every unordered pair of points falls into one of four buckets: the two labelings agree they
    are together (`same_same`), agree they are apart (`different_different`), or disagree
    (`same_different` where only the clustering groups them, `different_same` where only the
    reference does). These four counts are exactly what `adjusted_rand_score` and
    `fowlkes_mallows_score` are built from, and they say *which kind* of mistake a clustering makes
    — over-merging distinct classes versus splitting one class apart.

    Args:
        ds: The dataset holding both label columns.
        labels_true: The reference labeling.
        labels_pred: The clustering to score.

    Returns:
        A dict with keys ``same_same``, ``same_different``, ``different_same``, and
        ``different_different`` (matching scikit-learn's ``[[TN, FP], [FN, TP]]`` as
        ``different_different``, ``different_same``, ``same_different``, ``same_same``).

    Raises:
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import pair_confusion_matrix
            >>> ds = bt.from_pydict({"t": [0, 0, 1, 1], "p": [0, 0, 1, 1]})
            >>> result = pair_confusion_matrix(ds, "t", "p")
            >>> result["same_same"], result["same_different"]
            (4, 0)
    """
    import numpy as np

    matrix, a, b, n = _contingency(ds, labels_true, labels_pred)
    sum_squares = float(np.sum(matrix**2))
    sum_a = float(np.sum(a**2))
    sum_b = float(np.sum(b**2))
    same_same = sum_squares - n
    same_different = sum_b - sum_squares
    different_same = sum_a - sum_squares
    different_different = n * n - sum_a - sum_b + sum_squares
    return {
        "same_same": int(same_same),
        "same_different": int(same_different),
        "different_same": int(different_same),
        "different_different": int(different_different),
    }
