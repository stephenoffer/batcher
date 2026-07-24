"""Distribution drift between a reference dataset and a current one.

The question every deployed model eventually fails on: the code is unchanged, the metrics
are unavailable because the labels have not arrived yet, and the only thing observable is
whether the *inputs* still look like the ones the model was trained on.

Every measure here compares two datasets over the same column. The binning is the part
that matters and the part most implementations get wrong: the bin edges come from the
**reference** distribution's quantiles and are then applied unchanged to the current data,
so a shift shows up as mass moving between bins rather than as the bins themselves moving.
Deriving edges separately for each side would make two very different distributions look
identical.

Both sides are aggregated in the engine and only the per-bin counts — a handful of numbers
— reach the driver, so a drift check over a billion rows costs two grouped scans.

Reading the numbers, using the conventions the monitoring literature settled on:

===============  ============================================
PSI              < 0.1 stable · 0.1-0.25 moderate · > 0.25 significant
Information value  < 0.02 useless · 0.02-0.1 weak · 0.1-0.3 medium · > 0.3 strong
JS divergence    0 identical · 1 bit maximally different (base 2)
===============  ============================================
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.constructors import col, lit
from batcher.plan.functions.aggregate import count_if
from batcher.plan.functions.aggregate import sum as sum_

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = [
    "categorical_drift",
    "drift_report",
    "information_value",
    "js_divergence",
    "kl_divergence",
    "population_stability_index",
    "woe_table",
]

_BIN = "__bt_bin"
# A bin with no rows on one side would make a log ratio infinite, so an empty bin is given
# this share instead. 1e-6 is the value the credit-risk convention uses; it keeps a genuinely
# vanished bin visible as a large contribution without letting it swamp the total.
_EPSILON = 1e-6


def _numeric_edges(ds: Dataset, column: str, buckets: int) -> list[float]:
    """The interior quantile cut points of `column`, deduplicated and ordered.

    Duplicates are dropped rather than kept: a column where 60% of rows share one value has
    several identical quantiles, and keeping them would create empty bins whose zero counts
    dominate every ratio below.

    A **constant** reference column returns no edges at all, and the caller turns that into
    an error. It has to: every quantile of a constant column is the same number, so the
    dedup leaves one edge, every row on both sides lands above it, and the drift measure
    comes back as exactly 0.0 — reporting "no drift" for a column that moved from 1.0 to
    2.0. A silent zero is far worse here than a refusal.
    """
    fractions = [i / buckets for i in range(1, buckets)]
    aggregates: dict[str, Any] = {f"q{i}": col(column).quantile(f) for i, f in enumerate(fractions)}
    aggregates["__bt_min"] = col(column).min()
    aggregates["__bt_max"] = col(column).max()
    row = ds.agg(**aggregates).collect()
    low = row.column("__bt_min")[0].as_py()
    high = row.column("__bt_max")[0].as_py()
    if low is None or high is None or low == high:
        return []
    values = [row.column(f"q{i}")[0].as_py() for i in range(len(fractions))]
    edges: list[float] = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if not edges or number > edges[-1]:
            edges.append(number)
    return edges


def _bin_expr(column: str, edges: list[float]) -> Any:
    """An expression assigning each row to a 0-based bin index given interior `edges`."""
    # Sum of threshold indicators: bin index = how many edges the value is at or above. One
    # projection, no CASE ladder, and it stays a single vectorized expression at any width.
    index = lit(0)
    for edge in edges:
        index = index + (col(column) >= lit(edge)).cast("int64")
    return index


def _binned_shares(ds: Dataset, column: str, edges: list[float], name: str) -> Dataset:
    """Per-bin share of the non-null rows of `column`, as ``(__bt_bin, name)``."""
    present = ds.filter(col(column).is_not_null())
    binned = present.with_columns(**{_BIN: _bin_expr(column, edges)})
    counts = binned.group_by(_BIN).agg(__bt_n=col(_BIN).count())
    total = sum_(col("__bt_n")).over()
    return counts.with_columns(**{name: col("__bt_n").cast("float64") / total}).select(_BIN, name)


def _aligned_shares(
    reference: Dataset, current: Dataset, column: str, buckets: int
) -> list[tuple[float, float]]:
    """The ``(reference_share, current_share)`` pair for every bin, both sides aligned.

    A full outer join so a bin present on only one side still appears, with the missing
    side floored at `_EPSILON` — the case that *is* the drift signal.
    """
    edges = _numeric_edges(reference, column, buckets)
    if not edges:
        raise PlanError(
            f"column {column!r} is constant (or entirely null) in the reference data, so it "
            "has no distribution to compare against. Drop it from the check, or use "
            "`categorical_drift` if it is really a category."
        )
    left = _binned_shares(reference, column, edges, "__bt_ref")
    right = _binned_shares(current, column, edges, "__bt_cur")
    joined = left.join(right, on=[_BIN], how="outer").to_pydict()
    pairs: list[tuple[float, float]] = []
    for expected, observed in zip(joined["__bt_ref"], joined["__bt_cur"], strict=True):
        pairs.append(
            (
                _EPSILON if expected in (None, 0.0) else float(expected),
                _EPSILON if observed in (None, 0.0) else float(observed),
            )
        )
    return pairs


def population_stability_index(
    reference: Dataset, current: Dataset, column: str, *, buckets: int = 10
) -> float:
    """The population stability index — how far `current`'s distribution has moved.

    ``sum((current - reference) * ln(current / reference))`` over quantile bins of the
    reference. Symmetric, always non-negative, and zero only when the two distributions are
    identical. The industry standard for input drift because it is interpretable on a fixed
    scale regardless of the column: below 0.1 is stable, above 0.25 warrants retraining.

    Args:
        reference: The baseline dataset (usually the training data).
        current: The dataset to compare against it (usually recent production traffic).
        column: The numeric column to compare.
        buckets: How many quantile bins to build from the reference.

    Returns:
        The PSI, at least 0.

    Raises:
        PlanError: If the reference column is constant or entirely null.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import population_stability_index
            >>> train = bt.from_pydict({"x": [float(i) for i in range(100)]})
            >>> same = bt.from_pydict({"x": [float(i) for i in range(100)]})
            >>> round(population_stability_index(train, same, "x", buckets=4), 6)
            0.0
    """
    pairs = _aligned_shares(reference, current, column, buckets)
    return sum((c - r) * math.log(c / r) for r, c in pairs)


def kl_divergence(
    reference: Dataset, current: Dataset, column: str, *, buckets: int = 10, base: float = 2.0
) -> float:
    """Kullback-Leibler divergence from the reference to the current distribution.

    ``sum(current * log(current / reference))`` — the extra bits per observation you pay
    for describing today's data with yesterday's model of it. Asymmetric on purpose: it
    punishes current mass landing where the reference had almost none, which is exactly the
    failure that breaks a model.

    Args:
        reference: The baseline dataset.
        current: The dataset to compare.
        column: The numeric column to compare.
        buckets: How many quantile bins to build from the reference.
        base: The logarithm base; 2 gives bits.

    Returns:
        The divergence in `base` units, at least 0.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import kl_divergence
            >>> train = bt.from_pydict({"x": [float(i) for i in range(100)]})
            >>> same = bt.from_pydict({"x": [float(i) for i in range(100)]})
            >>> round(kl_divergence(train, same, "x", buckets=4), 6)
            0.0
    """
    pairs = _aligned_shares(reference, current, column, buckets)
    return sum(c * math.log(c / r) for r, c in pairs) / math.log(base)


def js_divergence(
    reference: Dataset, current: Dataset, column: str, *, buckets: int = 10, base: float = 2.0
) -> float:
    """Jensen-Shannon divergence — the symmetric, bounded sibling of `kl_divergence`.

    The KL divergence of each side against their average, halved. Bounded above by 1 bit,
    which is what makes it comparable across columns and across time: a JS of 0.3 means the
    same thing for a latency column as for a price column, where a KL of 0.3 does not.

    Args:
        reference: The baseline dataset.
        current: The dataset to compare.
        column: The numeric column to compare.
        buckets: How many quantile bins to build from the reference.
        base: The logarithm base; 2 bounds the result at 1.

    Returns:
        The divergence in ``[0, 1]`` for base 2.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import js_divergence
            >>> train = bt.from_pydict({"x": [float(i) for i in range(100)]})
            >>> shifted = bt.from_pydict({"x": [float(i) + 200 for i in range(100)]})
            >>> round(js_divergence(train, shifted, "x", buckets=4), 4)
            0.5488
    """
    pairs = _aligned_shares(reference, current, column, buckets)
    total = 0.0
    for expected, observed in pairs:
        mean = (expected + observed) / 2.0
        total += 0.5 * expected * math.log(expected / mean)
        total += 0.5 * observed * math.log(observed / mean)
    return total / math.log(base)


def categorical_drift(reference: Dataset, current: Dataset, column: str) -> float:
    """Total variation distance between two categorical distributions, in ``[0, 1]``.

    Half the sum of absolute differences in per-category share. Needs no binning, handles a
    category that appears only on one side, and reads directly: 0.15 means 15% of the mass
    would have to move to make the two distributions match.

    Args:
        reference: The baseline dataset.
        current: The dataset to compare.
        column: The categorical column to compare.

    Returns:
        The total variation distance in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import categorical_drift
            >>> train = bt.from_pydict({"c": ["a", "a", "b", "b"]})
            >>> now = bt.from_pydict({"c": ["a", "a", "a", "b"]})
            >>> categorical_drift(train, now, "c")
            0.25
    """
    left = _category_shares(reference, column, "__bt_ref")
    right = _category_shares(current, column, "__bt_cur")
    joined = left.join(right, on=[column], how="outer").to_pydict()
    total = 0.0
    for expected, observed in zip(joined["__bt_ref"], joined["__bt_cur"], strict=True):
        total += abs((expected or 0.0) - (observed or 0.0))
    return total / 2.0


def _category_shares(ds: Dataset, column: str, name: str) -> Dataset:
    """Per-category share of the non-null rows, as ``(column, name)``."""
    counts = ds.filter(col(column).is_not_null()).group_by(column).agg(__bt_n=col(column).count())
    total = sum_(col("__bt_n")).over()
    return counts.with_columns(**{name: col("__bt_n").cast("float64") / total}).select(column, name)


def woe_table(
    ds: Dataset, feature: str, label: str, *, buckets: int = 10, positive: Any = 1
) -> Dataset:
    """The weight-of-evidence table for a numeric feature against a binary label.

    The scorecard-building step: bin the feature, and for each bin report the log odds of a
    positive relative to the overall odds. A monotone WOE column across bins is what makes a
    feature usable in a linear scorecard, and the shape of the table is what tells you where
    to merge bins.

    Args:
        ds: The dataset holding both columns.
        feature: The numeric feature to bin.
        label: The binary label column.
        buckets: How many quantile bins to build.
        positive: The label value that counts as the positive class.

    Returns:
        A `Dataset` of ``bin``, ``rows``, ``positives``, ``negatives``, ``positive_rate``,
        ``woe``, ``iv_contribution``, ordered by bin.

    Raises:
        PlanError: If the feature is constant or entirely null.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import woe_table
            >>> ds = bt.from_pydict(
            ...     {"x": [1.0, 2.0, 3.0, 4.0], "y": [0, 0, 1, 1]}
            ... )
            >>> woe_table(ds, "x", "y", buckets=2).to_pydict()["positive_rate"]
            [0.0, 1.0]
    """
    edges = _numeric_edges(ds, feature, buckets)
    if not edges:
        raise PlanError(
            f"feature {feature!r} has no distinct quantile cut points, so it cannot be binned."
        )
    is_positive = col(label) == lit(positive)
    binned = ds.filter(col(feature).is_not_null()).with_columns(**{_BIN: _bin_expr(feature, edges)})
    grouped = binned.group_by(_BIN).agg(
        rows=col(_BIN).count(),
        positives=count_if(is_positive),
        negatives=count_if(~is_positive),
    )
    positive_share = col("positives").cast("float64") / sum_(col("positives")).over()
    negative_share = col("negatives").cast("float64") / sum_(col("negatives")).over()
    shares = grouped.with_columns(
        __bt_p=positive_share.clip(lit(_EPSILON), lit(1.0)),
        __bt_n=negative_share.clip(lit(_EPSILON), lit(1.0)),
    )
    weight = (col("__bt_p") / col("__bt_n")).ln()
    return shares.select(
        bin=col(_BIN),
        rows=col("rows"),
        positives=col("positives"),
        negatives=col("negatives"),
        positive_rate=col("positives").cast("float64") / col("rows").cast("float64"),
        woe=weight,
        iv_contribution=(col("__bt_p") - col("__bt_n")) * weight,
    ).sort("bin")


def information_value(
    ds: Dataset, feature: str, label: str, *, buckets: int = 10, positive: Any = 1
) -> float:
    """The information value of a feature — the total predictive power in its `woe_table`.

    The sum of the per-bin WOE contributions, and the standard feature-ranking number in
    credit risk. Unlike a correlation it captures a non-monotone relationship, and unlike a
    model-based importance it is computed once, independently of any model.

    Args:
        ds: The dataset holding both columns.
        feature: The numeric feature to bin.
        label: The binary label column.
        buckets: How many quantile bins to build.
        positive: The label value that counts as the positive class.

    Returns:
        The information value, at least 0.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import information_value
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0], "y": [0, 0, 1, 1]})
            >>> information_value(ds, "x", "y", buckets=2) > 1.0
            True
    """
    table = woe_table(ds, feature, label, buckets=buckets, positive=positive)
    row = table.agg(iv=sum_(col("iv_contribution"))).collect()
    value = row.column("iv")[0].as_py()
    return 0.0 if value is None else float(value)


def drift_report(
    reference: Dataset, current: Dataset, columns: list[str], *, buckets: int = 10
) -> Dataset:
    """One row per column: its PSI, JS divergence, and the shift in mean and null rate.

    The monitoring job's output. Written as a `Dataset` rather than a dict so it appends
    straight to a drift table with a timestamp, which is what makes the trend visible —
    a single PSI is far less informative than its history.

    Args:
        reference: The baseline dataset.
        current: The dataset to compare.
        columns: The numeric columns to check.
        buckets: How many quantile bins to build per column.

    Returns:
        A `Dataset` of ``column``, ``psi``, ``js_divergence``, ``mean_shift``,
        ``null_rate_shift``, ordered by descending PSI.

    Raises:
        PlanError: If `columns` is empty.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import drift_report
            >>> train = bt.from_pydict({"x": [float(i) for i in range(100)]})
            >>> now = bt.from_pydict({"x": [float(i) + 50 for i in range(100)]})
            >>> drift_report(train, now, ["x"], buckets=4).to_pydict()["psi"][0] > 0.25
            True
    """
    import batcher as bt

    if not columns:
        raise PlanError("drift_report needs at least one column to check")
    rows: dict[str, list[Any]] = {
        "column": [],
        "psi": [],
        "js_divergence": [],
        "mean_shift": [],
        "null_rate_shift": [],
    }
    for name in columns:
        summary = [
            frame.agg(m=col(name).mean(), nulls=bt.null_rate(col(name))).collect()
            for frame in (reference, current)
        ]
        rows["column"].append(name)
        rows["psi"].append(population_stability_index(reference, current, name, buckets=buckets))
        rows["js_divergence"].append(js_divergence(reference, current, name, buckets=buckets))
        rows["mean_shift"].append(
            _difference(summary[1].column("m")[0].as_py(), summary[0].column("m")[0].as_py())
        )
        rows["null_rate_shift"].append(
            _difference(
                summary[1].column("nulls")[0].as_py(), summary[0].column("nulls")[0].as_py()
            )
        )
    return bt.from_pydict(rows).sort("psi", descending=True)


def _difference(current: float | None, reference: float | None) -> float:
    """``current - reference``, or NaN when either side is undefined."""
    if current is None or reference is None:
        return float("nan")
    return float(current) - float(reference)
