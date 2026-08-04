"""Deciding which features to keep, before a model ever sees them.

Feature selection is normally done by fitting a model and reading its importances, which
is circular (the importances depend on the model you were trying to choose) and expensive
(one fit per candidate set). Everything here is model-free: it reads the data once and
answers a question about the *column*.

The three questions worth asking, in the order they are cheapest to answer:

`constant_columns`
    Which columns carry no information at all — zero variance, or one value in almost
    every row. These cost storage, scan time, and plan width and give nothing back, and a
    surprising number of them appear after a join or a partition filter.
`correlated_columns`
    Which columns are near-duplicates of each other. Two features at 0.99 correlation add
    no information, split the importance between them so neither looks useful, and make a
    linear model's coefficients unstable.
`feature_report`
    A ranked table of every candidate against the target, so the whole screen is one
    result you can sort, filter, and store rather than a sequence of ad-hoc calls.
`feature_profile`
    What each column *is*, before any target is involved: how much of it is present, how
    concentrated it is, how skewed, how many outliers. `Dataset.profile` answers the
    data-quality version of this; `feature_profile` answers the modelling version, and
    names the transform each column is asking for.

Each returns names or a `Dataset`, never a mutated dataset. Dropping is your decision, and
`ds.drop(*names)` is already the obvious way to make it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.constructors import col

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["constant_columns", "correlated_columns", "feature_profile", "feature_report"]


def _numeric_columns(ds: Dataset, columns: list[str] | None) -> list[str]:
    """The requested columns, or every numeric column when none are named."""
    if columns is not None:
        missing = [c for c in columns if c not in ds.columns]
        if missing:
            from batcher._internal.errors import ColumnNotFoundError, unknown_message

            raise ColumnNotFoundError(
                unknown_message("column", missing[0], ds.columns, hint="Pass an existing column.")
            )
        return list(columns)
    numeric = ds.select_dtypes("number").columns
    if not numeric:
        raise PlanError("no numeric columns to screen; name the columns explicitly")
    return list(numeric)


def constant_columns(
    ds: Dataset, columns: list[str] | None = None, *, max_mode_share: float = 1.0
) -> list[str]:
    """The columns carrying no usable information, by variance and by modal share.

    A zero-variance column is dead weight by definition. `max_mode_share` catches the more
    common and more insidious case: a column where 99.5% of rows share one value has a
    non-zero variance and is still a flag rather than a measurement, and a model that
    splits on it is fitting a handful of rows.

    Args:
        ds: The dataset to screen.
        columns: The columns to consider; every numeric column when omitted.
        max_mode_share: Flag a column whose most common value covers more than this share
            of rows. 1.0 (the default) flags only genuinely constant columns.

    Returns:
        The names of the columns to consider dropping, in the order given.

    Raises:
        PlanError: If `max_mode_share` is outside ``(0, 1]``.
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.selection import constant_columns
            >>> ds = bt.from_pydict({"useful": [1.0, 2.0, 3.0], "dead": [7.0, 7.0, 7.0]})
            >>> constant_columns(ds)
            ['dead']
    """
    if not 0.0 < max_mode_share <= 1.0:
        raise PlanError(f"max_mode_share must be in (0, 1], got {max_mode_share}")
    names = _numeric_columns(ds, columns)
    spreads = ds.agg(**{f"__bt_v_{i}": col(name).var() for i, name in enumerate(names)}).collect()
    flagged = []
    for index, name in enumerate(names):
        variance = spreads.column(f"__bt_v_{index}")[0].as_py()
        if variance is None or variance == 0.0:
            flagged.append(name)
            continue
        if max_mode_share < 1.0:
            from batcher.ml.stats import mode_share

            if mode_share(ds, name) > max_mode_share:
                flagged.append(name)
    return flagged


def correlated_columns(
    ds: Dataset,
    columns: list[str] | None = None,
    *,
    threshold: float = 0.95,
    keep: Sequence[str] = (),
) -> list[str]:
    """The columns to drop so that no surviving pair correlates above `threshold`.

    Runs over the correlation matrix and, for each pair above the threshold, drops the one
    appearing later in `columns`. That deterministic rule is the point: a "drop one of each
    pair" step that depends on dict ordering gives a different feature set on every run, and
    two pipelines that disagree about which of two identical columns survived are impossible
    to compare.

    `keep` is expressed through that same rule rather than as a filter over the result: a
    protected name is moved to the front of the ordering, so it wins every pair it is in and
    its partner is the one dropped. Removing protected names from the answer afterwards
    would instead leave *both* columns of the pair standing, which is the opposite of what
    the caller asked for.

    Args:
        ds: The dataset to screen.
        columns: The columns to consider; every numeric column when omitted.
        threshold: The absolute correlation above which a pair is redundant.
        keep: Columns that must survive; their correlated partners are dropped instead.

    Returns:
        The names to drop, so that the remaining columns are pairwise below `threshold`.

    Raises:
        PlanError: If `threshold` is outside ``(0, 1]``.
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.selection import correlated_columns
            >>> ds = bt.from_pydict(
            ...     {"a": [1.0, 2.0, 3.0], "copy": [2.0, 4.0, 6.0], "other": [5.0, 1.0, 4.0]}
            ... )
            >>> correlated_columns(ds)
            ['copy']
            >>> correlated_columns(ds, keep=["copy"])
            ['a']
    """
    if not 0.0 < threshold <= 1.0:
        raise PlanError(f"threshold must be in (0, 1], got {threshold}")
    import batcher as bt

    names = _numeric_columns(ds, columns)
    if keep:
        protected = [n for n in keep if n in names]
        names = protected + [n for n in names if n not in set(protected)]
    if len(names) < 2:
        return []
    pairs = {
        f"__bt_r_{i}_{j}": bt.corr(col(names[i]), col(names[j]))
        for i in range(len(names))
        for j in range(i + 1, len(names))
    }
    row = ds.agg(**pairs).collect()
    dropped: set[str] = set()
    for i in range(len(names)):
        if names[i] in dropped:
            continue
        for j in range(i + 1, len(names)):
            if names[j] in dropped:
                continue
            value = row.column(f"__bt_r_{i}_{j}")[0].as_py()
            if value is not None and abs(value) >= threshold:
                dropped.add(names[j])
    return [name for name in names if name in dropped]


def feature_report(
    ds: Dataset,
    target: str,
    columns: list[str] | None = None,
    *,
    buckets: int = 10,
    positive: Any = 1,
) -> Dataset:
    """Rank every candidate feature against a binary target, model-free.

    One row per feature with four numbers that answer different questions, so a feature
    strong on any of them survives the screen:

    ``information_value``
        Total predictive power over quantile bins. Catches a non-monotone relationship a
        correlation cannot see. Above 0.1 is worth keeping.
    ``point_biserial``
        The signed linear correlation with the target, so the *direction* is visible.
    ``signal_ratio``
        How far apart the two classes' means are, in standard deviations. Survives a
        relationship that reverses direction.
    ``null_rate``
        How much of the column is actually there. A feature with a strong signal on the 4%
        of rows that have it is a different proposition from one on all of them.

    Args:
        ds: The dataset holding the features and the target.
        target: The binary label column.
        columns: The features to screen; every numeric column but the target when omitted.
        buckets: Quantile bins for the information value.
        positive: The label value that counts as the positive class.

    Cost is one aggregate carrying every feature's correlation, signal ratio and null rate,
    plus one quantile-binned pass per feature for the information value, which needs the
    feature's own bin edges and so cannot ride the shared aggregate.

    Returns:
        A lazy `Dataset` of ``feature``, ``information_value``, ``point_biserial``,
        ``signal_ratio``, ``null_rate``, ordered by descending information value.

    Raises:
        ColumnNotFoundError: If `target` or a named feature is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.selection import feature_report
            >>> ds = bt.from_pydict(
            ...     {"good": [1.0, 2.0, 8.0, 9.0], "noise": [5.0, 1.0, 6.0, 2.0],
            ...      "y": [0, 0, 1, 1]}
            ... )
            >>> feature_report(ds, "y", buckets=2).to_pydict()["feature"][0]
            'good'
    """
    import batcher as bt

    if target not in ds.columns:
        from batcher._internal.errors import ColumnNotFoundError, unknown_message

        raise ColumnNotFoundError(
            unknown_message("column", target, ds.columns, hint="Pass the label column.")
        )
    names = [c for c in _numeric_columns(ds, columns) if c != target]
    outcome = col(target) == bt.lit(positive)
    scores = ds.agg(
        **{
            key: builder
            for i, name in enumerate(names)
            for key, builder in (
                (f"__bt_r_{i}", bt.point_biserial(name, outcome)),
                (f"__bt_s_{i}", bt.signal_ratio(name, outcome)),
                (f"__bt_n_{i}", bt.null_rate(col(name))),
            )
        }
    ).collect()
    rows: dict[str, list[Any]] = {
        "feature": [],
        "information_value": [],
        "point_biserial": [],
        "signal_ratio": [],
        "null_rate": [],
    }
    for index, name in enumerate(names):
        rows["feature"].append(name)
        rows["information_value"].append(_iv_or_nan(ds, name, target, buckets, positive))
        rows["point_biserial"].append(scores.column(f"__bt_r_{index}")[0].as_py())
        rows["signal_ratio"].append(scores.column(f"__bt_s_{index}")[0].as_py())
        rows["null_rate"].append(scores.column(f"__bt_n_{index}")[0].as_py())
    return bt.from_pydict(rows).sort("information_value", descending=True)


def _iv_or_nan(ds: Dataset, feature: str, target: str, buckets: int, positive: Any) -> float:
    """The feature's information value, or NaN when it cannot be binned.

    A constant or near-constant feature has no quantile cut points and cannot be scored;
    that is a fact about the feature, not a reason to fail the whole report.
    """
    from batcher.ml.stats import information_value

    try:
        return information_value(ds, feature, target, buckets=buckets, positive=positive)
    except PlanError:
        return float("nan")


# The shape thresholds `feature_profile` uses to name a treatment. They are conventions, not
# laws, and they are stated here rather than buried in the branch so a reader can disagree
# with the specific number without having to reverse-engineer it.
_CONSTANT_SHARE = 0.99
_SKEW_LIMIT = 1.0
_HEAVY_TAIL_KURTOSIS = 3.0


def feature_profile(ds: Dataset, columns: list[str] | None = None) -> Dataset:
    """What each numeric column looks like as a *feature*, and what it is asking for.

    `Dataset.profile` answers the data-quality question — how much is present, how many
    distinct values. This answers the modelling one: how concentrated the column is, how
    skewed, how heavy its tails are, and therefore which transform it wants before a model
    sees it.

    Cost is one aggregate carrying every column's null rate, skew, kurtosis and robust
    spread, plus **one grouped query per column** for the mode share. That last one cannot
    join the others: a mode is a `group_by` on the column itself, and two columns cannot
    share a grouping. Profile the columns you are actually considering rather than the whole
    frame when it is wide.

    The ``suggestion`` column is a convention rather than a rule, and deliberately blunt:

    ``"drop"``
        Constant, or one value in over 99% of rows. It is a flag, not a measurement.
    ``"impute"``
        More than half the rows are missing. Whatever else is true, that comes first.
    ``"power transform"``
        Strongly skewed. `PowerTransformer` or `LogTransformer`.
    ``"clip or quantile"``
        Heavy-tailed without much skew. `Clipper` or `QuantileTransformer`.
    ``"ready"``
        Nothing stands out.

    Args:
        ds: The dataset to profile.
        columns: The numeric columns to profile; every numeric column when omitted.

    Returns:
        A `Dataset` of ``column``, ``null_rate``, ``mode_share``, ``skew``,
        ``excess_kurtosis``, ``robust_cv``, ``suggestion``, ordered as `columns` was.

    Raises:
        PlanError: If there are no numeric columns to profile.
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.selection import feature_profile
            >>> ds = bt.from_pydict({"flat": [1.0] * 6, "fine": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
            >>> feature_profile(ds).sort("column").to_pydict()["suggestion"]
            ['ready', 'drop']
    """
    import batcher as bt

    names = _numeric_columns(ds, columns)
    aggregates: dict[str, Any] = {}
    for index, name in enumerate(names):
        aggregates[f"__bt_n_{index}"] = bt.null_rate(col(name))
        aggregates[f"__bt_s_{index}"] = bt.skewness(col(name))
        aggregates[f"__bt_k_{index}"] = bt.kurtosis(col(name))
        aggregates[f"__bt_c_{index}"] = bt.robust_cv(col(name))
    row = ds.agg(**aggregates).collect()
    from batcher.ml.stats import mode_share

    table: dict[str, list[Any]] = {
        "column": [],
        "null_rate": [],
        "mode_share": [],
        "skew": [],
        "excess_kurtosis": [],
        "robust_cv": [],
        "suggestion": [],
    }
    for index, name in enumerate(names):
        nulls = row.column(f"__bt_n_{index}")[0].as_py()
        skew = row.column(f"__bt_s_{index}")[0].as_py()
        kurtosis = row.column(f"__bt_k_{index}")[0].as_py()
        spread = row.column(f"__bt_c_{index}")[0].as_py()
        share = mode_share(ds, name)
        table["column"].append(name)
        table["null_rate"].append(nulls)
        table["mode_share"].append(share)
        table["skew"].append(skew)
        table["excess_kurtosis"].append(kurtosis)
        table["robust_cv"].append(spread)
        table["suggestion"].append(_suggest(nulls, share, skew, kurtosis))
    return bt.from_pydict(table)


def _suggest(
    null_rate: float | None, mode_share: float, skew: float | None, kurtosis: float | None
) -> str:
    """The one-word treatment a column's shape is asking for."""
    if mode_share >= _CONSTANT_SHARE:
        return "drop"
    if null_rate is not None and null_rate > 0.5:
        return "impute"
    if skew is not None and abs(skew) > _SKEW_LIMIT:
        return "power transform"
    if kurtosis is not None and kurtosis > _HEAVY_TAIL_KURTOSIS:
        return "clip or quantile"
    return "ready"
