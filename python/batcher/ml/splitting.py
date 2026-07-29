"""Cross-validation splits as filters — k-fold, stratified, grouped, and time-series.

The usual implementation of a k-fold split materializes an index array and shuffles it,
which caps it at one machine's memory and makes the assignment depend on the row order. All
four here are instead a **content hash** of each row compared against fold boundaries, so:

- a fold is a plain row-wise `Filter`, which the distributed executor treats as
  embarrassingly parallel and the streaming engine accepts;
- the same row lands in the same fold however the data is partitioned, on one node or a
  hundred, this run or next month's;
- nothing is materialized, so the training half of a fold stays lazy and is only ever read
  by whatever consumes it.

`time_series_split` is the exception, and necessarily so: a time series must be split by
*time*, not by hash, or the model trains on the future and every score is a lie.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.api.dataset._build import split_key
from batcher.plan.expr_ir.constructors import col, lit

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = [
    "fold_column",
    "group_kfold",
    "kfold",
    "stratified_kfold",
    "time_series_split",
]


def _check_folds(k: int) -> None:
    """Reject a fold count that cannot produce a train and a test part."""
    if k < 2:
        raise PlanError(f"a cross-validation split needs at least 2 folds, got {k}")


def _check_columns(ds: Dataset, *names: str) -> None:
    """Raise a `ColumnNotFoundError` naming the closest real column for any missing name."""
    available = ds.columns
    # Membership against a set: the check runs per requested name, and `available` is the
    # relation's full width — a wide feature table turned a handful of name checks into a
    # scan of thousands of columns each. The list is kept for the error message, which
    # needs the original order to suggest a close match.
    present = set(available)
    for name in names:
        if name not in present:
            from batcher._internal.errors import ColumnNotFoundError, unknown_message

            raise ColumnNotFoundError(
                unknown_message("column", name, available, hint="Pass an existing column.")
            )


def fold_column(
    ds: Dataset,
    k: int = 5,
    *,
    seed: int = 0,
    key: str | list[str] | None = None,
    name: str = "fold",
) -> Dataset:
    """Append a deterministic fold index in ``[0, k)`` to every row.

    The primitive the other splitters are built from, exposed because it is often what you
    actually want: one column, written once, that every downstream job can filter on
    without re-deriving the assignment — and that can be written to disk so the split
    survives the pipeline that created it.

    Args:
        ds: The dataset to assign folds over.
        k: How many folds to assign.
        seed: Seed for the assignment; the same seed reproduces it.
        key: The column(s) identifying a row. Prefer it on a real corpus: hashing only
            these keeps the assignment stable when other columns change.
        name: The name of the appended fold column.

    Returns:
        A new lazy `Dataset` with the fold column appended.

    Raises:
        PlanError: If `k` is less than 2.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.splitting import fold_column
            >>> ds = bt.range(0, 100)
            >>> folded = fold_column(ds, 5, key="value")
            >>> sorted(set(folded.to_pydict()["fold"]))
            [0, 1, 2, 3, 4]
    """
    _check_folds(k)
    keys = _as_key_columns(ds, key)
    uniform = split_key(ds, keys, seed)
    # floor(u * k) over a uniform [0, 1) is a uniform fold index; clipping guards the
    # measure-zero case where u rounds to exactly 1.0.
    index = (uniform * lit(float(k))).floor().cast("int64").clip(lit(0), lit(k - 1))
    return ds.with_columns(**{name: index})


def _as_key_columns(ds: Dataset, key: str | list[str] | None) -> list[str] | None:
    """Normalize and validate a hash-key argument."""
    if key is None:
        return None
    names = [key] if isinstance(key, str) else list(key)
    _check_columns(ds, *names)
    return names


def kfold(
    ds: Dataset, k: int = 5, *, seed: int = 0, key: str | list[str] | None = None
) -> list[tuple[Dataset, Dataset]]:
    """Split into `k` ``(train, validation)`` pairs, every row validating exactly once.

    The standard cross-validation split. Each pair is two lazy `Dataset`s built from the
    same fold assignment, so nothing is computed until a fold is actually consumed and an
    abandoned fold costs nothing.

    Args:
        ds: The dataset to split.
        k: How many folds.
        seed: Seed for the fold assignment.
        key: The column(s) identifying a row; see `fold_column`.

    Returns:
        `k` ``(train, validation)`` pairs, in fold order.

    Raises:
        PlanError: If `k` is less than 2.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.splitting import kfold
            >>> folds = kfold(bt.range(0, 100), 4, key="value")
            >>> sum(validate.count() for _, validate in folds)
            100
    """
    _check_folds(k)
    folded = fold_column(ds, k, seed=seed, key=key, name="__bt_fold")
    pairs = []
    for index in range(k):
        train = folded.filter(col("__bt_fold") != lit(index)).drop("__bt_fold")
        validate = folded.filter(col("__bt_fold") == lit(index)).drop("__bt_fold")
        pairs.append((train, validate))
    return pairs


def stratified_kfold(
    ds: Dataset, label: str, k: int = 5, *, seed: int = 0, key: str | list[str] | None = None
) -> list[tuple[Dataset, Dataset]]:
    """K-fold that keeps each label's proportion the same in every fold.

    The split to use whenever the label is imbalanced, which is most of the time. A plain
    `kfold` over a 1%-positive dataset gives folds whose positive counts vary by tens of
    percent, and at that point the fold-to-fold variance in the score is measuring the split
    rather than the model.

    Stratification comes free from the hash: the fold index is derived from the row hash
    *and* its label, so within each label the rows are spread uniformly across folds and
    every fold inherits the same label mix. No sorting, no grouping, no shuffle.

    Args:
        ds: The dataset to split.
        label: The column whose distribution each fold must preserve.
        k: How many folds.
        seed: Seed for the fold assignment.
        key: The column(s) identifying a row; see `fold_column`.

    Returns:
        `k` ``(train, validation)`` pairs, in fold order.

    Raises:
        PlanError: If `k` is less than 2.
        ColumnNotFoundError: If `label` is not a column of `ds`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.splitting import stratified_kfold
            >>> ds = bt.from_pydict({"y": [0] * 90 + [1] * 10, "x": list(range(100))})
            >>> folds = stratified_kfold(ds, "y", 5, key="x")
            >>> [v.filter(bt.col("y") == 1).count() for _, v in folds]
            [2, 2, 2, 2, 2]
    """
    _check_folds(k)
    _check_columns(ds, label)
    keys = _as_key_columns(ds, key)
    columns = keys if keys is not None else list(ds.columns)
    # Rank within the label stratum, then take that rank modulo k: consecutive ranks land in
    # consecutive folds, so each stratum is dealt round-robin and every fold gets the same
    # share of it. Hash-ordering the rank keeps the deal reproducible and order-independent.
    from batcher.plan.expr_ir.nodes import row_number

    uniform = split_key(ds, columns, seed)
    ranked = ds.with_columns(__bt_u=uniform).with_columns(
        __bt_fold=(row_number().over(partition_by=[label], order_by=["__bt_u"]) - lit(1)) % lit(k)
    )
    pairs = []
    for index in range(k):
        train = ranked.filter(col("__bt_fold") != lit(index)).drop("__bt_fold", "__bt_u")
        validate = ranked.filter(col("__bt_fold") == lit(index)).drop("__bt_fold", "__bt_u")
        pairs.append((train, validate))
    return pairs


def group_kfold(
    ds: Dataset, group: str, k: int = 5, *, seed: int = 0
) -> list[tuple[Dataset, Dataset]]:
    """K-fold where every row of a group lands in the same fold.

    The split that prevents the most common silent leak in applied ML: the same user,
    patient, session, or document appearing in both train and validation. When that happens
    the model memorizes the entity rather than learning the pattern, the cross-validated
    score is excellent, and production is not.

    The fold is a hash of the *group* value, so grouping is enforced by construction rather
    than by a shuffle that has to be trusted. Fold sizes are only approximately equal, since
    groups differ in size.

    Args:
        ds: The dataset to split.
        group: The column that must not span folds.
        k: How many folds.
        seed: Seed for the fold assignment.

    Returns:
        `k` ``(train, validation)`` pairs, in fold order.

    Raises:
        PlanError: If `k` is less than 2.
        ColumnNotFoundError: If `group` is not a column of `ds`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.splitting import group_kfold
            >>> users = [f"u{i // 10}" for i in range(100)]
            >>> ds = bt.from_pydict({"user": users, "x": list(range(100))})
            >>> folds = group_kfold(ds, "user", 5)
            >>> # No user appears in more than one fold's validation set.
            >>> sum(v.select("user").distinct().count() for _, v in folds)
            10
    """
    _check_folds(k)
    _check_columns(ds, group)
    return kfold(ds, k, seed=seed, key=group)


def time_series_split(
    ds: Dataset, time_column: str, n_splits: int = 5, *, expanding: bool = True
) -> list[tuple[Dataset, Dataset]]:
    """Chronological ``(train, validation)`` splits — never train on the future.

    The only correct cross-validation for a time series, and the one a `kfold` silently
    breaks: a random fold puts next week's rows in the training set, which lets the model
    see the future and produces a validation score no deployment will ever reproduce.

    The time range is cut at ``n_splits + 1`` quantiles of `time_column`. Split *i* trains
    on everything before cut *i* and validates on the window between cut *i* and cut
    *i + 1*. With `expanding` the training set grows with each split, which is what a model
    retrained on all history does; with ``expanding=False`` it is a fixed-width rolling
    window, which is what a model that deliberately forgets does.

    Args:
        ds: The dataset to split.
        time_column: The column defining chronological order.
        n_splits: How many train/validation pairs to produce.
        expanding: Grow the training window (default) rather than sliding it.

    Returns:
        `n_splits` ``(train, validation)`` pairs, earliest first.

    Raises:
        PlanError: If `n_splits` is less than 1.
        ColumnNotFoundError: If `time_column` is not a column of `ds`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.splitting import time_series_split
            >>> ds = bt.from_pydict({"t": list(range(100)), "x": list(range(100))})
            >>> splits = time_series_split(ds, "t", 4)
            >>> [(tr.count(), va.count()) for tr, va in splits]
            [(20, 20), (40, 20), (60, 20), (80, 19)]

    Note:
        Each validation window is half-open, ``[cut_i, cut_i+1)``, and the last cut is the
        maximum of `time_column`, so the single latest row falls outside every validation
        fold. That is why the final pair above validates on 19 rows rather than 20.
    """
    _check_columns(ds, time_column)
    if n_splits < 1:
        raise PlanError(f"time_series_split needs at least 1 split, got {n_splits}")
    fractions = [(i + 1) / (n_splits + 1) for i in range(n_splits + 1)]
    aggregates = {f"q{i}": col(time_column).quantile(f) for i, f in enumerate(fractions)}
    row = ds.agg(**aggregates).collect()
    cuts = [row.column(f"q{i}")[0].as_py() for i in range(len(fractions))]
    splits = []
    for index in range(n_splits):
        start, end = cuts[index], cuts[index + 1]
        train = ds.filter(col(time_column) < lit(start))
        if not expanding and index > 0:
            train = train.filter(col(time_column) >= lit(cuts[index - 1]))
        validate = ds.filter((col(time_column) >= lit(start)) & (col(time_column) < lit(end)))
        splits.append((train, validate))
    return splits
