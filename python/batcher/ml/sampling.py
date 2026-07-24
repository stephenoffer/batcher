"""Resampling for imbalanced learning — reshaping the class balance without leaving the engine.

A classifier trained on a 1%-positive dataset learns to predict "negative" and score well on
accuracy while being useless. The two standard fixes are to change the *data* (resample the
classes toward balance) or to change the *loss* (weight the rare class up). Both are here,
both as relational operations, so they run over a dataset larger than memory and distribute
like everything else.

Resampling is a filter or a concatenation over a content-hashed selection, never a
driver-side shuffle, so:

- `undersample` keeps a reproducible fraction of the majority — cheap, and it throws away
  data, which is fine when there is plenty.
- `oversample` duplicates the minority up to balance — keeps every row, at the cost of
  repeating some. It duplicates deterministically so a re-run gives the identical training
  set.
- `class_weights` / `sample_weights` change nothing about the rows and instead produce the
  weight a model's ``sample_weight`` argument wants — the option that neither discards nor
  duplicates, and the one to prefer when the model supports it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.constructors import col, lit

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = [
    "balanced_sample",
    "class_counts",
    "class_weights",
    "oversample",
    "sample_weights",
    "stratified_sample",
    "undersample",
]


def _require(ds: Dataset, *names: str) -> None:
    """Raise a `ColumnNotFoundError` naming the closest real column for any missing name."""
    for name in names:
        if name not in ds.columns:
            from batcher._internal.errors import ColumnNotFoundError, unknown_message

            raise ColumnNotFoundError(
                unknown_message("column", name, ds.columns, hint="Pass an existing column.")
            )


def class_counts(ds: Dataset, label: str) -> dict[Any, int]:
    """The number of rows in each class — the first thing to look at before resampling.

    Args:
        ds: The dataset to count.
        label: The class column.

    Returns:
        A ``{class: count}`` dict.

    Raises:
        ColumnNotFoundError: If `label` is not a column.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.sampling import class_counts
            >>> ds = bt.from_pydict({"y": [0, 0, 0, 1]})
            >>> class_counts(ds, "y")
            {0: 3, 1: 1}
    """
    _require(ds, label)
    import batcher as bt

    rows = ds.group_by(label).agg(__bt_n=bt.sum(lit(1))).sort(label).collect()
    keys = rows.column(label).to_pylist()
    counts = rows.column("__bt_n").to_pylist()
    return {k: int(n) for k, n in zip(keys, counts, strict=True)}


def undersample(ds: Dataset, label: str, *, seed: int = 0, target: str = "min") -> Dataset:
    """Downsample every class to the size of the smallest — balance by discarding.

    Each class is thinned to the target count by a reproducible content-hash filter, so the
    result is balanced, lazy, and identical however the data is partitioned. Cheap and the
    right choice when the majority class has data to spare; it discards rows, so prefer
    `class_weights` when every row matters.

    Args:
        ds: The dataset to resample.
        label: The class column.
        seed: Seed for the row selection; the same seed reproduces the sample.
        target: ``"min"`` to match the smallest class (the default), or an int count to
            thin every class to.

    Returns:
        A new lazy `Dataset` with each class downsampled toward balance.

    Raises:
        PlanError: If `target` is neither ``"min"`` nor a positive int.
        ColumnNotFoundError: If `label` is not a column.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.sampling import class_counts, undersample
            >>> ds = bt.from_pydict({"y": [0] * 100 + [1] * 10, "x": list(range(110))})
            >>> balanced = undersample(ds, "y", seed=1)
            >>> counts = class_counts(balanced, "y")
            >>> counts[0] == counts[1]
            True
    """
    counts = class_counts(ds, label)
    if not counts:
        return ds
    target_count = _resolve_target(target, min(counts.values()))
    ranked = _within_class_rank(ds, label, seed)
    return ranked.filter(col("__bt_rank") <= lit(target_count)).drop("__bt_rank")


def oversample(ds: Dataset, label: str, *, seed: int = 0, target: str = "max") -> Dataset:
    """Upsample every class to the size of the largest — balance by duplicating.

    Each minority class is topped up with deterministically chosen duplicate rows until it
    matches the target count. Keeps every original row, unlike `undersample`; the cost is
    repeated rows, which some models weight more heavily. The duplication is reproducible, so
    a re-run produces the identical training set.

    Args:
        ds: The dataset to resample.
        label: The class column.
        seed: Seed for the duplicate selection.
        target: ``"max"`` to match the largest class (the default), or an int count.

    Returns:
        A new lazy `Dataset` with each class oversampled toward balance.

    Raises:
        PlanError: If `target` is neither ``"max"`` nor a positive int.
        ColumnNotFoundError: If `label` is not a column.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.sampling import class_counts, oversample
            >>> ds = bt.from_pydict({"y": [0] * 10 + [1] * 2, "x": list(range(12))})
            >>> counts = class_counts(oversample(ds, "y", seed=1), "y")
            >>> counts[0] == counts[1] == 10
            True
    """
    counts = class_counts(ds, label)
    if not counts:
        return ds
    target_count = _resolve_target(target, max(counts.values()))
    return _oversample_to(ds, label, counts, target_count, seed)


def balanced_sample(ds: Dataset, label: str, *, seed: int = 0) -> Dataset:
    """Balance the classes toward the median class size — undersample the big, oversample the small.

    A middle path between `undersample` (throws away the majority) and `oversample`
    (duplicates the minority): every class is moved to the *median* class count, so the big
    classes lose a little and the small ones gain a little, rather than one extreme absorbing
    the whole adjustment. Often the most robust default when the imbalance is severe.

    Args:
        ds: The dataset to resample.
        label: The class column.
        seed: Seed for the selection and duplication.

    Returns:
        A new lazy `Dataset` with every class at (approximately) the median class size.

    Raises:
        ColumnNotFoundError: If `label` is not a column.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.sampling import balanced_sample, class_counts
            >>> ds = bt.from_pydict(
            ...     {"y": [0] * 100 + [1] * 20 + [2] * 4, "x": list(range(124))}
            ... )
            >>> counts = class_counts(balanced_sample(ds, "y"), "y")
            >>> counts[0] == counts[1] == counts[2] == 20
            True
    """
    counts = class_counts(ds, label)
    if not counts:
        return ds
    ordered = sorted(counts.values())
    median = ordered[len(ordered) // 2]
    # Undersample the classes above the median exactly, then oversample everything up to it.
    ranked = _within_class_rank(ds, label, seed)
    trimmed = ranked.filter(col("__bt_rank") <= lit(median)).drop("__bt_rank")
    trimmed_counts = {cls: min(count, median) for cls, count in counts.items()}
    return _oversample_to(trimmed, label, trimmed_counts, median, seed)


def class_weights(ds: Dataset, label: str) -> dict[Any, float]:
    """Balanced class weights — ``n_rows / (n_classes * class_count)`` (the sklearn default).

    The weight to hand a model's ``class_weight`` argument so the rare class counts as much as
    the common one without touching the data. A class half as frequent gets twice the weight;
    the weights average to 1. This is the option to prefer over resampling whenever the model
    supports it, because it neither discards nor duplicates a single row.

    Args:
        ds: The dataset to weight.
        label: The class column.

    Returns:
        A ``{class: weight}`` dict.

    Raises:
        ColumnNotFoundError: If `label` is not a column.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.sampling import class_weights
            >>> ds = bt.from_pydict({"y": [0, 0, 0, 1]})
            >>> class_weights(ds, "y")
            {0: 0.6666666666666666, 1: 2.0}
    """
    counts = class_counts(ds, label)
    total = sum(counts.values())
    k = len(counts)
    return {cls: total / (k * count) for cls, count in counts.items()}


def sample_weights(ds: Dataset, label: str, *, output_column: str = "sample_weight") -> Dataset:
    """Append a per-row weight column carrying each row's balanced class weight.

    The row-level form of `class_weights`: instead of a dict the caller has to map, this joins
    the weight onto every row as `output_column`, ready to pass straight to a training loop's
    ``sample_weight``. Computed as a lazy `Expr`, so it adds no pass.

    Args:
        ds: The dataset to weight.
        label: The class column.
        output_column: The name of the appended weight column.

    Returns:
        A new lazy `Dataset` with the weight column appended.

    Raises:
        ColumnNotFoundError: If `label` is not a column.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.sampling import sample_weights
            >>> ds = bt.from_pydict({"y": [0, 0, 0, 1]})
            >>> sample_weights(ds, "y").sort("y").to_pydict()["sample_weight"]
            [0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 2.0]
    """
    from batcher.plan.expr_ir.constructors import when

    weights = class_weights(ds, label)
    if not weights:
        return ds.with_columns(**{output_column: lit(1.0)})
    items = list(weights.items())
    expression = lit(items[-1][1])
    for cls, weight in items[:-1]:
        expression = when(col(label) == lit(cls)).then(lit(weight)).otherwise(expression)
    return ds.with_columns(**{output_column: expression})


def _resolve_target(target: str, default: int) -> int:
    """Resolve a ``"min"``/``"max"`` sentinel or an explicit int to a positive count."""
    if isinstance(target, str):
        return default
    if not (isinstance(target, int) and target > 0):
        raise PlanError(f"target must be 'min'/'max' or a positive int, got {target!r}")
    return target


def _within_class_rank(ds: Dataset, label: str, seed: int) -> Dataset:
    """Append a 1-based, hash-ordered rank within each class as ``__bt_rank``.

    A content hash of every row gives a deterministic order that does not depend on how the
    data is partitioned, and ``row_number`` over it numbers each class's rows 1..n. Filtering
    on that rank is an *exact* per-class size cut, unlike a hash fraction which is only
    binomially close to the target.
    """
    from batcher.api.dataset._build import split_key
    from batcher.plan.expr_ir.nodes import row_number

    return (
        ds.with_columns(__bt_hash=split_key(ds, None, seed))
        .with_columns(__bt_rank=row_number().over(partition_by=[label], order_by=["__bt_hash"]))
        .drop("__bt_hash")
    )


def _oversample_to(
    ds: Dataset, label: str, counts: dict[Any, int], target: int, seed: int
) -> Dataset:
    """Grow every class to exactly `target` rows by appending hash-ordered duplicates.

    Each class keeps all its rows, then appends copies — a whole copy at a time, and an exact
    hash-ranked prefix for the final partial copy — until it reaches the target. The prefix is
    the same rows every run, so the oversampled training set is reproducible.
    """
    result = ds
    for cls, count in counts.items():
        if count == 0 or count >= target:
            continue
        part = ds.filter(col(label) == lit(cls))
        ranked = _within_class_rank(part, label, seed)
        remaining = target - count
        while remaining > 0:
            take = min(count, remaining)
            chunk = (
                part
                if take == count
                else ranked.filter(col("__bt_rank") <= lit(take)).drop("__bt_rank")
            )
            result = result.union(chunk)
            remaining -= take
    return result


def stratified_sample(ds: Dataset, by: str, fraction: float, *, seed: int = 0) -> Dataset:
    """Keep the same `fraction` of the rows within every stratum — proportional subsampling.

    Unlike `undersample` and `balanced_sample`, which *equalize* the strata, this preserves
    their relative sizes: it keeps ``floor(fraction * n)`` rows from each group, so a 10% sample
    is 10% of every class rather than 10% of the whole (which would under-represent rare
    classes). The right way to shrink a dataset for a quick experiment while keeping its class
    balance, and the standard split for a stratified holdout.

    The selection is an exact per-stratum count cut over a content hash, so it is reproducible
    from the seed and identical however the data is partitioned. A stratum too small for
    ``floor(fraction * n)`` to reach 1 contributes no rows, which is the honest behavior of a
    proportional sample.

    Args:
        ds: The dataset to subsample.
        by: The stratifying column.
        fraction: The share of each stratum to keep, in ``(0, 1]``.
        seed: Seed for the row selection; the same seed reproduces the sample.

    Returns:
        A new lazy `Dataset` holding `fraction` of each stratum.

    Raises:
        PlanError: If `fraction` is not in ``(0, 1]``.
        ColumnNotFoundError: If `by` is not a column.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.sampling import class_counts, stratified_sample
            >>> ds = bt.from_pydict({"g": ["a"] * 100 + ["b"] * 20, "x": list(range(120))})
            >>> sampled = stratified_sample(ds, "g", 0.5, seed=1)
            >>> class_counts(sampled, "g")
            {'a': 50, 'b': 10}
    """
    _require(ds, by)
    if not 0.0 < fraction <= 1.0:
        raise PlanError(f"fraction must be in (0, 1], got {fraction}.")
    ranked = _within_class_rank(ds, by, seed).with_columns(
        __bt_grpn=col(by).count().over(partition_by=[by])
    )
    keep = col("__bt_rank") <= (lit(fraction) * col("__bt_grpn")).floor()
    return ranked.filter(keep).drop("__bt_rank", "__bt_grpn")
