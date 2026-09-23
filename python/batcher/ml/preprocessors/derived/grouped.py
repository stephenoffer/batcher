"""Group-aggregate features — what a row's group looks like, attached to the row.

The single most productive family of tabular features, and the one a raw table never has:
the average transaction amount *for this customer*, the failure rate *for this device model*,
the count of prior events *for this session*. Each is an aggregate over a group, joined back
onto every row of that group, and each routinely outperforms the raw columns it summarizes
because it encodes behaviour the individual row cannot.

`GroupStatEncoder` computes them, learning the per-group statistics on the training data and
applying them to any frame — so a serving row inherits the training set's view of its group,
not the serving batch's, which is what keeps the feature stable.

`GroupImputer` is the same machinery pointed at missing values: fill a null with its group's
mean rather than the global one, because the global mean is usually wrong for the group. A
customer's missing income looks more like their segment's income than like everyone's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.ml.preprocessors.base import Preprocessor, columns_arg, nan_as_null
from batcher.plan.expr_ir.constructors import col

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["GROUP_STATISTICS", "GroupImputer", "GroupStatEncoder"]

#: The per-group statistics `GroupStatEncoder` can compute, and the aggregate for each.
GROUP_STATISTICS = ("mean", "std", "min", "max", "count", "median")


def _keys(by: str | Sequence[str]) -> list[str]:
    """Normalize the grouping key(s) into a list."""
    keys = [by] if isinstance(by, str) else list(by)
    if not keys:
        raise PlanError("a group feature needs at least one grouping column")
    return keys


def _learn_lookup(grouped: Dataset, keys: list[str]) -> dict[str, list[Any]]:
    """Read a per-group aggregate to the driver as plain ``{column: values}`` state.

    The learned table has to be *state*, not a lazy `Dataset`: a lazy plan (even a cached one)
    cannot be written by `save` or `to_dict`, so both classes used to persist an empty state
    and fail on the first `transform` after a reload. As plain lists it round-trips through
    JSON and pickle like every other fitted attribute. The table has one row per group, the
    same size the cached plan held. It is sorted by the keys, because a group-by emits its
    groups in hash-table order, which differs between one node and several, and the saved
    state should not.
    """
    return grouped.sort(*keys).collect().to_pydict()


def _lookup_frame(lookup: dict[str, list[Any]], keys: list[str], ds: Dataset) -> Dataset:
    """The learned table as a `Dataset` whose key types match `ds`, ready to join onto it.

    A reloaded table is rebuilt from Python values, so an ``int32`` or ``date`` key would come
    back as whatever `from_pydict` infers. Casting each key to the frame's own type keeps the
    join from failing, or silently missing, on a type mismatch.
    """
    import batcher as bt

    frame = bt.from_pydict(lookup)
    schema = ds.schema
    casts = {k: col(k).cast(schema.field(k).type) for k in keys if k in schema.names}
    return frame.with_columns(**casts) if casts else frame


class GroupStatEncoder(Preprocessor):
    """Attach per-group statistics of a value column to every row of that group.

    The behaviour-encoding feature. ``GroupStatEncoder("amount", by="customer",
    statistics=["mean", "std"])`` gives every transaction its customer's average and spread,
    which is what separates a $500 purchase that is routine for one customer from the same
    purchase that is a five-sigma event for another.

    `fit` learns each group's statistics once, on the training data. `transform` joins them
    onto any frame, so a serving row is described by its group's *training* behaviour — a
    group unseen in training gets null, which a booster reads as "no history", the honest
    answer.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import GroupStatEncoder
            >>> ds = bt.from_pydict(
            ...     {"cust": ["a", "a", "b"], "amount": [10.0, 20.0, 100.0]}
            ... )
            >>> out = GroupStatEncoder("amount", by="cust", statistics=["mean"])
            >>> out.fit_transform(ds).sort("amount").to_pydict()["amount_mean_by_cust"]
            [15.0, 15.0, 100.0]

    Args:
        value: The column to summarize per group.
        by: The grouping column(s).
        statistics: Which statistics to attach; see `GROUP_STATISTICS`.
    """

    __slots__ = ("by", "lookup_", "statistics", "value")

    def __init__(
        self,
        value: str,
        *,
        by: str | Sequence[str],
        statistics: Sequence[str] = ("mean",),
    ) -> None:
        if not isinstance(value, str):
            raise PlanError(f"value must be a column name, got {value!r}")
        self.value = value
        self.by = _keys(by)
        stats = list(statistics)
        for name in stats:
            if name not in GROUP_STATISTICS:
                from batcher._internal.errors import suggestion

                hint = suggestion(name, GROUP_STATISTICS)
                tail = f" {hint}" if hint else ""
                raise PlanError(
                    f"unknown group statistic {name!r}; expected one of "
                    f"{sorted(GROUP_STATISTICS)}.{tail}"
                )
        if not stats:
            raise PlanError("GroupStatEncoder needs at least one statistic")
        self.statistics = stats
        self.lookup_: dict[str, list[Any]] = {}

    def _feature_name(self, statistic: str) -> str:
        """The column name for one statistic, e.g. ``amount_mean_by_cust``."""
        return f"{self.value}_{statistic}_by_{'_'.join(self.by)}"

    def fit(self, ds: Dataset) -> GroupStatEncoder:
        """Learn each group's statistics with one `group_by` aggregate.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import GroupStatEncoder
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "v": [1.0, 3.0, 9.0]})
                >>> pre = GroupStatEncoder("v", by="g", statistics=["mean"]).fit(ds)
                >>> pre.is_fitted
                True

        Args:
            ds: The dataset to learn the per-group statistics from.

        Returns:
            ``self``, fitted.
        """
        import batcher as bt

        aggregates = {}
        for statistic in self.statistics:
            builder = {
                "mean": bt.mean,
                "std": bt.std,
                "min": bt.min,
                "max": bt.max,
                "median": bt.median,
            }.get(statistic)
            name = self._feature_name(statistic)
            aggregates[name] = builder(col(self.value)) if builder else col(self.value).count()
        clean = nan_as_null(ds, [self.value])
        self.lookup_ = _learn_lookup(clean.group_by(*self.by).agg(**aggregates), self.by)
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Join the learned per-group statistics onto `ds`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import GroupStatEncoder
                >>> train = bt.from_pydict({"g": ["a", "a"], "v": [2.0, 4.0]})
                >>> pre = GroupStatEncoder("v", by="g", statistics=["mean"]).fit(train)
                >>> pre.transform(bt.from_pydict({"g": ["a"], "v": [99.0]})).to_pydict()[
                ...     "v_mean_by_g"
                ... ]
                [3.0]

        Args:
            ds: The dataset to attach the statistics to.

        Returns:
            A new lazy `Dataset` with one feature column per statistic joined on.
        """
        self._require_fitted()
        return ds.join(_lookup_frame(self.lookup_, self.by, ds), on=self.by, how="left")


class GroupImputer(Preprocessor):
    """Fill nulls with the value's mean *within its group*, not the global mean.

    A missing value looks more like its group than like the whole population: a customer's
    missing income resembles their segment's, a sensor's missing reading resembles that
    sensor's history. Filling with the global mean flattens exactly the signal a group
    feature exists to capture; filling with the group mean preserves it.

    `fit` learns each group's mean on the training data. A row whose group was unseen, or
    whose group is entirely null, falls back to the global mean rather than staying null —
    an unfillable value is worse than an approximate one here. In a float column a NaN is
    missing too: it is skipped by the means and filled like a null. The learned per-group
    table is plain state (``group_means_``), so it survives `save` / `load` and pickling.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import GroupImputer
            >>> ds = bt.from_pydict(
            ...     {"seg": ["a", "a", "b"], "income": [10.0, None, 50.0]}
            ... )
            >>> out = GroupImputer("income", by="seg").fit_transform(ds)
            >>> out.sort("seg").to_pydict()["income"]
            [10.0, 10.0, 50.0]

    Args:
        columns: The columns whose nulls to fill.
        by: The grouping column(s) whose per-group mean supplies the fill value.
    """

    __slots__ = ("by", "columns", "global_means_", "group_means_")

    def __init__(self, columns: str | Sequence[str], *, by: str | Sequence[str]) -> None:
        self.columns = columns_arg(columns, what="GroupImputer")
        self.by = _keys(by)
        self.group_means_: dict[str, list[Any]] = {}
        self.global_means_: dict[str, float] = {}

    def _fill_name(self, column: str) -> str:
        """The internal column carrying a column's per-group mean."""
        return f"__bt_gmean_{column}"

    def fit(self, ds: Dataset) -> GroupImputer:
        """Learn each group's per-column mean, and the global mean as a fallback.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import GroupImputer
                >>> pre = GroupImputer("v", by="g").fit(
                ...     bt.from_pydict({"g": ["a", "a"], "v": [2.0, 4.0]})
                ... )
                >>> pre.is_fitted
                True

        Args:
            ds: The dataset to learn the group and global means from.

        Returns:
            ``self``, fitted.
        """
        import batcher as bt

        ds = nan_as_null(ds, self.columns)
        group_aggs = {self._fill_name(c): bt.mean(col(c)) for c in self.columns}
        self.group_means_ = _learn_lookup(ds.group_by(*self.by).agg(**group_aggs), self.by)
        global_row = ds.agg(**{c: bt.mean(col(c)) for c in self.columns}).collect()
        self.global_means_ = {}
        for column in self.columns:
            value = global_row.column(column)[0].as_py()
            self.global_means_[column] = 0.0 if value is None else float(value)
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Fill each column's nulls with its group mean, falling back to the global mean.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import GroupImputer
                >>> train = bt.from_pydict({"g": ["a", "a"], "v": [2.0, 4.0]})
                >>> pre = GroupImputer("v", by="g").fit(train)
                >>> pre.transform(bt.from_pydict({"g": ["a"], "v": [None]})).to_pydict()["v"]
                [3.0]

        Args:
            ds: The dataset to fill.

        Returns:
            A new lazy `Dataset` with the nulls filled and no helper columns left behind.
        """
        from batcher.plan.expr_ir.constructors import lit

        self._require_fitted()
        ds = nan_as_null(ds, self.columns)
        joined = ds.join(_lookup_frame(self.group_means_, self.by, ds), on=self.by, how="left")
        projections = {}
        for column in self.columns:
            group_mean = col(self._fill_name(column))
            # A group unseen at fit joins to a null group-mean, so the global mean is the
            # second fallback. The column is cast to float64 first: a mean is a float, and a
            # serving batch whose column is entirely null types as `null`, which would clash
            # with the float fill value otherwise.
            fallback = group_mean.fill_null(lit(self.global_means_[column]))
            projections[column] = col(column).cast("float64").fill_null(fallback)
        filled = joined.with_columns(**projections)
        return filled.drop(*(self._fill_name(c) for c in self.columns))
