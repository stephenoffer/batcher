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

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.ml.preprocessors.base import Preprocessor, columns_arg
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

    __slots__ = ("by", "statistics", "value")

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
        # The learned lookup is a Dataset, not a scalar, so it lives on the instance rather
        # than a trailing-underscore field — persistence would have to serialize a table.
        self._lookup: Dataset | None = None

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
        self._lookup = ds.group_by(*self.by).agg(**aggregates).cache()
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
        assert self._lookup is not None
        return ds.join(self._lookup, on=self.by, how="left")


class GroupImputer(Preprocessor):
    """Fill nulls with the value's mean *within its group*, not the global mean.

    A missing value looks more like its group than like the whole population: a customer's
    missing income resembles their segment's, a sensor's missing reading resembles that
    sensor's history. Filling with the global mean flattens exactly the signal a group
    feature exists to capture; filling with the group mean preserves it.

    `fit` learns each group's mean on the training data. A row whose group was unseen, or
    whose group is entirely null, falls back to the global mean rather than staying null —
    an unfillable value is worse than an approximate one here.

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

    __slots__ = ("by", "columns")

    def __init__(self, columns: str | Sequence[str], *, by: str | Sequence[str]) -> None:
        self.columns = columns_arg(columns, what="GroupImputer")
        self.by = _keys(by)
        self._group_means: Dataset | None = None
        self._global_means: dict[str, float] = {}

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

        group_aggs = {self._fill_name(c): bt.mean(col(c)) for c in self.columns}
        self._group_means = ds.group_by(*self.by).agg(**group_aggs).cache()
        global_row = ds.agg(**{c: bt.mean(col(c)) for c in self.columns}).collect()
        for column in self.columns:
            value = global_row.column(column)[0].as_py()
            self._global_means[column] = 0.0 if value is None else float(value)
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
        assert self._group_means is not None
        joined = ds.join(self._group_means, on=self.by, how="left")
        projections = {}
        for column in self.columns:
            group_mean = col(self._fill_name(column))
            # A group unseen at fit joins to a null group-mean, so the global mean is the
            # second fallback. The column is cast to float64 first: a mean is a float, and a
            # serving batch whose column is entirely null types as `null`, which would clash
            # with the float fill value otherwise.
            fallback = group_mean.fill_null(lit(self._global_means[column]))
            projections[column] = col(column).cast("float64").fill_null(fallback)
        filled = joined.with_columns(**projections)
        return filled.drop(*(self._fill_name(c) for c in self.columns))
