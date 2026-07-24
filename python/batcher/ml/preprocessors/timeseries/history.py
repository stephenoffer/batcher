"""Lag and rolling features — history as columns, without leaking the future.

Every time-series model needs to know what happened before. The features that carry that
are lags (the value `n` steps ago) and rolling aggregates (the mean over the last `n`
steps), and building them is where forecasting pipelines leak most often.

The leak is always the same shape. A rolling mean that includes the current row has the
target's own value inside its own feature; a "last 7 days" window computed over the whole
table rather than within one entity mixes customers together. Both produce a
cross-validated score that no deployment reproduces, and neither raises.

So both preprocessors here **exclude the current row by construction** — the window ends at
the previous row, not the current one — and both require a `partition_by` when the table
holds more than one series. Neither is optional, because a default that silently mixed
series would be the exact bug this module exists to prevent.

Both are stateless: the window is a frame specification, not learned state, so training and
serving cannot disagree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.ml.preprocessors.base import Preprocessor, columns_arg
from batcher.plan.expr_ir.constructors import col
from batcher.plan.expr_ir.nodes import lag

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["ROLLING_AGGREGATES", "LagFeaturizer", "RollingFeaturizer"]

#: The aggregates a rolling window may compute, and the `Expr` builder for each.
ROLLING_AGGREGATES = ("mean", "sum", "min", "max", "count")


def _check_partition(partition_by: Sequence[str] | None) -> list[str]:
    """Normalize a partition specification (an empty list means one global series)."""
    if partition_by is None:
        return []
    return [partition_by] if isinstance(partition_by, str) else list(partition_by)


class LagFeaturizer(Preprocessor):
    """Append the value of a column `n` rows earlier, within each series.

    The most basic and most useful time-series feature: yesterday's demand is the single
    best predictor of today's, and a model cannot see it unless you put it in a column.

    ``lags=[1, 7, 28]`` on a daily table gives yesterday, the same weekday last week, and the
    same day four weeks ago — the standard set, and the one that lets a linear model express
    weekly and monthly seasonality without a Fourier basis.

    Rows near the start of a series have no history and get null, which is the honest answer.
    Drop them, or let a model that handles missing values (any booster) use the null as the
    signal it is.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import LagFeaturizer
            >>> ds = bt.from_pydict({"day": [1, 2, 3], "sales": [10.0, 20.0, 30.0]})
            >>> out = LagFeaturizer("sales", order_by="day", lags=[1]).fit_transform(ds)
            >>> out.sort("day").to_pydict()["sales_lag_1"]
            [None, 10.0, 20.0]

    Args:
        columns: The columns to lag.
        order_by: The column defining time order within a series.
        lags: How many rows back each lag column reaches.
        partition_by: The column(s) identifying one series. Required whenever the table
            holds more than one, or lags cross from one series into another.
    """

    __slots__ = ("columns", "lags", "order_by", "partition_by")

    def __init__(
        self,
        columns: str | Sequence[str],
        *,
        order_by: str,
        lags: Sequence[int] = (1,),
        partition_by: str | Sequence[str] | None = None,
    ) -> None:
        self.columns = columns_arg(columns, what="LagFeaturizer")
        self.order_by = order_by
        steps = list(lags)
        if not steps or any(n < 1 for n in steps):
            raise PlanError(f"lags must be positive row counts, got {list(lags)}")
        self.lags = steps
        self.partition_by = _check_partition(partition_by)

    def transform(self, ds: Dataset) -> Dataset:
        """Append one ``{column}_lag_{n}`` column per column and lag.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import LagFeaturizer
                >>> ds = bt.from_pydict({"t": [1, 2], "v": [5.0, 6.0]})
                >>> LagFeaturizer("v", order_by="t").transform(ds).columns
                ['t', 'v', 'v_lag_1']

        Args:
            ds: The dataset to extend.

        Returns:
            A new lazy `Dataset` with the lag columns appended.
        """
        projections = {
            f"{name}_lag_{n}": lag(col(name), n).over(
                partition_by=list(self.partition_by), order_by=[self.order_by]
            )
            for name in self.columns
            for n in self.lags
        }
        return ds.with_columns(**projections)


class RollingFeaturizer(Preprocessor):
    """Append a rolling aggregate over the `window` rows **before** each row.

    The window deliberately ends at the previous row. A rolling mean that includes the
    current row contains the target's own value, which is the single most common leak in a
    forecasting pipeline: the cross-validated error looks excellent, and production does not
    reproduce any of it. There is no option to include the current row, because there is no
    correct reason to.

    A rolling mean is what lets a model see level and trend without being given the raw
    timestamp; a rolling max or count is what surfaces a recent spike.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import RollingFeaturizer
            >>> ds = bt.from_pydict({"t": [1, 2, 3], "v": [10.0, 20.0, 60.0]})
            >>> pre = RollingFeaturizer("v", order_by="t", window=2, aggregates=["mean"])
            >>> pre.fit_transform(ds).sort("t").to_pydict()["v_rolling_mean_2"]
            [None, 10.0, 15.0]

    Args:
        columns: The columns to aggregate.
        order_by: The column defining time order within a series.
        window: How many preceding rows the window covers.
        aggregates: Which aggregates to compute; see `ROLLING_AGGREGATES`.
        partition_by: The column(s) identifying one series. Required whenever the table
            holds more than one, or the window reaches across series.
    """

    __slots__ = ("aggregates", "columns", "order_by", "partition_by", "window")

    def __init__(
        self,
        columns: str | Sequence[str],
        *,
        order_by: str,
        window: int = 7,
        aggregates: Sequence[str] = ("mean",),
        partition_by: str | Sequence[str] | None = None,
    ) -> None:
        self.columns = columns_arg(columns, what="RollingFeaturizer")
        self.order_by = order_by
        if window < 1:
            raise PlanError(f"window must be at least 1 row, got {window}")
        self.window = window
        names = list(aggregates)
        for name in names:
            if name not in ROLLING_AGGREGATES:
                from batcher._internal.errors import suggestion

                hint = suggestion(name, ROLLING_AGGREGATES)
                tail = f" {hint}" if hint else ""
                raise PlanError(
                    f"unknown rolling aggregate {name!r}; expected one of "
                    f"{sorted(ROLLING_AGGREGATES)}.{tail}"
                )
        if not names:
            raise PlanError("RollingFeaturizer needs at least one aggregate")
        self.aggregates = names
        self.partition_by = _check_partition(partition_by)

    def transform(self, ds: Dataset) -> Dataset:
        """Append one ``{column}_rolling_{agg}_{window}`` column per column and aggregate.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import RollingFeaturizer
                >>> ds = bt.from_pydict({"t": [1, 2], "v": [5.0, 6.0]})
                >>> RollingFeaturizer("v", order_by="t", window=1).transform(ds).columns
                ['t', 'v', 'v_rolling_mean_1']

        Args:
            ds: The dataset to extend.

        Returns:
            A new lazy `Dataset` with the rolling columns appended.
        """
        import batcher as bt

        builders = {
            "mean": bt.mean,
            "sum": bt.sum,
            "min": bt.min,
            "max": bt.max,
            "count": lambda e: e.count(),
        }
        # `(-window, -1)` is the frame ending one row *before* the current one, which is the
        # whole point: a window ending at 0 would put the target inside its own feature.
        frame = (-self.window, -1)
        projections = {}
        for name in self.columns:
            for aggregate in self.aggregates:
                projections[f"{name}_rolling_{aggregate}_{self.window}"] = builders[aggregate](
                    col(name)
                ).over(
                    partition_by=list(self.partition_by),
                    order_by=[self.order_by],
                    frame=frame,
                )
        return ds.with_columns(**projections)
