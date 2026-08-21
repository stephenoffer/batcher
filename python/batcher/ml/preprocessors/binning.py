"""Binning / discretization preprocessors.

`KBinsDiscretizer` learns bin edges in `fit` (min/max for ``"uniform"``, or quantiles
for ``"quantile"``, both one mergeable aggregate) and maps each value to its integer
bin index in `transform` via a `CASE` chain — no per-row Python.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.ml.preprocessors.base import Preprocessor, columns_arg, fit_aggregate
from batcher.plan.expr_ir import col, when

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["KBinsDiscretizer"]

# The ceiling on `n_bins`. A quantile fit builds one sketch per inner edge and the
# transform builds one CASE arm per inner edge, so an unbounded `n_bins` is an unbounded
# fit cost and an unbounded plan. 256 covers every realistic discretization.
MAX_BINS = 256


class KBinsDiscretizer(Preprocessor):
    """Bin continuous columns into ``n_bins`` integer bins (sklearn ``KBinsDiscretizer``).

    Matches ``encode="ordinal"``. ``strategy="quantile"`` (default) makes each bin hold
    roughly equal counts (edges are the quantiles); ``"uniform"`` makes equal-width
    bins (edges from min/max). The output column replaces the input with its bin index
    ``0..n_bins-1``.

    The two strategies do not agree with scikit-learn to the same degree, and it is worth
    knowing which one you are on. ``"uniform"`` is exact: its edges are the column's min
    and max. ``"quantile"`` places its edges with a **mergeable sketch**
    (`approx_quantile`) rather than the exact percentile `quantile` computes, so the edges
    sit near scikit-learn's without matching them, and a value close to an edge can land
    one bin either side of where ``KBinsDiscretizer`` puts it. On normal data that was
    0.2% of rows at ``n_bins=4`` and 7.6% at ``n_bins=25`` — the rate **rises** with
    `n_bins`, because the bins narrow while the per-edge error does not shrink with them.

    That is a deliberate trade, not an oversight, and it is the reason to leave it alone:
    the exact aggregate costs 4-9x more on the same fit (2M rows, 24 inner edges: 11.9 s
    exact against 1.3 s sketched), and it widens as edges are added. Reach for
    ``"uniform"`` when the edges must be exact, or `RobustScaler` / `QuantileTransformer`
    when you want exact percentiles and can pay for them.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import KBinsDiscretizer
            >>> ds = bt.from_pydict({"v": [0.0, 2.0, 6.0, 8.0, 10.0]})
            >>> KBinsDiscretizer(["v"], n_bins=2, strategy="uniform").fit_transform(ds).to_pydict()
            {'v': [0, 0, 1, 1, 1]}

    Args:
        columns: the numeric columns to discretize (replaced in place).
        n_bins: the number of bins (>= 2).
        strategy: ``"quantile"`` or ``"uniform"``.
    """

    numeric_only = True

    __slots__ = ("columns", "edges_", "n_bins", "strategy")

    def __init__(
        self, columns: str | Sequence[str], *, n_bins: int = 5, strategy: str = "quantile"
    ) -> None:
        self.columns = columns_arg(columns, what="KBinsDiscretizer")
        if n_bins < 2:
            raise PlanError(f"n_bins must be >= 2, got {n_bins}")
        if n_bins > MAX_BINS:
            raise PlanError(
                f"n_bins must be <= {MAX_BINS}, got {n_bins}. Each bin edge is a CASE arm in "
                f"the transform, and on the 'quantile' strategy also its own sketch in the "
                f"fit, so both the plan and the fit cost grow with n_bins."
            )
        if strategy not in ("quantile", "uniform"):
            raise PlanError(f"strategy must be 'quantile' or 'uniform', got {strategy!r}")
        self.n_bins = n_bins
        self.strategy = strategy
        # Per column: the n_bins-1 inner edges separating the bins.
        self.edges_: dict[str, list[float]] = {}

    def fit(self, ds: Dataset) -> KBinsDiscretizer:
        """Learn each column's ``n_bins - 1`` inner bin edges into `edges_`.

        For ``"uniform"`` the edges are equally spaced between min and max; for
        ``"quantile"`` they are the approximate quantiles — both one mergeable pass.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import KBinsDiscretizer
                >>> ds = bt.from_pydict({"v": [0.0, 2.0, 6.0, 8.0, 10.0]})
                >>> KBinsDiscretizer(["v"], n_bins=2, strategy="uniform").fit(ds).edges_
                {'v': [5.0]}

        Args:
            ds: The dataset to compute each column's bin edges from.

        Returns:
            ``self``, fitted.
        """
        self._check_numeric(ds)
        inner = self.n_bins - 1
        if self.strategy == "uniform":
            aggs = {}
            for c in self.columns:
                aggs[f"{c}__min"] = col(c).min()
                aggs[f"{c}__max"] = col(c).max()
            cell = fit_aggregate(ds, aggs)
            for c in self.columns:
                lo = float(cell[f"{c}__min"] or 0.0)
                hi = float(cell[f"{c}__max"] or 0.0)
                width = (hi - lo) / self.n_bins
                self.edges_[c] = [lo + width * (i + 1) for i in range(inner)]
        else:  # quantile
            aggs = {}
            for c in self.columns:
                for i in range(inner):
                    aggs[f"{c}__q{i}"] = col(c).approx_quantile((i + 1) / self.n_bins)
            cell = fit_aggregate(ds, aggs)
            for c in self.columns:
                self.edges_[c] = [float(cell[f"{c}__q{i}"] or 0.0) for i in range(inner)]
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Replace each fitted column with its integer bin index ``0..n_bins-1``.

        The index is how many learned edges the value meets or exceeds, computed by a
        `CASE` chain. A null stays null, and does not become a bin.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import KBinsDiscretizer
                >>> ds = bt.from_pydict({"v": [0.0, 2.0, 6.0, 8.0, 10.0]})
                >>> kb = KBinsDiscretizer(["v"], n_bins=2, strategy="uniform").fit(ds)
                >>> kb.transform(ds).to_pydict()
                {'v': [0, 0, 1, 1, 1]}

                >>> gaps = bt.from_pydict({"v": [0.0, None, 10.0]})
                >>> kb.transform(gaps).to_pydict()
                {'v': [0, None, 1]}

        Args:
            ds: The dataset to discretize.

        Returns:
            A new lazy `Dataset` with each fitted column replaced by its bin index.
        """
        self._require_fitted()
        new = {}
        for c in self.columns:
            edges = self.edges_[c]
            # Bin index = how many edges the value meets or exceeds (first match wins).
            expr = self.n_bins - 1
            for i in range(len(edges) - 1, -1, -1):
                expr = when(col(c) < edges[i]).then(i).otherwise(expr)
            # A null compares false against every edge, so the CASE chain fell all the way
            # through to the `otherwise` and binned every missing value into the TOP bin.
            # Nothing errored and nothing warned: a model then trained on fabricated values
            # sitting at one end of the feature's range, which is the worst place to put
            # them. Every other preprocessor here leaves a null alone, and so does sklearn.
            # The `then` arm is reached only where the value IS null, so casting that null
            # to the bin type yields a null of the right type. The IR has no null literal,
            # and this needs none.
            new[c] = when(col(c).is_null()).then(col(c).cast("int64")).otherwise(expr)
        return ds.with_columns(**new)
