"""Binning / discretization preprocessors.

`KBinsDiscretizer` learns bin edges in `fit` (min/max for ``"uniform"``, or quantiles
for ``"quantile"``, both one mergeable aggregate) and maps each value to its integer
bin index in `transform` via a `CASE` chain — no per-row Python.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.ml.preprocessors.base import Preprocessor, fit_aggregate
from batcher.plan.expr_ir import col, when

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["KBinsDiscretizer"]


class KBinsDiscretizer(Preprocessor):
    """Bin continuous columns into ``n_bins`` integer bins (sklearn
    ``KBinsDiscretizer`` with ``encode="ordinal"``).

    ``strategy="quantile"`` (default) makes each bin hold roughly equal counts (edges
    are the quantiles); ``"uniform"`` makes equal-width bins (edges from min/max). The
    output column replaces the input with its bin index ``0..n_bins-1``.

    Args:
        columns: the numeric columns to discretize (replaced in place).
        n_bins: the number of bins (>= 2).
        strategy: ``"quantile"`` or ``"uniform"``.
    """

    __slots__ = ("columns", "edges_", "n_bins", "strategy")

    def __init__(
        self, columns: Sequence[str], *, n_bins: int = 5, strategy: str = "quantile"
    ) -> None:
        self.columns = list(columns)
        if not self.columns:
            raise PlanError("KBinsDiscretizer requires at least one column")
        if n_bins < 2:
            raise PlanError(f"n_bins must be >= 2, got {n_bins}")
        if strategy not in ("quantile", "uniform"):
            raise PlanError(f"strategy must be 'quantile' or 'uniform', got {strategy!r}")
        self.n_bins = n_bins
        self.strategy = strategy
        # Per column: the n_bins-1 inner edges separating the bins.
        self.edges_: dict[str, list[float]] = {}

    def fit(self, ds: Dataset) -> KBinsDiscretizer:
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
        self._require_fitted()
        new = {}
        for c in self.columns:
            edges = self.edges_[c]
            # Bin index = how many edges the value meets or exceeds (first match wins).
            expr = self.n_bins - 1
            for i in range(len(edges) - 1, -1, -1):
                expr = when(col(c) < edges[i]).then(i).otherwise(expr)
            new[c] = expr
        return ds.with_columns(**new)
