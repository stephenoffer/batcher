"""Binning / discretization preprocessors.

`KBinsDiscretizer` learns bin edges in `fit` (min/max for ``"uniform"``, or quantiles
for ``"quantile"``, both one mergeable aggregate) and maps each value to its integer
bin index in `transform` via a `CASE` chain — no per-row Python.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.ml.preprocessors.base import (
    Preprocessor,
    columns_arg,
    fit_aggregate,
    nan_as_null,
    output_columns_arg,
    output_pairs,
)
from batcher.plan.expr_ir import col, when

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from batcher.api.dataset import Dataset

__all__ = ["KBinsDiscretizer"]

# The ceiling on `n_bins`. A quantile fit builds one sketch per inner edge and the
# transform builds one CASE arm per inner edge, so an unbounded `n_bins` is an unbounded
# fit cost and an unbounded plan. 256 covers every realistic discretization.
MAX_BINS = 256

#: What `duplicates=` accepts. ``"keep"`` leaves a repeated edge in place, so the bin between
#: the two copies is empty and its index is skipped. ``"drop"`` removes the repeat and numbers
#: the surviving bins consecutively, and ``"raise"`` refuses the fit. The last two are pandas'
#: ``cut(duplicates=)``, which Ray Data's discretizers pass straight through.
_DUPLICATES = ("keep", "drop", "raise")


def _check_bins(n_bins: int) -> int:
    """Validate one bin count against the ``[2, MAX_BINS]`` range."""
    if isinstance(n_bins, bool) or not isinstance(n_bins, int):
        raise PlanError(f"n_bins must be an integer, got {n_bins!r}")
    if n_bins < 2:
        raise PlanError(f"n_bins must be >= 2, got {n_bins}")
    if n_bins > MAX_BINS:
        raise PlanError(
            f"n_bins must be <= {MAX_BINS}, got {n_bins}. Each bin edge is a CASE arm in "
            f"the transform, and on the 'quantile' strategy also its own sketch in the "
            f"fit, so both the plan and the fit cost grow with n_bins."
        )
    return n_bins


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

            >>> # Ray Data's right-closed bins, written beside the input.
            >>> kb = KBinsDiscretizer(
            ...     "v", n_bins=2, strategy="uniform", right=True, output_columns="v_bin"
            ... )
            >>> kb.fit_transform(bt.from_pydict({"v": [0.0, 5.0, 10.0]})).to_pydict()
            {'v': [0.0, 5.0, 10.0], 'v_bin': [0, 0, 1]}

    Args:
        columns: the numeric columns to discretize (replaced in place).
        n_bins: the number of bins (>= 2), or a mapping from each column to its own count.
        strategy: ``"quantile"`` or ``"uniform"``.
        right: which bin a value lying exactly on an inner edge joins. ``False`` (the
            default, scikit-learn's reading) puts it in the upper bin, so bins are
            ``[lo, hi)``; ``True`` puts it in the lower one, so bins are ``(lo, hi]``, as
            pandas' ``cut`` and Ray Data's ``UniformKBinsDiscretizer`` do.
        duplicates: what to do when two learned edges coincide: ``"keep"`` (the default)
            leaves the empty bin and skips its index, ``"drop"`` removes the repeated edge
            and numbers the remaining bins consecutively, ``"raise"`` refuses the fit.
        output_columns: write each bin-index column to this name instead of over its
            input, one name per column in order, keeping the inputs (Ray Data's
            ``output_columns``). ``None`` (the default) replaces the columns in place.
    """

    numeric_only = True

    __slots__ = ("columns", "duplicates", "edges_", "n_bins", "output_columns", "right", "strategy")

    def __init__(
        self,
        columns: str | Sequence[str],
        *,
        n_bins: int | Mapping[str, int] = 5,
        strategy: str = "quantile",
        right: bool = False,
        duplicates: str = "keep",
        output_columns: str | Sequence[str] | None = None,
    ) -> None:
        self.columns = columns_arg(columns, what="KBinsDiscretizer")
        if isinstance(n_bins, int):
            _check_bins(n_bins)
        else:
            n_bins = {str(k): _check_bins(v) for k, v in dict(n_bins).items()}
            missing = [c for c in self.columns if c not in n_bins]
            if missing:
                raise PlanError(
                    f"KBinsDiscretizer: n_bins is a mapping but has no entry for {missing!r}; "
                    "give every column its bin count"
                )
        if strategy not in ("quantile", "uniform"):
            raise PlanError(f"strategy must be 'quantile' or 'uniform', got {strategy!r}")
        if duplicates not in _DUPLICATES:
            raise PlanError(f"duplicates must be one of {_DUPLICATES}, got {duplicates!r}")
        self.n_bins = n_bins
        self.strategy = strategy
        self.right = bool(right)
        self.duplicates = duplicates
        self.output_columns = output_columns_arg(
            self.columns, output_columns, what="KBinsDiscretizer"
        )
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
        ds = nan_as_null(ds, self.columns)
        if self.strategy == "uniform":
            aggs = {}
            for c in self.columns:
                aggs[f"{c}__min"] = col(c).min()
                aggs[f"{c}__max"] = col(c).max()
            cell = fit_aggregate(ds, aggs)
            for c in self.columns:
                bins = self._bins(c)
                lo = float(cell[f"{c}__min"] or 0.0)
                hi = float(cell[f"{c}__max"] or 0.0)
                width = (hi - lo) / bins
                self.edges_[c] = self._dedupe(c, [lo + width * (i + 1) for i in range(bins - 1)])
        else:  # quantile
            aggs = {}
            for c in self.columns:
                bins = self._bins(c)
                for i in range(bins - 1):
                    aggs[f"{c}__q{i}"] = col(c).approx_quantile((i + 1) / bins)
            cell = fit_aggregate(ds, aggs)
            for c in self.columns:
                edges = [float(cell[f"{c}__q{i}"] or 0.0) for i in range(self._bins(c) - 1)]
                self.edges_[c] = self._dedupe(c, edges)
        self._fitted = True
        return self

    def _bins(self, column: str) -> int:
        """The bin count for `column`, from the shared count or the per-column mapping."""
        return self.n_bins if isinstance(self.n_bins, int) else self.n_bins[column]

    def _dedupe(self, column: str, edges: list[float]) -> list[float]:
        """Apply the `duplicates` policy to one column's sorted inner edges."""
        if self.duplicates == "keep" or len(set(edges)) == len(edges):
            return edges
        if self.duplicates == "raise":
            raise PlanError(
                f"KBinsDiscretizer: column {column!r} learned repeated bin edges {edges!r}, "
                "so some bins are empty. Pass duplicates='drop' to merge them, or fewer n_bins."
            )
        return sorted(set(edges))

    def transform(self, ds: Dataset) -> Dataset:
        """Replace each fitted column with its integer bin index ``0..n_bins-1``.

        The index is how many learned edges the value meets or exceeds (only exceeds, with
        ``right=True``), computed by a `CASE` chain. A null stays null, and does not become a bin.

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
        for c, out in output_pairs(self.columns, self.output_columns):
            edges = self.edges_[c]
            # Bin index = how many edges the value passes (first match wins). Left-closed
            # bins count an edge the value meets; right-closed ones only an edge it exceeds.
            expr = len(edges)
            for i in range(len(edges) - 1, -1, -1):
                below = col(c) <= edges[i] if self.right else col(c) < edges[i]
                expr = when(below).then(i).otherwise(expr)
            # A null compares false against every edge, so the CASE chain fell all the way
            # through to the `otherwise` and binned every missing value into the TOP bin.
            # Nothing errored and nothing warned: a model then trained on fabricated values
            # sitting at one end of the feature's range, which is the worst place to put
            # them. Every other preprocessor here leaves a null alone, and so does sklearn.
            # The `then` arm is reached only where the value IS null, so casting that null
            # to the bin type yields a null of the right type. The IR has no null literal,
            # and this needs none.
            new[out] = when(col(c).is_null()).then(col(c).cast("int64")).otherwise(expr)
        return ds.with_columns(**new)
