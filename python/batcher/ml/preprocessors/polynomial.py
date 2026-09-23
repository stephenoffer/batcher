"""Basis expansion — polynomial/interaction terms, and B-splines.

`PolynomialFeatures` adds the products of the input columns up to a given degree — the
interaction and power terms (``a*b``, ``a^2``) that let a linear model fit curvature and
feature crosses. It is the scikit-learn ``PolynomialFeatures`` transform, but every term
is an ordinary `Expr` product the engine evaluates column-wise: no per-row Python, and the
whole expansion stays mergeable and distributable. `fit` is a no-op (the term set is fixed
by the column list and `degree`); the degree-1 terms are the original columns, already in
the frame, so only the degree-≥2 terms are added.
"""

from __future__ import annotations

from functools import reduce
from itertools import combinations_with_replacement
from operator import mul
from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.ml.preprocessors.base import (
    Preprocessor,
    columns_arg,
    fit_aggregate,
    nan_as_null,
)
from batcher.plan.expr_ir import Expr, col, lit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["PolynomialFeatures", "SplineTransformer"]

# The default ceiling on how many terms an expansion may add. Every term is a separate
# projection the engine evaluates over every row, so the term count is the width of the
# output *and* the per-row work. 1,000 is generous — it admits degree 2 over 44 columns
# and degree 3 over 17 — while still catching the combinatorial blowups.
MAX_TERMS = 1000


def _poly_name(combo: tuple[str, ...]) -> str:
    """Name a monomial, collapsing repeated factors to powers: ``(a, a, b)`` → ``a^2*b``."""
    parts = []
    for name in dict.fromkeys(combo):  # first-seen order, deduplicated
        power = combo.count(name)
        parts.append(name if power == 1 else f"{name}^{power}")
    return "*".join(parts)


class PolynomialFeatures(Preprocessor):
    """Add interaction and power terms of `columns` up to `degree` (scikit-learn style).

    A stateless transform (``fit`` is a no-op). For columns ``[a, b]`` and ``degree=2`` it
    adds ``a^2``, ``a*b``, ``b^2``; the degree-1 terms are the original columns and are left
    as they are. `interaction_only` keeps only cross terms (no ``a^2``); `include_bias` adds
    a constant ``bias`` column of 1.0. Each term is an `Expr` product, so the expansion is
    evaluated in the engine with no per-row Python.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import PolynomialFeatures
            >>> ds = bt.from_pydict({"a": [2.0], "b": [3.0]})
            >>> PolynomialFeatures(["a", "b"], degree=2).fit_transform(ds).to_pydict()
            {'a': [2.0], 'b': [3.0], 'a^2': [4.0], 'a*b': [6.0], 'b^2': [9.0]}

    Args:
        columns: the numeric columns to expand, in order.
        degree: the maximum total degree of the added terms (must be ≥ 2).
        interaction_only: keep only distinct-factor cross terms (drop pure powers) when True.
        include_bias: add a constant ``bias`` column of 1.0 when True.
        max_terms: the ceiling on how many terms the expansion may add. The term count is
            combinatorial in ``degree`` and the column count (degree 3 over 20 columns is
            1,540 new columns), so this fails an accidental explosion instead of silently
            building it.
    """

    numeric_only = True

    __slots__ = ("columns", "degree", "include_bias", "interaction_only", "max_terms")

    def __init__(
        self,
        columns: str | Sequence[str],
        *,
        degree: int = 2,
        interaction_only: bool = False,
        include_bias: bool = False,
        max_terms: int = MAX_TERMS,
    ) -> None:
        self.columns = columns_arg(columns, what="PolynomialFeatures")
        if not self.columns:
            raise PlanError("PolynomialFeatures requires at least one column")
        if degree < 2:
            raise PlanError(
                "PolynomialFeatures degree must be >= 2 (degree-1 terms are the input columns)"
            )
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.max_terms = max_terms

    def transform(self, ds: Dataset) -> Dataset:
        """Add the degree-2..`degree` product terms of the fitted columns.

        The original columns pass through unchanged; `include_bias` adds a ``bias`` column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import PolynomialFeatures
                >>> ds = bt.from_pydict({"a": [2.0], "b": [3.0]})
                >>> pre = PolynomialFeatures(["a", "b"], degree=2, interaction_only=True)
                >>> pre.fit_transform(ds).to_pydict()
                {'a': [2.0], 'b': [3.0], 'a*b': [6.0]}

        Args:
            ds: The dataset whose columns to expand.

        Returns:
            A new lazy `Dataset` with the interaction/power columns added.

        Raises:
            PlanError: If the expansion would add more than `max_terms` terms.
        """
        self._require_fitted()
        combos = [
            combo
            for total_degree in range(2, self.degree + 1)
            for combo in combinations_with_replacement(self.columns, total_degree)
            if not (self.interaction_only and len(set(combo)) != len(combo))
        ]
        if len(combos) > self.max_terms:
            raise PlanError(
                f"PolynomialFeatures: degree={self.degree} over {len(self.columns)} columns "
                f"expands to {len(combos)} terms, above max_terms={self.max_terms}. Each term "
                f"is another projection evaluated over every row. Lower the degree, pass "
                f"interaction_only=True, expand fewer columns, or raise max_terms to accept "
                f"the cost."
            )
        new: dict[str, Expr] = {}
        if self.include_bias:
            new["bias"] = lit(1.0)
        for combo in combos:
            new[_poly_name(combo)] = reduce(mul, (col(c) for c in combo))
        return ds.with_columns(**new)


class SplineTransformer(Preprocessor):
    """Expand each column into a B-spline basis, so a linear model can fit a curve.

    The alternative to `PolynomialFeatures` when the relationship bends but is not
    polynomial. A degree-3 polynomial fits curvature by making the whole column's shape one
    cubic, which oscillates at the edges and lets a point at one end of the range move the
    fit at the other. A spline basis is *local*: each basis function is non-zero over a few
    knots only, so a wiggle in one region stays in that region. This is the standard way to
    give a generalized additive model its smooth terms.

    The basis is scikit-learn's ``SplineTransformer`` default: `n_knots` knots spaced
    evenly across the training range (``knots="uniform"``), extended by `degree` more knots
    at each end at the same spacing, and ``extrapolation="constant"``, so a value outside
    the training range gets the basis value at the nearest boundary rather than an all-zero
    row. That makes ``n_knots + degree - 1`` output columns per input, the same count
    scikit-learn emits with ``include_bias=True``. ``knots="quantile"`` places the knots at
    the column's exact percentiles instead, so they follow the data's density; that
    placement is what `fit` learns, in one aggregate. Nulls stay null in every basis
    column, and there is no ``inverse_transform``.

    Each output column ``<name>_sp0 … <name>_sp{n}`` is an ordinary `Expr` over the source,
    so the expansion stays lazy, streams, and distributes.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import SplineTransformer
            >>> ds = bt.from_pydict({"x": [0.0, 1.0, 2.0, 3.0, 4.0]})
            >>> out = SplineTransformer("x", n_knots=3, degree=1).fit_transform(ds)
            >>> [c for c in out.columns if c.startswith("x_sp")]
            ['x_sp0', 'x_sp1', 'x_sp2']

    Args:
        columns: The numeric columns to expand.
        n_knots: How many knots to place. More knots means a more flexible curve and more
            output columns.
        degree: The spline degree; 1 is piecewise linear, 3 the usual cubic.
        knots: ``"uniform"`` (the default, as in scikit-learn) to space knots evenly across
            the observed range, or ``"quantile"`` to place them at the column's percentiles.
        drop_original: Remove the source columns after expanding them.
    """

    numeric_only = True

    __slots__ = ("columns", "degree", "drop_original", "knots", "knots_", "n_knots")

    def __init__(
        self,
        columns: str | Sequence[str],
        *,
        n_knots: int = 5,
        degree: int = 3,
        knots: str = "uniform",
        drop_original: bool = False,
    ) -> None:
        self.columns = columns_arg(columns, what="SplineTransformer")
        if n_knots < 2:
            raise PlanError(f"SplineTransformer: n_knots must be at least 2, got {n_knots}")
        if degree < 1:
            raise PlanError(f"SplineTransformer: degree must be at least 1, got {degree}")
        if knots not in ("quantile", "uniform"):
            raise PlanError(
                f"SplineTransformer: knots must be 'quantile' or 'uniform', got {knots!r}"
            )
        self.n_knots = n_knots
        self.degree = degree
        self.knots = knots
        self.drop_original = drop_original
        self.knots_: dict[str, list[float]] = {}

    def fit(self, ds: Dataset) -> SplineTransformer:
        """Learn each column's knot positions with one aggregate.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import SplineTransformer
                >>> ds = bt.from_pydict({"x": [0.0, 1.0, 2.0, 3.0, 4.0]})
                >>> SplineTransformer("x", n_knots=3, knots="uniform").fit(ds).knots_["x"]
                [0.0, 2.0, 4.0]

        Args:
            ds: The dataset to learn the knot positions from.

        Returns:
            ``self``, fitted.

        Raises:
            PlanError: If a column is entirely null, so no knots can be placed.
        """
        self._check_numeric(ds)
        ds = nan_as_null(ds, self.columns)
        aggs: dict[str, Expr] = {}
        for name in self.columns:
            if self.knots == "uniform":
                aggs[f"{name}__lo"] = col(name).min()
                aggs[f"{name}__hi"] = col(name).max()
            else:
                for i in range(self.n_knots):
                    fraction = i / (self.n_knots - 1)
                    aggs[f"{name}__k{i}"] = col(name).quantile(fraction)
        read = fit_aggregate(ds, aggs)
        for name in self.columns:
            if self.knots == "uniform":
                low, high = read[f"{name}__lo"], read[f"{name}__hi"]
                if low is None or high is None:
                    raise PlanError(
                        f"SplineTransformer: column {name!r} has no non-null values, so there "
                        "is nothing to place knots between."
                    )
                span = (float(high) - float(low)) or 1.0
                positions = [
                    float(low) + span * i / (self.n_knots - 1) for i in range(self.n_knots)
                ]
            else:
                positions = [read[f"{name}__k{i}"] for i in range(self.n_knots)]
                if any(p is None for p in positions):
                    raise PlanError(
                        f"SplineTransformer: column {name!r} has no non-null values, so there "
                        "is nothing to place knots between."
                    )
                positions = [float(p) for p in positions]
            self.knots_[name] = _distinct_knots(positions)
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Append the spline basis columns for each fitted column.

        A value outside the training range is clamped to it first, which is scikit-learn's
        ``extrapolation="constant"``: the row repeats the boundary's basis values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import SplineTransformer
                >>> ds = bt.from_pydict({"x": [0.0, 1.0, 2.0]})
                >>> pre = SplineTransformer("x", n_knots=3, degree=1).fit(ds)
                >>> pre.transform(bt.from_pydict({"x": [1.0, 9.0]})).to_pydict()["x_sp1"]
                [1.0, 0.0]

        Args:
            ds: The dataset to expand.

        Returns:
            A new lazy `Dataset` with the basis columns appended.
        """
        self._require_fitted()
        new: dict[str, Expr] = {}
        for name in self.columns:
            for index, basis in enumerate(_basis_expressions(name, self.knots_[name], self.degree)):
                new[f"{name}_sp{index}"] = basis
        out = ds.with_columns(**new)
        return out.drop(*self.columns) if self.drop_original else out


def _distinct_knots(positions: list[float]) -> list[float]:
    """Deduplicate knot positions, keeping order.

    A quantile placement on a column with a heavy point mass — a zero-inflated amount, a
    mostly-constant flag — returns the same value several times over. Left in, the repeated
    knots make consecutive basis functions identical, so the expansion produces perfectly
    collinear columns and any linear fit over them is singular.
    """
    seen: list[float] = []
    for value in positions:
        if not seen or value > seen[-1]:
            seen.append(value)
    if len(seen) == 1:
        seen.append(seen[0] + 1.0)
    return seen


def _basis_expressions(name: str, knots: list[float], degree: int) -> list[Expr]:
    """The B-spline basis over `knots`, as one expression per basis function.

    Built by the Cox-de Boor recursion, which defines a degree-`d` basis function as a
    weighted sum of two degree-``d-1`` ones. Running the recursion at *plan* time rather
    than per row means the result is a plain arithmetic tree over the column, so the engine
    evaluates it column-wise and the JIT can compile it.

    The knot vector is scikit-learn's: the fitted knots extended by `degree` more at each end,
    spaced like the first and last fitted interval. A clamped vector (the boundary knot
    repeated) gives a different basis near the edges. The column is clamped into the fitted
    range before the recursion, which is scikit-learn's ``extrapolation="constant"``; without
    it a value past the last knot fell outside every interval and its whole row came back
    zero.
    """
    low_step = knots[1] - knots[0]
    high_step = knots[-1] - knots[-2]
    padded = (
        [knots[0] - low_step * (degree - i) for i in range(degree)]
        + list(knots)
        + [knots[-1] + high_step * (i + 1) for i in range(degree)]
    )
    column = col(name).cast("float64").clip(lit(knots[0]), lit(knots[-1]))
    # Degree 0: an indicator per interval, each half-open on the right so a value on a knot
    # belongs to exactly one. The clamped maximum sits on the last fitted knot and so falls in
    # the first extension interval, which exists because `degree >= 1`.
    current: list[Expr] = [
        _as_float((column >= lit(padded[i])) & (column < lit(padded[i + 1])))
        for i in range(len(padded) - 1)
    ]
    for d in range(1, degree + 1):
        nxt: list[Expr] = []
        for i in range(len(current) - 1):
            left_span = padded[i + d] - padded[i]
            right_span = padded[i + d + 1] - padded[i + 1]
            left = (column - lit(padded[i])) / lit(left_span) * current[i]
            right = (lit(padded[i + d + 1]) - column) / lit(right_span) * current[i + 1]
            nxt.append(left + right)
        current = nxt
    return current


def _as_float(expression: Expr) -> Expr:
    """Cast a boolean indicator to a float, so the recursion arithmetic is numeric."""
    return expression.cast("float64")
