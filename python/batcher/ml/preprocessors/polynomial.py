"""Polynomial and interaction feature expansion (stateless).

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
from batcher.ml.preprocessors.base import Preprocessor, columns_arg
from batcher.plan.expr_ir import Expr, col, lit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["PolynomialFeatures"]

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
