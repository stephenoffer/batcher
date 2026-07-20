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
from batcher.ml.preprocessors.base import Preprocessor
from batcher.plan.expr_ir import Expr, col, lit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["PolynomialFeatures"]


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
    """

    __slots__ = ("columns", "degree", "include_bias", "interaction_only")

    def __init__(
        self,
        columns: Sequence[str],
        *,
        degree: int = 2,
        interaction_only: bool = False,
        include_bias: bool = False,
    ) -> None:
        self.columns = list(columns)
        if not self.columns:
            raise PlanError("PolynomialFeatures requires at least one column")
        if degree < 2:
            raise PlanError(
                "PolynomialFeatures degree must be >= 2 (degree-1 terms are the input columns)"
            )
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias

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
        """
        self._require_fitted()
        new: dict[str, Expr] = {}
        if self.include_bias:
            new["bias"] = lit(1.0)
        for total_degree in range(2, self.degree + 1):
            for combo in combinations_with_replacement(self.columns, total_degree):
                if self.interaction_only and len(set(combo)) != len(combo):
                    continue
                new[_poly_name(combo)] = reduce(mul, (col(c) for c in combo))
        return ds.with_columns(**new)
