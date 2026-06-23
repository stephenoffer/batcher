"""The `Preprocessor` contract and `Chain` — sklearn-style fit/transform on a Dataset.

A preprocessor learns state from a dataset (`fit`, which *executes* a small aggregate
— the measure step, like `describe`) and then applies a **lazy** column rewrite
(`transform`, which returns a new `Dataset` and runs no work until a terminal op).
The fitted state lives on the object, so you fit on the training set and `transform`
the validation/test set with the *same* statistics — the reason a preprocessor is an
object, not a `Dataset` method.

The win is that `fit` lowers to the existing relational aggregates (`mean`, `min`,
`max`, `median`, `distinct`) and `transform` to ordinary `Expr` projections — so the
whole path is mergeable, distributed, and spillable for free, with no per-row Python.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset
    from batcher.plan.expr_ir import Expr

__all__ = ["Chain", "Preprocessor"]


def fit_aggregate(ds: Dataset, aggs: dict[str, Expr]) -> dict[str, Any]:
    """Run a single global aggregate and return its one row as ``{name: scalar}``.

    The shared `fit` primitive: every scaler/imputer learns its statistics in one
    mergeable pass over the data (the same engine path as `describe`), then reads the
    scalars back to the driver as plain Python values.
    """
    row = ds.agg(**aggs).collect()
    return {name: row.column(name)[0].as_py() for name in row.column_names}


def distinct_values(ds: Dataset, column: str) -> list[Any]:
    """The sorted, non-null distinct values of `column` (an encoder's categories).

    Executes a `distinct` over the single column and reads the values to the driver —
    the `fit` step for categorical encoders. Nulls are dropped (they map to the
    encoder's unknown value at transform time).
    """
    values = ds.select(column).distinct().collect().column(column).to_pylist()
    return sorted(v for v in values if v is not None)


class Preprocessor(abc.ABC):
    """A stateful column transform with a `fit` / `transform` / `fit_transform` API.

    Subclasses implement `fit` (learn state, return ``self``) and `transform`
    (return a new lazy `Dataset`). `fit` executes; `transform` stays lazy.
    """

    _fitted: bool = False

    def fit(self, ds: Dataset) -> Preprocessor:
        """Learn this preprocessor's state from `ds` and return ``self`` (fitted).

        The default is the stateless case: there is nothing to learn, so it just marks
        the preprocessor fitted. Stateful preprocessors (scalers, encoders, imputers)
        override this to run their aggregate.
        """
        _ = ds  # stateless default — no statistics to learn
        self._fitted = True
        return self

    @abc.abstractmethod
    def transform(self, ds: Dataset) -> Dataset:
        """Apply the fitted transform to `ds`, returning a new lazy `Dataset`."""

    def fit_transform(self, ds: Dataset) -> Dataset:
        """`fit(ds)` then `transform(ds)` — the common single-dataset path."""
        return self.fit(ds).transform(ds)

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise PlanError(
                f"{type(self).__name__} must be fitted before transform(); "
                "call fit(ds) or fit_transform(ds) first"
            )


class Chain(Preprocessor):
    """Compose preprocessors into one pipeline, applied left to right.

    Each step is fitted on the output of the previous step (so a scaler downstream of
    an imputer sees imputed values), mirroring scikit-learn's ``Pipeline``.
    """

    __slots__ = ("steps",)

    def __init__(self, steps: Sequence[Preprocessor]) -> None:
        self.steps = list(steps)
        if not self.steps:
            raise PlanError("Chain() requires at least one preprocessor")

    def fit(self, ds: Dataset) -> Chain:
        cur = ds
        for step in self.steps:
            cur = step.fit_transform(cur)
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        self._require_fitted()
        cur = ds
        for step in self.steps:
            cur = step.transform(cur)
        return cur
