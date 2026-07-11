"""The `Preprocessor` contract — sklearn-style fit/transform on a Dataset.

A preprocessor learns state from a dataset (`fit`, which *executes* a small aggregate
— the measure step, like `describe`) and then applies a **lazy** column rewrite
(`transform`, which returns a new `Dataset` and runs no work until a terminal op).
The fitted state lives on the object, so you fit on the training set and `transform`
the validation/test set with the *same* statistics — the reason a preprocessor is an
object, not a `Dataset` method.

The win is that `fit` lowers to the existing relational aggregates (`mean`, `min`,
`max`, `median`, `distinct`) and `transform` to ordinary `Expr` projections — so the
whole path is mergeable, distributed, and spillable for free, with no per-row Python.
Compose preprocessors by sequencing them: ``fit_transform`` the first on train, feed
its output to the next, then ``transform`` the test set through the same fitted
objects — the same chaining the `Dataset` builder already gives every other transform.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset
    from batcher.plan.expr_ir import Expr

__all__ = ["Preprocessor"]


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

    Subclasses implement `fit` (learn state from a dataset, return ``self``) and
    `transform` (return a new lazy `Dataset` that applies the learned rewrite). `fit`
    executes a small mergeable aggregate; `transform` stays lazy and adds only `Expr`
    projections, so it runs no work until a terminal op. Fit on the training split and
    `transform` the held-out split with the *same* statistics.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import StandardScaler
            >>> train = bt.from_pydict({"x": [1.0, 3.0]})
            >>> pre = StandardScaler(["x"]).fit(train)
            >>> pre.transform(train).to_pydict()
            {'x': [-1.0, 1.0]}
    """

    _fitted: bool = False

    def fit(self, ds: Dataset) -> Preprocessor:
        """Learn this preprocessor's state from `ds` and return ``self`` (fitted).

        The default is the stateless case: there is nothing to learn, so it just marks
        the preprocessor fitted. Stateful preprocessors (scalers, encoders, imputers)
        override this to run their aggregate over `ds`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import StandardScaler
                >>> pre = StandardScaler(["x"]).fit(bt.from_pydict({"x": [1.0, 3.0]}))
                >>> pre.mean_, pre.scale_
                ({'x': 2.0}, {'x': 1.0})

        Args:
            ds: The dataset to learn the statistics from (the training split).

        Returns:
            ``self``, marked fitted, so `fit` chains straight into `transform`.
        """
        _ = ds  # stateless default — no statistics to learn
        self._fitted = True
        return self

    @abc.abstractmethod
    def transform(self, ds: Dataset) -> Dataset:
        """Apply the fitted transform to `ds`, returning a new lazy `Dataset`.

        Each subclass contributes `Expr` projections (via `with_columns` / `select`),
        so the returned dataset is lazy and runs no work until a terminal op. Must be
        called after `fit` (or `fit_transform`).

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import StandardScaler
                >>> pre = StandardScaler(["x"]).fit(bt.from_pydict({"x": [1.0, 3.0]}))
                >>> pre.transform(bt.from_pydict({"x": [2.0, 4.0]})).to_pydict()
                {'x': [0.0, 2.0]}

        Args:
            ds: The dataset to rewrite (may differ from the one `fit` saw).

        Returns:
            A new lazy `Dataset` with the fitted transform applied.
        """

    def fit_transform(self, ds: Dataset) -> Dataset:
        """`fit(ds)` then `transform(ds)` — the common single-dataset path.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import StandardScaler
                >>> ds = bt.from_pydict({"x": [1.0, 3.0]})
                >>> StandardScaler(["x"]).fit_transform(ds).to_pydict()
                {'x': [-1.0, 1.0]}

        Args:
            ds: The dataset to fit on and then transform.

        Returns:
            A new lazy `Dataset` with the just-fitted transform applied.
        """
        return self.fit(ds).transform(ds)

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise PlanError(
                f"{type(self).__name__} must be fitted before transform(); "
                "call fit(ds) or fit_transform(ds) first"
            )
