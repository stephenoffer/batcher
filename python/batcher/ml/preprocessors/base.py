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
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset
    from batcher.plan.expr_ir import Expr

__all__ = ["Preprocessor", "columns_arg"]


def columns_arg(columns: str | Sequence[str], *, what: str) -> list[str]:
    """Normalize a `columns` argument, accepting a single name or a sequence of names.

    A single string is the common case (``StandardScaler("age")``), so it is accepted
    alongside the list form (``StandardScaler(["age", "height"])``). Passing the bare
    string straight to ``list(...)`` would silently split it into characters, so this
    guard is what makes the one-column spelling do the obvious thing rather than fit a
    scaler over ``["a", "g", "e"]``.

    Args:
        columns: A single column name, or a sequence of column names.
        what: The caller's class name, used in the error message.

    Returns:
        The column names as a list.

    Raises:
        PlanError: If no columns are given, or a non-string sneaks into the sequence.

    Examples:
        .. doctest::

            >>> from batcher.ml.preprocessors.base import columns_arg
            >>> columns_arg("age", what="StandardScaler")
            ['age']
            >>> columns_arg(["a", "b"], what="StandardScaler")
            ['a', 'b']
    """
    cols = [columns] if isinstance(columns, str) else list(columns)
    if not cols:
        raise PlanError(f"{what} requires at least one column")
    bad = [c for c in cols if not isinstance(c, str)]
    if bad:
        raise PlanError(
            f"{what} columns must be strings (a column name), got {bad[0]!r}. "
            "Pass a single column name or a list of names."
        )
    return cols


# The default ceiling on a learned category set. Every categorical encoder lowers to a
# per-category `CASE` arm (ordinal/target) or a per-category output column (one-hot), so
# the fitted cardinality is also the size of the resulting expression or schema. 1,000 is
# deliberately generous — it admits any ordinary categorical column while still failing
# fast on the accidental fit over a primary key, which would otherwise build a
# million-arm expression or a million-column plan.
MAX_CATEGORIES = 1000


def check_cardinality(what: str, column: str, found: int, limit: int, *, exact: bool) -> None:
    """Raise if a learned category set is too large to lower into an expression.

    Args:
        what: The preprocessor class name, for the message.
        column: The column whose category set was learned.
        found: How many distinct categories were seen (a lower bound if not `exact`).
        limit: The configured `max_categories` ceiling.
        exact: Whether `found` is the true cardinality or only a lower bound.

    Raises:
        PlanError: If `found` exceeds `limit`.
    """
    if found <= limit:
        return
    seen = f"{found} distinct categories" if exact else f"more than {limit} distinct categories"
    raise PlanError(
        f"{what}: column {column!r} has {seen}, above max_categories={limit}. Each "
        f"category becomes a CASE arm or an output column, so the plan grows with the "
        f"cardinality and the category set is materialized on the driver. Reduce the "
        f"cardinality first (bucket rare values, or use a hashing/target encoding for a "
        f"high-cardinality column), or raise max_categories to accept the cost."
    )


def fit_aggregate(ds: Dataset, aggs: dict[str, Expr]) -> dict[str, Any]:
    """Run a single global aggregate and return its one row as ``{name: scalar}``.

    The shared `fit` primitive: every scaler/imputer learns its statistics in one
    mergeable pass over the data (the same engine path as `describe`), then reads the
    scalars back to the driver as plain Python values.
    """
    row = ds.agg(**aggs).collect()
    return {name: row.column(name)[0].as_py() for name in row.column_names}


def distinct_values(ds: Dataset, column: str, *, what: str, max_categories: int) -> list[Any]:
    """The sorted, non-null distinct values of `column` (an encoder's categories).

    Executes a `distinct` over the single column and reads the values to the driver —
    the `fit` step for categorical encoders. Nulls are dropped (they map to the
    encoder's unknown value at transform time).

    The read is bounded: it pulls at most ``max_categories + 1`` values, so a fit over
    an unbounded column (a key, a free-text field) fails with an actionable error
    instead of materializing the whole domain on the driver.

    Args:
        ds: The dataset to learn the category set from.
        column: The column whose distinct values to read.
        what: The calling preprocessor's class name, for the error message.
        max_categories: The ceiling on the learned cardinality.

    Returns:
        The sorted, non-null distinct values.

    Raises:
        PlanError: If `column` has more than `max_categories` distinct values.
    """
    bounded = ds.select(column).distinct().limit(max_categories + 1)
    values = bounded.collect().column(column).to_pylist()
    categories = sorted(v for v in values if v is not None)
    check_cardinality(what, column, len(categories), max_categories, exact=False)
    return categories


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

    @property
    def is_fitted(self) -> bool:
        """Whether `fit` (or `fit_transform`) has run, so `transform` is ready.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import StandardScaler
                >>> pre = StandardScaler("x")
                >>> pre.is_fitted
                False
                >>> pre.fit(bt.from_pydict({"x": [1.0, 3.0]})).is_fitted
                True

        Returns:
            ``True`` once fitted, ``False`` before.
        """
        return self._fitted

    def get_params(self) -> dict[str, Any]:
        """The constructor hyperparameters, in scikit-learn's ``get_params`` shape.

        Returns the configuration passed at construction (columns, strategy, and so
        on), not the state learned by `fit` — the latter is exposed on the
        trailing-underscore attributes scikit-learn uses (``mean_``, ``scale_``).

        Examples:
            .. doctest::

                >>> from batcher.ml.preprocessors import StandardScaler
                >>> sorted(StandardScaler("x").get_params())
                ['columns', 'with_mean', 'with_std']

        Returns:
            A ``{name: value}`` dict of the constructor hyperparameters.
        """
        names: dict[str, None] = {}
        for klass in type(self).__mro__:
            for slot in getattr(klass, "__slots__", ()):
                names.setdefault(slot, None)
        for attr in getattr(self, "__dict__", {}):
            names.setdefault(attr, None)
        return {
            n: getattr(self, n)
            for n in names
            if not n.startswith("_") and not n.endswith("_") and hasattr(self, n)
        }

    def __repr__(self) -> str:
        """``StandardScaler(columns=['x'], fitted=True)`` — params plus the fitted flag."""
        params = self.get_params()
        rendered = ", ".join(f"{k}={v!r}" for k, v in params.items())
        state = "fitted" if self._fitted else "unfitted"
        return f"{type(self).__name__}({rendered}) [{state}]"

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

    def save(self, path: str) -> None:
        """Write this fitted preprocessor to `path` as readable JSON.

        The state has to outlive the process that fitted it, or the scaler standardizing a
        request at serving time uses the request's own mean instead of the training set's.
        JSON rather than a pickle so the file is reviewable, diffable, portable to another
        language, and safe to load from a store you do not fully control. Accepts a cloud
        URI as well as a local path.

        Examples:
            .. doctest::

                >>> import batcher as bt, os, tempfile
                >>> from batcher.ml.preprocessors import StandardScaler
                >>> pre = StandardScaler("x").fit(bt.from_pydict({"x": [1.0, 3.0]}))
                >>> target = os.path.join(tempfile.mkdtemp(), "scaler.json")
                >>> pre.save(target)
                >>> StandardScaler.load(target).mean_
                {'x': 2.0}

        Args:
            path: Where to write it; a local path or a cloud URI.
        """
        from batcher.ml.preprocessors.persistence import save

        save(self, path)

    @staticmethod
    def load(path: str) -> Preprocessor:
        """Read a preprocessor written by `save`, learned state included.

        A static method on the base class rather than on each subclass: the document names
        its own class, so the caller never has to know which one to ask.

        Examples:
            .. doctest::

                >>> import batcher as bt, os, tempfile
                >>> from batcher.ml.preprocessors import MinMaxScaler, Preprocessor
                >>> pre = MinMaxScaler("x").fit(bt.from_pydict({"x": [0.0, 10.0]}))
                >>> target = os.path.join(tempfile.mkdtemp(), "scaler.json")
                >>> pre.save(target)
                >>> type(Preprocessor.load(target)).__name__
                'MinMaxScaler'

        Args:
            path: The local path or cloud URI to read.

        Returns:
            The reconstructed preprocessor, fitted if the saved one was.
        """
        from batcher.ml.preprocessors.persistence import load

        return load(path)

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise PlanError(
                f"{type(self).__name__} must be fitted before transform(); "
                "call fit(ds) or fit_transform(ds) first"
            )
