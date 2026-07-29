"""Distribution-reshaping preprocessors — quantile, power, log, and clipping transforms.

Scaling changes a column's units; these change its *shape*. That is what a linear model,
a distance metric, and a neural net all actually need, and what a `StandardScaler` cannot
give them: standardizing a log-normal column leaves it just as skewed, with a mean that
still sits at the 70th percentile.

Every one of them keeps the `Preprocessor` contract — `fit` is a bounded number of
aggregates, `transform` is a lazy `Expr` — which for these is less obvious than it sounds
and is the interesting part of each implementation:

- `QuantileTransformer` learns `n_quantiles` cut points in one aggregate and transforms
  with a **sum of threshold indicators**, so the rank lookup is one vectorized expression
  rather than a search per row.

The power transform is the fourth member of this family; it lives in the sibling `power`
module, because its one-pass maximum-likelihood fit needs more explaining than the
transform itself does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.ml.preprocessors.base import Preprocessor, columns_arg, fit_aggregate
from batcher.plan.expr_ir.constructors import col, lit, nullif, when
from batcher.plan.functions.analysis._normal import normal_ppf

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset
    from batcher.plan.expr_ir import Expr

__all__ = ["Clipper", "LogTransformer", "MissingIndicator", "QuantileTransformer"]


class QuantileTransformer(Preprocessor):
    """Map a column onto a uniform or normal distribution by its rank.

    The most aggressive and most reliable of the shape transforms: it discards everything
    about the column except the *order* of its values, so the result is uniform by
    construction whatever the input looked like. Outliers cannot survive it — an extreme
    value becomes simply "the largest", at the top of the output range.

    `fit` learns `n_quantiles` cut points in one aggregate. `transform` counts how many cut
    points each value is at or above, which is one vectorized expression: no per-row search,
    no lookup table, and it stays a projection so it runs distributed unchanged.

    The mapping is a **step** function with `n_quantiles` steps rather than an interpolated
    one, so two values inside the same step get the same output — the step's midpoint. That
    is a deliberate trade for staying inside the expression language; raise `n_quantiles`
    for a finer grid.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import QuantileTransformer
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 1000.0]})
            >>> QuantileTransformer("x", n_quantiles=4).fit_transform(ds).to_pydict()
            {'x': [0.125, 0.375, 0.625, 0.875]}

    Args:
        columns: The numeric columns to transform (replaced in place).
        n_quantiles: How many cut points to learn; the output resolution.
        output_distribution: ``"uniform"`` for ``(0, 1)``, or ``"normal"`` for a
            standard-normal-shaped output.
    """

    __slots__ = ("columns", "n_quantiles", "output_distribution", "quantiles_")

    def __init__(
        self,
        columns: str | Sequence[str],
        *,
        n_quantiles: int = 100,
        output_distribution: str = "uniform",
    ) -> None:
        self.columns = columns_arg(columns, what="QuantileTransformer")
        if n_quantiles < 2:
            raise PlanError(f"n_quantiles must be at least 2, got {n_quantiles}")
        if output_distribution not in ("uniform", "normal"):
            raise PlanError(
                f"output_distribution must be 'uniform' or 'normal', got {output_distribution!r}"
            )
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.quantiles_: dict[str, list[float]] = {}

    def fit(self, ds: Dataset) -> QuantileTransformer:
        """Learn each column's `n_quantiles` cut points in one aggregate.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import QuantileTransformer
                >>> pre = QuantileTransformer("x", n_quantiles=2).fit(
                ...     bt.from_pydict({"x": [0.0, 10.0]})
                ... )
                >>> pre.quantiles_
                {'x': [0.0, 5.0]}

        Args:
            ds: The dataset to learn the cut points from.

        Returns:
            ``self``, fitted.
        """
        fractions = [i / self.n_quantiles for i in range(self.n_quantiles)]
        aggregates = {
            f"{name}__{i}": col(name).quantile(f)
            for name in self.columns
            for i, f in enumerate(fractions)
        }
        cell = fit_aggregate(ds, aggregates)
        for name in self.columns:
            values = [cell[f"{name}__{i}"] for i in range(len(fractions))]
            self.quantiles_[name] = [0.0 if v is None else float(v) for v in values]
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Replace each fitted column with its rank-based position in the learned grid.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import QuantileTransformer
                >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0]})
                >>> pre = QuantileTransformer("x", n_quantiles=2).fit(ds)
                >>> pre.transform(ds).to_pydict()
                {'x': [0.25, 0.25, 0.75, 0.75]}

        Args:
            ds: The dataset to transform.

        Returns:
            A new lazy `Dataset` with the fitted columns reshaped.
        """
        self._require_fitted()
        return ds.with_columns(**{name: self._expr(name) for name in self.columns})

    def _expr(self, name: str) -> Expr:
        """The step-function expression mapping `name` onto the target distribution."""
        cuts = self.quantiles_[name]
        outputs = _step_outputs(len(cuts), self.output_distribution)
        # Each cut contributes its own step height once the value reaches it, so the whole
        # piecewise mapping is one sum of indicators rather than a CASE ladder.
        expression = lit(outputs[0])
        for index in range(1, len(cuts)):
            height = outputs[index] - outputs[index - 1]
            expression = expression + (col(name) >= lit(cuts[index])).cast("float64") * lit(height)
        return expression


def _step_outputs(steps: int, distribution: str) -> list[float]:
    """The output value of each step, for a uniform or normal target distribution.

    Each step reports its **midpoint**, ``(i + 0.5) / steps``, not its lower edge. That is
    the unbiased estimate of the cumulative probability for a value known only to lie
    inside the step, and it is what makes the mapping symmetric: taking the lower edge
    shifts every output down by half a step, which for a normal target moved the mean of a
    uniform input to -0.098 instead of 0. It also keeps every fraction strictly inside
    ``(0, 1)``, where the normal quantile is finite.
    """
    fractions = [(i + 0.5) / steps for i in range(steps)]
    if distribution == "uniform":
        return fractions
    # Evaluate the normal quantile on the driver once per step; the transform expression
    # then carries the results as constants and never computes an inverse CDF per row.
    return [normal_ppf(f) for f in fractions]


class LogTransformer(Preprocessor):
    """Apply ``log1p`` (or ``log``) to a column — the cheap, explainable shape fix.

    `PowerTransformer` is stronger and data-driven, but its lambda is a number nobody can
    explain to a stakeholder. ``log1p`` is the transform an analyst can defend, it is
    exactly right for a multiplicative quantity, and it is stateless — the same expression
    on training and serving data with nothing fitted in between.

    Values that would make the logarithm undefined become null rather than NaN or an error,
    so one bad row does not poison a column.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import LogTransformer
            >>> ds = bt.from_pydict({"x": [0.0, 1.0, 3.0]})
            >>> LogTransformer("x").fit_transform(ds).to_pydict()["x"]
            [0.0, 0.6931471805599453, 1.3862943611198906]

    Args:
        columns: The numeric columns to transform (replaced in place).
        offset: Added before the logarithm; 1.0 gives ``log1p``, which handles zeros.
    """

    __slots__ = ("columns", "offset")

    def __init__(self, columns: str | Sequence[str], *, offset: float = 1.0) -> None:
        self.columns = columns_arg(columns, what="LogTransformer")
        self.offset = offset

    def transform(self, ds: Dataset) -> Dataset:
        """Replace each column with the logarithm of ``value + offset``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import LogTransformer
                >>> ds = bt.from_pydict({"x": [0.0, 1.0]})
                >>> LogTransformer("x").fit_transform(ds).to_pydict()["x"]
                [0.0, 0.6931471805599453]

        Args:
            ds: The dataset to transform.

        Returns:
            A new lazy `Dataset` with the fitted columns log-transformed.
        """
        # The `otherwise` arm covers both an undefined logarithm and a null input, and it used
        # to emit `nan` — which the class docstring has always said it must not. The difference
        # is not cosmetic: a NaN escapes `is_null()`, so a downstream null check reports the
        # column clean, and it propagates through `mean`/`sum` to poison every aggregate over
        # it. `nullif(x, x)` is an always-null expression of the right numeric type, which the
        # IR has no literal for.
        shifted = {name: col(name) + lit(self.offset) for name in self.columns}
        null = nullif(lit(0.0), lit(0.0))
        return ds.with_columns(
            **{
                name: when(expression > lit(0.0)).then(expression.ln()).otherwise(null)
                for name, expression in shifted.items()
            }
        )


class Clipper(Preprocessor):
    """Clamp columns to a learned quantile range — the winsorizing preprocessor.

    The bluntest and most effective outlier defense for a model that is sensitive to scale:
    nothing is dropped, so the row count and every join key survive, but no value can be
    more extreme than the `lower`/`upper` quantiles of the training data. Applying the
    *training* cut points to serving data is the point — a new record-breaking value gets
    clamped rather than extrapolated into a region the model never saw.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import Clipper
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 1000.0]})
            >>> Clipper("x", upper=0.8).fit_transform(ds).to_pydict()["x"][-1]
            203.2000000000002

    Args:
        columns: The numeric columns to clamp (replaced in place).
        lower: The lower quantile to clamp at, or None for no lower bound.
        upper: The upper quantile to clamp at, or None for no upper bound.
    """

    __slots__ = ("bounds_", "columns", "lower", "upper")

    def __init__(
        self,
        columns: str | Sequence[str],
        *,
        lower: float | None = 0.01,
        upper: float | None = 0.99,
    ) -> None:
        self.columns = columns_arg(columns, what="Clipper")
        if lower is None and upper is None:
            raise PlanError("Clipper needs at least one of lower= or upper=")
        for name, value in (("lower", lower), ("upper", upper)):
            if value is not None and not 0.0 <= value <= 1.0:
                raise PlanError(f"{name} must be a quantile in [0, 1], got {value}")
        self.lower = lower
        self.upper = upper
        self.bounds_: dict[str, tuple[float | None, float | None]] = {}

    def fit(self, ds: Dataset) -> Clipper:
        """Learn each column's clamp bounds from the requested quantiles.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import Clipper
                >>> pre = Clipper("x", lower=0.0, upper=1.0).fit(
                ...     bt.from_pydict({"x": [1.0, 5.0]})
                ... )
                >>> pre.bounds_
                {'x': (1.0, 5.0)}

        Args:
            ds: The dataset to learn the bounds from.

        Returns:
            ``self``, fitted.
        """
        aggregates = {}
        for name in self.columns:
            if self.lower is not None:
                aggregates[f"{name}__lo"] = col(name).quantile(self.lower)
            if self.upper is not None:
                aggregates[f"{name}__hi"] = col(name).quantile(self.upper)
        cell = fit_aggregate(ds, aggregates)
        for name in self.columns:
            low = cell.get(f"{name}__lo")
            high = cell.get(f"{name}__hi")
            self.bounds_[name] = (
                None if low is None else float(low),
                None if high is None else float(high),
            )
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Clamp each fitted column to its learned bounds.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import Clipper
                >>> ds = bt.from_pydict({"x": [1.0, 5.0]})
                >>> pre = Clipper("x", lower=0.0, upper=1.0).fit(ds)
                >>> pre.transform(bt.from_pydict({"x": [-9.0, 99.0]})).to_pydict()
                {'x': [1.0, 5.0]}

        Args:
            ds: The dataset to clamp.

        Returns:
            A new lazy `Dataset` with the fitted columns clamped.
        """
        self._require_fitted()
        projections = {}
        for name in self.columns:
            low, high = self.bounds_[name]
            projections[name] = col(name).clip(
                None if low is None else lit(low),
                None if high is None else lit(high),
            )
        return ds.with_columns(**projections)


class MissingIndicator(Preprocessor):
    """Add a boolean column recording which values were missing, before they are imputed.

    Missingness is usually a signal, not an accident: a blank income field means something
    different from a low one. Imputing first destroys that signal permanently, so the flag
    has to be created *before* the imputer runs. Stateless, so the same expression applies
    to training and serving data.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import MissingIndicator
            >>> ds = bt.from_pydict({"x": [1.0, None]})
            >>> MissingIndicator("x").fit_transform(ds).to_pydict()
            {'x': [1.0, None], 'x_missing': [False, True]}

    Args:
        columns: The columns to flag.
        suffix: Appended to each column name to build the flag column's name.
    """

    __slots__ = ("columns", "suffix")

    def __init__(self, columns: str | Sequence[str], *, suffix: str = "_missing") -> None:
        self.columns = columns_arg(columns, what="MissingIndicator")
        self.suffix = suffix

    def transform(self, ds: Dataset) -> Dataset:
        """Append one boolean flag column per fitted column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import MissingIndicator
                >>> ds = bt.from_pydict({"x": [1.0, None]})
                >>> MissingIndicator("x").transform(ds).to_pydict()["x_missing"]
                [False, True]

        Args:
            ds: The dataset to flag.

        Returns:
            A new lazy `Dataset` with one flag column appended per input column.
        """
        return ds.with_columns(
            **{f"{name}{self.suffix}": col(name).is_null() for name in self.columns}
        )
