"""Linear-regression aggregate functions (DuckDB/PostgreSQL ``regr_*`` family).

Each summarizes the least-squares fit of a dependent column ``y`` on an independent
column ``x`` over the rows of a group. They are *expressions over aggregates* — built
from the mergeable `covar_pop` / `corr` / `mean` / `count` primitives and the arithmetic
`AggExpr` now supports — so `group_by().agg()` runs them as one aggregate pass plus a
projection, identical single-node and distributed.

Following SQL semantics, every function considers only rows where **both** ``x`` and
``y`` are non-null; the pairing is done by null-propagating arithmetic (``x + y*0`` is
``x`` when both are present, null otherwise) so a null in either column drops the pair
from all of `count`, the means, and the sums of squares. A group with fewer than two
valid pairs has an undefined fit — slope/intercept are ``NaN`` there.
"""

from __future__ import annotations

from batcher.plan.expr_ir.core import AggExpr, Expr, IntoExpr, Lit
from batcher.plan.functions.aggregate import _as_column, corr, covar_pop


def _paired(y: IntoExpr, x: IntoExpr) -> tuple[Expr, Expr]:
    """Return ``(y, x)`` each masked to null on rows where *either* is null.

    ``a + b*0`` is ``a`` when both are non-null and null otherwise (nulls propagate
    through arithmetic), so aggregating the masked columns restricts to the paired rows
    the ``regr_*`` family is defined over — no null literal required.
    """
    yy, xx = _as_column(y), _as_column(x)
    return yy + xx * Lit(0), xx + yy * Lit(0)


def regr_count(y: IntoExpr, x: IntoExpr) -> AggExpr:
    """Number of ``(x, y)`` rows where both are non-null — the regression sample size.

    Args:
        y: The dependent column.
        x: The independent column.

    Returns:
        The count of non-null pairs per group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 3.0, 5.0, 7.0]})
            >>> ds.agg(n=bt.regr_count(bt.col("y"), bt.col("x"))).to_pydict()
            {'n': [4]}
    """
    _, xp = _paired(y, x)
    return xp.count()


def regr_avgx(y: IntoExpr, x: IntoExpr) -> AggExpr:
    """Mean of ``x`` over the rows where both ``x`` and ``y`` are non-null.

    Args:
        y: The dependent column.
        x: The independent column.

    Returns:
        The mean of the independent column over the paired rows.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 3.0, 5.0, 7.0]})
            >>> ds.agg(a=bt.regr_avgx(bt.col("y"), bt.col("x"))).to_pydict()
            {'a': [2.5]}
    """
    _, xp = _paired(y, x)
    return xp.mean()


def regr_avgy(y: IntoExpr, x: IntoExpr) -> AggExpr:
    """Mean of ``y`` over the rows where both ``x`` and ``y`` are non-null.

    Args:
        y: The dependent column.
        x: The independent column.

    Returns:
        The mean of the dependent column over the paired rows.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 3.0, 5.0, 7.0]})
            >>> ds.agg(a=bt.regr_avgy(bt.col("y"), bt.col("x"))).to_pydict()
            {'a': [4.0]}
    """
    yp, _ = _paired(y, x)
    return yp.mean()


def regr_slope(y: IntoExpr, x: IntoExpr) -> Expr:
    """Slope of the least-squares line ``y = slope*x + intercept`` (``covar_pop(y,x)/var_pop(x)``).

    Args:
        y: The dependent column.
        x: The independent column.

    Returns:
        The regression slope per group (``NaN`` if fewer than two paired rows).

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 3.0, 5.0, 7.0]})
            >>> ds.agg(m=bt.regr_slope(bt.col("y"), bt.col("x"))).to_pydict()
            {'m': [2.0]}
    """
    yp, xp = _paired(y, x)
    return covar_pop(yp, xp) / covar_pop(xp, xp)


def regr_intercept(y: IntoExpr, x: IntoExpr) -> Expr:
    """Intercept of the least-squares line — ``avg(y) - slope * avg(x)`` over the pairs.

    Args:
        y: The dependent column.
        x: The independent column.

    Returns:
        The regression intercept per group (``NaN`` if fewer than two paired rows).

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 3.0, 5.0, 7.0]})
            >>> ds.agg(b=bt.regr_intercept(bt.col("y"), bt.col("x"))).to_pydict()
            {'b': [-1.0]}
    """
    yp, xp = _paired(y, x)
    slope = covar_pop(yp, xp) / covar_pop(xp, xp)
    return yp.mean() - slope * xp.mean()


def regr_r2(y: IntoExpr, x: IntoExpr) -> Expr:
    """Coefficient of determination R² — the squared Pearson correlation of the pairs.

    Args:
        y: The dependent column.
        x: The independent column.

    Returns:
        The fraction of variance explained, in ``[0, 1]`` (null if undefined).

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 3.0, 5.0, 7.0]})
            >>> round(ds.agg(r=bt.regr_r2(bt.col("y"), bt.col("x"))).to_pydict()["r"][0], 6)
            1.0
    """
    yp, xp = _paired(y, x)
    return corr(yp, xp) ** 2


def regr_sxx(y: IntoExpr, x: IntoExpr) -> Expr:
    """Sum of squared deviations of ``x`` — ``sum((x - avg(x))**2)`` over the pairs.

    Args:
        y: The dependent column.
        x: The independent column.

    Returns:
        The ``x`` sum of squares per group (the regression's ``Sxx``).

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 3.0, 5.0, 7.0]})
            >>> ds.agg(s=bt.regr_sxx(bt.col("y"), bt.col("x"))).to_pydict()
            {'s': [5.0]}
    """
    _, xp = _paired(y, x)
    return covar_pop(xp, xp) * xp.count()


def regr_syy(y: IntoExpr, x: IntoExpr) -> Expr:
    """Sum of squared deviations of ``y`` — ``sum((y - avg(y))**2)`` over the pairs.

    Args:
        y: The dependent column.
        x: The independent column.

    Returns:
        The ``y`` sum of squares per group (the regression's ``Syy``).

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 3.0, 5.0, 7.0]})
            >>> ds.agg(s=bt.regr_syy(bt.col("y"), bt.col("x"))).to_pydict()
            {'s': [20.0]}
    """
    yp, xp = _paired(y, x)
    return covar_pop(yp, yp) * xp.count()


def regr_sxy(y: IntoExpr, x: IntoExpr) -> Expr:
    """Sum of cross-deviations — ``sum((x - avg(x)) * (y - avg(y)))`` over the pairs.

    Args:
        y: The dependent column.
        x: The independent column.

    Returns:
        The cross sum of products per group (the regression's ``Sxy``).

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 3.0, 5.0, 7.0]})
            >>> ds.agg(s=bt.regr_sxy(bt.col("y"), bt.col("x"))).to_pydict()
            {'s': [10.0]}
    """
    yp, xp = _paired(y, x)
    return covar_pop(yp, xp) * xp.count()
