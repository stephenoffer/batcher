"""Two-sample comparison and interval estimation as single-pass aggregates.

The "is this difference real" family. An A/B test, a cohort comparison, or a
before-and-after check is normally done by pulling both samples to a driver and calling
`scipy.stats`; here each statistic is arithmetic over *conditional* aggregates, so both
samples are summarized in one pass over the rows and never leave the engine.

The pattern that makes it work is the conditional mean:
``sum(when(group) then x else 0) / count_if(group)``. Two of those, plus two conditional
variances, give every two-sample statistic below.

What is deliberately **not** here is a p-value. Turning a t statistic into a probability
needs the incomplete beta function, which is not an arithmetic expression, and returning a
wrong one would be worse than returning none. Compare the statistic against a critical
value, or hand the statistic and the degrees of freedom to `scipy.stats` on the driver —
which is a scalar operation on two numbers, not a pass over the data.
"""

from __future__ import annotations

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.constructors import lit, when
from batcher.plan.expr_ir.core import Expr, IntoExpr
from batcher.plan.functions.aggregate import _as_column, count_if
from batcher.plan.functions.analysis._normal import normal_ppf

__all__ = [
    "cohens_d",
    "group_mean",
    "hedges_g",
    "mean_ci_half_width",
    "proportion_ci_half_width",
    "proportion_z_statistic",
    "welch_df",
    "welch_t_statistic",
]


def _two_sided_z(confidence: float, *, func: str) -> float:
    """The two-sided normal multiplier for a `confidence` level, exactly.

    This used to be ``1.959963984540054 if confidence < 0.97 else 2.5758293035489004``, which
    served *every* level below 0.97 with the 95% multiplier. A 90% interval came back 15% too
    wide and a 50% interval nearly three times too wide, both reported as if they were the level
    asked for:

        confidence   reported half-width   correct
        0.50         0.0673                0.0234
        0.90         0.0673                0.0570
        0.95         0.0673                0.0673

    The driver already has an accurate inverse normal CDF, so there is no reason to approximate:
    `confidence` is a Python float known at plan-build time, so this is one scalar evaluation
    per plan, not per row.

    Args:
        confidence: The two-sided confidence level, strictly between 0 and 1.
        func: The calling function's name, for the error message.

    Returns:
        The multiplier ``z`` such that a standard normal lies within ``+/- z`` with probability
        `confidence`.

    Raises:
        PlanError: If `confidence` is not strictly between 0 and 1.
    """
    if not 0.0 < confidence < 1.0:
        raise PlanError(
            f"{func}() needs a confidence strictly between 0 and 1, got {confidence!r}. "
            "Pass 0.95 for a 95% interval."
        )
    return normal_ppf(1.0 - (1.0 - confidence) / 2.0)


def group_mean(value: IntoExpr, group: Expr) -> Expr:
    """The mean of `value` over the rows where `group` is true.

    The building block for every two-sample statistic here: one conditional sum over one
    conditional count, so both samples are measured in the same pass over the same rows.

    Args:
        value: The measured column.
        group: A boolean expression selecting the sample.

    Returns:
        The conditional mean.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 3.0, 100.0], "g": ["a", "a", "b"]})
            >>> ds.agg(m=bt.group_mean("x", bt.col("g") == bt.lit("a"))).to_pydict()
            {'m': [2.0]}
    """
    column = _as_column(value)
    kept = group & column.is_not_null()
    return when(kept).then(column).otherwise(lit(0.0)).sum() / count_if(kept)


def _group_var(value: IntoExpr, group: Expr) -> Expr:
    """The sample variance of `value` over the rows where `group` is true.

    ``E[x^2] - E[x]^2`` scaled to the ``n - 1`` denominator. The naive form is used here
    rather than the engine's Welford aggregate because a *conditional* variance cannot be
    expressed with the aggregate directly, and a window expression -- which is how
    `Expr.zscore` and `Expr.expanding_var` center their inputs -- is rejected inside an
    aggregate.

    .. warning::

       The difference of two nearly equal large numbers loses a digit of precision for
       every digit by which the mean exceeds the spread, and taking a difference of means
       downstream does **not** recover it: the cancellation has already happened here. On
       ``[k+1, ..., k+6]`` split into two groups, `welch_t_statistic` returns ``-inf`` at
       ``k=1e9`` (both variances cancelled to 0) and ``NaN`` at ``k=1e12`` (they cancelled
       negative). An epoch-second timestamp is ~1.7e9, so subtract a reference point from
       such a column before testing it. Fixing this properly needs a weighted/conditional
       Welford aggregate in `bc-runtime`.
    """
    column = _as_column(value)
    kept = group & column.is_not_null()
    n = count_if(kept)
    masked = when(kept).then(column).otherwise(lit(0.0))
    mean = masked.sum() / n
    mean_square = when(kept).then(column * column).otherwise(lit(0.0)).sum() / n
    return (mean_square - mean * mean) * n / (n - lit(1.0))


def welch_t_statistic(value: IntoExpr, group: Expr) -> Expr:
    """Welch's t statistic comparing `group` against its complement.

    The unequal-variance two-sample t test, which is the one to use by default: Student's
    version assumes both samples have the same variance, and when they do not it is
    anti-conservative in exactly the case that matters (unequal sample sizes).

    Positive means the selected group has the larger mean. Pair with `welch_df` to look up
    a critical value.

    Args:
        value: The measured column.
        group: A boolean expression selecting the first sample; its negation is the second.

    Returns:
        Welch's t statistic.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"x": [10.0, 11.0, 12.0, 20.0, 21.0, 22.0], "g": ["a"] * 3 + ["b"] * 3}
            ... )
            >>> round(ds.agg(t=bt.welch_t_statistic("x", bt.col("g") == bt.lit("a")))
            ...       .to_pydict()["t"][0], 4)
            -12.2474
    """
    column = _as_column(value)
    n_first = count_if(group & column.is_not_null())
    n_second = count_if(~group & column.is_not_null())
    difference = group_mean(value, group) - group_mean(value, ~group)
    standard_error = (
        _group_var(value, group) / n_first + _group_var(value, ~group) / n_second
    ).sqrt()
    return difference / standard_error


def welch_df(value: IntoExpr, group: Expr) -> Expr:
    """The Welch-Satterthwaite degrees of freedom for `welch_t_statistic`.

    Needed to turn the t statistic into a probability, and not an integer: the whole point
    of Welch's correction is that the effective sample size falls between the two groups'.

    Args:
        value: The measured column.
        group: A boolean expression selecting the first sample.

    Returns:
        The effective degrees of freedom.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"x": [10.0, 11.0, 12.0, 20.0, 21.0, 22.0], "g": ["a"] * 3 + ["b"] * 3}
            ... )
            >>> ds.agg(d=bt.welch_df("x", bt.col("g") == bt.lit("a"))).to_pydict()
            {'d': [4.0]}
    """
    column = _as_column(value)
    n_first = count_if(group & column.is_not_null())
    n_second = count_if(~group & column.is_not_null())
    first = _group_var(value, group) / n_first
    second = _group_var(value, ~group) / n_second
    numerator = (first + second) * (first + second)
    denominator = first * first / (n_first - lit(1.0)) + second * second / (n_second - lit(1.0))
    return numerator / denominator


def cohens_d(value: IntoExpr, group: Expr) -> Expr:
    """Cohen's d — the difference in means in pooled standard deviations.

    The effect size, which answers the question a t statistic does not: *how big* is the
    difference, independent of how many rows you collected. At a large `n` every difference
    is "significant" and only this number distinguishes a real effect from a detectable one.
    Conventionally 0.2 is small, 0.5 medium, 0.8 large.

    Args:
        value: The measured column.
        group: A boolean expression selecting the first sample.

    Returns:
        Cohen's d.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"x": [10.0, 11.0, 12.0, 20.0, 21.0, 22.0], "g": ["a"] * 3 + ["b"] * 3}
            ... )
            >>> round(ds.agg(d=bt.cohens_d("x", bt.col("g") == bt.lit("a")))
            ...       .to_pydict()["d"][0], 4)
            -10.0
    """
    column = _as_column(value)
    n_first = count_if(group & column.is_not_null())
    n_second = count_if(~group & column.is_not_null())
    pooled = (
        (
            (n_first - lit(1.0)) * _group_var(value, group)
            + (n_second - lit(1.0)) * _group_var(value, ~group)
        )
        / (n_first + n_second - lit(2.0))
    ).sqrt()
    return (group_mean(value, group) - group_mean(value, ~group)) / pooled


def hedges_g(value: IntoExpr, group: Expr) -> Expr:
    """Hedges' g — `cohens_d` with the small-sample bias removed.

    Cohen's d over-states the effect when either group is small; the correction factor
    ``1 - 3 / (4n - 9)`` fixes it. Identical to d for large samples, so it is the safer
    default when group sizes vary — which they do in any real cohort comparison.

    Args:
        value: The measured column.
        group: A boolean expression selecting the first sample.

    Returns:
        Hedges' g.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"x": [10.0, 11.0, 12.0, 20.0, 21.0, 22.0], "g": ["a"] * 3 + ["b"] * 3}
            ... )
            >>> round(ds.agg(g=bt.hedges_g("x", bt.col("g") == bt.lit("a")))
            ...       .to_pydict()["g"][0], 4)
            -8.0
    """
    column = _as_column(value)
    total = count_if(group & column.is_not_null()) + count_if(~group & column.is_not_null())
    correction = lit(1.0) - lit(3.0) / (lit(4.0) * total - lit(9.0))
    return cohens_d(value, group) * correction


def proportion_z_statistic(outcome: Expr, group: Expr) -> Expr:
    """The two-proportion z statistic — the conversion-rate A/B test.

    Compares the rate at which `outcome` is true between the rows `group` selects and the
    rest, against the pooled rate. This is the statistic behind every "is variant B
    actually converting better" question, and it is one pass over the events table.

    Args:
        outcome: A boolean expression that is true on a success.
        group: A boolean expression selecting the first sample.

    Returns:
        The z statistic; positive means the selected group converts better.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"won": [True] * 30 + [False] * 70 + [True] * 50 + [False] * 50,
            ...      "arm": ["a"] * 100 + ["b"] * 100}
            ... )
            >>> round(ds.agg(z=bt.proportion_z_statistic(bt.col("won"),
            ...       bt.col("arm") == bt.lit("a"))).to_pydict()["z"][0], 4)
            -2.8868
    """
    n_first = count_if(group)
    n_second = count_if(~group)
    first_rate = count_if(group & outcome) / n_first
    second_rate = count_if(~group & outcome) / n_second
    pooled = count_if(outcome) / (n_first + n_second)
    standard_error = (
        pooled * (lit(1.0) - pooled) * (lit(1.0) / n_first + lit(1.0) / n_second)
    ).sqrt()
    return (first_rate - second_rate) / standard_error


def mean_ci_half_width(column: str | Expr, *, confidence: float = 0.95) -> Expr:
    """Half the width of the normal-approximation confidence interval for a mean.

    Add and subtract it from the mean to get the interval. Reported separately because it
    is the useful number on its own: it is the error bar, and comparing it against the
    difference you are interested in tells you whether the sample is large enough.

    Uses the normal quantile rather than Student's t, which is a negligible difference past
    a few dozen rows and avoids needing the t distribution in an expression.

    Args:
        column: The measured column.
        confidence: The two-sided confidence level, strictly between 0 and 1. Any level is
            exact; the multiplier is the normal quantile, evaluated once when the plan is
            built.

    Returns:
        The interval half-width.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [float(i) for i in range(100)]})
            >>> round(ds.agg(w=bt.mean_ci_half_width("x")).to_pydict()["w"][0], 4)
            5.6861
    """
    value = _as_column(column)
    multiplier = _two_sided_z(confidence, func="mean_ci_half_width")
    return lit(multiplier) * value.std() / value.count().cast("float64").sqrt()


def proportion_ci_half_width(outcome: Expr, *, confidence: float = 0.95) -> Expr:
    """Half the width of the confidence interval for a rate (the Wald interval).

    The error bar on a conversion rate, a null rate, or a failure rate. Note the standard
    caveat: near 0 or 1 the Wald interval is too narrow and can extend outside ``[0, 1]``,
    so treat a rate below a few percent with a correspondingly large grain of salt.

    Args:
        outcome: A boolean expression that is true on a success.
        confidence: The two-sided confidence level, strictly between 0 and 1.

    Returns:
        The interval half-width.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"won": [True] * 30 + [False] * 70})
            >>> round(ds.agg(w=bt.proportion_ci_half_width(bt.col("won")))
            ...       .to_pydict()["w"][0], 4)
            0.0898
    """
    total = count_if(outcome | ~outcome)
    rate = count_if(outcome) / total
    multiplier = _two_sided_z(confidence, func="proportion_ci_half_width")
    return lit(multiplier) * (rate * (lit(1.0) - rate) / total).sqrt()
