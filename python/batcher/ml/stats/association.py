"""Association between two columns — contingency tables, chi-squared, and ANOVA.

Everything here answers "do these two columns tell me anything about each other", for the
cases a Pearson correlation cannot: two categorical columns, or a numeric column against a
categorical one. Each is a `group_by` plus an aggregate, so a screen over every feature
against the label is a handful of scans rather than a driver-side loop.

The contingency table is built over the **full grid** of category pairs, including the
pairs no row lands in. That is the whole subtlety: a chi-squared statistic sums over every
cell, and an empty cell is where the deviation from independence is largest. Summing only
the cells a `group_by` returned halved the statistic on a perfectly associated table.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from batcher.ml.stats._shared import require_columns, scalar
from batcher.plan.expr_ir.constructors import col, lit, when
from batcher.plan.functions.aggregate import count_if
from batcher.plan.functions.aggregate import sum as sum_

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = [
    "anova_f",
    "chi_square",
    "cohens_f",
    "cramers_v",
    "epsilon_squared",
    "eta_squared",
    "mutual_information",
    "omega_squared",
    "theils_u",
]


def _contingency(ds: Dataset, x: str, y: str) -> Dataset:
    """The observed and expected counts of every ``(x, y)`` cell under independence.

    Every cell of the full grid appears, including the ones no row lands in. That is not a
    detail: a chi-squared statistic sums ``(observed - expected)^2 / expected`` over *all*
    cells, and an empty cell is exactly where the deviation from independence is largest.
    Summing only the cells a `group_by` produced halved the statistic on a perfectly
    associated 2x2 table, which reads as "no relationship" instead of "total relationship".
    """
    present = ds.filter(col(x).is_not_null() & col(y).is_not_null())
    cells = present.group_by(x, y).agg(__bt_seen=col(x).count())
    grid = present.select(x).distinct().cross_join(present.select(y).distinct())
    full = grid.join(cells, on=[x, y], how="left").with_columns(
        observed=col("__bt_seen").fill_null(lit(0))
    )
    return full.with_columns(
        __bt_row_total=sum_(col("observed")).over(partition_by=[x]),
        __bt_col_total=sum_(col("observed")).over(partition_by=[y]),
        __bt_total=sum_(col("observed")).over(),
    ).with_columns(
        __bt_expected=col("__bt_row_total").cast("float64")
        * col("__bt_col_total").cast("float64")
        / col("__bt_total").cast("float64")
    )


def chi_square(ds: Dataset, x: str, y: str) -> float:
    """The chi-squared statistic for independence between two categorical columns.

    ``sum((observed - expected)^2 / expected)`` over the contingency table. Large means the
    two columns are related; compare it against a chi-squared critical value with
    ``(rows - 1) * (columns - 1)`` degrees of freedom. Use `cramers_v` when you want a
    magnitude that does not grow with the row count.

    Args:
        ds: The dataset holding both columns.
        x: The first categorical column.
        y: The second categorical column.

    Returns:
        The chi-squared statistic.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import chi_square
            >>> ds = bt.from_pydict({"a": ["x", "x", "y", "y"], "b": ["p", "p", "q", "q"]})
            >>> chi_square(ds, "a", "b")
            4.0
    """
    require_columns(ds, x, y)
    table = _contingency(ds, x, y)
    difference = col("observed").cast("float64") - col("__bt_expected")
    return scalar(table.agg(c=sum_(difference * difference / col("__bt_expected"))), "c")


def cramers_v(ds: Dataset, x: str, y: str) -> float:
    """Cramer's V — the chi-squared statistic rescaled into ``[0, 1]``.

    ``sqrt(chi2 / (n * (min(rows, columns) - 1)))``. This is the categorical counterpart of
    a correlation, and the one to rank features by: unlike `chi_square` it does not grow
    with the dataset size, so a value from a million rows is comparable with one from a
    thousand. 0 is independent, 1 is a perfect association.

    Args:
        ds: The dataset holding both columns.
        x: The first categorical column.
        y: The second categorical column.

    Returns:
        Cramer's V in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import cramers_v
            >>> ds = bt.from_pydict({"a": ["x", "x", "y", "y"], "b": ["p", "p", "q", "q"]})
            >>> cramers_v(ds, "a", "b")
            1.0
    """

    statistic = chi_square(ds, x, y)
    sizes = ds.agg(
        n=col(x).count(),
        rows=col(x).n_unique(),
        columns=col(y).n_unique(),
    ).collect()
    total = sizes.column("n")[0].as_py()
    smaller = min(sizes.column("rows")[0].as_py(), sizes.column("columns")[0].as_py())
    if total == 0 or smaller < 2:
        return float("nan")
    return math.sqrt(statistic / (total * (smaller - 1)))


def mutual_information(ds: Dataset, x: str, y: str, *, base: float = 2.0) -> float:
    """Mutual information between two categorical columns, in `base` units.

    How much knowing one column tells you about the other — 0 when they are independent,
    and equal to either column's `entropy` when one determines the other. Unlike a
    correlation it detects *any* dependence, including a non-monotone one, which is what
    makes it the right screen for a categorical feature against a categorical label.

    Args:
        ds: The dataset holding both columns.
        x: The first categorical column.
        y: The second categorical column.
        base: The logarithm base; 2 gives bits.

    Returns:
        The mutual information in `base` units, never negative.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import mutual_information
            >>> ds = bt.from_pydict({"a": ["x", "x", "y", "y"], "b": ["p", "p", "q", "q"]})
            >>> mutual_information(ds, "a", "b")
            1.0
    """

    require_columns(ds, x, y)
    table = _contingency(ds, x, y)
    joint = col("observed").cast("float64") / col("__bt_total").cast("float64")
    ratio = col("observed").cast("float64") / col("__bt_expected")
    # An empty cell contributes 0 (the limit of p*ln(p) as p goes to 0), not the NaN that
    # `0 * ln(0)` would produce and poison the whole sum with.
    contribution = when(col("observed") > lit(0)).then(joint * ratio.ln()).otherwise(lit(0.0))
    return scalar(table.agg(m=sum_(contribution)), "m") / math.log(base)


def anova_f(ds: Dataset, value: str, group: str) -> float:
    """The one-way ANOVA F statistic — between-group variance over within-group variance.

    The numeric-feature-against-categorical-label screen, and the generalization of a
    two-sample t test to more than two groups. Large means the group means differ by more
    than the within-group noise explains.

    Args:
        ds: The dataset holding both columns.
        value: The numeric column.
        group: The grouping column.

    Returns:
        The F statistic.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import anova_f
            >>> ds = bt.from_pydict(
            ...     {"v": [1.0, 2.0, 10.0, 11.0], "g": ["a", "a", "b", "b"]}
            ... )
            >>> anova_f(ds, "v", "g")
            162.0
    """
    require_columns(ds, value, group)
    present = ds.filter(col(value).is_not_null() & col(group).is_not_null())
    with_means = present.with_columns(
        __bt_group_mean=sum_(col(value)).over(partition_by=[group])
        / count_if(col(value).is_not_null()).over(partition_by=[group]),
        __bt_grand_mean=sum_(col(value)).over() / count_if(col(value).is_not_null()).over(),
    )
    between = col("__bt_group_mean") - col("__bt_grand_mean")
    within = col(value) - col("__bt_group_mean")
    summary = with_means.agg(
        ss_between=sum_(between * between),
        ss_within=sum_(within * within),
        n=col(value).count(),
        k=col(group).n_unique(),
    ).collect()
    ss_between = summary.column("ss_between")[0].as_py()
    ss_within = summary.column("ss_within")[0].as_py()
    n = summary.column("n")[0].as_py()
    k = summary.column("k")[0].as_py()
    if k < 2 or n <= k or ss_within == 0:
        return float("nan")
    return (ss_between / (k - 1)) / (ss_within / (n - k))


def theils_u(ds: Dataset, x: str, y: str, *, base: float = 2.0) -> float:
    """Theil's U — the fraction of `y`'s uncertainty that knowing `x` removes.

    The *asymmetric* association measure for two categorical columns: ``U(y | x)`` is the
    mutual information divided by the entropy of `y`, so it lands in ``[0, 1]`` and reads as a
    proportion. Because it is directional, ``theils_u(ds, x, y)`` and ``theils_u(ds, y, x)``
    differ, which is exactly what you want when one column is a candidate predictor of the
    other. Cramer's V is the symmetric alternative when direction is not meaningful.

    Args:
        ds: The dataset holding both columns.
        x: The conditioning (predictor) column.
        y: The column whose uncertainty is being explained.
        base: The logarithm base; it cancels in the ratio, so the result is base-independent.

    Returns:
        Theil's ``U(y | x)`` in ``[0, 1]``; 0 when `x` says nothing about `y`, 1 when `x`
        determines `y`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import theils_u
            >>> ds = bt.from_pydict({"x": ["a", "a", "b", "b"], "y": ["p", "p", "q", "q"]})
            >>> theils_u(ds, "x", "y")
            1.0
    """
    from batcher.ml.stats.descriptive import entropy

    require_columns(ds, x, y)
    h_y = entropy(ds, y, base=base)
    if h_y == 0:
        return float("nan")
    return mutual_information(ds, x, y, base=base) / h_y


def eta_squared(ds: Dataset, value: str, group: str) -> float:
    """Eta-squared — the share of a numeric column's variance explained by the grouping.

    The bounded effect-size companion to `anova_f`: where F is unbounded and grows with the
    sample size, ``eta^2`` is the between-group sum of squares over the total, so it stays in
    ``[0, 1]`` and reads directly as "this grouping explains 30% of the variance". Reach for it
    when you need to compare the *strength* of an effect across groupings of different sizes,
    which a raw F cannot do.

    Args:
        ds: The dataset holding both columns.
        value: The numeric column.
        group: The grouping column.

    Returns:
        Eta-squared in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import eta_squared
            >>> ds = bt.from_pydict({"v": [1.0, 2.0, 9.0, 10.0], "g": ["a", "a", "b", "b"]})
            >>> round(eta_squared(ds, "v", "g"), 4)
            0.9846
    """
    f = anova_f(ds, value, group)
    if math.isnan(f):
        return float("nan")
    row = ds.agg(n=col(value).count(), k=col(group).n_unique()).collect()
    n = row.column("n")[0].as_py()
    k = row.column("k")[0].as_py()
    df1, df2 = k - 1, n - k
    return df1 * f / (df1 * f + df2)


def epsilon_squared(ds: Dataset, value: str, group: str) -> float:
    """Epsilon-squared — the bias-corrected version of `eta_squared`.

    Eta-squared is optimistic: it never decreases as you add groups, so on a small sample it
    reports an effect that is partly noise. Epsilon-squared subtracts the variance a random
    grouping would explain, which can push it slightly below 0 for a genuinely null effect —
    that is the honesty, not a bug. Prefer it over `eta_squared` when the group count is large
    relative to the sample.

    Args:
        ds: The dataset holding both columns.
        value: The numeric column.
        group: The grouping column.

    Returns:
        Epsilon-squared, usually in ``[0, 1]`` but slightly negative for a null effect.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import epsilon_squared
            >>> ds = bt.from_pydict({"v": [1.0, 2.0, 9.0, 10.0], "g": ["a", "a", "b", "b"]})
            >>> round(epsilon_squared(ds, "v", "g"), 4)
            0.9769
    """
    f = anova_f(ds, value, group)
    if math.isnan(f):
        return float("nan")
    row = ds.agg(n=col(value).count(), k=col(group).n_unique()).collect()
    n = row.column("n")[0].as_py()
    k = row.column("k")[0].as_py()
    df1, df2 = k - 1, n - k
    return df1 * (f - 1.0) / (df1 * f + df2)


def omega_squared(ds: Dataset, value: str, group: str) -> float:
    """Omega-squared — the least-biased ANOVA effect size, the population variance explained.

    Both `eta_squared` and `epsilon_squared` overestimate the effect in a sample; omega-squared
    subtracts the most of the sampling noise, so it is the estimate to report when generalizing
    beyond the data at hand. It can dip slightly below zero for a genuinely null effect. Computed
    from the `anova_f` statistic and the group and sample sizes.

    Args:
        ds: The dataset holding both columns.
        value: The numeric column.
        group: The grouping column.

    Returns:
        Omega-squared, usually in ``[0, 1]`` but slightly negative for a null effect.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import omega_squared
            >>> ds = bt.from_pydict({"v": [1.0, 2.0, 9.0, 10.0], "g": ["a", "a", "b", "b"]})
            >>> round(omega_squared(ds, "v", "g"), 4)
            0.9695
    """
    f = anova_f(ds, value, group)
    if math.isnan(f):
        return float("nan")
    row = ds.agg(n=col(value).count(), k=col(group).n_unique()).collect()
    n = row.column("n")[0].as_py()
    k = row.column("k")[0].as_py()
    df1, df2 = k - 1, n - k
    return df1 * (f - 1.0) / (df1 * f + df2 + 1.0)


def cohens_f(ds: Dataset, value: str, group: str) -> float:
    """Cohen's f — the ANOVA effect size used for power and sample-size calculation.

    The ratio of between-group to within-group standard deviation, ``sqrt(eta^2 / (1 - eta^2))``.
    It is the effect-size scale a power analysis for an ANOVA is specified on (Cohen's conventional
    small/medium/large are 0.1 / 0.25 / 0.4), which is what makes it worth reporting alongside the
    variance-explained measures. Computed from the `anova_f` statistic and the degrees of freedom.

    Args:
        ds: The dataset holding both columns.
        value: The numeric column.
        group: The grouping column.

    Returns:
        Cohen's f, at least 0.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import cohens_f
            >>> ds = bt.from_pydict({"v": [1.0, 2.0, 9.0, 10.0], "g": ["a", "a", "b", "b"]})
            >>> round(cohens_f(ds, "v", "g"), 4)
            8.0
    """
    f = anova_f(ds, value, group)
    if math.isnan(f):
        return float("nan")
    row = ds.agg(n=col(value).count(), k=col(group).n_unique()).collect()
    n = row.column("n")[0].as_py()
    k = row.column("k")[0].as_py()
    df1, df2 = k - 1, n - k
    if df2 <= 0:
        return float("nan")
    return math.sqrt(df1 * f / df2)
