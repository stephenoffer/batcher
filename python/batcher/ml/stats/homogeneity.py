"""Tests of equal variance across groups — the assumption ANOVA and the t-test quietly rely on.

A two-sample t-test and a one-way ANOVA both assume the groups share a common variance; when
they do not, the p-value is wrong, sometimes badly. These tests check that assumption directly.
Bartlett's test is the powerful choice when the groups are themselves normal; Levene's test
(in its median-centered Brown-Forsythe form) trades some power for robustness to non-normality,
and is the safer default.

Both reduce to per-group aggregates — a count and a variance for Bartlett, a spread-from-center
for Levene — so the whole test is a couple of scans plus scalar math on the driver. Each returns
a `TestResult` and is checked against SciPy.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from batcher.ml.stats._shared import complete_rows
from batcher.ml.stats._special import chi2_sf, f_sf
from batcher.ml.stats.hypothesis import TestResult, _require_groups
from batcher.plan.expr_ir.constructors import col

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["bartlett_test", "levene_test"]


def _group_stats(ds: Dataset, value: str, group: str) -> list[tuple[int, float]]:
    """Per-group ``(count, sample_variance)``; the variance is NaN for a one-row group."""
    grouped = (
        complete_rows(ds, value, group)
        .group_by(group)
        .agg(__bt_n=col(value).count(), __bt_v=col(value).var())
        .collect()
    )
    return [
        (
            int(grouped.column("__bt_n")[i].as_py()),
            math.nan if (v := grouped.column("__bt_v")[i].as_py()) is None else float(v),
        )
        for i in range(grouped.num_rows)
    ]


def bartlett_test(ds: Dataset, value: str, group: str) -> TestResult:
    """Test whether several groups share one variance (Bartlett's test).

    The likelihood-ratio test for equal variances, assuming the groups are normal. A small p-value
    says the group variances differ, which invalidates the equal-variance assumption of a pooled
    t-test or a one-way ANOVA. Sensitive to non-normality, so prefer `levene_test` when normality
    is in doubt. The statistic is chi-squared with ``k - 1`` degrees of freedom for ``k`` groups.

    Args:
        ds: The dataset holding both columns.
        value: The numeric column whose variance is compared across groups.
        group: The grouping column.

    Returns:
        A `TestResult` with the Bartlett statistic, ``k - 1`` degrees of freedom, and the
        upper-tail p-value. NaN when a group has one row or every group is constant.

    Raises:
        PlanError: If fewer than two groups remain after dropping null values and labels.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import bartlett_test
            >>> ds = bt.from_pydict(
            ...     {"g": ["a", "a", "a", "b", "b", "b"], "x": [1.0, 2.0, 3.0, 1.0, 5.0, 9.0]}
            ... )
            >>> bartlett_test(ds, "x", "g").df
            1.0
    """
    rows = _group_stats(ds, value, group)
    k = len(rows)
    _require_groups(k, "bartlett_test", group)
    df = float(k - 1)
    total = sum(n for n, _ in rows)
    variances = [v for _, v in rows]
    # A one-row group has no variance, and all-constant groups have no spread to compare:
    # both are NaN in `scipy.stats.bartlett`. One constant group among varying ones is an
    # infinitely significant difference, which SciPy also reports.
    if any(math.isnan(v) for v in variances) or all(v == 0 for v in variances):
        return TestResult(statistic=math.nan, pvalue=math.nan, df=df)
    if any(v == 0 for v in variances):
        return TestResult(statistic=math.inf, pvalue=0.0, df=df)
    pooled = sum((n - 1) * v for n, v in rows) / (total - k)
    numerator = (total - k) * math.log(pooled) - sum((n - 1) * math.log(v) for n, v in rows)
    correction = 1.0 + (sum(1.0 / (n - 1) for n, _ in rows) - 1.0 / (total - k)) / (3.0 * (k - 1))
    statistic = numerator / correction
    return TestResult(statistic=statistic, pvalue=chi2_sf(statistic, df), df=df)


def levene_test(ds: Dataset, value: str, group: str) -> TestResult:
    """Test whether several groups share one variance (Levene's test, median-centered).

    The robust alternative to `bartlett_test`: it runs a one-way ANOVA on each row's absolute
    deviation from its group's median (the Brown-Forsythe variant), which is far less sensitive to
    non-normal, heavy-tailed data. A small p-value says the group spreads differ. This is the
    equal-variance check to reach for by default. The statistic is F with ``(k - 1, n - k)``
    degrees of freedom.

    Args:
        ds: The dataset holding both columns.
        value: The numeric column whose spread is compared across groups.
        group: The grouping column.

    Returns:
        A `TestResult` with the Levene statistic, its ``(df1, df2)`` pair, and the upper-tail
        p-value. NaN when no group has any spread, or there are no more rows than groups.

    Raises:
        PlanError: If fewer than two groups remain after dropping null values and labels.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import levene_test
            >>> ds = bt.from_pydict(
            ...     {"g": ["a", "a", "a", "b", "b", "b"], "x": [1.0, 2.0, 3.0, 1.0, 5.0, 9.0]}
            ... )
            >>> levene_test(ds, "x", "g").df
            (1.0, 4.0)
    """
    from batcher.ml.stats.association import _anova

    present = complete_rows(ds, value, group)
    # Brown-Forsythe: the ANOVA of each row's distance from its own group's median. A window
    # median keeps the group label a column, where the per-label `when` chain this replaced
    # spliced each label in as a literal -- and a null label became an Int64 null compared
    # against a Utf8 column, which the engine rejects.
    spread = present.with_columns(
        __bt_z=(col(value) - col(value).median().over(partition_by=[group])).abs()
    )
    parts = _anova(spread, "__bt_z", group)
    _require_groups(parts.k, "levene_test", group)
    df1, df2 = float(parts.k - 1), float(parts.n - parts.k)
    statistic = parts.f
    return TestResult(statistic=statistic, pvalue=f_sf(statistic, df1, df2), df=(df1, df2))
