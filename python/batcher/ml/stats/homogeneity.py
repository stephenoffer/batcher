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

from batcher.ml.stats._special import chi2_sf, f_sf
from batcher.ml.stats.hypothesis import TestResult
from batcher.plan.expr_ir.constructors import col, lit, when

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["bartlett_test", "levene_test"]


def _group_stats(ds: Dataset, value: str, group: str):
    """Per-group ``(label, count, sample_variance)`` rows, and the total count and group count."""
    from batcher.ml.stats._shared import require_columns

    require_columns(ds, value, group)
    grouped = (
        ds.filter(col(value).is_not_null())
        .group_by(group)
        .agg(__bt_n=col(value).count(), __bt_v=col(value).var())
        .collect()
    )
    rows = [
        (
            grouped.column(group)[i].as_py(),
            int(grouped.column("__bt_n")[i].as_py()),
            float(grouped.column("__bt_v")[i].as_py() or 0.0),
        )
        for i in range(grouped.num_rows)
    ]
    total = sum(n for _, n, _ in rows)
    return rows, total, len(rows)


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
        upper-tail p-value.

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
    rows, total, k = _group_stats(ds, value, group)
    pooled = sum((n - 1) * v for _, n, v in rows) / (total - k)
    numerator = (total - k) * math.log(pooled) - sum(
        (n - 1) * math.log(v) for _, n, v in rows if v > 0
    )
    correction = 1.0 + (sum(1.0 / (n - 1) for _, n, _ in rows) - 1.0 / (total - k)) / (
        3.0 * (k - 1)
    )
    statistic = numerator / correction
    df = float(k - 1)
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
        p-value.

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
    from batcher.ml.stats._shared import require_columns
    from batcher.plan.functions.analysis import correlation_ratio

    require_columns(ds, value, group)
    present = ds.filter(col(value).is_not_null())
    medians = present.group_by(group).agg(__bt_median=col(value).median()).collect()
    center = lit(0.0)
    for i in range(medians.num_rows):
        label = medians.column(group)[i].as_py()
        median = float(medians.column("__bt_median")[i].as_py())
        center = when(col(group) == lit(label)).then(lit(median)).otherwise(center)
    spread = (col(value) - center).abs()
    with_spread = present.with_columns(__bt_z=spread)
    # Levene's F is the one-way ANOVA F of the spreads; reuse the ANOVA machinery via a group mean.
    with_means = with_spread.with_columns(
        __bt_group_mean=col("__bt_z").mean().over(partition_by=[group])
    )
    eta = correlation_ratio("__bt_z", "__bt_group_mean")
    ratio = float(with_means.agg(__bt_eta=eta).collect().column("__bt_eta")[0].as_py())
    counts = present.group_by(group).agg(__bt_n=col(value).count()).collect()
    n = sum(int(counts.column("__bt_n")[i].as_py()) for i in range(counts.num_rows))
    k = counts.num_rows
    df1, df2 = float(k - 1), float(n - k)
    # eta^2 is SS_between / SS_total; F = (eta^2 / df1) / ((1 - eta^2) / df2).
    if ratio >= 1.0 or ratio <= 0.0:
        statistic = math.inf if ratio >= 1.0 else 0.0
    else:
        statistic = (ratio / df1) / ((1.0 - ratio) / df2)
    return TestResult(statistic=statistic, pvalue=f_sf(statistic, df1, df2), df=(df1, df2))
