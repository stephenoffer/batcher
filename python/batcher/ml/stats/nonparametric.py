"""Rank-based tests — comparing groups without assuming a distribution.

A t-test asks whether two group *means* differ and trusts that the data is roughly normal. When
that assumption is shaky — a skewed, heavy-tailed, or ordinal column — the rank-based tests here
ask the distribution-free question instead: does one group tend to produce larger *ranks* than
another. The Mann-Whitney U test is the two-group case, Kruskal-Wallis its extension to several.

Both work off a single set of ranks over the pooled values, computed in the engine with a window,
and reduce to a handful of aggregates. Ties are given average ranks and the tie correction is one
extra aggregate (``sum(tie_size^2 - 1)`` equals the textbook ``sum(t^3 - t)``), so the statistics
match SciPy's exactly.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from batcher.ml.stats._special import chi2_sf, normal_two_sided_p
from batcher.ml.stats.hypothesis import TestResult
from batcher.plan.expr_ir.constructors import col, lit, when
from batcher.plan.expr_ir.nodes import rank as rank_
from batcher.plan.functions.aggregate import sum as sum_

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = [
    "cliffs_delta",
    "common_language_effect_size",
    "friedman_test",
    "kruskal_wallis",
    "mann_whitney_u",
    "wilcoxon_signed_rank",
]


def _average_ranks(ds: Dataset, value: str) -> Dataset:
    """Append the pooled average rank of `value` (ties averaged) and each value's tie size."""
    tie_size = col(value).count().over(partition_by=[value])
    min_rank = rank_().over(order_by=[value])
    return ds.filter(col(value).is_not_null()).with_columns(
        __bt_rank=min_rank + (tie_size - lit(1.0)) / lit(2.0),
        __bt_tie=tie_size,
    )


def _u_summary(ds: Dataset, value: str, group: str, what: str):
    """The Mann-Whitney ``(u1, n1, n2, n, tie_correction)`` for a two-group split."""
    from batcher._internal.errors import PlanError

    levels = sorted(
        v.as_py()
        for v in ds.select(group).distinct().collect().column(group)
        if v.as_py() is not None
    )
    if len(levels) != 2:
        raise PlanError(f"{what} needs exactly two groups in {group!r}, found {len(levels)}.")
    ranked = _average_ranks(ds, value)
    first = col(group) == lit(levels[0])
    summary = ranked.agg(
        __bt_r1=sum_(when(first).then(col("__bt_rank")).otherwise(lit(0.0))),
        __bt_n1=sum_(when(first).then(lit(1.0)).otherwise(lit(0.0))),
        __bt_n=col(value).count(),
        __bt_tie=sum_(col("__bt_tie") * col("__bt_tie") - lit(1.0)),
    ).collect()
    r1 = float(summary.column("__bt_r1")[0].as_py())
    n1 = int(summary.column("__bt_n1")[0].as_py())
    n = int(summary.column("__bt_n")[0].as_py())
    tie = float(summary.column("__bt_tie")[0].as_py())
    return r1 - n1 * (n1 + 1) / 2, n1, n - n1, n, tie


def mann_whitney_u(ds: Dataset, value: str, group: str) -> TestResult:
    """Test whether one of two groups tends to produce larger values (Mann-Whitney U).

    The distribution-free alternative to a two-sample t-test: it ranks the pooled values and asks
    whether the ranks of one group sit systematically above the other's, so it is unaffected by
    skew, heavy tails, or an ordinal scale where a mean is meaningless. The p-value is the
    tie-corrected normal approximation with a continuity correction, matching SciPy's asymptotic
    ``mannwhitneyu``. The `group` column must take exactly two values.

    Args:
        ds: The dataset holding both columns.
        value: The numeric or ordinal column to compare.
        group: The two-valued column that splits the sample.

    Returns:
        A `TestResult` whose statistic is the U of the first group (sorted order), with the normal
        limit as degrees of freedom and the two-sided p-value.

    Raises:
        PlanError: If `group` does not have exactly two distinct values.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import mann_whitney_u
            >>> ds = bt.from_pydict(
            ...     {"g": ["a", "a", "a", "b", "b", "b"], "x": [1.0, 2.0, 3.0, 7.0, 8.0, 9.0]}
            ... )
            >>> mann_whitney_u(ds, "x", "g").pvalue < 0.1
            True
    """
    u1, n1, n2, n, tie = _u_summary(ds, value, group, "mann_whitney_u")
    mu = n1 * n2 / 2
    variance = n1 * n2 / 12 * ((n + 1) - tie / (n * (n - 1)))
    if variance <= 0:
        return TestResult(statistic=u1, pvalue=math.nan, df=math.inf)
    z = (u1 - mu - math.copysign(0.5, u1 - mu)) / math.sqrt(variance)
    return TestResult(statistic=u1, pvalue=normal_two_sided_p(z), df=math.inf)


def kruskal_wallis(ds: Dataset, value: str, group: str) -> TestResult:
    """Test whether several groups share one distribution by their ranks (Kruskal-Wallis).

    The rank-based extension of Mann-Whitney to more than two groups, and the distribution-free
    counterpart of one-way ANOVA. It pools and ranks every value, then asks whether the average
    rank differs across groups. Tie-corrected, and chi-squared with ``k - 1`` degrees of freedom
    for ``k`` groups, matching SciPy's ``kruskal``.

    Args:
        ds: The dataset holding both columns.
        value: The numeric or ordinal column to compare.
        group: The grouping column.

    Returns:
        A `TestResult` with the H statistic, ``k - 1`` degrees of freedom, and the upper-tail
        p-value.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import kruskal_wallis
            >>> ds = bt.from_pydict(
            ...     {"g": ["a", "a", "b", "b", "c", "c"], "x": [1.0, 2.0, 5.0, 6.0, 9.0, 10.0]}
            ... )
            >>> kruskal_wallis(ds, "x", "g").df
            2.0
    """
    ranked = _average_ranks(ds, value)
    per_group = (
        ranked.group_by(group)
        .agg(__bt_r=sum_(col("__bt_rank")), __bt_n=col(value).count())
        .collect()
    )
    totals = ranked.agg(
        __bt_n=col(value).count(),
        __bt_tie=sum_(col("__bt_tie") * col("__bt_tie") - lit(1.0)),
    ).collect()
    n = int(totals.column("__bt_n")[0].as_py())
    tie = float(totals.column("__bt_tie")[0].as_py())
    k = per_group.num_rows
    rank_sum = 0.0
    for i in range(k):
        group_r = float(per_group.column("__bt_r")[i].as_py())
        group_n = int(per_group.column("__bt_n")[i].as_py())
        rank_sum += group_r * group_r / group_n
    statistic = 12.0 / (n * (n + 1)) * rank_sum - 3.0 * (n + 1)
    correction = 1.0 - tie / (n**3 - n) if n > 1 else 1.0
    if correction != 0:
        statistic /= correction
    df = float(k - 1)
    return TestResult(statistic=statistic, pvalue=chi2_sf(statistic, df), df=df)


def common_language_effect_size(ds: Dataset, value: str, group: str) -> float:
    """The probability that a random member of the first group exceeds one from the second.

    The most intuitive effect size for two groups: ``P(x > y)`` estimated over all cross-group
    pairs, with ties counted as half. 0.5 means the groups are indistinguishable, 1.0 means every
    member of the first group beats every member of the second. It is the Mann-Whitney U rescaled
    to a probability, and it says *how much* two groups differ where the test only says *whether*.

    Args:
        ds: The dataset holding both columns.
        value: The numeric or ordinal column to compare.
        group: The two-valued column that splits the sample.

    Returns:
        The common-language effect size in ``[0, 1]``.

    Raises:
        PlanError: If `group` does not have exactly two distinct values.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import common_language_effect_size
            >>> ds = bt.from_pydict(
            ...     {"g": ["a", "a", "a", "b", "b", "b"], "x": [1.0, 2.0, 3.0, 7.0, 8.0, 9.0]}
            ... )
            >>> common_language_effect_size(ds, "x", "g")
            0.0
    """
    u1, n1, n2, _, _ = _u_summary(ds, value, group, "common_language_effect_size")
    return u1 / (n1 * n2) if n1 and n2 else float("nan")


def cliffs_delta(ds: Dataset, value: str, group: str) -> float:
    """Cliff's delta — the non-parametric effect size for two groups, in ``[-1, 1]``.

    ``P(x > y) - P(x < y)`` over all cross-group pairs: 0 when the groups overlap completely, +1
    when every first-group value exceeds every second-group value, -1 for the reverse. Unlike
    Cohen's d it makes no assumption about the distribution or equal variances, so it is the effect
    size to report alongside a `mann_whitney_u` result, and equals
    ``2 * common_language_effect_size - 1``.

    Args:
        ds: The dataset holding both columns.
        value: The numeric or ordinal column to compare.
        group: The two-valued column that splits the sample.

    Returns:
        Cliff's delta in ``[-1, 1]``.

    Raises:
        PlanError: If `group` does not have exactly two distinct values.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import cliffs_delta
            >>> ds = bt.from_pydict(
            ...     {"g": ["a", "a", "a", "b", "b", "b"], "x": [1.0, 2.0, 3.0, 7.0, 8.0, 9.0]}
            ... )
            >>> cliffs_delta(ds, "x", "g")
            -1.0
    """
    u1, n1, n2, _, _ = _u_summary(ds, value, group, "cliffs_delta")
    return 2.0 * u1 / (n1 * n2) - 1.0 if n1 and n2 else float("nan")


def wilcoxon_signed_rank(ds: Dataset, x: str, y: str) -> TestResult:
    """Test whether the paired differences between two columns are centered at zero (Wilcoxon).

    The paired, distribution-free alternative to a paired t-test: it ranks the *magnitudes* of the
    per-row differences ``x - y`` and asks whether the positive and negative differences carry
    equal rank weight. Use it for a before/after comparison, a matched-pair design, or any two
    measurements on the same unit when the differences are too skewed for a t-test. Zero
    differences are dropped, ties share average ranks, and the p-value is the continuity- and
    tie-corrected normal approximation, matching SciPy's ``wilcoxon`` with ``mode="approx"``.

    Args:
        ds: The dataset holding both columns.
        x: The first measurement column.
        y: The second (paired) measurement column.

    Returns:
        A `TestResult` whose statistic is the smaller signed-rank sum, with the normal limit as
        degrees of freedom and the two-sided p-value.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import wilcoxon_signed_rank
            >>> ds = bt.from_pydict(
            ...     {"before": [5.0, 6.0, 7.0, 8.0], "after": [6.0, 8.0, 9.0, 11.0]}
            ... )
            >>> wilcoxon_signed_rank(ds, "before", "after").statistic
            0.0
    """
    diff = ds.with_columns(__bt_d=col(x) - col(y))
    nonzero = diff.filter((col("__bt_d") != lit(0.0)) & col("__bt_d").is_not_null())
    absolute = nonzero.with_columns(__bt_ad=col("__bt_d").abs())
    tie_size = col("__bt_ad").count().over(partition_by=["__bt_ad"])
    rank_expr = rank_().over(order_by=["__bt_ad"]) + (tie_size - lit(1.0)) / lit(2.0)
    ranked = absolute.with_columns(__bt_rank=rank_expr, __bt_tie=tie_size)
    positive = col("__bt_d") > lit(0.0)
    summary = ranked.agg(
        __bt_wp=sum_(when(positive).then(col("__bt_rank")).otherwise(lit(0.0))),
        __bt_wm=sum_(when(~positive).then(col("__bt_rank")).otherwise(lit(0.0))),
        __bt_n=col("__bt_d").count(),
        __bt_tie=sum_(col("__bt_tie") * col("__bt_tie") - lit(1.0)),
    ).collect()
    w_plus = float(summary.column("__bt_wp")[0].as_py())
    w_minus = float(summary.column("__bt_wm")[0].as_py())
    n = int(summary.column("__bt_n")[0].as_py())
    tie = float(summary.column("__bt_tie")[0].as_py())
    statistic = min(w_plus, w_minus)
    mu = n * (n + 1) / 4
    variance = n * (n + 1) * (2 * n + 1) / 24 - tie / 48
    if variance <= 0:
        return TestResult(statistic=statistic, pvalue=math.nan, df=math.inf)
    z = (abs(statistic - mu) - 0.5) / math.sqrt(variance)
    return TestResult(statistic=statistic, pvalue=normal_two_sided_p(z), df=math.inf)


def friedman_test(ds: Dataset, value: str, block: str, treatment: str) -> TestResult:
    """Test whether several treatments differ across matched blocks (Friedman).

    The non-parametric counterpart of a repeated-measures ANOVA: each `block` (a subject, a day, a
    matched set) is measured under every `treatment`, and the test ranks the treatments *within*
    each block and asks whether one tends to rank higher across blocks. Because it compares within a
    block it removes the block-to-block variation a between-groups test would be swamped by. The
    tie-corrected statistic is chi-squared with ``k - 1`` degrees of freedom for `k` treatments,
    matching SciPy's ``friedmanchisquare``.

    Args:
        ds: The dataset in long form — one row per ``(block, treatment)`` measurement.
        value: The measured column.
        block: The column identifying a matched block.
        treatment: The column identifying which treatment a row measures.

    Returns:
        A `TestResult` with the Friedman statistic, ``k - 1`` degrees of freedom, and the
        upper-tail p-value.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import friedman_test
            >>> ds = bt.from_pydict(
            ...     {"subject": [1, 1, 2, 2, 3, 3], "drug": ["a", "b", "a", "b", "a", "b"],
            ...      "score": [1.0, 2.0, 1.0, 3.0, 2.0, 4.0]}
            ... )
            >>> friedman_test(ds, "score", "subject", "drug").df
            1.0
    """
    present = ds.filter(col(value).is_not_null())
    tie_size = col(value).count().over(partition_by=[block, value])
    within_rank = rank_().over(partition_by=[block], order_by=[value]) + (
        tie_size - lit(1.0)
    ) / lit(2.0)
    ranked = present.with_columns(__bt_rank=within_rank, __bt_tie=tie_size)
    per_treatment = ranked.group_by(treatment).agg(__bt_r=sum_(col("__bt_rank"))).collect()
    sizes = ranked.agg(
        __bt_blocks=col(block).n_unique(),
        __bt_tie=sum_(col("__bt_tie") * col("__bt_tie") - lit(1.0)),
    ).collect()
    n = int(sizes.column("__bt_blocks")[0].as_py())
    tie = float(sizes.column("__bt_tie")[0].as_py())
    k = per_treatment.num_rows
    rank_sum_squares = sum(float(per_treatment.column("__bt_r")[i].as_py()) ** 2 for i in range(k))
    statistic = 12.0 / (n * k * (k + 1)) * rank_sum_squares - 3.0 * n * (k + 1)
    correction = 1.0 - tie / (n * k * (k * k - 1)) if k > 1 else 1.0
    if correction != 0:
        statistic /= correction
    df = float(k - 1)
    return TestResult(statistic=statistic, pvalue=chi2_sf(statistic, df), df=df)
