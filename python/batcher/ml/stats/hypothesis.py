"""Hypothesis tests — a statistic plus the p-value that makes it a decision.

A test statistic on its own says how large an effect is in its own units; the p-value says how
surprising that is under the null hypothesis, and it is the p-value a data scientist actually
acts on. The statistics already live as mergeable aggregates (`anova_f`, `chi_square`, the
Welch expressions, `jarque_bera`); this module pairs each with its reference-distribution tail
probability, computed on the single aggregated scalar in the control plane.

Every test here reduces the data to a handful of aggregates in one pass, then evaluates a
survival function on the result — no per-row work and no third-party runtime dependency. Each
returns a `TestResult` carrying the statistic, its degrees of freedom, and the p-value, and
each is checked against SciPy in the tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from batcher.ml.stats._shared import indicator, scalar
from batcher.ml.stats._special import (
    chi2_sf,
    f_sf,
    normal_two_sided_p,
    students_t_two_sided_p,
)
from batcher.ml.stats.association import anova_f, chi_square
from batcher.plan.expr_ir.constructors import col
from batcher.plan.functions.aggregate import corr, count_if, mean, n_unique, std

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = [
    "TestResult",
    "anova_test",
    "binomial_test",
    "chi_square_test",
    "mcnemar_test",
    "normality_test",
    "pearson_test",
    "proportion_ztest",
    "spearman_test",
    "t_test_1samp",
    "t_test_ind",
]


@dataclass(frozen=True)
class TestResult:
    """The outcome of a hypothesis test: the statistic, its degrees of freedom, and the p-value.

    The p-value is the number to threshold (reject the null below your alpha); the statistic and
    degrees of freedom are kept so the result can be reported in full or fed to a power
    calculation.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import t_test_1samp
            >>> result = t_test_1samp(bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 5.0]}), 3.0)
            >>> result.pvalue > 0.99
            True

    Args:
        statistic: The test statistic.
        pvalue: The p-value under the null hypothesis.
        df: The degrees of freedom, a single value or an ``(df1, df2)`` pair for an F test.
    """

    statistic: float
    pvalue: float
    df: float | tuple[float, float]


def t_test_1samp(ds: Dataset, popmean: float, column: str = "x") -> TestResult:
    """Test whether a column's mean differs from a hypothesized value (one-sample t test).

    The two-sided test of ``H0: mean == popmean``. Reduces the column to its mean, standard
    deviation, and count in one pass, then reads the p-value off a Student's t with ``n - 1``
    degrees of freedom.

    Args:
        ds: The dataset to test.
        popmean: The hypothesized population mean.
        column: The numeric column to test.

    Returns:
        A `TestResult` with the t statistic, ``n - 1`` degrees of freedom, and the two-sided
        p-value.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import t_test_1samp
            >>> ds = bt.from_pydict({"x": [2.0, 4.0, 6.0, 8.0, 10.0]})
            >>> round(t_test_1samp(ds, 6.0).pvalue, 3)
            1.0
    """
    row = ds.agg(m=mean(col(column)), s=std(col(column)), n=col(column).count()).collect()
    m = float(row.column("m")[0].as_py())
    s = float(row.column("s")[0].as_py())
    n = int(row.column("n")[0].as_py())
    df = n - 1
    t = (m - popmean) / (s / math.sqrt(n)) if s > 0 else math.inf
    return TestResult(statistic=t, pvalue=students_t_two_sided_p(t, df), df=float(df))


def t_test_ind(ds: Dataset, value: str, group: str) -> TestResult:
    """Test whether two groups have different means (Welch's two-sample t test).

    The unequal-variance (Welch) form, which is the safe default over Student's pooled t. The
    `group` column must take exactly two distinct values; the p-value comes from a Student's t
    with the Welch-Satterthwaite degrees of freedom.

    Args:
        ds: The dataset to test.
        value: The numeric column whose means are compared.
        group: The two-valued column that splits the sample.

    Returns:
        A `TestResult` with the Welch t statistic, its fractional degrees of freedom, and the
        two-sided p-value.

    Raises:
        PlanError: If `group` does not have exactly two distinct values.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import t_test_ind
            >>> ds = bt.from_pydict(
            ...     {"g": ["a", "a", "a", "b", "b", "b"], "x": [1.0, 2.0, 3.0, 8.0, 9.0, 10.0]}
            ... )
            >>> t_test_ind(ds, "x", "g").pvalue < 0.05
            True
    """
    from batcher._internal.errors import PlanError

    levels = sorted(
        v.as_py()
        for v in ds.select(group).distinct().collect().column(group)
        if v.as_py() is not None
    )
    if len(levels) != 2:
        raise PlanError(f"t_test_ind needs exactly two groups in {group!r}, found {len(levels)}.")
    stats = {}
    for level in levels:
        sub = ds.filter(col(group) == level)
        row = sub.agg(m=mean(col(value)), v=std(col(value)) ** 2, n=col(value).count()).collect()
        stats[level] = (
            float(row.column("m")[0].as_py()),
            float(row.column("v")[0].as_py()),
            int(row.column("n")[0].as_py()),
        )
    (m1, v1, n1), (m2, v2, n2) = stats[levels[0]], stats[levels[1]]
    se = math.sqrt(v1 / n1 + v2 / n2)
    t = (m1 - m2) / se if se > 0 else math.inf
    df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    return TestResult(statistic=t, pvalue=students_t_two_sided_p(t, df), df=df)


def anova_test(ds: Dataset, value: str, group: str) -> TestResult:
    """Test whether several groups share one mean (one-way ANOVA).

    Extends the two-sample t test to more than two groups: ``H0`` is that every group mean is
    equal. Reuses the mergeable `anova_f` statistic and reads the p-value off an F distribution
    with ``(k - 1, n - k)`` degrees of freedom for ``k`` groups and ``n`` rows.

    Args:
        ds: The dataset to test.
        value: The numeric column whose group means are compared.
        group: The grouping column.

    Returns:
        A `TestResult` with the F statistic, its ``(df1, df2)`` pair, and the upper-tail
        p-value.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import anova_test
            >>> ds = bt.from_pydict(
            ...     {"g": ["a", "a", "b", "b", "c", "c"], "x": [1.0, 2.0, 5.0, 6.0, 9.0, 10.0]}
            ... )
            >>> anova_test(ds, "x", "g").pvalue < 0.05
            True
    """
    f = anova_f(ds, value, group)
    row = ds.agg(k=n_unique(col(group)), n=col(value).count()).collect()
    k = int(row.column("k")[0].as_py())
    n = int(row.column("n")[0].as_py())
    df1, df2 = float(k - 1), float(n - k)
    return TestResult(statistic=f, pvalue=f_sf(f, df1, df2), df=(df1, df2))


def chi_square_test(ds: Dataset, x: str, y: str) -> TestResult:
    """Test whether two categorical columns are independent (Pearson's chi-squared).

    ``H0`` is that `x` and `y` are independent. Reuses the mergeable `chi_square` statistic and
    reads the p-value off a chi-squared with ``(cardinality(x) - 1) * (cardinality(y) - 1)``
    degrees of freedom.

    Args:
        ds: The dataset to test.
        x: The first categorical column.
        y: The second categorical column.

    Returns:
        A `TestResult` with the chi-squared statistic, its degrees of freedom, and the
        upper-tail p-value.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import chi_square_test
            >>> ds = bt.from_pydict(
            ...     {"a": ["x", "x", "y", "y"] * 25, "b": ["p", "q", "p", "q"] * 25}
            ... )
            >>> chi_square_test(ds, "a", "b").pvalue > 0.05
            True
    """
    statistic = chi_square(ds, x, y)
    row = ds.agg(cx=n_unique(col(x)), cy=n_unique(col(y))).collect()
    cx = int(row.column("cx")[0].as_py())
    cy = int(row.column("cy")[0].as_py())
    df = float((cx - 1) * (cy - 1))
    return TestResult(statistic=statistic, pvalue=chi2_sf(statistic, df), df=df)


def normality_test(ds: Dataset, column: str) -> TestResult:
    """Test whether a column is normally distributed (Jarque-Bera).

    A large statistic means the sample skew or kurtosis departs from a normal's, so a small
    p-value rejects normality. The Jarque-Bera statistic is asymptotically chi-squared with two
    degrees of freedom, which is where the p-value comes from.

    Args:
        ds: The dataset to test.
        column: The numeric column to test.

    Returns:
        A `TestResult` with the Jarque-Bera statistic, two degrees of freedom, and the
        upper-tail p-value.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import normality_test
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})
            >>> normality_test(ds, "x").df
            2.0
    """
    from batcher.plan.functions.analysis import jarque_bera

    statistic = scalar(ds.agg(jb=jarque_bera(column)), "jb")
    return TestResult(statistic=statistic, pvalue=chi2_sf(statistic, 2.0), df=2.0)


def _corr_significance(r: float, n: int) -> TestResult:
    """Turn a sample correlation into a `TestResult` via the ``t = r*sqrt((n-2)/(1-r^2))`` test."""
    df = n - 2
    if df <= 0 or abs(r) >= 1.0:
        t = math.inf if abs(r) >= 1.0 else math.nan
        return TestResult(statistic=r, pvalue=0.0 if t == math.inf else math.nan, df=float(df))
    t = r * math.sqrt(df / (1.0 - r * r))
    return TestResult(statistic=r, pvalue=students_t_two_sided_p(t, df), df=float(df))


def pearson_test(ds: Dataset, x: str, y: str) -> TestResult:
    """Test whether two numeric columns are linearly correlated (Pearson).

    Pairs the Pearson correlation with its significance: under the null of no correlation,
    ``t = r * sqrt((n - 2) / (1 - r^2))`` is a Student's t with ``n - 2`` degrees of freedom.
    The `statistic` field carries the correlation itself, so a significant tiny `r` on a huge
    sample is visible as exactly that.

    Args:
        ds: The dataset holding both columns.
        x: The first numeric column.
        y: The second numeric column.

    Returns:
        A `TestResult` whose statistic is the Pearson `r`, with ``n - 2`` degrees of freedom and
        the two-sided p-value.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import pearson_test
            >>> ds = bt.from_pydict(
            ...     {"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.0, 4.1, 5.9, 8.0, 10.1]}
            ... )
            >>> pearson_test(ds, "x", "y").pvalue < 0.01
            True
    """
    n = ds.count()
    r = ds.agg(r=corr(col(x), col(y))).collect().column("r")[0].as_py()
    return _corr_significance(float(r) if r is not None else math.nan, n)


def spearman_test(ds: Dataset, x: str, y: str) -> TestResult:
    """Test whether two columns are monotonically associated (Spearman).

    The rank-based counterpart of `pearson_test`: it correlates the ranks, so it detects any
    monotone relationship, not just a linear one, and shrugs off outliers. The p-value uses the
    same ``t`` approximation on the rank correlation, matching SciPy's ``spearmanr`` for a sample
    of any real size.

    Args:
        ds: The dataset holding both columns.
        x: The first column.
        y: The second column.

    Returns:
        A `TestResult` whose statistic is Spearman's rho, with ``n - 2`` degrees of freedom and
        the two-sided p-value.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import spearman_test
            >>> ds = bt.from_pydict(
            ...     {"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [1.0, 4.0, 9.0, 16.0, 25.0]}
            ... )
            >>> round(spearman_test(ds, "x", "y").statistic, 6)
            1.0
    """
    from batcher.ml.stats.descriptive import spearman_corr

    n = ds.count()
    return _corr_significance(spearman_corr(ds, x, y), n)


def proportion_ztest(ds: Dataset, success: str, p0: float = 0.5) -> TestResult:
    """Test whether a 0/1 column's success rate differs from a hypothesized proportion.

    The one-sample proportion z-test: ``z = (phat - p0) / sqrt(p0 (1 - p0) / n)``, with the
    p-value from the standard normal. The `success` column is a 0/1 (or boolean) indicator, so a
    conversion column, a click column, or a correct/incorrect flag all fit directly.

    Args:
        ds: The dataset holding the indicator column.
        success: The 0/1 or boolean success column.
        p0: The hypothesized success proportion.

    Returns:
        A `TestResult` with the z statistic, infinite degrees of freedom (the normal limit), and
        the two-sided p-value.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import proportion_ztest
            >>> ds = bt.from_pydict({"won": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]})
            >>> proportion_ztest(ds, "won", 0.5).pvalue < 0.1
            True
    """
    row = ds.agg(k=count_if(indicator(success)), n=col(success).count()).collect()
    k = int(row.column("k")[0].as_py())
    n = int(row.column("n")[0].as_py())
    phat = k / n
    se = math.sqrt(p0 * (1.0 - p0) / n)
    z = (phat - p0) / se if se > 0 else math.inf
    return TestResult(statistic=z, pvalue=normal_two_sided_p(z), df=math.inf)


def mcnemar_test(ds: Dataset, correct_a: str, correct_b: str) -> TestResult:
    """Test whether two classifiers have different error rates on the same rows (McNemar).

    The paired test for comparing two models: given a boolean column per model marking whether
    it got each row right, it looks only at the rows where the two disagree — ``b`` where A is
    wrong and B right, ``c`` where A is right and B wrong — because the rows they both get right
    or both get wrong carry no information about which is better. The continuity-corrected
    statistic ``(|b - c| - 1)^2 / (b + c)`` is chi-squared with one degree of freedom.

    Args:
        ds: The dataset holding both correctness columns.
        correct_a: The boolean column marking model A correct per row.
        correct_b: The boolean column marking model B correct per row.

    Returns:
        A `TestResult` with the chi-squared statistic, one degree of freedom, and the p-value.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import mcnemar_test
            >>> ds = bt.from_pydict(
            ...     {"a": [True, True, False, False], "b": [True, False, True, True]}
            ... )
            >>> mcnemar_test(ds, "a", "b").df
            1.0
    """
    a_right = indicator(correct_a)
    b_right = indicator(correct_b)
    row = ds.agg(
        b=count_if(~a_right & b_right),
        c=count_if(a_right & ~b_right),
    ).collect()
    b = int(row.column("b")[0].as_py())
    c = int(row.column("c")[0].as_py())
    if b + c == 0:
        return TestResult(statistic=0.0, pvalue=1.0, df=1.0)
    statistic = (abs(b - c) - 1.0) ** 2 / (b + c)
    return TestResult(statistic=statistic, pvalue=chi2_sf(statistic, 1.0), df=1.0)


def _binomial_pmf(k: int, n: int, p: float) -> float:
    """The binomial probability mass ``P(X = k)`` for ``X ~ Binomial(n, p)``."""
    if p <= 0.0:
        return 1.0 if k == 0 else 0.0
    if p >= 1.0:
        return 1.0 if k == n else 0.0
    log_mass = (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        + k * math.log(p)
        + (n - k) * math.log1p(-p)
    )
    return math.exp(log_mass)


def binomial_test(ds: Dataset, success: str, p0: float = 0.5) -> TestResult:
    """Test whether a 0/1 column's success rate differs from a hypothesized proportion (exact).

    The exact binomial test: unlike `proportion_ztest`, which uses the normal approximation, this
    sums the exact binomial probabilities and so is correct even for a tiny sample where the
    approximation is unreliable. The two-sided p-value is the total probability of every outcome no
    more likely than the observed one, matching SciPy's ``binomtest``.

    Args:
        ds: The dataset holding the indicator column.
        success: The 0/1 or boolean success column.
        p0: The hypothesized success proportion.

    Returns:
        A `TestResult` whose statistic is the observed success count, with the trial count as
        degrees of freedom and the exact two-sided p-value.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import binomial_test
            >>> ds = bt.from_pydict({"won": [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]})
            >>> round(binomial_test(ds, "won", 0.5).pvalue, 4)
            0.3437
    """
    row = ds.agg(k=count_if(indicator(success)), n=col(success).count()).collect()
    k = int(row.column("k")[0].as_py())
    n = int(row.column("n")[0].as_py())
    observed = _binomial_pmf(k, n, p0)
    tolerance = observed * (1.0 + 1e-7)
    pvalue = sum(
        _binomial_pmf(j, n, p0) for j in range(n + 1) if _binomial_pmf(j, n, p0) <= tolerance
    )
    return TestResult(statistic=float(k), pvalue=min(pvalue, 1.0), df=float(n))
