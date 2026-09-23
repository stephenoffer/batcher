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

Missing and degenerate input follows SciPy's defaults, one rule for every test in `ml.stats`:

* a **null** is a missing observation and is dropped -- in the value column, and in the group
  column, where a null label is never a group of its own;
* a **NaN** is a value, and it propagates: the statistic and p-value are NaN, never a p of 0
  (SciPy's ``nan_policy="propagate"``);
* **too little data** -- one row, a group of one, constant data, no pairs -- gives NaN, not an
  exception, and constant groups that differ give an infinite statistic with ``p = 0``;
* **too few groups** for the question (fewer than two, or not exactly two for a two-sample
  test) raises `PlanError`, as SciPy raises for it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from batcher.ml.stats._shared import complete_rows, indicator, scalar
from batcher.ml.stats._special import (
    chi2_sf,
    f_sf,
    normal_two_sided_p,
    safe_ratio,
    students_t_ppf,
    students_t_two_sided_p,
)
from batcher.ml.stats.association import _anova, chi_square
from batcher.plan.expr_ir.constructors import col
from batcher.plan.functions.aggregate import corr, count_distinct, count_if, mean, std

if TYPE_CHECKING:
    import pyarrow as pa

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

    The four trailing fields are optional because not every test can fill them: a chi-squared
    test of independence has no single effect to put an interval around, while a t test does.
    They exist so a test that *can* report an interval or an effect size does not need a second
    result type to carry it — a p-value alone answers "is this distinguishable from noise" and
    says nothing about how large the effect is, which is the question an experiment readout
    actually needs.

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
        ci: The confidence interval for the estimated effect, or ``None`` if the test does not
            estimate one. Its coverage is `ci_level`.
        ci_level: The coverage of `ci` — 0.95 unless the caller asked for another.
        n: The number of rows the test consumed, or ``None`` if it was not counted.
        alternative: Which departures from the null the p-value covers: ``"two-sided"``,
            ``"greater"``, or ``"less"``.
        effect_size: A standardized effect (Cohen's d for the t tests), or ``None``.
    """

    statistic: float
    pvalue: float
    df: float | tuple[float, float]
    ci: tuple[float, float] | None = None
    ci_level: float = 0.95
    n: int | None = None
    alternative: str = "two-sided"
    effect_size: float | None = None


def _float(table: pa.Table, name: str) -> float:
    """A one-row aggregate as a float, NaN where the engine returned null (no rows, or n=1)."""
    value = table.column(name)[0].as_py()
    return math.nan if value is None else float(value)


def _require_groups(k: int, what: str, group: str, *, exactly_two: bool = False) -> None:
    """Raise a `PlanError` unless `group` split the rows into enough groups for the test.

    SciPy raises for fewer than two samples too (``f_oneway``, ``levene``, ``bartlett`` and
    ``kruskal`` all do), because a between-groups test with one group has no question to answer.
    Everything short of that -- a group of one row, constant data -- is a NaN result instead.
    """
    from batcher._internal.errors import PlanError

    if exactly_two and k != 2:
        raise PlanError(
            f"{what} needs exactly two groups in {group!r} after dropping null labels and null "
            f"values, found {k}. Filter {group!r} to the two levels you want to compare."
        )
    if k < 2:
        raise PlanError(
            f"{what} needs at least two groups in {group!r} after dropping null labels and null "
            f"values, found {k}. Check that {group!r} is the grouping column and is not all null."
        )


def _mean_ci(estimate: float, se: float, df: float, level: float = 0.95) -> tuple[float, float]:
    """A two-sided t interval around `estimate`, at `level` coverage.

    The t quantile rather than the normal one: `plan.functions.analysis.inference` already
    ships `mean_ci_half_width` as a single-pass *aggregate*, which is the right tool when the
    interval is a column. Here the mean, its standard error and the degrees of freedom are
    already scalars in hand, and at small `df` the normal quantile is too narrow.
    """
    if se == 0.0:
        return (estimate, estimate)
    if math.isnan(estimate) or not math.isfinite(se) or math.isnan(df) or df <= 0:
        return (math.nan, math.nan)
    half = students_t_ppf(0.5 * (1.0 + level), df) * se
    return (estimate - half, estimate + half)


def t_test_1samp(ds: Dataset, popmean: float, column: str = "x") -> TestResult:
    """Test whether a column's mean differs from a hypothesized value (one-sample t test).

    The two-sided test of ``H0: mean == popmean``. Reduces the column to its mean, standard
    deviation, and count in one pass, then reads the p-value off a Student's t with ``n - 1``
    degrees of freedom. Nulls are skipped; a NaN in the column makes the result NaN, as does a
    column of fewer than two values.

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
    m, s, n = _float(row, "m"), _float(row, "s"), int(row.column("n")[0].as_py())
    df = float(n - 1) if n > 0 else math.nan
    se = s / math.sqrt(n) if n > 0 else math.nan
    t = safe_ratio(m - popmean, se)
    return TestResult(
        statistic=t,
        pvalue=students_t_two_sided_p(t, df),
        df=df,
        ci=_mean_ci(m - popmean, se, df),
        n=n,
        effect_size=safe_ratio(m - popmean, s),
    )


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
        PlanError: If `group` does not have exactly two distinct non-null values among the rows
            with a value.

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
    present = complete_rows(ds, value, group)
    levels = sorted(v.as_py() for v in present.select(group).distinct().collect().column(group))
    _require_groups(len(levels), "t_test_ind", group, exactly_two=True)
    stats = []
    for level in levels:
        sub = present.filter(col(group) == level)
        row = sub.agg(m=mean(col(value)), v=std(col(value)) ** 2, n=col(value).count()).collect()
        stats.append((_float(row, "m"), _float(row, "v"), int(row.column("n")[0].as_py())))
    (m1, v1, n1), (m2, v2, n2) = stats
    a, b = v1 / n1, v2 / n2
    se = math.sqrt(a + b)
    t = safe_ratio(m1 - m2, se)
    # Welch-Satterthwaite. A group of one row has no variance (NaN), which carries through.
    spread = a * a / (n1 - 1) + b * b / (n2 - 1) if n1 > 1 and n2 > 1 else math.nan
    df = (a + b) ** 2 / spread if spread > 0 else math.nan
    # Cohen's d takes the *pooled* SD even though the test itself is Welch's: the interval is
    # about the difference in the data's own units, while d is about a common scale.
    pooled = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)) if n1 + n2 > 2 else math.nan
    return TestResult(
        statistic=t,
        pvalue=students_t_two_sided_p(t, df),
        df=df,
        ci=_mean_ci(m1 - m2, se, df),
        n=n1 + n2,
        effect_size=safe_ratio(m1 - m2, pooled),
    )


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

    Raises:
        PlanError: If fewer than two groups remain after dropping null values and labels.

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
    parts = _anova(ds, value, group)
    _require_groups(parts.k, "anova_test", group)
    f = parts.f
    df1, df2 = float(parts.k - 1), float(parts.n - parts.k)
    return TestResult(statistic=f, pvalue=f_sf(f, df1, df2), df=(df1, df2), n=parts.n)


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
    row = ds.agg(cx=count_distinct(col(x)), cy=count_distinct(col(y))).collect()
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
