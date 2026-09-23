"""Hypothesis tests on messy data: nulls, NaN, tiny groups, and the one-group mistake.

The tests in `batcher.ml.stats` follow SciPy's defaults on input that is not a clean sample,
and this script walks through each rule on data small enough to check by eye:

- a null is a missing observation and is dropped, including a null *group label*, which never
  becomes a group of its own;
- a NaN is a value and it propagates, so the result is NaN rather than a confident p-value;
- too little data (a group of one row, constant groups, no pairs) is NaN, not an exception;
- too few groups for the question is a typed `PlanError` that says what to fix.

The functions live in `batcher.ml.stats`, not on `bt` itself: import the module, then pass
a `Dataset`. Each returns a `TestResult` (statistic, p-value, degrees of freedom).

    python examples/statistics/hypothesis_tests_edge_cases.py
"""

from __future__ import annotations

import math

import batcher as bt
import batcher.ml.stats as stats


def null_labels_are_dropped() -> None:
    """A row whose group is missing is not a third group: the result equals the clean data's."""
    clean = bt.from_pydict(
        {"latency": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0], "arm": list("aaabbbbb")}
    )
    messy = bt.from_pydict(
        {
            "latency": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 10.0],
            "arm": [*"aaabbbbb", None],
        }
    )
    for test in (stats.kruskal_wallis, stats.anova_test, stats.levene_test, stats.bartlett_test):
        on_clean = test(clean, "latency", "arm")
        on_messy = test(messy, "latency", "arm")
        print(f"{test.__name__:>15}: stat={on_messy.statistic:.4f} p={on_messy.pvalue:.4f}")
        assert math.isclose(on_messy.statistic, on_clean.statistic, rel_tol=1e-9)
        assert math.isclose(on_messy.pvalue, on_clean.pvalue, rel_tol=1e-9)
    # Kruskal-Wallis on the two real groups: H = 5.0 exactly, as scipy.stats.kruskal reports.
    assert math.isclose(stats.kruskal_wallis(messy, "latency", "arm").statistic, 5.0)
    # The effect sizes read off the same rows, so they drop the null label too.
    assert stats.eta_squared(messy, "latency", "arm") == stats.eta_squared(clean, "latency", "arm")


def nan_propagates() -> None:
    """One NaN makes the answer NaN. It never reads as ``p = 0``."""
    with_nan = bt.from_pydict({"x": [1.0, 2.0, 3.0, float("nan"), 4.0]})
    result = stats.t_test_1samp(with_nan, 0.0, "x")
    print("t test with a NaN:", result.statistic, result.pvalue)
    assert math.isnan(result.statistic) and math.isnan(result.pvalue)

    # A null is different: it is a missing value, skipped like any SQL aggregate skips it.
    with_null = bt.from_pydict({"x": [1.0, 2.0, 3.0, None, 4.0]})
    skipped = stats.t_test_1samp(with_null, 0.0, "x")
    assert skipped.n == 4 and skipped.df == 3.0
    assert 0.0 < skipped.pvalue < 0.05

    # The robust estimators follow the same rule.
    assert math.isnan(stats.trimmed_mean(with_nan, "x"))
    assert stats.trimmed_mean(with_null, "x", proportion=0.0) == 2.5


def degenerate_samples_are_nan() -> None:
    """Too little data is an undefined statistic, not a crash."""
    one_row_each = bt.from_pydict({"v": [1.0, 2.0], "g": ["a", "b"]})
    assert math.isnan(stats.t_test_ind(one_row_each, "v", "g").pvalue)
    assert math.isnan(stats.bartlett_test(one_row_each, "v", "g").pvalue)

    # Constant groups that differ are an infinitely significant difference, as in SciPy.
    constant = bt.from_pydict({"v": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0], "g": list("aaabbb")})
    welch = stats.t_test_ind(constant, "v", "g")
    assert welch.statistic == -math.inf and welch.pvalue == 0.0
    # ...and constant groups that agree have nothing to test.
    same = bt.from_pydict({"v": [5.0] * 6, "g": list("aaabbb")})
    assert math.isnan(stats.anova_test(same, "v", "g").pvalue)
    assert math.isnan(stats.kruskal_wallis(same, "v", "g").pvalue)

    # Paired differences that are all zero leave nothing to rank: statistic 0, p-value NaN.
    unchanged = bt.from_pydict({"before": [1.0, 2.0, 3.0], "after": [1.0, 2.0, 3.0]})
    paired = stats.wilcoxon_signed_rank(unchanged, "before", "after")
    assert paired.statistic == 0.0 and math.isnan(paired.pvalue)

    # Two rows cannot say how three columns inflate each other.
    tiny = bt.from_pydict({"a": [1.0, 2.0], "b": [1.0, 3.0], "c": [2.0, 1.0]})
    assert all(
        math.isnan(v) for v in stats.variance_inflation_factor(tiny, ["a", "b", "c"]).values()
    )


def one_group_is_a_plan_error() -> None:
    """A between-groups test with one group has no question to answer, so it says so."""
    one_group = bt.from_pydict({"v": [1.0, 2.0, 3.0], "g": ["a", "a", "a"]})
    for test in (stats.anova_test, stats.kruskal_wallis, stats.levene_test, stats.t_test_ind):
        try:
            test(one_group, "v", "g")
        except bt.PlanError as error:
            print(f"{test.__name__:>15}: {error}")
        else:
            raise AssertionError(f"{test.__name__} accepted a single group")


def main() -> None:
    null_labels_are_dropped()
    nan_propagates()
    degenerate_samples_are_nan()
    one_group_is_a_plan_error()


if __name__ == "__main__":
    main()
