"""Edge cases of the statistics, weighted aggregates, text metrics and prompt helpers, vs oracles.

The happy path of every function here is already pinned against SciPy or a hand count. What
this file pins is the input the happy path never sees: a NaN, a null group label, a group of
one row, constant data, an empty relation, a two-word generation scored at 4-gram BLEU. Every
case below was a wrong answer before it was a test -- a one-sample t test on a column holding a
NaN reported ``t = inf, p = 0``; Kruskal-Wallis counted a null label as a third group; a
weighted variance read 1.56 where the complete rows give 1.0; BLEU scored 1.0 for a pair it
defines as 0.

The oracles are the reference implementations, never the engine: ``scipy.stats`` and
``numpy`` for the statistics, and for BLEU/ROUGE a short reference written from the
definitions (sacrebleu and nltk are not installed here), applying the same SQuAD
normalization the engine documents so the comparison is of the metric rather than of two
tokenizers.
"""

from __future__ import annotations

import collections
import math
import re
import string
import warnings

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
import batcher.ml.stats as S
from batcher._internal.errors import PlanError

scipy_stats = pytest.importorskip("scipy.stats")

pytestmark = pytest.mark.differential

NAN = float("nan")

#: String schemas, so a column that is all null in one case is still text, not type null.
_PR = pa.schema([("p", pa.string()), ("r", pa.string())])
_S = pa.schema([("s", pa.string())])


def _same(got: float, want: float, rel: float = 1e-9) -> bool:
    """Equal as floats, with NaN equal to NaN and matching infinities equal."""
    got, want = float(got), float(want)
    if math.isnan(want):
        return math.isnan(got)
    if math.isinf(want):
        return got == want
    return math.isclose(got, want, rel_tol=rel, abs_tol=1e-12)


def _scipy(fn, *args, **kwargs):
    """Call a SciPy test with its small-sample and degenerate-input warnings silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def _groups(values: list[float], labels: list[str | None]) -> bt.Dataset:
    return bt.from_pydict({"v": values, "g": labels})


# --------------------------------------------------------------------------------------------
# 1. A NaN is a value, and it propagates -- never a p-value of 0
# --------------------------------------------------------------------------------------------


def test_one_sample_t_on_a_nan_matches_scipy_propagate():
    column = [1.0, 2.0, 3.0, NAN, 4.0]
    got = S.t_test_1samp(bt.from_pydict({"v": column}), 0.0, "v")
    want = _scipy(scipy_stats.ttest_1samp, column, 0.0)
    assert math.isnan(want.statistic) and math.isnan(want.pvalue)
    assert math.isnan(got.statistic) and math.isnan(got.pvalue)
    assert math.isnan(got.effect_size)


def test_one_sample_t_skips_a_null_like_scipy_omits_it():
    got = S.t_test_1samp(bt.from_pydict({"v": [1.0, 2.0, 3.0, None, 4.0]}), 0.0, "v")
    want = scipy_stats.ttest_1samp([1.0, 2.0, 3.0, 4.0], 0.0)
    assert _same(got.statistic, want.statistic) and _same(got.pvalue, want.pvalue)
    assert got.df == 3.0 and got.n == 4


_NAN_TWO_GROUP = ([1.0, 2.0, 3.0, NAN, 4.0, 5.0, 6.0, 7.0, 9.0], ["a"] * 4 + ["b"] * 5)


@pytest.mark.parametrize(
    "name",
    ["t_test_ind", "mann_whitney_u", "kruskal_wallis", "anova_test", "levene_test",
     "bartlett_test"],
)  # fmt: skip
def test_a_nan_value_makes_every_group_test_nan(name):
    """SciPy's default ``nan_policy="propagate"`` answers NaN for all six; none may report a
    finite p-value, and above all not the ``p = 0`` a NaN standard error used to produce."""
    values, labels = _NAN_TWO_GROUP
    got = getattr(S, name)(_groups(values, labels), "v", "g")
    a = [v for v, g in zip(values, labels, strict=True) if g == "a"]
    b = [v for v, g in zip(values, labels, strict=True) if g == "b"]
    oracle = {
        "t_test_ind": lambda: scipy_stats.ttest_ind(a, b, equal_var=False),
        "mann_whitney_u": lambda: scipy_stats.mannwhitneyu(a, b, method="asymptotic"),
        "kruskal_wallis": lambda: scipy_stats.kruskal(a, b),
        "anova_test": lambda: scipy_stats.f_oneway(a, b),
        "levene_test": lambda: scipy_stats.levene(a, b),
        "bartlett_test": lambda: scipy_stats.bartlett(a, b),
    }[name]
    want = _scipy(oracle)
    assert math.isnan(want.pvalue)
    assert math.isnan(got.pvalue), got
    assert math.isnan(got.statistic), got


def test_wilcoxon_on_a_nan_difference_is_nan():
    x, y = [1.0, 2.0, NAN, 4.0, 7.0], [2.0, 1.0, 3.0, 1.0, 1.0]
    got = S.wilcoxon_signed_rank(bt.from_pydict({"x": x, "y": y}), "x", "y")
    want = _scipy(scipy_stats.wilcoxon, x, y, method="approx")
    assert math.isnan(want.pvalue)
    assert math.isnan(got.statistic) and math.isnan(got.pvalue)


@pytest.mark.parametrize(
    ("name", "oracle"),
    [
        ("trimmed_mean", lambda z: scipy_stats.trim_mean(z, 0.1)),
        ("median_abs_deviation", lambda z: scipy_stats.median_abs_deviation(z, scale="normal")),
        ("mean_abs_deviation", lambda z: float(np.mean(np.abs(z - np.mean(z))))),
        ("winsorized_mean", lambda z: float(np.mean(z))),
    ],
)
def test_a_robust_estimate_over_a_nan_is_nan(name, oracle):
    """A quantile sorts NaN past every number, so the trim and the median used to discard it
    quietly and answer as if it were not there. NumPy and SciPy propagate it."""
    column = np.array([1.0, 2.0, 3.0, NAN, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    assert math.isnan(_scipy(oracle, column))
    assert math.isnan(getattr(S, name)(bt.from_pydict({"x": list(column)}), "x"))


# --------------------------------------------------------------------------------------------
# 2. Degenerate input: NaN like SciPy, or a typed PlanError where SciPy raises
# --------------------------------------------------------------------------------------------

_EMPTY = bt.from_pydict({"v": [1.0], "g": ["a"], "x": [1.0], "y": [1.0]}).filter(
    bt.col("v") > bt.lit(9.0)
)


@pytest.mark.parametrize(
    ("label", "values", "labels"),
    [
        ("one row per group", [1.0, 2.0], ["a", "b"]),
        ("all constant", [5.0] * 6, ["a"] * 3 + ["b"] * 3),
        ("constant, groups differ", [1.0, 1.0, 1.0, 2.0, 2.0, 2.0], ["a"] * 3 + ["b"] * 3),
        ("one constant group", [5.0, 5.0, 5.0, 1.0, 5.0, 9.0], ["a"] * 3 + ["b"] * 3),
        ("singleton beside a sample", [1.0, 2.0, 3.0, 4.0], ["a", "b", "b", "b"]),
    ],
)
@pytest.mark.parametrize(
    ("name", "oracle"),
    [
        ("t_test_ind", lambda a, b: scipy_stats.ttest_ind(a, b, equal_var=False)),
        ("anova_test", scipy_stats.f_oneway),
        ("levene_test", scipy_stats.levene),
        ("bartlett_test", scipy_stats.bartlett),
        ("kruskal_wallis", scipy_stats.kruskal),
    ],
)
def test_a_degenerate_two_group_sample_matches_scipy(label, values, labels, name, oracle):
    """Each of these raised a raw ``TypeError``, ``ZeroDivisionError`` or a math domain error
    somewhere in the grid; SciPy answers every one with a number, NaN or infinite."""
    got = getattr(S, name)(_groups(values, labels), "v", "g")
    a = [v for v, g in zip(values, labels, strict=True) if g == "a"]
    b = [v for v, g in zip(values, labels, strict=True) if g == "b"]
    want = _scipy(oracle, a, b)
    assert _same(got.statistic, want.statistic, rel=1e-7), (label, got, want)
    assert _same(got.pvalue, want.pvalue, rel=1e-6), (label, got, want)


@pytest.mark.parametrize(
    "name", ["t_test_ind", "anova_test", "levene_test", "bartlett_test", "kruskal_wallis"]
)
@pytest.mark.parametrize(
    ("label", "ds"),
    [
        ("empty", _EMPTY),
        ("one group", _groups([1.0, 2.0, 3.0], ["a", "a", "a"])),
        ("only null labels", _groups([1.0, 2.0], [None, None])),
    ],
)
def test_too_few_groups_is_a_plan_error_not_a_raw_exception(name, label, ds):
    """SciPy raises for fewer than two samples too; the error here is typed and says what to do."""
    with pytest.raises(PlanError, match="groups in 'g'"):
        getattr(S, name)(ds, "v", "g")


def test_t_test_ind_still_rejects_three_groups():
    with pytest.raises(PlanError, match="exactly two groups"):
        S.t_test_ind(_groups([1.0, 2.0, 3.0], ["a", "b", "c"]), "v", "g")


@pytest.mark.parametrize(
    ("label", "column", "popmean"),
    [
        ("empty", [], 0.0),
        ("one row", [1.0], 0.0),
        ("constant, off the mean", [2.0, 2.0, 2.0], 0.0),
        ("constant, on the mean", [2.0, 2.0, 2.0], 2.0),
    ],
)
def test_one_sample_t_degenerate_matches_scipy(label, column, popmean):
    ds = bt.from_pydict({"v": [*column, 99.0]}).filter(bt.col("v") < bt.lit(99.0))
    got = S.t_test_1samp(ds, popmean, "v")
    want = _scipy(scipy_stats.ttest_1samp, column, popmean)
    assert _same(got.statistic, want.statistic), (label, got, want)
    assert _same(got.pvalue, want.pvalue), (label, got, want)
    assert _same(got.df, want.df), (label, got, want)


@pytest.mark.parametrize(
    ("label", "x", "y"),
    [
        ("every difference zero", [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
        ("one pair", [1.0], [2.0]),
    ],
)
def test_wilcoxon_degenerate_matches_scipy(label, x, y):
    got = S.wilcoxon_signed_rank(bt.from_pydict({"x": x, "y": y}), "x", "y")
    want = _scipy(scipy_stats.wilcoxon, x, y, method="approx", correction=True)
    assert _same(got.statistic, want.statistic), (label, got, want)
    assert _same(got.pvalue, want.pvalue), (label, got, want)


def test_wilcoxon_on_no_pairs_is_nan():
    got = S.wilcoxon_signed_rank(_EMPTY, "x", "y")
    assert math.isnan(got.statistic) and math.isnan(got.pvalue)


@pytest.mark.parametrize(
    "name", ["trimmed_mean", "winsorized_mean", "median_abs_deviation", "mean_abs_deviation"]
)
def test_a_robust_estimate_of_no_rows_is_nan(name):
    assert math.isnan(_scipy(scipy_stats.trim_mean, np.array([]), 0.1))
    assert math.isnan(getattr(S, name)(_EMPTY, "v"))


def _vif_by_regression(x: np.ndarray, j: int) -> float:
    others = np.column_stack([np.ones(len(x)), np.delete(x, j, axis=1)])
    coef, *_ = np.linalg.lstsq(others, x[:, j], rcond=None)
    residual = x[:, j] - others @ coef
    return 1.0 / (residual.var() / x[:, j].var())


def test_vif_matches_the_regression_definition_and_is_at_least_one():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(200, 4))
    x[:, 3] = 0.8 * x[:, 0] + 0.3 * x[:, 1] + rng.normal(0.0, 0.3, 200)
    ds = bt.from_pydict({c: list(x[:, i]) for i, c in enumerate("abcd")})
    got = S.variance_inflation_factor(ds, list("abcd"))
    for j, name in enumerate("abcd"):
        assert got[name] >= 1.0
        assert got[name] == pytest.approx(_vif_by_regression(x, j), rel=1e-6)


def test_vif_on_no_more_rows_than_columns_is_nan_not_below_one():
    """Two rows, three columns: every regression fits exactly, and the inverse of the
    correlation matrix used to report 0.11 -- a VIF below its floor of 1."""
    ds = bt.from_pydict({"x": [1.0, 2.0], "y": [1.0, 3.0], "v": [2.0, 1.0]})
    got = S.variance_inflation_factor(ds, ["x", "y", "v"])
    assert all(math.isnan(v) for v in got.values()), got


def test_vif_of_an_exact_linear_combination_is_infinite():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(50, 3))
    x[:, 2] = x[:, 0] + x[:, 1]
    ds = bt.from_pydict({c: list(x[:, i]) for i, c in enumerate("abc")})
    assert all(math.isinf(v) for v in S.variance_inflation_factor(ds, list("abc")).values())


def test_vif_with_a_constant_column_is_nan_not_a_linalg_error():
    ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [1.0] * 5})
    assert all(math.isnan(v) for v in S.variance_inflation_factor(ds, ["x", "y"]).values())


# --------------------------------------------------------------------------------------------
# 3. A null group label is dropped, never a group of its own
# --------------------------------------------------------------------------------------------

_NULL_LABEL_VALUES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 10.0]
_NULL_LABELS = ["a", "a", "a", "b", "b", "b", "b", "b", None]
_A, _B = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0, 9.0]


@pytest.mark.parametrize(
    ("name", "oracle"),
    [
        ("kruskal_wallis", scipy_stats.kruskal),
        ("anova_test", scipy_stats.f_oneway),
        ("levene_test", scipy_stats.levene),
        ("bartlett_test", scipy_stats.bartlett),
        ("t_test_ind", lambda a, b: scipy_stats.ttest_ind(a, b, equal_var=False)),
        ("mann_whitney_u", lambda a, b: scipy_stats.mannwhitneyu(a, b, method="asymptotic")),
    ],
)
def test_a_null_label_row_is_dropped_like_scipy_never_sees_it(name, oracle):
    """Kruskal-Wallis read H = 7.64 by ranking the null row as a third group (5.0 without it);
    Levene raised ``Utf8 == Int64`` splicing the null label in as a literal."""
    got = getattr(S, name)(_groups(_NULL_LABEL_VALUES, _NULL_LABELS), "v", "g")
    want = oracle(_A, _B)
    assert _same(got.statistic, want.statistic, rel=1e-9), (got, want)
    assert _same(got.pvalue, want.pvalue, rel=1e-6), (got, want)


def test_anova_degrees_of_freedom_count_only_labelled_rows():
    got = S.anova_test(_groups(_NULL_LABEL_VALUES, _NULL_LABELS), "v", "g")
    assert got.df == (1.0, 6.0)
    assert got.n == 8


def _anova_parts(groups: list[list[float]]) -> tuple[float, float, int, int]:
    pooled = np.concatenate(groups)
    grand = pooled.mean()
    ss_between = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups)
    ss_within = sum(((np.asarray(g) - np.mean(g)) ** 2).sum() for g in groups)
    return ss_between, ss_within, len(pooled), len(groups)


@pytest.mark.parametrize("name", ["eta_squared", "epsilon_squared", "omega_squared", "cohens_f"])
def test_anova_effect_sizes_drop_null_labels(name):
    ssb, ssw, n, k = _anova_parts([_A, _B])
    msw = ssw / (n - k)
    want = {
        "eta_squared": ssb / (ssb + ssw),
        "epsilon_squared": (ssb - (k - 1) * msw) / (ssb + ssw),
        "omega_squared": (ssb - (k - 1) * msw) / (ssb + ssw + msw),
        "cohens_f": math.sqrt(ssb / ssw),
    }[name]
    got = getattr(S, name)(_groups(_NULL_LABEL_VALUES, _NULL_LABELS), "v", "g")
    assert got == pytest.approx(want, rel=1e-9)


def test_levene_accepts_an_integer_group_column_with_a_null():
    ds = bt.from_pydict({"v": [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0], "g": [0, 0, 0, 1, 1, 1, None]})
    got = S.levene_test(ds, "v", "g")
    want = scipy_stats.levene([1.0, 2.0, 3.0], [4.0, 5.0, 7.0])
    assert _same(got.statistic, want.statistic) and _same(got.pvalue, want.pvalue, rel=1e-6)


def test_three_groups_with_a_null_label_match_scipy():
    rng = np.random.default_rng(0)
    groups = [
        np.round(rng.normal(m, s, n), 2) for m, s, n in [(0, 1, 20), (0.5, 2, 15), (1, 1.5, 25)]
    ]
    values = [float(v) for g in groups for v in g] + [100.0, -100.0]
    labels = [k for k, g in zip("abc", groups, strict=True) for _ in g] + [None, None]
    ds = _groups(values, labels)
    for name, oracle in [
        ("kruskal_wallis", scipy_stats.kruskal),
        ("anova_test", scipy_stats.f_oneway),
        ("levene_test", scipy_stats.levene),
        ("bartlett_test", scipy_stats.bartlett),
    ]:
        got, want = getattr(S, name)(ds, "v", "g"), oracle(*groups)
        assert _same(got.statistic, want.statistic, rel=1e-9), name
        assert _same(got.pvalue, want.pvalue, rel=1e-6), name


# --------------------------------------------------------------------------------------------
# 4. Weighted aggregates drop a null row from every sum at once
# --------------------------------------------------------------------------------------------


def _complete(*columns: list) -> list[np.ndarray]:
    keep = [all(c[i] is not None for c in columns) for i in range(len(columns[0]))]
    return [np.array([v for v, k in zip(c, keep, strict=True) if k], dtype=float) for c in columns]


def _wcov(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    return float(np.cov(x, y, aweights=w, bias=True)[0, 1])


_WEIGHTED_CASES = [
    ("null value", [1.0, None, 3.0], [1.0, 5.0, 3.0], [1.0, 1.0, 1.0]),
    ("null weight", [1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 4.0, 3.0], [1.0, 2.0, None, 1.0]),
    (
        "nulls everywhere",
        [1.0, 2.0, None, 4.0, 5.0, 7.0],
        [1.0, None, 3.0, 4.0, 4.5, 8.0],
        [1.0, 1.0, 1.0, None, 2.0, 0.5],
    ),
]


@pytest.mark.parametrize(("label", "x", "y", "w"), _WEIGHTED_CASES)
def test_weighted_moments_match_numpy_over_complete_rows(label, x, y, w):
    ds = bt.from_pydict({"x": x, "y": y, "w": w})
    got = ds.agg(
        var=bt.weighted_var("x", "w"),
        std=bt.weighted_std("x", "w"),
        cov=bt.weighted_covariance("x", "y", "w"),
        corr=bt.weighted_correlation("x", "y", "w"),
    ).to_pydict()
    xv, wv = _complete(x, w)
    mean = np.average(xv, weights=wv)
    want_var = float(np.average((xv - mean) ** 2, weights=wv))
    assert got["var"][0] == pytest.approx(want_var, rel=1e-9), label
    assert got["std"][0] == pytest.approx(math.sqrt(want_var), rel=1e-9), label
    xc, yc, wc = _complete(x, y, w)
    cov = _wcov(xc, yc, wc)
    assert got["cov"][0] == pytest.approx(cov, rel=1e-9, abs=1e-12), label
    corr = cov / math.sqrt(_wcov(xc, xc, wc) * _wcov(yc, yc, wc))
    assert got["corr"][0] == pytest.approx(corr, rel=1e-9), label


def test_the_audit_repro_reads_the_complete_rows():
    ds = bt.from_pydict({"x": [1.0, None, 3.0], "y": [1.0, 2.0, 3.0], "w": [1.0, 1.0, 1.0]})
    got = ds.agg(v=bt.weighted_var("x", "w"), r=bt.weighted_correlation("x", "y", "w"))
    out = got.to_pydict()
    assert out["v"][0] == pytest.approx(1.0)
    assert out["r"][0] == pytest.approx(1.0)


def test_weighted_moments_per_group_match_numpy():
    ds = bt.from_pydict(
        {
            "k": ["p", "p", "p", "q", "q", "q", "q"],
            "x": [1.0, None, 3.0, 2.0, 4.0, 6.0, None],
            "w": [1.0, 2.0, 1.0, 1.0, None, 2.0, 1.0],
        }
    )
    got = ds.group_by("k").agg(v=bt.weighted_var("x", "w")).sort("k").to_pydict()
    want = []
    for xs, ws in [([1.0, 3.0], [1.0, 1.0]), ([2.0, 6.0], [1.0, 2.0])]:
        m = np.average(xs, weights=ws)
        want.append(float(np.average((np.array(xs) - m) ** 2, weights=ws)))
    assert got["k"] == ["p", "q"]
    assert got["v"] == pytest.approx(want)


# --------------------------------------------------------------------------------------------
# 5. Text metrics against a reference written from the definitions
# --------------------------------------------------------------------------------------------


def _norm(s: str) -> str:
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def _toks(s: str | None) -> list[str]:
    return _norm(s or "").split()


def _grams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _clip(p: list[str], r: list[str], n: int) -> tuple[int, int, int]:
    pred, ref = collections.Counter(_grams(p, n)), collections.Counter(_grams(r, n))
    return sum(min(c, ref[g]) for g, c in pred.items()), sum(pred.values()), sum(ref.values())


def _ref_bleu(p: str | None, r: str | None, max_n: int = 4) -> float:
    pt, rt = _toks(p), _toks(r)
    if not pt:
        return 0.0
    product = 1.0
    for n in range(1, max_n + 1):
        overlap, pred_len, _ = _clip(pt, rt, n)
        product *= overlap / pred_len if pred_len else 0.0
    bp = 1.0 if len(pt) >= len(rt) else math.exp(1 - len(rt) / len(pt))
    return bp * product ** (1 / max_n)


def _ref_brevity(p: str | None, r: str | None) -> float:
    pt, rt = _toks(p), _toks(r)
    if not pt:
        return 0.0
    return 1.0 if len(pt) >= len(rt) else math.exp(1 - len(rt) / len(pt))


def _lcs(a: list[str], b: list[str]) -> int:
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            table[i + 1][j + 1] = (
                table[i][j] + 1 if x == y else max(table[i][j + 1], table[i + 1][j])
            )
    return table[-1][-1]


def _ref_rouge_l(p: str | None, r: str | None) -> float:
    pt, rt = _toks(p), _toks(r)
    lcs = _lcs(pt, rt)
    precision = lcs / len(pt) if pt else 0.0
    recall = lcs / len(rt) if rt else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _ref_precision(p: str | None, r: str | None, n: int) -> float:
    overlap, pred_len, _ = _clip(_toks(p), _toks(r), n)
    return overlap / pred_len if pred_len else 0.0


def _ref_recall(p: str | None, r: str | None, n: int) -> float:
    overlap, _, ref_len = _clip(_toks(p), _toks(r), n)
    return overlap / ref_len if ref_len else 0.0


_PAIRS = [
    ("the cat sat on the mat", "the cat sat on the mat"),
    ("cat", "cat"),
    ("cat sat", "cat sat"),
    ("cat sat down", "cat"),
    ("cat", "cat sat down now"),
    ("", ""),
    ("", "cat"),
    ("cat", ""),
    ("Hello, World!", "hello world"),
    ("a b c d e", "a b c d f"),
    ("x y z", "z y x"),
    ("one two three four five", "one two three four five six"),
    ("one two three four five", "two three four five"),
    (None, "cat sat"),
    ("cat sat", None),
    (None, None),
]


@pytest.mark.parametrize(("p", "r"), _PAIRS)
def test_ngram_metrics_match_the_reference_definitions(p, r):
    """A row shorter than `n` tokens has no n-gram of that order. It used to be padded into
    one, so ``bleu("cat sat", "cat sat")`` read 1.0 where the definition gives 0."""
    ds = bt.from_pydict({"p": [p], "r": [r]}, schema=_PR)
    got = ds.agg(
        b4=bt.bleu("p", "r"),
        b2=bt.bleu("p", "r", max_n=2),
        bp=bt.brevity_penalty("p", "r"),
        rl=bt.rouge_l_f1("p", "r"),
        p2=bt.ngram_precision("p", "r", n=2),
        r2=bt.ngram_recall("p", "r", n=2),
    ).to_pydict()
    want = {
        "b4": _ref_bleu(p, r),
        "b2": _ref_bleu(p, r, 2),
        "bp": _ref_brevity(p, r),
        "rl": _ref_rouge_l(p, r),
        "p2": _ref_precision(p, r, 2),
        "r2": _ref_recall(p, r, 2),
    }
    for key, value in want.items():
        assert got[key][0] == pytest.approx(value, abs=1e-12), (key, p, r, got[key][0], value)


def test_corpus_bleu_is_the_mean_of_the_sentence_scores():
    pairs = list(_PAIRS)
    ds = bt.from_pydict(
        {"p": [p for p, _ in pairs], "r": [r for _, r in pairs]},
        schema=_PR,
    )
    got = ds.agg(b=bt.bleu("p", "r", max_n=2), rl=bt.rouge_l_f1("p", "r")).to_pydict()
    assert got["b"][0] == pytest.approx(np.mean([_ref_bleu(p, r, 2) for p, r in pairs]))
    assert got["rl"][0] == pytest.approx(np.mean([_ref_rouge_l(p, r) for p, r in pairs]))


def test_brevity_penalty_of_a_null_prediction_is_zero_not_e():
    ds = bt.from_pydict({"p": [None], "r": ["cat sat"]}, schema=_PR)
    got = ds.agg(bp=bt.brevity_penalty("p", "r")).to_pydict()["bp"][0]
    assert got == 0.0
    assert got <= 1.0, "a brevity penalty above 1 is outside its own definition"


def test_distinct_and_novel_ngrams_of_a_too_short_row_are_empty():
    ds = bt.from_pydict({"p": ["cat"], "r": ["dog"]})
    got = ds.agg(
        d=bt.distinct_ngram_ratio("p", n=2), nv=bt.ngram_novelty("p", "r", n=2)
    ).to_pydict()
    assert got == {"d": [0.0], "nv": [0.0]}


# --------------------------------------------------------------------------------------------
# 6. One null rule: a null prediction or reference scores exactly as an empty string does
# --------------------------------------------------------------------------------------------

_PAIRED_METRICS = {
    "bleu": lambda: bt.bleu("p", "r", max_n=2),
    "brevity_penalty": lambda: bt.brevity_penalty("p", "r"),
    "ngram_precision": lambda: bt.ngram_precision("p", "r"),
    "ngram_recall": lambda: bt.ngram_recall("p", "r"),
    "ngram_f1": lambda: bt.ngram_f1("p", "r"),
    "ngram_novelty": lambda: bt.ngram_novelty("p", "r", n=1),
    "rouge_l_f1": lambda: bt.rouge_l_f1("p", "r"),
    "rouge_l_precision": lambda: bt.rouge_l_precision("p", "r"),
    "rouge_l_recall": lambda: bt.rouge_l_recall("p", "r"),
    "token_set_f1": lambda: bt.token_set_f1("p", "r"),
    "token_set_jaccard": lambda: bt.token_set_jaccard("p", "r"),
    "char_ngram_f1": lambda: bt.char_ngram_f1("p", "r"),
    "char_ngram_jaccard": lambda: bt.char_ngram_jaccard("p", "r"),
    "length_ratio": lambda: bt.length_ratio("p", "r"),
}

_WITH_NULLS = {"p": ["cat sat", None, "cat sat on mat", None], "r": ["cat sat", "cat", None, None]}
_AS_EMPTY = {"p": ["cat sat", "", "cat sat on mat", ""], "r": ["cat sat", "cat", "", ""]}


@pytest.mark.parametrize("name", sorted(_PAIRED_METRICS))
def test_a_null_side_scores_as_the_empty_string(name):
    """BLEU dropped a row with a null reference while ROUGE-L scored it 0, so the two read
    different corpora. Now both sides of every paired metric read null as ``""``."""
    expr = _PAIRED_METRICS[name]
    got = bt.from_pydict(_WITH_NULLS).agg(m=expr()).to_pydict()["m"][0]
    want = bt.from_pydict(_AS_EMPTY).agg(m=expr()).to_pydict()["m"][0]
    assert got is not None
    assert got == pytest.approx(want), (name, got, want)


def test_exact_match_never_matches_a_null():
    """The exact-match family keeps SQL equality: ``null == null`` is not a match, and a null
    row still counts in the denominator as a miss."""
    ds = bt.from_pydict({"p": ["a", None, None, "b"], "r": ["a", "a", None, "c"]})
    got = ds.agg(em=bt.exact_match("p", "r"), nem=bt.normalized_exact_match("p", "r"))
    assert got.to_pydict() == {"em": [0.25], "nem": [0.25]}


# --------------------------------------------------------------------------------------------
# 5 (cont.). Readability, sentence and word shape, and the prompt and safety helpers
# --------------------------------------------------------------------------------------------


def _ref_sentences(s: str) -> int:
    """Runs of terminators ending at whitespace or end of text; CJK terminators always end."""
    count, i = 0, 0
    while i < len(s):
        if s[i] in "\u3002\uff01\uff1f":
            while i < len(s) and s[i] in "\u3002\uff01\uff1f":
                i += 1
            count += 1
        elif s[i] in ".!?\u2026":
            while i < len(s) and s[i] in ".!?\u2026":
                i += 1
            while i < len(s) and s[i] in "\"')\u201d\u2019]":
                i += 1
            if i == len(s) or s[i] in " \t\n\r":
                count += 1
        else:
            i += 1
    return count


def _ref_ari(s: str | None) -> float | None:
    if not s or not s.split():
        return None
    chars = sum(c.isalnum() for c in s)
    words = len(s.split())
    sentences = max(_ref_sentences(s), 1)
    return 4.71 * chars / words + 0.5 * words / sentences - 21.43


_PROSE = [
    "The cat sat on the mat. It was warm.",
    "Wait... what?",
    "Dr. Smith, a well-known surgeon, operated for 3.5 hours!",
    "no terminator at all",
    "\u732b\u304c\u5ea7\u3063\u305f\u3002\u732b\u304c\u5bdd\u305f\u3002",
    'He said "stop." Then he left.',
    "ok!!! fine???",
    "Pi is 3.14 and e is 2.71.",
    "",
    "   ",
    None,
]


@pytest.mark.parametrize("text", _PROSE)
def test_sentence_count_matches_the_documented_definition(text):
    got = bt.from_pydict({"s": [text]}, schema=_S).select(n=bt.col("s").str.sentence_count())
    want = None if text is None else _ref_sentences(text)
    assert got.to_pydict()["n"][0] == want, text


def test_sentence_count_on_the_audit_cases():
    ds = bt.from_pydict({"s": ["Wait... what?", "\u732b\u304c\u5ea7\u3063\u305f\u3002", "3.14"]})
    assert ds.select(n=bt.col("s").str.sentence_count()).to_pydict() == {"n": [2, 1, 0]}


@pytest.mark.parametrize("text", _PROSE)
def test_ari_counts_letters_and_digits_and_skips_empty_rows(text):
    ds = bt.from_pydict({"s": [text]}, schema=_S)
    got = ds.agg(a=bt.automated_readability_index("s")).to_pydict()["a"][0]
    want = _ref_ari(text)
    if want is None:
        assert got is None, (text, got)
    else:
        assert got == pytest.approx(want), (text, got, want)


def test_ari_corpus_mean_ignores_the_empty_rows():
    ds = bt.from_pydict({"s": _PROSE}, schema=_S)
    got = ds.agg(a=bt.automated_readability_index("s")).to_pydict()["a"][0]
    scored = [a for a in map(_ref_ari, _PROSE) if a is not None]
    assert got == pytest.approx(sum(scored) / len(scored))


_WORDS = ["héllo wörld", "Привет мир", "alpha beta", "abc123 x_y", "café au lait", "", None]


@pytest.mark.parametrize("text", _WORDS)
def test_avg_word_length_counts_unicode_letters(text):
    """``[A-Za-z]`` read "héllo" as four letters and a Cyrillic word as none."""
    ds = bt.from_pydict({"s": [text]}, schema=_S)
    got = ds.select(w=bt.col("s").str.avg_word_length()).to_pydict()["w"][0]
    if text is None or not text.split():
        assert got is None
        return
    words = text.split()
    assert got == pytest.approx(sum(c.isalpha() for c in text) / len(words)), text


def test_truncate_middle_keeps_a_null_null():
    ds = bt.from_pydict({"t": ["abcdefghijklmnop", None, "abc"]})
    got = ds.select(r=bt.truncate_middle("t", budget=2, marker="~")).to_pydict()
    assert got == {"r": ["abcd~nop", None, "abc"]}


@pytest.mark.parametrize(
    ("text", "flagged"),
    [
        ("-----BEGIN RSA PRIVATE KEY-----", False),
        ("Intro\n---\nBody", False),
        ("a -- b", False),
        ("x' OR '1'='1", True),
        ("admin'--", True),
        ("1; DROP TABLE users", True),
    ],
)
def test_sql_injection_rate_ignores_armor_and_rules(text, flagged):
    got = bt.from_pydict({"t": [text]}).agg(r=bt.sql_injection_rate("t")).to_pydict()["r"][0]
    assert got == (1.0 if flagged else 0.0), text


@pytest.mark.parametrize(
    ("text", "repeated"),
    [
        ("First paragraph.\n\nSecond paragraph.\n\nThird paragraph.", False),
        ("x\ny\nz", False),
        ("a\na", True),
        ("same line\nother\nsame line  ", True),
    ],
)
def test_repeated_line_rate_ignores_paragraph_breaks(text, repeated):
    got = bt.from_pydict({"t": [text]}).agg(r=bt.repeated_line_rate("t")).to_pydict()["r"][0]
    assert got == (1.0 if repeated else 0.0), text
