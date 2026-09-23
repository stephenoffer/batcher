"""The model metrics at their edges, against scikit-learn.

Every case here is an input on which a metric used to answer differently from scikit-learn,
and each one is ordinary rather than exotic: a group holding one class, a null score, a tie,
a query with nothing relevant, a label that is not 0/1, a count past two billion squared.
The happy path agreed all along, which is how these survived.

scikit-learn is the oracle, per function and including its conventions for undefined
values: ``zero_division=0`` for the proportions, the classes present in ``y_true`` for
``balanced_accuracy_score``, NaN from ``class_likelihood_ratios`` and ``cohen_kappa_score``,
tie-averaged gains in ``ndcg_score``. Where scikit-learn *raises* (ROC AUC on one class, a
constant target in ``d2_tweedie_score``), Batcher answers NaN so a grouped report survives,
and the test pins that NaN explicitly rather than skipping the case.
"""

from __future__ import annotations

import math
import time
import warnings

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher.ml import metrics as M
from batcher.plan.expr_ir.nodes import rank

sk = pytest.importorskip("sklearn.metrics", reason="scikit-learn is the metrics oracle")

pytestmark = pytest.mark.differential

nan = float("nan")


def _agg(ds: bt.Dataset, expr: bt.Expr) -> float:
    return ds.agg(m=expr).to_pydict()["m"][0]


def _pair(y: list, p: list) -> bt.Dataset:
    return bt.from_pydict({"y": y, "p": p})


def _same(got: float, want: float, rel: float = 1e-12) -> None:
    assert got == pytest.approx(want, rel=rel, abs=rel, nan_ok=True), (got, want)


def _quiet(fn, *args, **kwargs):
    """Call a scikit-learn metric without its UndefinedMetricWarning noise."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


# --- 1. counts past int64 --------------------------------------------------------------


@pytest.mark.parametrize("half", [60_000, 100_000])
def test_matthews_corrcoef_does_not_overflow_on_a_large_group(half: int) -> None:
    # tp*tn*... for 120,000 perfectly correlated rows is ~1.3e19, past int64: it came back
    # NaN at 120k and 4.38 at 200k.
    y = [1] * half + [0] * half
    assert _agg(_pair(y, y), bt.matthews_corrcoef("y", "p")) == 1.0


def test_matthews_corrcoef_matches_sklearn_at_scale() -> None:
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 400_000)
    p = np.where(rng.random(y.size) < 0.8, y, 1 - y)
    got = _agg(bt.from_arrow(pa.table({"y": y, "p": p})), bt.matthews_corrcoef("y", "p"))
    _same(got, sk.matthews_corrcoef(y, p), rel=1e-9)
    kappa = _agg(bt.from_arrow(pa.table({"y": y, "p": p})), bt.cohen_kappa("y", "p"))
    _same(kappa, sk.cohen_kappa_score(y, p), rel=1e-9)


# --- 2. the 0/0 convention -------------------------------------------------------------

EDGE_CASES = [
    ([1, 1, 1], [1, 1, 0]),  # no negatives
    ([0, 0, 0], [0, 1, 0]),  # no positives
    ([1, 1], [1, 1]),  # one class, perfect
    ([0, 1, 0, 1], [0, 0, 0, 0]),  # nothing predicted positive
    ([0, 1, 0, 1], [1, 1, 1, 1]),  # nothing predicted negative
    ([1, 0, 1, 0], [1, 0, 1, 0]),  # perfect, both classes
    ([1, 0, 1, 1, 0, 0, 1], [1, 1, 0, 1, 0, 1, 1]),  # ordinary
]


@pytest.mark.parametrize(("y", "p"), EDGE_CASES)
def test_proportions_follow_zero_division_zero(y: list, p: list) -> None:
    ds = _pair(y, p)
    expected = {
        "precision": _quiet(sk.precision_score, y, p, zero_division=0),
        "recall": _quiet(sk.recall_score, y, p, zero_division=0),
        "f1_score": _quiet(sk.f1_score, y, p, zero_division=0),
        "jaccard_score": _quiet(sk.jaccard_score, y, p, zero_division=0),
        "specificity": _quiet(sk.recall_score, y, p, pos_label=0, zero_division=0),
        "negative_predictive_value": _quiet(sk.precision_score, y, p, pos_label=0, zero_division=0),
        "matthews_corrcoef": _quiet(sk.matthews_corrcoef, y, p),
        # fp / (fp + tn) with an empty denominator is 0: 1 - recall of class 0, where that
        # recall's own zero-division value is 1.
        "false_positive_rate": 1.0 - _quiet(sk.recall_score, y, p, pos_label=0, zero_division=1),
        "false_negative_rate": 1.0 - _quiet(sk.recall_score, y, p, zero_division=1),
        "false_discovery_rate": 1.0 - _quiet(sk.precision_score, y, p, zero_division=1),
        "false_omission_rate": 1.0 - _quiet(sk.precision_score, y, p, pos_label=0, zero_division=1),
    }
    for name, want in expected.items():
        _same(_agg(ds, getattr(bt, name)("y", "p")), want)
    fmi = math.sqrt(expected["precision"] * expected["recall"])
    _same(_agg(ds, bt.fowlkes_mallows_index("y", "p")), fmi)


@pytest.mark.parametrize(("y", "p"), EDGE_CASES)
def test_class_averaged_scores_use_the_classes_present(y: list, p: list) -> None:
    ds = _pair(y, p)
    _same(_agg(ds, bt.balanced_accuracy("y", "p")), _quiet(sk.balanced_accuracy_score, y, p))
    # The geometric mean over the same class set: the recalls of the classes y_true has.
    recalls = [
        _quiet(sk.recall_score, y, p, pos_label=c, zero_division=0) for c in (0, 1) if c in y
    ]
    _same(
        _agg(ds, bt.geometric_mean_score("y", "p")), float(np.prod(recalls)) ** (1 / len(recalls))
    )


@pytest.mark.parametrize(("y", "p"), EDGE_CASES)
def test_ratio_metrics_are_nan_where_sklearn_is(y: list, p: list) -> None:
    ds = _pair(y, p)
    lr_pos, lr_neg = _quiet(sk.class_likelihood_ratios, y, p, labels=[0, 1])
    _same(_agg(ds, bt.positive_likelihood_ratio("y", "p")), lr_pos)
    _same(_agg(ds, bt.negative_likelihood_ratio("y", "p")), lr_neg)
    _same(_agg(ds, bt.cohen_kappa("y", "p")), _quiet(sk.cohen_kappa_score, y, p))
    both_classes = 0 in y and 1 in y
    informed = _agg(ds, bt.informedness("y", "p"))
    if both_classes:
        _same(informed, _quiet(sk.balanced_accuracy_score, y, p, adjusted=True))
    else:
        # scikit-learn's adjusted balanced accuracy is -inf here; undefined is NaN.
        assert math.isnan(informed)
        assert math.isnan(_agg(ds, bt.prevalence_threshold("y", "p")))


def test_the_single_class_group_from_the_audit() -> None:
    ds = _pair([1, 1, 1], [1, 1, 0])
    assert _agg(ds, bt.false_positive_rate("y", "p")) == 0.0  # was 1.0
    _same(_agg(ds, bt.balanced_accuracy("y", "p")), 2 / 3)  # was 1/3
    _same(_agg(_pair([1, 1], [1, 1]), bt.balanced_accuracy("y", "p")), 1.0)  # was 0.5
    _same(_agg(ds, bt.geometric_mean_score("y", "p")), 2 / 3)  # was 0.0


def test_cohen_kappa_is_multiclass_with_labels_and_refuses_to_binarize_without() -> None:
    y, p = ["a", "b", "a", "c", "b", "c"], ["a", "a", "a", "c", "b", "b"]
    ds = _pair(y, p)
    got = _agg(ds, bt.cohen_kappa("y", "p", labels=["a", "b", "c"]))
    _same(got, sk.cohen_kappa_score(y, p))
    # Restricting the class set drops the other rows, as sklearn's labels= does.
    got = _agg(ds, bt.cohen_kappa("y", "p", labels=["a", "b"]))
    _same(got, sk.cohen_kappa_score(y, p, labels=["a", "b"]))
    # Without labels a three-class column is not silently scored a-versus-rest (0.5).
    assert math.isnan(_agg(ds, bt.cohen_kappa("y", "p", positive="a")))
    # ...while a genuinely binary string column still is.
    yb, pb = ["x", "y", "x", "y"], ["x", "x", "x", "y"]
    _same(_agg(_pair(yb, pb), bt.cohen_kappa("y", "p", positive="x")), sk.cohen_kappa_score(yb, pb))


# --- 3/4/10. rank metrics: one class, nulls, partitions ---------------------------------


def test_roc_auc_is_nan_for_a_single_class_group_on_every_shape() -> None:
    ds = bt.from_pydict(
        {
            "k": ["a"] * 3 + ["b"] * 4,
            "y": [1, 1, 1, 0, 1, 0, 1],
            "s": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.4],
        }
    )
    grouped = dict(zip(*M.roc_auc(ds, "y", "s", by="k").to_pydict().values(), strict=True))
    assert math.isnan(grouped["a"])
    _same(grouped["b"], sk.roc_auc_score([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4]))
    single = ds.filter(bt.col("k") == "a")
    assert math.isnan(M.roc_auc(single, "y", "s"))
    assert math.isnan(M.gini_coefficient(single, "y", "s"))
    assert math.isnan(M.ks_statistic(single, "y", "s"))
    report = M.evaluate(ds, "y", y_score="s", by="k", metrics=["roc_auc", "gini"]).to_pydict()
    at = report["k"].index("a")
    assert math.isnan(report["roc_auc"][at]) and math.isnan(report["gini"][at])
    # scikit-learn's average precision is 0.0 on a group with no positives.
    no_positive = bt.from_pydict({"y": [0, 0], "s": [0.1, 0.2]})
    assert M.average_precision(no_positive, "y", "s") == _quiet(
        sk.average_precision_score, [0, 0], [0.1, 0.2]
    )


@pytest.mark.parametrize(
    ("extra_y", "extra_s"),
    [(None, 0.95), (1, None), (0, None), (0, nan), (1, nan)],
)
def test_rank_metrics_drop_null_labels_and_null_or_nan_scores(extra_y, extra_s) -> None:
    y, s = [1, 0, 1, 0, 0, 1], [0.9, 0.1, 0.8, 0.3, 0.2, 0.25]
    ds = bt.from_arrow(
        pa.table(
            {"y": pa.array([*y, extra_y], pa.int64()), "s": pa.array([*s, extra_s], pa.float64())}
        )
    )
    _same(M.roc_auc(ds, "y", "s"), sk.roc_auc_score(y, s))
    _same(M.average_precision(ds, "y", "s"), sk.average_precision_score(y, s))
    positive, negative = np.array(s)[np.array(y) == 1], np.array(s)[np.array(y) == 0]
    thresholds = np.unique(s)
    ks = max(abs((positive <= t).mean() - (negative <= t).mean()) for t in thresholds)
    _same(M.ks_statistic(ds, "y", "s"), ks)
    ranking = bt.from_arrow(
        pa.table(
            {
                "q": [0] * 7,
                "s": pa.array([*s, extra_s], pa.float64()),
                "y": pa.array([*y, extra_y], pa.int64()),
            }
        )
    )
    _same(M.ndcg_at_k(ranking, "q", "s", "y", k=3), sk.ndcg_score([y], [s], k=3))


def _window_partitions(node: object) -> list[list]:
    """The ``partition_keys`` of every window operator in an IR tree."""
    found: list[list] = []
    if isinstance(node, dict):
        if node.get("op") == "window":
            found.append(node.get("partition_keys", []))
        for value in node.values():
            found.extend(_window_partitions(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_window_partitions(value))
    return found


def test_roc_auc_by_ranks_within_each_partition() -> None:
    rng = np.random.default_rng(7)
    keys = np.repeat(["a", "b", "c"], 200)
    y = rng.integers(0, 2, keys.size)
    # Group c's scores all sit above group a's, so a global rank would change a's AUC.
    s = rng.random(keys.size) + np.select([keys == "a", keys == "b"], [0.0, 1.0], 2.0)
    ds = bt.from_arrow(pa.table({"k": keys, "y": y, "s": s}))
    per_group = M.roc_auc(ds, "y", "s", by="k")
    got = dict(zip(*per_group.to_pydict().values(), strict=True))
    for key in "abc":
        _same(got[key], sk.roc_auc_score(y[keys == key], s[keys == key]), rel=1e-12)
    # explain() labels this window "[global]", which is a rendering slip (the describer
    # reads an aggregate's `group_keys`); the IR the engine runs partitions it by `k`.
    partitioned = _window_partitions(per_group._plan.to_ir())
    assert partitioned and all(p == [{"e": "col", "name": "k"}] for p in partitioned)
    # Positive control: the same probe does see an unpartitioned window as one.
    flat = ds.with_columns(r=rank().over(order_by=["s"]))
    assert _window_partitions(flat._plan.to_ir()) == [[]]


# --- 5/6. evaluate and compare_models -------------------------------------------------


@pytest.mark.parametrize(
    ("labels", "positive"), [(["no", "yes"], "yes"), ([1, 2], 2), ([False, True], True)]
)
def test_evaluate_accuracy_from_a_score_on_non_binary_encoded_labels(labels, positive) -> None:
    rng = np.random.default_rng(11)
    y = [labels[i] for i in rng.integers(0, 2, 2000)]
    s = rng.random(2000)
    ds = bt.from_pydict({"y": y, "s": s.tolist()})
    negative = labels[0]
    hard = [positive if v >= 0.5 else negative for v in s]
    got = M.evaluate(
        ds,
        "y",
        y_score="s",
        positive=positive,
        task="binary",
        metrics=["accuracy", "precision", "recall", "balanced_accuracy"],
    )
    _same(got["accuracy"], sk.accuracy_score(y, hard))  # was 0.246 against 0.503
    _same(got["precision"], sk.precision_score(y, hard, pos_label=positive))
    _same(got["recall"], sk.recall_score(y, hard, pos_label=positive))
    _same(got["balanced_accuracy"], sk.balanced_accuracy_score(y, hard))


def test_evaluate_leaves_an_unscored_row_out() -> None:
    ds = bt.from_arrow(
        pa.table({"y": [1, 0, 1, 0], "s": pa.array([0.9, 0.2, None, nan], pa.float64())})
    )
    got = M.evaluate(ds, "y", y_score="s", task="binary", metrics=["accuracy", "true_negatives"])
    assert got == {"accuracy": 1.0, "true_negatives": 1}


def test_compare_models_defaults_work_on_a_binary_task() -> None:
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 1000)
    a, b = rng.random(1000), rng.random(1000)
    ds = bt.from_pydict({"y": y.tolist(), "a": a.tolist(), "b": b.tolist()})
    table = M.compare_models(ds, "y", {"a": "a", "b": "b"}).to_pydict()  # raised PlanError
    assert table["model"] == ["a", "b"]
    assert "roc_auc" not in table and "log_loss" in table
    for i, column in enumerate((a, b)):
        _same(table["accuracy"][i], sk.accuracy_score(y, (column >= 0.5).astype(int)))
        _same(table["log_loss"][i], sk.log_loss(y, column), rel=1e-9)
    hard = M.compare_models(
        ds.with_columns(h=(bt.col("a") >= 0.5).cast("int64")), "y", {"h": "h"}, scores=False
    )
    assert "log_loss" not in hard.to_pydict() and "accuracy" in hard.to_pydict()
    with pytest.raises(bt.PlanError):
        M.compare_models(ds, "y", {"a": "a"}, metrics=["roc_auc"])


# --- 7. ranking ties, graded gains, empty queries -------------------------------------


def _ranking_grid(
    seed: int, queries: int = 60, per: int = 8
) -> tuple[np.ndarray, np.ndarray, bt.Dataset]:
    rng = np.random.default_rng(seed)
    # Scores on a coarse grid so most queries carry ties, some relevant and some not.
    scores = rng.integers(0, 4, (queries, per)) / 4.0
    gains = rng.integers(0, 4, (queries, per)) * (rng.random((queries, per)) < 0.5)
    gains[:3] = 0  # three queries with nothing relevant
    ds = bt.from_pydict(
        {
            "q": np.repeat(np.arange(queries), per).tolist(),
            "s": scores.ravel().tolist(),
            "g": gains.ravel().tolist(),
            "rel": (gains.ravel() > 0).astype(int).tolist(),
        }
    )
    return scores, gains, ds


@pytest.mark.parametrize("k", [1, 3, 8])
def test_ndcg_matches_sklearn_under_ties_graded_and_empty_queries(k: int) -> None:
    scores, gains, ds = _ranking_grid(21)
    binary = (gains > 0).astype(int)
    _same(M.ndcg_at_k(ds, "q", "s", "rel", k=k), sk.ndcg_score(binary, scores, k=k), rel=1e-9)
    _same(
        M.ndcg_at_k(ds, "q", "s", "g", k=k, graded=True),
        sk.ndcg_score(gains, scores, k=k),
        rel=1e-9,
    )


def test_ndcg_and_precision_average_an_all_tied_query() -> None:
    ds = bt.from_pydict({"q": [0] * 4, "s": [0.5] * 4, "y": [1, 0, 0, 0]})
    _same(M.ndcg_at_k(ds, "q", "s", "y", k=1), sk.ndcg_score([[1, 0, 0, 0]], [[0.5] * 4], k=1))
    _same(M.precision_at_k(ds, "q", "s", "y", k=1), 0.25)  # was 1.0
    _same(M.recall_at_k(ds, "q", "s", "y", k=2), 0.5)
    # The first-hit metrics place the tie at its last position, as sklearn's LRAP does.
    _same(
        M.mean_reciprocal_rank(ds, "q", "s", "y"),
        sk.label_ranking_average_precision_score([[1, 0, 0, 0]], [[0.5] * 4]),
    )
    assert M.hit_rate_at_k(ds, "q", "s", "y", k=3) == 0.0
    assert M.hit_rate_at_k(ds, "q", "s", "y", k=4) == 1.0


def test_map_matches_sklearn_average_precision_per_query_under_ties() -> None:
    scores, gains, ds = _ranking_grid(8)
    binary = (gains > 0).astype(int)
    per_query = [
        _quiet(sk.average_precision_score, b, s) if b.any() else 0.0
        for b, s in zip(binary, scores, strict=True)
    ]
    _same(M.map_at_k(ds, "q", "s", "rel", k=scores.shape[1]), float(np.mean(per_query)), rel=1e-9)


def test_mrr_matches_sklearn_lrap_with_one_relevant_item_under_ties() -> None:
    rng = np.random.default_rng(4)
    scores = rng.integers(0, 3, (40, 6)) / 3.0
    relevant = np.zeros((40, 6), dtype=int)
    relevant[np.arange(40), rng.integers(0, 6, 40)] = 1
    ds = bt.from_pydict(
        {
            "q": np.repeat(np.arange(40), 6).tolist(),
            "s": scores.ravel().tolist(),
            "y": relevant.ravel().tolist(),
        }
    )
    _same(
        M.mean_reciprocal_rank(ds, "q", "s", "y"),
        sk.label_ranking_average_precision_score(relevant, scores),
    )


def test_every_ranking_metric_keeps_a_query_with_nothing_relevant() -> None:
    ds = bt.from_pydict({"q": [0, 0, 1, 1], "s": [0.9, 0.1, 0.9, 0.1], "y": [1, 0, 0, 0]})
    for metric in (M.recall_at_k, M.ndcg_at_k, M.map_at_k, M.hit_rate_at_k):
        assert metric(ds, "q", "s", "y", k=2) == 0.5, metric.__name__
    assert M.mean_reciprocal_rank(ds, "q", "s", "y") == 0.5


@pytest.mark.parametrize("k", [1, 2, 3])
def test_top_k_accuracy_breaks_ties_like_sklearn(k: int) -> None:
    rng = np.random.default_rng(9)
    probs = rng.integers(0, 3, (300, 4)) / 3.0
    y = rng.integers(0, 4, 300)
    ds = bt.from_pydict({"y": y.tolist(), **{f"c{i}": probs[:, i].tolist() for i in range(4)}})
    got = M.top_k_accuracy(ds, "y", [f"c{i}" for i in range(4)], k=k)
    _same(got, sk.top_k_accuracy_score(y, probs, k=k, labels=[0, 1, 2, 3]))


# --- 8. regression edges ------------------------------------------------------------


def test_nash_sutcliffe_on_a_constant_target_follows_r2() -> None:
    for pred in ([3.0, 2.0, 4.0], [3.0, 3.0, 3.0]):
        ds = _pair([3.0, 3.0, 3.0], pred)
        want = sk.r2_score([3.0, 3.0, 3.0], pred)
        _same(_agg(ds, bt.nash_sutcliffe_efficiency("y", "p")), want)  # was -inf
        _same(_agg(ds, bt.r2("y", "p")), want)


def test_error_metrics_pair_their_rows() -> None:
    ds = bt.from_pydict({"y": [1.0, None, 3.0, 4.0], "p": [1.5, 2.0, None, 5.0]})
    y, p = np.array([1.0, 4.0]), np.array([1.5, 5.0])
    _same(_agg(ds, bt.normalized_rmse("y", "p")), math.sqrt(sk.mean_squared_error(y, p)) / y.mean())
    _same(_agg(ds, bt.medae("y", "p")), sk.median_absolute_error(y, p))
    assert math.isnan(_agg(_pair([0.0, 0.0], [1.0, 0.0]), bt.wape("y", "p")))
    assert math.isnan(_agg(_pair([1.0, -1.0], [1.0, 0.0]), bt.normalized_rmse("y", "p")))


def test_medae_propagates_nan_like_mae() -> None:
    ds = _pair([1.0, nan, 3.0], [1.0, 2.0, 3.5])
    assert math.isnan(_agg(ds, bt.mae("y", "p")))
    assert math.isnan(_agg(ds, bt.medae("y", "p")))  # was 0.5


def test_d2_scores_use_the_paired_rows_for_the_baseline() -> None:
    ds = bt.from_pydict({"y": [1.0, 2.0, 3.0, 5.0, 100.0], "p": [1.5, 2.0, 3.0, 4.0, None]})
    y, p = [1.0, 2.0, 3.0, 5.0], [1.5, 2.0, 3.0, 4.0]
    _same(M.d2_absolute_error_score(ds, "y", "p"), sk.d2_absolute_error_score(y, p))  # was 0.985
    _same(M.d2_pinball_score(ds, "y", "p", alpha=0.3), sk.d2_pinball_score(y, p, alpha=0.3))
    _same(M.d2_tweedie_score(ds, "y", "p", power=1.0), sk.d2_tweedie_score(y, p, power=1), rel=1e-9)
    constant = _pair([1.0, 1.0], [1.0, 2.0])
    assert M.d2_absolute_error_score(constant, "y", "p") == sk.d2_absolute_error_score(
        [1.0, 1.0], [1.0, 2.0]
    )
    assert M.d2_pinball_score(_pair([1.0, 1.0], [1.0, 1.0]), "y", "p") == 1.0
    assert math.isnan(M.d2_tweedie_score(constant, "y", "p", power=1.0))  # sklearn raises


# --- 9. clustering at scale ---------------------------------------------------------


def test_adjusted_mutual_info_is_vectorized_and_exact() -> None:
    rng = np.random.default_rng(0)
    t = rng.integers(0, 200, 100_000)
    p = np.where(rng.random(t.size) < 0.5, t, rng.integers(0, 200, t.size))
    ds = bt.from_arrow(pa.table({"t": t, "p": p}))
    start = time.perf_counter()
    got = M.adjusted_mutual_info_score(ds, "t", "p")
    elapsed = time.perf_counter() - start
    _same(got, sk.adjusted_mutual_info_score(t, p), rel=1e-8)
    # The Python triple loop took ~56 s here; the numpy version takes well under one.
    assert elapsed < 20.0, elapsed
    small_t, small_p = [0, 0, 1, 1, 2, 2, 2], [0, 0, 1, 2, 2, 2, 0]
    small = _pair(small_t, small_p)
    _same(
        M.adjusted_mutual_info_score(small, "y", "p"),
        sk.adjusted_mutual_info_score(small_t, small_p),
    )
    _same(M.mutual_info_score(small, "y", "p"), sk.mutual_info_score(small_t, small_p))
