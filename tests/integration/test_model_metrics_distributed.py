"""Model metrics agree single-node and distributed, including on the groups that break them.

The metric expressions are mergeable aggregates, and the rank metrics are partitioned
windows plus an aggregate, so each should give the same answer on one node or four. That
is only worth asserting on data that exercises the awkward cases, so the fixture carries
them: a group holding one class (where ROC AUC used to be ``inf`` on one path and NaN on the
other), a three-row group, null predictions, tied scores, and 200,000 rows over eight
Parquet files so the work really is split.

Every comparison names ``distributed=False`` on one side and ``distributed=True,
num_workers=4`` on the other, and `test_the_cluster_really_splits_the_work` is the positive
control: a ``LIMIT`` over an unordered ``group_by``, which is known to return different
groups once more than one worker runs it. Without that control a comparison could pass
because both sides ran on one worker.

Run with a fresh local cluster: ``RAY_ADDRESS=local python -m pytest <this file> -q``.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _ray_cluster import init_test_ray, shutdown_test_ray
from batcher.config import option_context
from batcher.ml import metrics as M

pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("batcher._native", reason="native engine not built")

pytestmark = pytest.mark.integration

ROWS = 200_000
FILES = 8
WORKERS = 4


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    started = init_test_ray(WORKERS)
    yield
    shutdown_test_ray(started)


@pytest.fixture(scope="module")
def source(tmp_path_factory) -> str:
    """200,000 scored rows over eight Parquet files, with the edge-case groups mixed in."""
    from batcher.dist.shuffle_io import shared_scratch_root

    root = shared_scratch_root()
    if root is not None:
        base = os.path.join(root, f"model_metrics_{os.getpid()}")
        os.makedirs(base, exist_ok=True)
    else:
        base = str(tmp_path_factory.mktemp("model_metrics"))
    rng = np.random.default_rng(0)
    key = np.char.add("g", np.char.zfill(rng.integers(0, 40, ROWS).astype(str), 2))
    y = rng.integers(0, 2, ROWS)
    key[:500], y[:500] = "g40", 1  # a single-class group
    key[500:503] = "g41"  # a three-row group
    score = np.round(rng.random(ROWS) * 0.5 + 0.4 * y, 2)  # two decimals: heavy ties
    yreg = np.abs(rng.normal(10, 3, ROWS)) + 0.1
    preg = yreg + rng.normal(0, 1, ROWS)
    table = pa.table(
        {
            "k": key,
            "y": y,
            "yhat": (score > 0.5).astype(int),
            "score": score,
            "yreg": yreg,
            "preg": pa.array(preg, mask=rng.random(ROWS) < 0.01),
            "q": rng.integers(0, 20_000, ROWS),
            "rel": rng.integers(0, 4, ROWS) * (rng.random(ROWS) < 0.3),
        }
    ).take(rng.permutation(ROWS))
    for i in range(FILES):
        pq.write_table(table.slice(i * ROWS // FILES, ROWS // FILES), f"{base}/part{i}.parquet")
    return base


def _close(a: object, b: object) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return a == b or abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))
    return a == b


def _assert_same_by_key(ds: bt.Dataset, key: str) -> dict[str, list]:
    """Collect `ds` both ways and assert every column agrees row for row, keyed by `key`."""
    local = ds.collect(distributed=False).to_pydict()
    dist = ds.collect(distributed=True, num_workers=WORKERS).to_pydict()
    assert list(local) == list(dist)
    assert sorted(local[key]) == sorted(dist[key])
    at = {value: i for i, value in enumerate(dist[key])}
    for name, column in local.items():
        for i, value in enumerate(column):
            other = dist[name][at[local[key][i]]]
            assert _close(value, other), (name, local[key][i], value, other)
    return local


def test_the_cluster_really_splits_the_work(source: str) -> None:
    query = bt.read.parquet(source).group_by("k").agg(n=bt.count()).limit(3)
    local = query.collect(distributed=False).to_pydict()["k"]
    dist = query.collect(distributed=True, num_workers=WORKERS).to_pydict()["k"]
    assert len(local) == len(dist) == 3
    # Same count, different groups: the unordered LIMIT only diverges when more than one
    # worker produces the groups, which is what makes the comparisons below meaningful.
    assert local != dist


def test_grouped_classification_metrics(source: str) -> None:
    metrics = {
        name: getattr(bt, name)("y", "yhat")
        for name in (
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "matthews_corrcoef",
            "cohen_kappa",
            "balanced_accuracy",
            "specificity",
            "false_positive_rate",
            "geometric_mean_score",
            "informedness",
            "markedness",
            "positive_likelihood_ratio",
            "negative_likelihood_ratio",
            "prevalence_threshold",
            "jaccard_score",
            "fowlkes_mallows_index",
        )
    }
    local = _assert_same_by_key(bt.read.parquet(source).group_by("k").agg(**metrics), "k")
    single = local["k"].index("g40")
    # The single-class group has no negatives: its specificity-based ratios are undefined,
    # while the class-averaged scores fall back to the recall of the class it has.
    assert math.isnan(local["informedness"][single])
    assert math.isnan(local["positive_likelihood_ratio"][single])
    assert local["false_positive_rate"][single] == 0.0
    assert local["balanced_accuracy"][single] == local["recall"][single]


def test_grouped_regression_metrics(source: str) -> None:
    metrics = {
        name: getattr(bt, name)("yreg", "preg")
        for name in (
            "mae",
            "rmse",
            "r2",
            "medae",
            "wape",
            "normalized_rmse",
            "nash_sutcliffe_efficiency",
            "kling_gupta_efficiency",
            "concordance_correlation",
        )
    }
    _assert_same_by_key(bt.read.parquet(source).group_by("k").agg(**metrics), "k")


def test_grouped_rank_metrics(source: str) -> None:
    ds = bt.read.parquet(source)
    for metric in (M.roc_auc, M.average_precision, M.ks_statistic, M.gini_coefficient):
        local = _assert_same_by_key(metric(ds, "y", "score", by="k"), "k")
        value = local[next(c for c in local if c != "k")][local["k"].index("g40")]
        if metric is M.average_precision:
            assert value == 1.0  # every row positive: every precision is 1
        else:
            assert math.isnan(value), metric.__name__  # was +inf single-node, NaN distributed


def test_grouped_evaluate(source: str) -> None:
    report = M.evaluate(bt.read.parquet(source), "y", y_pred="yhat", y_score="score", by="k")
    local = _assert_same_by_key(report, "k")
    assert math.isnan(local["roc_auc"][local["k"].index("g40")])


def test_ranking_window_quantities(source: str) -> None:
    # The per-query ranking metrics return a float, so what is compared here is the ranked
    # frame they reduce: every window quantity they read, row for row. `_POSITION` is left
    # out on purpose -- it is a row_number, arbitrary within a tie on either path, and no
    # metric reads it except alongside the tie-averaged gain.
    from batcher.ml.metrics import ranking

    ranked = ranking._ranked(bt.read.parquet(source), "q", "score", "rel", 1, graded=True)
    per_query = ranked.group_by("q").agg(
        last=bt.col(ranking._LAST).max(),
        cum=bt.col(ranking._CUM_GAIN).max(),
        tie=bt.col(ranking._TIE_GAIN).sum(),
        hits=ranking._expected_hits(5),
    )
    _assert_same_by_key(per_query, "q")


def test_ranking_functions_under_forced_distribution(source: str) -> None:
    # The Dataset-level functions collect internally, so they route through the
    # `distributed.mode` option rather than a collect() argument.
    ds = bt.read.parquet(source)
    calls = {
        "ndcg": lambda: M.ndcg_at_k(ds, "q", "score", "rel", k=5, graded=True),
        "map": lambda: M.map_at_k(ds, "q", "score", "rel", k=5),
        "mrr": lambda: M.mean_reciprocal_rank(ds, "q", "score", "rel"),
        "p@5": lambda: M.precision_at_k(ds, "q", "score", "rel", k=5),
    }
    local = {name: call() for name, call in calls.items()}
    with option_context("distributed.mode", "always"):
        dist = {name: call() for name, call in calls.items()}
    for name in calls:
        assert _close(local[name], dist[name]), (name, local[name], dist[name])
