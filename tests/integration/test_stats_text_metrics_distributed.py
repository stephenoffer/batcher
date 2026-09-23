"""Weighted aggregates, text metrics and the `ml.stats` tests agree on one node and on several.

Three surfaces, reached three ways:

- the weighted aggregates (`bt.weighted_*`) and the text metrics (`bt.bleu`, `bt.rouge_l_f1`,
  `bt.automated_readability_index`, the n-gram family) are ordinary aggregate expressions, so
  a grouped query over them is collected with ``distributed=False`` and with
  ``distributed=True, num_workers=4`` and compared group by group;
- the `batcher.ml.stats` tests take no `distributed=` argument. Each one runs its aggregates
  through ``collect()``'s ``distributed="auto"``, so the only way onto the cluster is the
  session pin ``distributed.mode``, and a spy on the routing resolver proves the pin reached
  every terminal rather than assuming it did.

The fan-out is proved rather than assumed, because ``collect(distributed=True)`` with no
`num_workers` runs one worker and computes what single-node computes
(`.claude/rules/testing.md`). The module opens with a control known to diverge across workers:
a `LIMIT` over an unordered `group_by` on this fixture.

The fixture carries the edges the single-node differential file pins -- null values, null
weights, null group labels, null and empty texts, texts shorter than the n-gram order -- so the
mergeable forms of the new null and short-row rules are exercised, not only their local form.

CI installs no Ray, so this suite never runs in the PR gate — see `just lint-skips`.
"""

from __future__ import annotations

import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
import batcher.ml.stats as S
from _ray_cluster import init_test_ray, shutdown_test_ray
from batcher.config import option_context

pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("batcher._native", reason="native engine not built")

_WORKERS = 4
_FILES = 8
_ROWS_PER_FILE = 25_000

#: Words, plus the two articles the SQuAD normalization drops, so some rows shrink under it.
_VOCAB = np.array([f"word{i}" for i in range(24)] + ["the", "a"])


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    started = init_test_ray(_WORKERS)
    yield
    shutdown_test_ray(started)


def _texts(rng: np.random.Generator, n: int) -> list[str | None]:
    """Short sentences from a small vocabulary: some empty, some null, many under four words."""
    lengths = rng.integers(0, 12, n)
    out: list[str | None] = []
    for length in lengths:
        words = rng.choice(_VOCAB, length)
        end = rng.choice([".", "!", "?", "", ". Then"])
        out.append((" ".join(words) + end) if length else "")
    mask = rng.random(n) < 0.03
    return [None if m else t for t, m in zip(out, mask, strict=True)]


@pytest.fixture(scope="module")
def parquet_dir(tmp_path_factory) -> str:
    """Eight files, 200,000 rows: past `MIN_ROWS_TO_SHARD`, with nulls in every column."""
    root = tmp_path_factory.mktemp("stats_text")
    rng = np.random.default_rng(11)
    for i in range(_FILES):
        n = _ROWS_PER_FILE
        x = rng.normal(50.0, 20.0, n)
        y = 0.6 * x + rng.normal(0.0, 10.0, n)
        label = rng.choice(["a", "b", "c"], n).astype(object)
        label[rng.random(n) < 0.02] = None
        pq.write_table(
            pa.table(
                {
                    "g": pa.array(rng.integers(0, 17, n).astype("int64")),
                    "row": pa.array(np.arange(i * n, (i + 1) * n, dtype="int64")),
                    "label": pa.array(list(label), pa.string()),
                    "x": pa.array(x, mask=rng.random(n) < 0.02),
                    "y": pa.array(y, mask=rng.random(n) < 0.02),
                    "w": pa.array(rng.uniform(0.1, 3.0, n), mask=rng.random(n) < 0.02),
                    "pred": pa.array(_texts(rng, n), pa.string()),
                    "ref": pa.array(_texts(rng, n), pa.string()),
                }
            ),
            root / f"part-{i}.parquet",
        )
    return str(root)


@pytest.fixture
def ds(parquet_dir) -> bt.Dataset:
    return bt.read.parquet(parquet_dir)


def _local(d: bt.Dataset) -> pa.Table:
    return d.collect(distributed=False)


def _remote(d: bt.Dataset) -> pa.Table:
    return d.collect(distributed=True, num_workers=_WORKERS)


def _close(a, b, rel: float = 1e-9) -> bool:
    """Equal, or equal up to float reassociation -- the one tolerance distribution may take."""
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=rel, abs_tol=1e-9)
    return a == b


def _assert_rows_close(left: pa.Table, right: pa.Table, key: str) -> None:
    """Row-for-row equality after ordering both sides by `key` (never order-blind)."""
    assert left.column_names == right.column_names
    assert left.schema.types == right.schema.types
    lrows = sorted(left.to_pylist(), key=lambda r: r[key])
    rrows = sorted(right.to_pylist(), key=lambda r: r[key])
    assert len(lrows) == len(rrows) > 0
    for lr, rr in zip(lrows, rrows, strict=True):
        for col in left.column_names:
            assert _close(lr[col], rr[col]), (col, lr[key], lr[col], rr[col])


def test_the_fixture_really_fans_out(ds):
    """Positive control: across workers an unordered LIMIT keeps different groups.

    If this ever agrees, the comparisons below are one worker against one worker and prove
    nothing about distribution.
    """
    q = ds.group_by("g").agg(s=bt.col("x").sum()).limit(3)
    local = sorted(_local(q).column("g").to_pylist())
    remote = sorted(_remote(q).column("g").to_pylist())
    assert len(local) == len(remote) == 3
    assert local != remote


def test_grouped_weighted_aggregates_agree_across_workers(ds):
    q = ds.group_by("g").agg(
        var=bt.weighted_var("x", "w"),
        std=bt.weighted_std("x", "w"),
        cov=bt.weighted_covariance("x", "y", "w"),
        corr=bt.weighted_correlation("x", "y", "w"),
    )
    local = _local(q)
    _assert_rows_close(local, _remote(q), "g")
    # Pairwise null dropping holds on the merged path too: one group checked against NumPy.
    rows = _local(ds.filter(bt.col("g") == bt.lit(3)).select("x", "y", "w")).to_pydict()
    keep = [
        all(v is not None for v in t) for t in zip(rows["x"], rows["y"], rows["w"], strict=True)
    ]
    x, y, w = (np.array([v for v, k in zip(rows[c], keep, strict=True) if k]) for c in "xyw")
    cov = np.cov(x, y, aweights=w, bias=True)
    corr = cov[0, 1] / math.sqrt(cov[0, 0] * cov[1, 1])
    got = {r["g"]: r for r in local.to_pylist()}[3]
    assert got["corr"] == pytest.approx(corr, rel=1e-9)


def test_grouped_text_metrics_agree_across_workers(ds):
    q = ds.group_by("g").agg(
        bleu=bt.bleu("pred", "ref"),
        bleu2=bt.bleu("pred", "ref", max_n=2),
        brevity=bt.brevity_penalty("pred", "ref"),
        rouge=bt.rouge_l_f1("pred", "ref"),
        p2=bt.ngram_precision("pred", "ref", n=2),
        r2=bt.ngram_recall("pred", "ref", n=2),
        distinct=bt.distinct_ngram_ratio("pred"),
        ari=bt.automated_readability_index("pred"),
        sentences=bt.mean_sentence_count("pred"),
    )
    local = _local(q)
    _assert_rows_close(local, _remote(q), "g")
    # Not a vacuous agreement: the scores move between groups and none is null.
    bleu2 = local.column("bleu2").to_pylist()
    assert len(set(bleu2)) > 1 and None not in bleu2
    assert all(0.0 <= b <= 1.0 for b in local.column("brevity").to_pylist())


def test_corpus_text_metrics_agree_across_workers(ds):
    q = ds.agg(
        bleu=bt.bleu("pred", "ref", max_n=2),
        rouge=bt.rouge_l_f1("pred", "ref"),
        ari=bt.automated_readability_index("pred"),
        n=bt.col("row").count(),
    ).with_columns(k=bt.lit(0))
    _assert_rows_close(_local(q), _remote(q), "k")


def _spy_routes(monkeypatch) -> list[bool]:
    """Record every `distributed="auto"` routing decision a terminal makes."""
    from batcher.api.terminal import routing

    routes: list[bool] = []
    original = routing._resolve_distributed

    def spy(distributed, plan=None, sources=None):
        decision = original(distributed, plan, sources)
        routes.append(decision)
        return decision

    monkeypatch.setattr(routing, "_resolve_distributed", spy)
    return routes


def _under(mode: str, fn):
    with option_context("distributed.mode", mode):
        return fn()


def _fields(result) -> list[float]:
    """A `TestResult` or a float or a dict as a flat list of floats, for comparison."""
    if isinstance(result, dict):
        return [float(result[k]) for k in sorted(result)]
    if isinstance(result, float):
        return [result]
    df = result.df if isinstance(result.df, tuple) else (result.df,)
    return [result.statistic, result.pvalue, *map(float, df)]


_STATS_CALLS = {
    "t_test_ind": lambda d: S.t_test_ind(d.filter(bt.col("label") != bt.lit("c")), "x", "label"),
    "t_test_1samp": lambda d: S.t_test_1samp(d, 50.0, "x"),
    "anova_test": lambda d: S.anova_test(d, "x", "label"),
    "kruskal_wallis": lambda d: S.kruskal_wallis(d, "x", "label"),
    "levene_test": lambda d: S.levene_test(d, "x", "label"),
    "bartlett_test": lambda d: S.bartlett_test(d, "x", "label"),
    "mann_whitney_u": lambda d: S.mann_whitney_u(
        d.filter(bt.col("label") != bt.lit("c")), "x", "label"
    ),
    "wilcoxon_signed_rank": lambda d: S.wilcoxon_signed_rank(d, "x", "y"),
    "eta_squared": lambda d: S.eta_squared(d, "x", "label"),
    "omega_squared": lambda d: S.omega_squared(d, "x", "label"),
    "median_abs_deviation": lambda d: S.median_abs_deviation(d, "x"),
    "mean_abs_deviation": lambda d: S.mean_abs_deviation(d, "x"),
    "variance_inflation_factor": lambda d: S.variance_inflation_factor(d, ["x", "y", "w"]),
}


@pytest.mark.parametrize("name", sorted(_STATS_CALLS))
def test_ml_stats_reach_ray_under_mode_always_and_agree(ds, monkeypatch, name):
    """Without the pin every aggregate stays local; with it every one runs on Ray, and the
    answer is the same number up to float reassociation."""
    call = _STATS_CALLS[name]
    routes = _spy_routes(monkeypatch)
    never = _under("never", lambda: call(ds))
    assert routes and not any(routes), (name, routes)
    routes.clear()
    always = _under("always", lambda: call(ds))
    assert routes and all(routes), (name, routes)
    for a, b in zip(_fields(never), _fields(always), strict=True):
        assert _close(a, b, rel=1e-7), (name, never, always)
    # Not a vacuous agreement: the statistic is a finite number on this fixture.
    assert math.isfinite(_fields(never)[0]), (name, never)
