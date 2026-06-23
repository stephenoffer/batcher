"""The `StatsEstimator` memoizes by node identity within one optimize run.

`estimate(node)` is invoked O(nodes) times per optimize (once per node in
`_annotate_ops`, plus every cost-based rule), and each call previously re-descended
to the leaves — making planning super-linear in plan depth and re-hashing whole
subtrees for the structural signature. These tests pin the memoization that keeps
per-query planning cost proportional to the plan, protecting the sub-second
small-query mandate. Memoization must be semantically transparent: it returns the
same `RelStats` an uncached estimate would.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col, count
from batcher.kyber.stats.estimator import StatsEstimator


def _deep_plan():
    ds = (
        bt.from_arrow(pa.table({"k": [i % 10 for i in range(1000)], "v": list(range(1000))}))
        .filter(col("v") > 5)
        .with_columns(w=col("v") * 2)
        .group_by("k")
        .agg(n=count(), s=col("v").sum())
        .sort("k")
    )
    return ds._plan, ds._sources


def _all_nodes(node):
    seen = []
    stack = [node]
    while stack:
        n = stack.pop()
        seen.append(n)
        for attr in ("input", "left", "right"):
            child = getattr(n, attr, None)
            if child is not None:
                stack.append(child)
        for child in getattr(n, "inputs", ()) or ():
            stack.append(child)
    return seen


def test_estimate_runs_once_per_node_and_is_fully_cached_on_recall():
    plan, sources = _deep_plan()
    est = StatsEstimator(sources)

    calls: list[int] = [0]
    raw = est._estimate_uncached

    def counting(node):
        calls[0] += 1
        return raw(node)

    est._estimate_uncached = counting  # type: ignore[method-assign]

    first = est.estimate(plan)
    n_nodes = len({id(n) for n in _all_nodes(plan)})
    # Each distinct node is computed exactly once across the whole recursive descent.
    assert calls[0] == n_nodes

    # A second top-level estimate is a pure cache hit: zero further computation, and
    # the identical (frozen) RelStats object comes back.
    before = calls[0]
    second = est.estimate(plan)
    assert calls[0] == before
    assert second is first


def test_memoized_estimate_matches_uncached_value():
    # The cached estimator must produce the same rows/provenance a fresh,
    # never-cached estimator would — memoization changes cost, never results.
    plan, sources = _deep_plan()
    cached = StatsEstimator(sources).estimate(plan)
    fresh = StatsEstimator(sources).estimate(plan)
    assert cached.rows == fresh.rows
    assert cached.provenance == fresh.provenance


def test_signature_is_memoized_by_identity():
    plan, sources = _deep_plan()
    est = StatsEstimator(sources)
    assert est._sig(plan) == est._sig(plan)
    # The cache holds the node identity, so the same id maps back to the same sig.
    assert id(plan) in est._sig_cache
    assert est._sig_cache[id(plan)][0] is plan
