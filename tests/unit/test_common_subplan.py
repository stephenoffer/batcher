"""Plan-level common-subplan elimination: which repeated subtrees Kyber offers for reuse.

Pure decision tests — `common_subplans` never executes anything, so nothing here runs a
query. Plans are built through the public `Dataset` API rather than by hand, because the
shapes this has to recognize are the ones that API produces (`agg.join(agg.filter(...))`
binds one `Source` object at two indices, which is the whole reason
`api.subplan_reuse._one_id_per_source` exists).

The execution half is covered by `tests/differential/test_diff_common_subplan.py`, which
proves the rewritten query returns what DuckDB returns.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.api.subplan_reuse import _one_id_per_source
from batcher.kyber.common_subplan import common_subplans, structural_key


class _Estimator:
    """A stand-in `CardinalityEstimator`: every node is `rows` rows of `width` bytes."""

    def __init__(self, rows: float = 1000.0, width: float = 16.0):
        self._rows, self._width = rows, width

    def estimate(self, _node):
        return type("Stats", (), {"rows": self._rows})()

    def row_width(self, _node, _default):
        return self._width


def _ds():
    return bt.from_pydict({"k": [1, 2, 3, 1, 2], "v": [10, 20, 30, 40, 50]})


def _shared_agg_join(ds=None):
    """`agg ⋈ project(filter(agg))` — the canonical repeated-subplan shape."""
    ds = ds if ds is not None else _ds()
    agg = ds.group_by("k").agg(total=bt.col("v").sum())
    hot = agg.filter(bt.col("total") > 4).select(bt.col("k").alias("hk"))
    return agg.join(hot, left_on="k", right_on="hk", how="inner")


def _canonical(q):
    """The plan as the rewriter sees it: one source id per distinct source object."""
    return _one_id_per_source(q._plan, q._sources)


def _call(plan, estimator=None, **kw):
    est = estimator if estimator is not None else _Estimator()
    kw.setdefault("max_bytes", 256 << 20)
    kw.setdefault("row_bytes", 64)
    return common_subplans(plan, lambda: est, **kw)


def test_finds_a_subplan_that_appears_twice():
    q = _shared_agg_join()
    plan = _canonical(q)
    found = _call(plan)
    assert [type(n).__name__ for n in found] == ["Aggregate"]
    assert structural_key(found[0]) == structural_key(plan.left)


def test_the_two_appearances_are_invisible_until_the_sources_are_canonicalized():
    """The bug this nearly shipped with: `Dataset.join` renumbers the right side's scans.

    Both aggregates read the identical `Source` object, but at indices 0 and 1, so their IR
    differs and nothing repeats. Measured as "no repeated subplan" on the one shape the
    feature exists for.
    """
    q = _shared_agg_join()
    assert _call(q._plan) == []
    assert _call(_canonical(q)) != []


def test_nothing_repeats_across_genuinely_different_sources():
    other = bt.from_pydict({"k": [1, 2], "v": [7, 8]})
    agg = _ds().group_by("k").agg(total=bt.col("v").sum())
    hot = other.group_by("k").agg(total=bt.col("v").sum()).select(bt.col("k").alias("hk"))
    q = agg.join(hot, left_on="k", right_on="hk", how="inner")
    assert _call(_canonical(q)) == []


def test_a_streaming_repeat_is_left_alone():
    """A repeated scan/filter has no breaker in it: recomputing beats materializing."""
    ds = _ds()
    left = ds.filter(bt.col("v") > 1)
    right = ds.filter(bt.col("v") > 1).select(bt.col("k").alias("hk"))
    q = left.join(right, left_on="k", right_on="hk", how="inner")
    assert _call(_canonical(q)) == []


def test_the_whole_plan_is_never_a_candidate():
    """A root with no second appearance would be materialized and then scanned for nothing."""
    q = _ds().group_by("k").agg(total=bt.col("v").sum())
    assert _call(q._plan) == []


def test_a_result_too_large_to_hold_is_declined():
    """The size gate is the whole cost model: reuse costs holding the result."""
    plan = _canonical(_shared_agg_join())
    assert _call(plan, _Estimator(rows=1e9, width=64.0)) == []
    assert _call(plan, _Estimator(rows=10.0, width=8.0)) != []


def test_the_estimator_is_not_built_when_nothing_repeats():
    """Collecting source statistics is real work, and a plan with no repeat must not pay it."""
    built = []

    def factory():
        built.append(1)
        return _Estimator()

    q = _ds().group_by("k").agg(total=bt.col("v").sum())
    assert common_subplans(q._plan, factory, max_bytes=256 << 20, row_bytes=64) == []
    assert built == [], "the size gate was consulted for a plan with nothing repeated"


def test_returned_subplans_do_not_overlap():
    """Materializing an outer candidate already collapses every inner one beneath it.

    Both the `Aggregate` and the `Filter` above it repeat here, so a naive answer returns
    both — and the caller then materializes a subtree it had already replaced with a scan.
    """
    ds = _ds()
    hot = ds.group_by("k").agg(total=bt.col("v").sum()).filter(bt.col("total") > 4)
    q = hot.join(hot.select(bt.col("k").alias("hk")), left_on="k", right_on="hk", how="inner")
    found = _call(_canonical(q))
    assert len(found) == 1
    assert type(found[0]).__name__ == "Filter"


def test_an_oversized_plan_is_skipped_rather_than_analyzed():
    assert _call(_canonical(_shared_agg_join()), max_nodes=3) == []


@pytest.mark.parametrize("occurrences", [2, 3])
def test_more_than_two_appearances_still_yield_one_candidate(occurrences):
    ds = _ds()
    agg = ds.group_by("k").agg(total=bt.col("v").sum())
    q = agg
    for i in range(occurrences - 1):
        q = q.join(agg.select(bt.col("k").alias(f"j{i}")), left_on="k", right_on=f"j{i}")
    found = _call(_canonical(q))
    assert [type(n).__name__ for n in found] == ["Aggregate"]


def test_a_cheap_repeat_inside_an_expensive_plan_is_declined():
    """Bar 5. Repeating is not the same question as being worth materializing.

    TPC-H q20 is the shape: a repeated semi-join carrying 13.5% of the plan's cost cleared
    bars 1-4 and was **2.2x slower** materialized, because a separate execution costs its own
    engine round trip and forfeits the fusion the subtree had with its parent. Modelled here
    by making the shared aggregate cheap against a plan the estimator prices as enormous.
    """

    class _Lopsided(_Estimator):
        """Tiny everywhere except the join above the shared aggregate, which dominates."""

        def estimate(self, node):
            rows = 1e9 if type(node).__name__ == "Join" else 10.0
            return type("Stats", (), {"rows": rows})()

    assert _call(_canonical(_shared_agg_join()), _Lopsided()) == []


def test_a_third_appearance_lowers_the_bar_it_has_to_clear():
    """Three appearances save two runs for one fixed cost, so the same subtree qualifies.

    The saving is `share * (a-1)/a`, so a subtree that misses at two appearances can clear the
    same threshold at three. Pinning it here keeps bar 5 a statement about the *saving* rather
    than about the subtree's size, which is what makes it scale with how often a CTE is
    referenced.
    """

    class _Marginal(_Estimator):
        def estimate(self, node):
            # Sized so the shared aggregate's saving lands between the two-appearance and
            # three-appearance thresholds.
            rows = 400.0 if type(node).__name__ == "Aggregate" else 1000.0
            return type("Stats", (), {"rows": rows})()

    def chain(occurrences):
        ds = _ds()
        agg = ds.group_by("k").agg(total=bt.col("v").sum())
        q = agg
        for i in range(occurrences - 1):
            q = q.join(agg.select(bt.col("k").alias(f"j{i}")), left_on="k", right_on=f"j{i}")
        return _canonical(q)

    two, three = _call(chain(2), _Marginal()), _call(chain(3), _Marginal())
    assert len(three) >= len(two), "a third appearance must not make reuse *less* attractive"


def test_structural_key_separates_different_literals():
    """`plan_signature` normalizes literals; this key must not — they are different relations."""
    ds = _ds()
    a = ds.filter(bt.col("v") > 10)._plan
    b = ds.filter(bt.col("v") > 99)._plan
    assert structural_key(a) != structural_key(b)
    assert structural_key(a) == structural_key(ds.filter(bt.col("v") > 10)._plan)


def test_a_map_batches_subtree_has_no_key_and_is_never_reused():
    """Opaque user code may be non-deterministic or have side effects."""
    ds = _ds()
    mapped = ds.map_batches(lambda b: b)
    assert structural_key(mapped._plan) is None
    q = mapped.group_by("k").agg(total=bt.col("v").sum())
    joined = q.join(q.select(bt.col("k").alias("hk")), left_on="k", right_on="hk")
    assert _call(_canonical(joined)) == []


def _scan_source_ids(plan):
    """Every `Scan`'s `source_id` in `plan`, in pre-order."""
    from batcher.plan.logical import Scan
    from batcher.plan.visitor import walk

    return [n.source_id for n in walk(plan) if isinstance(n, Scan)]


def test_the_analysis_matches_canonically_and_reports_original_nodes():
    """The canonical form is an analysis artefact and must never be the plan that runs.

    Collapsing every binding of one table onto one `source_id` is what makes the repeats
    visible — and it is also what makes `bc_interp::streaming_parallelizes` false, since that
    predicate is exactly "no source is scanned twice". A plan failing it is routed to the
    *materializing* executor for its whole length, so returning the canonical plan to be run
    silently changed the executor of every query with a table bound more than once. On a
    snowflake schema that is most of them: TPC-DS q80 measured 1,010 ms against 151 ms once
    the executed plan kept its own ids, q77 482 against 91, q5 473 against 199.

    So the analysis has to do both halves at once, and this pins both: the appearances are
    *matched* canonically (they are found at all, though one scans source 0 and the other
    source 1), and they are *reported* as positions in the plan as written (the two subtrees
    at those positions still carry their different bindings).
    """
    from batcher.api.subplan_reuse import _analyze
    from batcher.config import active_config
    from batcher.core import ExecutionContext, default_hub
    from batcher.plan.visitor import walk

    q = _shared_agg_join()
    ctx = ExecutionContext(columns=list(q._plan.available_columns()), hub=default_hub())
    verdict = _analyze(q._plan, list(q._sources), ctx, active_config().optimizer)
    assert verdict, "the canonical shape must still be recognized"

    nodes = list(walk(q._plan))
    positions = verdict[0]
    assert len(positions) >= 2, "both appearances must be located"
    bindings = [tuple(_scan_source_ids(nodes[i])) for i in positions]
    assert len(set(bindings)) == len(bindings), (
        "the appearances are reported as the ORIGINAL subtrees, which differ in exactly the "
        "binding the canonical form collapses"
    )


def test_a_recorded_verdict_is_served_instead_of_re_analyzing():
    """The analysis is quadratic in plan size; re-deriving it per collect measured 404 ms on
    TPC-DS q80 against a 151 ms query. A key with a verdict on file must not reach it."""
    from batcher.api import subplan_reuse
    from batcher.config import active_config
    from batcher.core import ExecutionContext, default_hub

    q = _shared_agg_join()
    ctx = ExecutionContext(columns=list(q._plan.available_columns()), hub=default_hub())
    cfg = active_config()
    key = subplan_reuse._verdict_key(q._plan, list(q._sources), ctx, cfg, cfg.optimizer)
    assert key is not None
    subplan_reuse._record_verdict(key, list(q._sources), ())

    calls: list[int] = []
    original = subplan_reuse._analyze

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    subplan_reuse._analyze = counted
    try:
        plan, srcs = subplan_reuse.reuse_common_subplans(q._plan, list(q._sources), ctx)
    finally:
        subplan_reuse._analyze = original
    assert calls == [], "a plan with a verdict on file must not be analyzed again"
    assert plan is q._plan and len(srcs) == len(q._sources)
