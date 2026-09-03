"""The control plane must not pay for work whose result nothing reads.

Four optimizations here are invisible to every functional test: the code is correct
either way, and only the clock tells them apart. Each one was worth between a fifth and
four fifths of a real query's control-plane time, and each would be silently undone by an
ordinary-looking edit -- rendering an error message eagerly, calling the exhaustive join
DP, propagating column statistics at construction, writing a learned table per source.
So each gets a test that fails on the *work done*, not on the answer.
"""

from __future__ import annotations

import json

import pytest

from batcher import kyber
from batcher._internal.errors import public_members
from batcher.metadata import MetadataHub
from batcher.metadata.backends.in_process import InProcessBackend
from batcher.plan.stats import ColumnStat, LazyColumns, Provenance

pytestmark = pytest.mark.unit


# --- 1. a discarded AttributeError must not render its guidance ----------------------


def test_probing_an_absent_expr_attribute_does_not_build_the_message(monkeypatch):
    """`getattr(expr, name, None)` must not run the did-you-mean fuzzy match.

    The optimizer probes `Expr` nodes for optional IR fields by name, discarding the
    `AttributeError` each miss raises. Rendering the message runs `difflib` against all
    ~300 members of the type, and on a six-join plan that was **82% of the whole query**
    -- 2,172 messages built and thrown away.
    """
    import batcher as bt
    from batcher._internal.errors import suggest

    rendered = []
    real = suggest.did_you_mean
    monkeypatch.setattr(
        suggest, "did_you_mean", lambda *a, **k: (rendered.append(a[0]), real(*a, **k))[1]
    )
    expr = bt.col("x") * 2
    for probe in ("value", "arg", "input", "condition", "otherwise", "operands", "branches"):
        # Exactly the shape the optimizer uses: ask every node for every optional IR field.
        assert getattr(expr, probe, None) is None
        assert not hasattr(expr, "definitely_not_a_method")
    assert rendered == [], f"a discarded probe rendered {len(rendered)} guidance message(s)"


def test_the_message_is_still_exactly_right_when_something_reads_it():
    """Laziness must be invisible: same type, same text, same `args`, same `repr`."""
    import batcher as bt

    with pytest.raises(AttributeError) as caught:
        getattr(bt.col("x"), "meen")  # noqa: B009 - the lookup itself is what is under test
    exc = caught.value
    assert type(exc) is AttributeError  # not a private subclass: tracebacks say so
    assert str(exc).startswith("Expr has no attribute 'meen'.")
    assert "Did you mean" in str(exc)
    assert exc.args[0] == str(exc)
    assert repr(exc) == f"AttributeError({str(exc)!r})"


def test_public_members_is_cached_per_type():
    """The did-you-mean pool is a property of the class, so `dir` runs once per type."""
    before = public_members.cache_info()
    for _ in range(20):
        public_members(str)
    after = public_members.cache_info()
    assert after.hits - before.hits >= 19


# --- 2. join ordering must use the connected-subset DP ------------------------------


def test_join_reorder_never_calls_the_exhaustive_dp():
    """`_rebuild_dp` is the oracle for `tests/unit/test_dphyp_join_order.py`, not a path.

    It enumerates all 2ⁿ subsets and 3ⁿ splits and discovers a subset is disconnected only
    after trying to build a join for it. On a ten-way star join that was 72% of the query.

    Asserted as "`order` does not import the name", not by patching `order_search` and
    counting calls. That spelling looks stronger and is a tautology: `order` binds
    `_rebuild_dp` into its own namespace at import, so patching the *defining* module
    catches nothing -- it counted zero calls against the old code that called it on every
    join. The import is the thing that decides which search runs, so it is the thing to pin.
    """
    import batcher as bt
    from batcher.kyber.rules.joins import order, order_search

    assert not hasattr(order, "_rebuild_dp"), (
        "join ordering imported the exhaustive O(3ⁿ) DP again; it is the oracle, not a path"
    )
    assert order._rebuild_dphyp is order_search._rebuild_dphyp

    seen = []
    original = order._rebuild_dphyp
    order._rebuild_dphyp = lambda *a, **k: (seen.append(1), original(*a, **k))[1]
    try:
        fact = bt.from_pydict({"v": [1, 2], **{f"k{d}": [1, 2] for d in range(4)}})
        ds = fact
        for d in range(4):
            ds = ds.join(bt.from_pydict({f"k{d}": [1, 2], f"a{d}": [7, 8]}), on=f"k{d}")
        ds.collect()
    finally:
        order._rebuild_dphyp = original
    assert seen, "the connected-subset DP never ran, so this test proved nothing"


# --- 3. a join's output column statistics are propagated only if read ---------------


def test_lazy_columns_does_not_build_until_read():
    built = []

    def build():
        built.append(1)
        return {"a": ColumnStat(min=1, max=2)}

    lazy = LazyColumns(build)
    assert built == []
    assert lazy["a"].min == 1
    assert len(lazy) == 1 and list(lazy) == ["a"]
    assert built == [1], "the column map was rebuilt on a later read"
    assert lazy == {"a": ColumnStat(min=1, max=2)}  # Mapping supplies no __eq__; we do
    assert json.loads(json.dumps(list(lazy))) == ["a"]


def test_carried_through_join_equals_the_downgrade_and_replace_it_fuses():
    """One construction, byte-identical to `downgrade(DEFAULT)` plus the join's drops."""
    import dataclasses
    import random

    rng = random.Random(20260829)
    for _ in range(500):
        stat = ColumnStat(
            min=rng.choice([None, 1, "a"]),
            max=rng.choice([None, 9]),
            null_count=rng.choice([None, 3.0]),
            ndv=rng.choice([None, 8.0]),
            total_sum=rng.choice([None, 5.0]),
            mean=rng.choice([None, 2.0]),
            provenance=rng.choice(list(Provenance)),
            bloom=rng.choice([None, b"x"]),
            mcv=rng.choice([None, {"a": 1.0}]),
            avg_bytes=rng.choice([None, 12.0]),
            ndv_provenance=rng.choice([None, *Provenance]),
            null_count_provenance=rng.choice([None, *Provenance]),
            moments_provenance=rng.choice([None, *Provenance]),
        )
        out_ndv = rng.choice([None, 4.0])
        reference = dataclasses.replace(
            stat.downgrade(Provenance.DEFAULT),
            null_count=None,
            ndv=out_ndv,
            total_sum=None,
            mean=None,
            mcv=None,
        )
        assert stat.carried_through_join(out_ndv) == reference


# --- 4. learned column tables are written once, not once per source -----------------


def _hub() -> MetadataHub:
    return MetadataHub(InProcessBackend())


def test_column_stats_for_many_sources_cost_one_write_per_table():
    """Each table is a single blob, so a per-source write re-serializes all of it.

    Eleven sources rewrote four tables eleven times -- 833 KB of JSON for a query over 64
    rows, growing with everything the session had already learned.
    """
    hub = _hub()
    before = hub._params._writes
    kyber.record_column_stats_batch(
        hub,
        [
            kyber.MeasuredColumns(f"src{i}", {f"c{i}": 10.0}, {}, {f"c{i}": 4.0}, {})
            for i in range(11)
        ],
    )
    writes = hub._params._writes - before
    assert writes == 2, f"11 sources x 2 tables took {writes} writes, expected 2"
    table = hub.get_keyed_param("kyber.stats", kyber.NDV_KEY)
    assert len(table) == 11, "coalescing must not drop a source's entries"


def test_row_byte_widths_for_many_sources_cost_one_write():
    hub = _hub()
    before = hub._params._writes
    kyber.record_column_row_bytes_batch(hub, [(f"src{i}", {"c": float(i + 1)}) for i in range(11)])
    writes = hub._params._writes - before
    assert writes == 1, f"11 sources took {writes} writes, expected 1"
    assert len(hub.get_keyed_param("kyber.stats", kyber.ROW_BYTES_KEY)) == 11
