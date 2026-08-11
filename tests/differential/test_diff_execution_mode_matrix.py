"""Every operator x every *execution mode*, on edge-case data large enough to reach them.

`test_diff_operator_matrix.py` crosses every operator with every *scheduling* — `collect()`,
`collect(spill=True)`, `iter_batches()` — on a 15-row input loaded with nulls, `-0.0`/NaN, and
duplicates. That input is the point of it, and it is also its blind spot: at 15 rows the engine
takes exactly one path to the answer, so 45 operators x 3 shapes all measure the same mode.

Two engine decisions are keyed off size and configuration, and neither one engages there:

* **Sharding.** `execute_streaming_parallel` splits the driving scan across workers only above
  `MIN_ROWS_TO_SHARD` (`crates/bc-interp/src/stream/parallel.rs`, 4 morsels = 65,536 rows).
  Below it, it returns `fallback_with(...)` — the *sequential* streaming pipeline. So the whole
  Python differential suite, which is where the edge-case inputs live, runs unsharded.
* **Which executor.** `bc_py::prepare_exec` picks the streaming executor or the materializing
  one. The streaming path is the default and effectively the only one Python exercises; the
  materializing executor is what a self-join routes to, what a plan over budget falls back to,
  and what actually spills.

The Rust side covers the modes but not the data: `crates/bc-interp/tests/stream_oracle.rs`
shards a 200,000-row input, and every column in it is clean (`i % 100`, no nulls, no NaN, no
`-0.0`). So edge-case data and the modes that only appear at scale were covered in two
different suites and never met. Every bug the small matrix's docstring lists — a float group
key splitting `-0.0` from `0.0`, a group split across reducers, a spilled descending sort
emitting nulls mid-result — is a bug at exactly that intersection.

This file is that intersection: the small matrix's operator table and its edge-case columns,
replicated past the sharding threshold, run through both executors and every scheduling, and
checked against each other and against DuckDB.

The operator table is *imported* from the small matrix rather than restated, so an operator
added there is covered at scale here too and the two cannot drift apart.
"""

from __future__ import annotations

import dataclasses
import itertools
import re
from pathlib import Path

import pyarrow as pa
import pytest

from _harness import assert_same, assert_same_ordered, assert_tables_equal

pytestmark = pytest.mark.differential

bt = pytest.importorskip("batcher")

from test_diff_operator_matrix import BASE, ORDERINGS, RIGHT, UNORDERED_OPS  # noqa: E402

from batcher.config import active_config, set_config  # noqa: E402  (after importorskip)

#: Repeats of `BASE`'s 15-row edge pattern. 15 does not divide the 16,384-row morsel, so the
#: pattern phases across every morsel boundary instead of aligning to it — nulls, NaN, `-0.0`
#: and duplicate keys land at different offsets in each morsel, which is what makes a
#: boundary-sensitive bug (a group split across shards, a partial folded in the wrong order)
#: reachable at all.
_REPS = 4_700  # 70,500 rows — clear of the 65,536-row sharding threshold with headroom.

BIG = pa.table(
    {
        name: pa.array(BASE.column(name).to_pylist() * _REPS, BASE.schema.field(name).type)
        for name in BASE.column_names
    }
)

#: The join's build side, widened the same way so the probe spans many morsels.
BIG_RIGHT = pa.table(
    {
        name: pa.array(RIGHT.column(name).to_pylist() * 8, RIGHT.schema.field(name).type)
        for name in RIGHT.column_names
    }
)

#: `BIG` plus a unique `rid`, so `ORDER BY k, rid` is a **total** order.
#:
#: `ORDER BY k` alone does not determine the result at this scale: every key value repeats
#: 4,700 times, and SQL leaves the order *within* a tie group unspecified. Batcher and DuckDB
#: genuinely differ there (measured: identical multiset, identical key sequence, different
#: payload order inside the NULL group) and neither is wrong. Asserting row-by-row on that
#: would be pinning unspecified behavior, so the ordered comparisons sort on a total order
#: instead — which determines every row and is therefore a *stricter* assertion, not a
#: relaxed one. What `ORDER BY k` alone does specify — the key sequence and the multiset —
#: is asserted separately by `test_sort_key_sequence_matches_duckdb_at_shard_scale`.
BIG_TOTAL = BIG.append_column("rid", pa.array(range(BIG.num_rows), pa.int64()))


def _shard_threshold_rows() -> int:
    """`MIN_ROWS_TO_SHARD`, read out of the Rust source that defines it.

    Parsed rather than hardcoded so the coupling is mechanical. If someone raises the
    threshold past `BIG`, this file would quietly stop testing the sharded path and become a
    slower copy of the small matrix — passing while covering nothing new, which is the exact
    trap `stream_oracle.rs` documents on its own row count. `test_the_fixture_actually_shards`
    turns that into a failure instead.
    """
    root = Path(__file__).resolve().parents[2]
    parallel = (root / "crates/bc-interp/src/stream/parallel.rs").read_text()
    morsel = (root / "crates/bc-arrow/src/lib.rs").read_text()
    mult = re.search(
        r"MIN_ROWS_TO_SHARD:\s*usize\s*=\s*(\d+)\s*\*\s*bc_arrow::DEFAULT_MORSEL_ROWS", parallel
    )
    rows = re.search(r"DEFAULT_MORSEL_ROWS:\s*usize\s*=\s*([\d_]+)", morsel)
    if not mult or not rows:
        pytest.skip("could not read MIN_ROWS_TO_SHARD from the Rust sources")
    return int(mult.group(1)) * int(rows.group(1).replace("_", ""))


def test_the_fixture_actually_shards():
    """The input must clear the engine's sharding threshold, or this file tests nothing new."""
    threshold = _shard_threshold_rows()
    assert BIG.num_rows > threshold, (
        f"BIG is {BIG.num_rows} rows but the engine only shards above {threshold}. "
        "Raise _REPS — otherwise every test here silently re-runs the sequential path."
    )


def _configured(*, streaming: bool):
    """The engine config with the executor pinned — `streaming=False` forces materializing."""
    prev = active_config()
    return prev, prev.replace(execution=dataclasses.replace(prev.execution, streaming=streaming))


def _collect(build, table, *, streaming: bool, spill: bool = False) -> pa.Table:
    prev, pinned = _configured(streaming=streaming)
    set_config(pinned)
    try:
        return build(bt.from_arrow(table)).collect(spill=spill)
    finally:
        set_config(prev)


def _stream(build, table, *, streaming: bool) -> pa.Table:
    prev, pinned = _configured(streaming=streaming)
    set_config(pinned)
    try:
        ds = build(bt.from_arrow(table))
        batches = list(ds.iter_batches())
        if not batches:
            return ds.collect().slice(0, 0)
        return pa.Table.from_batches(batches, schema=batches[0].schema)
    finally:
        set_config(prev)


#: The join types `_build` can rebuild against the widened `BIG_RIGHT`.
#:
#: Named explicitly rather than derived by stripping the `join_` prefix. The prefix strip
#: assumed every `join_*` key in the shared matrix is spelled `join_<how>`, and the moment
#: byte-keyed cases (`join_inner_str`, ...) were added there it produced `how="inner_str"`
#: and every one of them raised `unsupported join type`. Those cases join on a *string* key
#: against their own right-hand table, so there is nothing here to widen: `BIG` already
#: drives them past the sharding threshold, which is the property this module adds.
_WIDENED_JOINS = ("inner", "left", "right", "full", "outer", "semi", "anti")


def _build(op):
    """The small matrix's builder, with the join's build side widened to match `BIG`."""
    build, sql = UNORDERED_OPS[op]
    how = op.removeprefix("join_")
    if op.startswith("join_") and how in _WIDENED_JOINS:
        return (lambda d: d.join(bt.from_arrow(BIG_RIGHT), left_on="k", right_on="k", how=how)), sql
    return build, sql


@pytest.mark.parametrize("op", sorted(UNORDERED_OPS))
def test_both_executors_agree_at_shard_scale(op):
    """The materializing executor is the oracle the streaming one is licensed against.

    They are two implementations of the same algebra, and above the sharding threshold the
    streaming one also splits the driving scan across workers — so this is simultaneously a
    streaming-vs-materializing check and a sharded-vs-unsharded one.
    """
    build, _ = _build(op)
    oracle = _collect(build, BIG, streaming=False)
    assert_tables_equal(_collect(build, BIG, streaming=True), oracle)


@pytest.mark.parametrize("op", sorted(UNORDERED_OPS))
def test_every_scheduling_agrees_at_shard_scale(op):
    """Spill and streaming are schedulings of `collect()` at scale too, not just at 15 rows."""
    build, _ = _build(op)
    oracle = _collect(build, BIG, streaming=True)
    assert_tables_equal(_collect(build, BIG, streaming=True, spill=True), oracle)
    assert_tables_equal(_stream(build, BIG, streaming=True), oracle)


@pytest.mark.parametrize("op", sorted(o for o in UNORDERED_OPS if UNORDERED_OPS[o][1]))
@pytest.mark.parametrize("streaming", [True, False], ids=["streaming", "materializing"])
def test_both_executors_match_duckdb_at_shard_scale(duck, op, streaming):
    """...and neither executor merely agrees with the other — both match the external oracle."""
    build, sql = _build(op)
    duck.register("t", BIG)
    assert_same(_collect(build, BIG, streaming=streaming), duck.sql(sql))


@pytest.mark.parametrize(("descending", "nulls_first"), ORDERINGS)
@pytest.mark.parametrize("streaming", [True, False], ids=["streaming", "materializing"])
def test_sort_matches_duckdb_at_shard_scale(duck, streaming, descending, nulls_first):
    """An ordered assertion — the only kind that can see a sort bug — above the shard threshold.

    A sort over 70,500 rows merges many sorted runs rather than sorting one block, and the
    spilled sort range-partitions them. The small matrix's 15 rows exercise neither. The
    secondary `rid` key makes the order total, so this pins every row's position rather than
    a tie order SQL does not define (see `BIG_TOTAL`).
    """
    duck.register("t", BIG_TOTAL)
    build = lambda d: d.sort(  # noqa: E731
        bt.col("k"), bt.col("rid"), descending=descending, nulls_first=nulls_first
    )
    out = _collect(build, BIG_TOTAL, streaming=streaming)
    d, n = ("DESC" if descending else "ASC"), ("FIRST" if nulls_first else "LAST")
    assert_same_ordered(
        out, duck.sql(f"SELECT * FROM t ORDER BY k {d} NULLS {n}, rid {d} NULLS {n}")
    )


@pytest.mark.parametrize(("descending", "nulls_first"), ORDERINGS)
def test_sort_key_sequence_matches_duckdb_at_shard_scale(duck, descending, nulls_first):
    """What a single-key sort *does* specify: the key sequence, and the rows carried with it.

    Tie order is unspecified, but the sequence of key values is not — that is where the
    historical bugs lived (a spilled `descending` sort emitting nulls mid-result). Asserting
    the key column in order catches every one of those without depending on tie order, and
    the multiset check confirms no row was dropped, duplicated, or corrupted on the way.
    """
    duck.register("t", BIG)
    out = _collect(
        lambda d: d.sort(bt.col("k"), descending=descending, nulls_first=nulls_first),
        BIG,
        streaming=True,
    )
    d, n = ("DESC" if descending else "ASC"), ("FIRST" if nulls_first else "LAST")
    want = duck.sql(f"SELECT * FROM t ORDER BY k {d} NULLS {n}").to_arrow_table()
    assert out.column("k").to_pylist() == want.column("k").to_pylist()
    assert_tables_equal(out, want.select(out.column_names))


@pytest.mark.parametrize(("descending", "nulls_first"), ORDERINGS)
@pytest.mark.parametrize("key", ["k", "g", "f"])  # int range-partitions; string and float differ
def test_sort_schedulings_agree_at_shard_scale(key, descending, nulls_first):
    """The spilled and streamed sort equal the in-memory sort at a scale that really merges."""
    build = lambda d: d.sort(  # noqa: E731
        bt.col(key), descending=descending, nulls_first=nulls_first
    )
    oracle = _collect(build, BIG, streaming=False)
    assert_tables_equal(_collect(build, BIG, streaming=True), oracle, ordered=True)
    assert_tables_equal(_collect(build, BIG, streaming=True, spill=True), oracle, ordered=True)
    assert_tables_equal(_stream(build, BIG, streaming=True), oracle, ordered=True)


#: Group-key shapes whose partial state must combine associatively across shards. At 15 rows
#: the fold runs once and associativity is untestable; here each key spans every morsel.
_FOLD_KEYS = ["k", "g", "f"]


@pytest.mark.parametrize("key", _FOLD_KEYS)
@pytest.mark.parametrize("streaming", [True, False], ids=["streaming", "materializing"])
def test_grouped_partials_combine_associatively(duck, key, streaming):
    """`partial -> combine -> finalize` over many morsels equals DuckDB's one-shot answer.

    Invariant #7 in one assertion per key type: the aggregate folds a partial per morsel and
    merges them in completion order, so a `combine` that is not associative and commutative —
    or a key identity that disagrees between the fast path and the fallback, which is how
    `-0.0` once split from `0.0` — produces a different answer here and only here.
    """
    duck.register("t", BIG)
    build = lambda d: d.group_by(key).agg(  # noqa: E731
        s=bt.col("v").sum(), n=bt.col("v").count(), lo=bt.col("v").min(), hi=bt.col("v").max()
    )
    got = _collect(build, BIG, streaming=streaming)
    assert_same(
        got,
        duck.sql(
            f"SELECT {key}, sum(v) AS s, count(v) AS n, min(v) AS lo, max(v) AS hi "
            f"FROM t GROUP BY {key}"
        ),
    )


def test_a_single_hot_key_still_agrees_across_modes(duck):
    """Every row in one group — the skew shape that puts one shard's partial against all others."""
    hot = BIG.append_column("one", pa.array([0] * BIG.num_rows, pa.int64()))
    duck.register("t", hot)
    build = lambda d: d.group_by("one").agg(s=bt.col("v").sum(), n=bt.col("k").count())  # noqa: E731
    for streaming in (True, False):
        assert_same(
            _collect(build, hot, streaming=streaming),
            duck.sql("SELECT one, sum(v) AS s, count(k) AS n FROM t GROUP BY one"),
        )


@pytest.mark.parametrize("how", ["inner", "left", "outer", "semi", "anti"])
def test_joins_agree_across_modes_at_shard_scale(how):
    """A probe spanning many morsels against a build side hashed once, on every join type."""
    build = lambda d: d.join(  # noqa: E731
        bt.from_arrow(BIG_RIGHT), left_on="k", right_on="k", how=how
    )
    oracle = _collect(build, BIG, streaming=False)
    assert_tables_equal(_collect(build, BIG, streaming=True), oracle)
    assert_tables_equal(_stream(build, BIG, streaming=True), oracle)


def test_a_self_join_agrees_across_modes():
    """The shape `prepare_exec` deliberately routes to the *materializing* executor.

    A source scanned twice makes the streaming executor fall to its sequential pipeline, so
    the engine sends this plan to the materializing one instead. That routing is a performance
    decision, and it is only sound if both executors still produce the same relation.

    Joined on the unique `rid` so the self-join stays 1:1. On a duplicated key it is quadratic
    — `k` repeats 4,700 times here, which fans 70,500 rows out to ~27M and turns one test into
    eight minutes without testing anything the 1:1 shape does not.
    """
    build = lambda d: d.join(d, left_on="rid", right_on="rid", how="inner")  # noqa: E731
    oracle = _collect(build, BIG_TOTAL, streaming=False)
    assert_tables_equal(_collect(build, BIG_TOTAL, streaming=True), oracle)


@pytest.mark.parametrize(
    "morsel_rows", [1_024, 16_384], ids=["many-small-morsels", "default-morsel"]
)
def test_results_are_independent_of_the_morsel_size(duck, morsel_rows):
    """Morsel size is a scheduling knob, so it must not change a single row of the answer.

    Shrinking the morsel multiplies the number of partials an aggregate folds and the number
    of runs a sort merges. Anything order-sensitive that survives the default morsel because
    the fold happened to run few times fails here.
    """
    duck.register("t", BIG)
    prev = active_config()
    set_config(prev.replace(execution=dataclasses.replace(prev.execution, morsel_rows=morsel_rows)))
    try:
        got = bt.from_arrow(BIG).group_by("f").agg(s=bt.col("v").sum()).collect()
    finally:
        set_config(prev)
    assert_same(got, duck.sql("SELECT f, sum(v) AS s FROM t GROUP BY f"))


@pytest.mark.parametrize("parallelism", [1, 2, 3], ids=lambda p: f"workers-{p}")
def test_results_are_independent_of_the_worker_count(duck, parallelism):
    """Shard count must not change the answer — including the `workers == 1` unsharded path.

    `execute_streaming_parallel` declines to shard at one worker and takes the sequential
    pipeline, so this also pins sharded-vs-sequential agreement on the same input, and an odd
    worker count catches a shard boundary that assumes an even split.
    """
    duck.register("t", BIG_TOTAL)
    prev = active_config()
    set_config(prev.replace(execution=dataclasses.replace(prev.execution, parallelism=parallelism)))
    try:
        got = bt.from_arrow(BIG_TOTAL).group_by("g", "k").agg(s=bt.col("v").sum()).collect()
        srt = (
            bt.from_arrow(BIG_TOTAL)
            .sort(bt.col("k"), bt.col("rid"), descending=True, nulls_first=True)
            .collect()
        )
    finally:
        set_config(prev)
    assert_same(got, duck.sql("SELECT g, k, sum(v) AS s FROM t GROUP BY g, k"))
    assert_same_ordered(
        srt, duck.sql("SELECT * FROM t ORDER BY k DESC NULLS FIRST, rid DESC NULLS FIRST")
    )


def test_window_functions_agree_across_modes():
    """Window partitions are built per shard, so their frames must survive being split."""
    build = lambda d: d.with_columns(  # noqa: E731
        rn=bt.row_number().over(partition_by="g", order_by="k"),
        s=bt.col("v").sum().over(partition_by="g"),
    )
    oracle = _collect(build, BIG, streaming=False)
    assert_tables_equal(_collect(build, BIG, streaming=True), oracle)
    assert_tables_equal(_stream(build, BIG, streaming=True), oracle)


@pytest.mark.parametrize(
    ("op_a", "op_b"),
    list(itertools.product(["filter", "project"], ["aggregate", "distinct"])),
)
def test_a_breaker_under_a_pipeline_agrees_across_modes(op_a, op_b):
    """A linear run feeding a breaker — the shape the streaming executor exists to make cheap."""
    linear = {
        "filter": lambda d: d.filter(bt.col("v") > 2),
        "project": lambda d: d.select(bt.col("g"), bt.col("k"), (bt.col("v") * 3).alias("v")),
    }[op_a]
    breaker = {
        "aggregate": lambda d: d.group_by("g").agg(s=bt.col("v").sum()),
        "distinct": lambda d: d.select(bt.col("g"), bt.col("k")).distinct(),
    }[op_b]
    build = lambda d: breaker(linear(d))  # noqa: E731
    oracle = _collect(build, BIG, streaming=False)
    assert_tables_equal(_collect(build, BIG, streaming=True), oracle)
    assert_tables_equal(_collect(build, BIG, streaming=True, spill=True), oracle)
    assert_tables_equal(_stream(build, BIG, streaming=True), oracle)
