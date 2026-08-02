"""Every operator x the *distributed* execution path, on the edge-case inputs.

This completes the cross-product `CLAUDE.md` names — `{collect, spill, iter_batches,
distributed}` x `{nulls, empty, one row, duplicates, -0.0/NaN, descending}`. The other three
paths are covered per-operator by `test_diff_operator_matrix.py` (at 15 rows) and
`test_diff_execution_mode_matrix.py` (above the sharding threshold, on both single-node
executors). Distributed was the one path with no per-operator coverage: `test_dist_hunt2_matrix.py`
checks it on 17 hand-written shapes, which is the shapes someone thought of rather than the
operator table.

Distribution is a *scheduling* concern over the same mergeable primitives (invariant #7), so a
two-worker result MUST equal the single-node one and both MUST equal DuckDB. What makes that
non-trivial is exactly the edge-case data: a group key has to hash to the same reducer on
every worker, which is where a `Float64` key once split `-0.0` from `0.0` across reducers and
where a nullable key once split one group in two. Those are `partial -> partition -> combine ->
finalize` bugs, invisible single-node.

The operator table is *imported* from `test_diff_operator_matrix`, so an operator added there
is automatically checked distributed too and the matrices cannot drift apart.

`num_workers` is pinned at 2 — the schedulable fan-out here, and the mergeable algebra makes
two-way equivalence what exercises the shuffle that any higher fan-out reuses verbatim (the
same reasoning `test_dist_hunt2_matrix.py` records).
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from _harness import assert_same, assert_same_ordered, assert_tables_equal

pytestmark = pytest.mark.differential

bt = pytest.importorskip("batcher")
pytest.importorskip("ray", reason="the distributed path needs Ray")

from test_diff_operator_matrix import (  # noqa: E402
    BASE,
    INPUTS,
    ORDERINGS,
    RIGHT,
    UNORDERED_OPS,
    assert_row_number_contract,
)

#: The schedulable worker fan-out for the suite.
_W = 2

#: `BASE` cut into four uneven pieces (4, 1, 7 and 3 rows) and repeated six times: 24 record
#: batches, 90 rows.
#:
#: `INPUTS` reaches many batches only by *volume* — its `multibatch` shape is `BASE` repeated
#: past the 16,384-row morsel, so the batches it splits into are uniform and morsel-aligned.
#: This one is small and arrives as many *uneven* batches carrying duplicate keys,
#: which is what makes the map side split a group *across* batches before the shuffle ever sees
#: it — the arrangement under the "group split across reducers" class of bug. The uneven lengths
#: matter: equal-sized batches align every group to a boundary the same way, so they cannot
#: catch an off-by-one in the partitioner's batch handling.
_SLICES = [BASE.slice(0, 4), BASE.slice(4, 1), BASE.slice(5, 7), BASE.slice(12, 3)]
MULTIBATCH = pa.Table.from_batches([batch for piece in _SLICES * 6 for batch in piece.to_batches()])

#: Every edge-case shape, with the uneven-batch arrangement above replacing `INPUTS`' own
#: volume-driven `multibatch`: at two workers it is the batch *boundaries* that decide which
#: rows a mapper groups together, and uneven ones are what catch an off-by-one there. The
#: morsel-crossing shape stays covered single-node by `test_diff_operator_matrix.py`.
SHAPES = {**INPUTS, "multibatch": MULTIBATCH}

#: A hot key: every row in one group, so one reducer takes the whole relation and the others
#: take nothing. Skew is where a partitioner that assumes an even split shows itself.
HOT = MULTIBATCH.append_column("one", pa.array([0] * MULTIBATCH.num_rows, pa.int64()))

#: `MULTIBATCH` plus a unique `rid`, so an ordered comparison against DuckDB has a total order
#: to compare on. Tie order within a duplicated sort key is unspecified and the two engines
#: legitimately differ there, so an ordered row-by-row assertion on a non-unique key would be
#: pinning undefined behavior rather than the sort.
TOTAL = MULTIBATCH.append_column("rid", pa.array(range(MULTIBATCH.num_rows), pa.int64()))


def _dist(ds, **kw) -> pa.Table:
    return ds.collect(distributed=True, num_workers=_W, **kw)


def _build(op):
    """The small matrix's builder and its DuckDB SQL, unchanged."""
    return UNORDERED_OPS[op]


def test_the_multibatch_fixture_really_is_multi_batch():
    """The point of `MULTIBATCH` is its batch boundaries — assert they survived construction.

    `pa.Table` operations readily coalesce chunks, and a single-chunk fixture here would make
    every "split across batches" test below a duplicate of the single-batch ones, passing while
    covering nothing.
    """
    assert MULTIBATCH.column("k").num_chunks > 1, MULTIBATCH.column("k").num_chunks
    assert MULTIBATCH.num_rows > BASE.num_rows


@pytest.mark.parametrize("op", sorted(UNORDERED_OPS))
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_distributed_equals_single_node(op, shape):
    """Two workers must produce exactly what one produces — the mergeable-algebra invariant."""
    build, _ = _build(op)
    table = SHAPES[shape]
    oracle = build(bt.from_arrow(table)).collect()
    assert_tables_equal(_dist(build(bt.from_arrow(table))), oracle)


@pytest.mark.parametrize("op", sorted(o for o in UNORDERED_OPS if UNORDERED_OPS[o][1]))
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_distributed_matches_duckdb(duck, op, shape):
    """...and it is not merely self-consistent — the distributed result matches the oracle.

    Single-node agreement alone cannot catch a bug both paths share; DuckDB can.
    """
    build, sql = _build(op)
    table = SHAPES[shape]
    duck.register("t", table)
    assert_same(_dist(build(bt.from_arrow(table))), duck.sql(sql))


@pytest.mark.parametrize("op", sorted(UNORDERED_OPS))
def test_distributed_with_spill_equals_single_node(op):
    """The two paths that were never crossed: distributed *and* out-of-core at once.

    Spill and distribution are independent knobs, and each is a scheduling concern, so their
    combination must still be the single-node answer. A reducer that spills its partial state
    and reloads it is where the two interact.
    """
    build, _ = _build(op)
    oracle = build(bt.from_arrow(MULTIBATCH)).collect()
    assert_tables_equal(_dist(build(bt.from_arrow(MULTIBATCH)), spill=True), oracle)


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_distributed_row_number_is_a_permutation_consistent_with_its_order_key(shape):
    """The shuffled window, on the tie-heavy order key, asserted on its real contract.

    This is the shape that exposed the over-assertion: on `MULTIBATCH`, 12 rows of partition
    `g='a'` tie on `k IS NULL`, and which of them takes rank 19 versus rank 30 is a free
    choice the two paths make differently and legitimately. What must hold on every worker
    count is that the partition still lands whole on one reducer -- which is exactly "the
    numbers are 1..n and ordered by k".
    """
    plan = lambda d: d.with_columns(rn=bt.row_number().over(partition_by="g", order_by="k"))  # noqa: E731
    assert_row_number_contract(_dist(plan(bt.from_arrow(SHAPES[shape]))), partition="g", order="k")


@pytest.mark.parametrize("num_partitions", [1, 2, 3, 5, 7])
def test_the_partition_count_never_changes_the_answer(duck, num_partitions):
    """Any reducer count is result-correct under the mergeable algebra — including odd ones.

    `shuffle_partitions` normally derives the bucket count from the fan-out. Overriding it with
    a non-divisor of the worker count is what catches a partitioner that assumes buckets divide
    evenly among workers, and `1` collapses the shuffle to a single reducer.
    """
    duck.register("t", MULTIBATCH)
    ds = bt.from_arrow(MULTIBATCH).group_by("g", "k").agg(s=bt.col("v").sum())
    got = ds.collect(distributed=True, num_workers=_W, num_partitions=num_partitions)
    assert_same(got, duck.sql("SELECT g, k, sum(v) AS s FROM t GROUP BY g, k"))


@pytest.mark.parametrize("key", ["k", "g", "f"])
def test_group_keys_hash_consistently_across_reducers(duck, key):
    """One group must land on one reducer, whatever its key type.

    This is the shuffle's key-identity contract, and the historical failures are all here: a
    `Float64` key splitting `-0.0` from `0.0`, a nullable key splitting NULL from NULL, a NaN
    hashing differently per payload. Each would show up as *two* output rows for one group,
    which DuckDB's answer pins exactly.
    """
    duck.register("t", MULTIBATCH)
    ds = bt.from_arrow(MULTIBATCH).group_by(key).agg(s=bt.col("v").sum(), n=bt.col("v").count())
    got = _dist(ds)
    assert_same(got, duck.sql(f"SELECT {key}, sum(v) AS s, count(v) AS n FROM t GROUP BY {key}"))
    # The group *count* directly: a group split across two reducers emits an extra row whose
    # partial sums still look individually plausible, so pin the row count as well as the values.
    want = duck.sql(f"SELECT count(*) FROM (SELECT {key} FROM t GROUP BY {key})").fetchone()[0]
    assert got.num_rows == want, f"expected {want} groups on {key}, got {got.num_rows}"


def test_a_hot_key_puts_the_whole_relation_on_one_reducer(duck):
    """Maximum skew: one group, so one reducer does everything and the rest do nothing."""
    duck.register("t", HOT)
    ds = bt.from_arrow(HOT).group_by("one").agg(s=bt.col("v").sum(), n=bt.col("k").count())
    assert_same(_dist(ds), duck.sql("SELECT one, sum(v) AS s, count(k) AS n FROM t GROUP BY one"))


@pytest.mark.parametrize(("descending", "nulls_first"), ORDERINGS)
def test_distributed_sort_matches_duckdb_on_every_ordering(duck, descending, nulls_first):
    """A distributed sort range-partitions, so its boundaries carry the ordering flags.

    Ordered assertion on a total order (`k, rid`) — the only kind that can see a sort bug, and
    the only kind that is well-defined when the primary key repeats.
    """
    duck.register("t", TOTAL)
    out = _dist(
        bt.from_arrow(TOTAL).sort(
            bt.col("k"), bt.col("rid"), descending=descending, nulls_first=nulls_first
        )
    )
    d, n = ("DESC" if descending else "ASC"), ("FIRST" if nulls_first else "LAST")
    assert_same_ordered(
        out, duck.sql(f"SELECT * FROM t ORDER BY k {d} NULLS {n}, rid {d} NULLS {n}")
    )


@pytest.mark.parametrize(("descending", "nulls_first"), ORDERINGS)
@pytest.mark.parametrize("key", ["k", "g", "f"])
def test_distributed_sort_equals_single_node_sort(key, descending, nulls_first):
    """Batcher-vs-Batcher, row for row: the range partitioner must reproduce the local sort.

    A numeric key range-partitions; a string key falls back to a different path. Both must
    equal the single-node sort exactly — a stricter statement than matching DuckDB, because
    both sides are the same engine and the comparison is ordered.

    The sort is on `(key, rid)`, not `key` alone, and that is the whole point rather than a
    convenience. `docs/user-guide/transform/rows/sorting.md` states the contract outright: "Two rows
    with the same key can come back in either order, and the order can change between a
    sequential run, a multi-core run, and a distributed one", and names `-0.0` against `0.0` as
    a tie like any other. Asserting row-for-row on a duplicated key therefore pins behaviour the
    engine explicitly disclaims — and it did: `MULTIBATCH.f` holds both `-0.0` and `0.0`, and
    the two paths interleave them differently while every pair compares equal. Every key here
    repeats, so without `rid` all twelve parameterisations assert undefined behaviour. `rid`
    makes the
    order total, which is what turns this into a real statement about the *partitioner*: with
    ties broken, any surviving difference is a row in the wrong bucket or a bucket concatenated
    out of key order, which is exactly the class of bug this file exists to catch.
    """
    plan = lambda d: d.sort(  # noqa: E731
        bt.col(key), bt.col("rid"), descending=descending, nulls_first=nulls_first
    )
    oracle = plan(bt.from_arrow(TOTAL)).collect()
    assert_tables_equal(_dist(plan(bt.from_arrow(TOTAL))), oracle, ordered=True)


@pytest.mark.parametrize("how", ["inner", "left", "outer", "semi", "anti"])
def test_distributed_joins_equal_single_node(how):
    """A join co-partitions both sides, so every join type must survive the shuffle."""
    build = lambda d: d.join(  # noqa: E731
        bt.from_arrow(RIGHT), left_on="k", right_on="k", how=how
    )
    oracle = build(bt.from_arrow(MULTIBATCH)).collect()
    assert_tables_equal(_dist(build(bt.from_arrow(MULTIBATCH))), oracle)


def test_a_distributed_aggregate_over_a_join_equals_single_node():
    """The fused shuffle-reduce shape: the reducer joins its bucket, then aggregates it."""
    build = lambda d: (  # noqa: E731
        d.join(bt.from_arrow(RIGHT), left_on="k", right_on="k", how="inner")
        .group_by("k")
        .agg(s=bt.col("v").sum())
    )
    oracle = build(bt.from_arrow(MULTIBATCH)).collect()
    assert_tables_equal(_dist(build(bt.from_arrow(MULTIBATCH))), oracle)


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_a_windowed_result_equals_single_node(shape):
    """Window partitions must be whole on one worker, or a frame spans a partition boundary.

    Ordered by `(k, rid)` rather than `k`, for the reason `assert_row_number_contract` in the
    single-node matrix already states: which of two rows tied on the ordering key gets the
    lower number is unspecified, so comparing the emitted rows across two paths over-asserts.
    It did — `multibatch` carries twelve rows with `g = 'a'` and a **null** `k`, all tied, and
    the two paths numbered them differently while every column's multiset stayed identical.
    A unique tiebreak makes `row_number` a function of the data, so a surviving difference
    means a partition was split across workers, which is what this test is for.
    """
    table = SHAPES[shape]
    table = table.append_column("rid", pa.array(range(table.num_rows), pa.int64()))
    build = lambda d: d.with_columns(  # noqa: E731
        rn=bt.row_number().over(partition_by="g", order_by=["k", "rid"]),
        s=bt.col("v").sum().over(partition_by="g"),
    )
    oracle = build(bt.from_arrow(table)).collect()
    assert_tables_equal(_dist(build(bt.from_arrow(table))), oracle)
