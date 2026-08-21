"""Every relational operator x every execution path, on one edge-case-loaded input.

`collect()`, `collect(spill=True)` and `iter_batches()` are three *schedulings* of the same
operator semantics (invariant #7), so they must agree with each other and with DuckDB — on
nulls, on empty input, on a single row, on `-0.0`/NaN float keys, on every ordering flag, and
on an input long enough to cross a morsel boundary (`MULTIBATCH`, without which the three
"paths" are three names for a single batch).

The reshape and nearest-match operators — `unnest`, `unpivot`, `sample`, `asof_join`,
`range_join` — live in `test_diff_reshape_matrix.py`, and
`test_diff_operator_matrix_coverage.py` fails if any `RelOp` tag is missing from the two.

This matrix exists because the per-operator tests each covered their own operator on its own
happy path, and the *combinations* were nobody's job. Four wrong-answer bugs lived in that gap:

* a spilled `descending` sort emitted nulls mid-result (the out-of-core sort re-derived the
  range partitioner instead of calling the shared one);
* a nullable `Float64` group key split `-0.0` from `0.0` into two groups (the null-free fast
  path canonicalized; the `RowConverter` fallback it fell through to did not);
* a shuffled `Float64`/nullable-`Int64` group key split groups across reducers;
* a keyless aggregate over an empty input yielded 0 rows from `iter_batches()` and 1 from
  `collect()`.

Every one is a *cross-product* failure: operator x path x edge case. Hence a matrix, not more
per-operator tests.
"""

from __future__ import annotations

import itertools

import pyarrow as pa
import pytest

from _harness import assert_same, assert_same_ordered, assert_tables_equal

pytestmark = pytest.mark.differential

bt = pytest.importorskip("batcher")

# One input carrying every edge the operators must survive: nulls in the sort/group key, a
# string key, both zeros and a NaN in the float key, duplicates, and an unsorted order.
BASE = pa.table(
    {
        "k": pa.array([3, 1, None, 10, 7, None, 0, 5, 9, None, 2, 8, 4, None, 6], pa.int64()),
        "g": pa.array(
            ["a", "b", "a", "c", None, "b", "a", "c", "b", None, "a", "c", "b", "a", "c"]
        ),
        "f": pa.array(
            [
                1.5,
                -0.0,
                0.0,
                None,
                2.5,
                -1.0,
                float("nan"),
                3.0,
                1.5,
                -0.0,
                2.5,
                None,
                0.0,
                7.5,
                -2.5,
            ],
            pa.float64(),
        ),
        "v": pa.array([5, 3, 9, 1, 4, 8, 2, 7, 6, 0, 5, 3, 8, 1, 2], pa.int64()),
    }
)
#: `BASE` repeated past two 16,384-row morsels, so the paths this matrix compares are
#: actually *different* rather than three names for one batch.
#:
#: Every other shape here is 15 rows, which is one morsel: `iter_batches()` yields a single
#: batch, and there is no boundary for a row to be dropped at, double-emitted at, or
#: reordered across. That made "the streaming scheduling" a scheduling of one batch, and
#: left the whole class of morsel-boundary bugs — the class this file exists for —
#: structurally invisible. (The `window_row_number` note below has referred to a
#: `MULTIBATCH` shape since it was written; the shape itself was never added.)
#:
#: Repetition, rather than fresh values, is deliberate: it keeps exactly the edges `BASE`
#: was built for (the nulls, the `-0.0`/NaN float key, the duplicates) and multiplies the
#: tie groups, which is what a partitioned or spilled path has to get right.
MULTIBATCH = pa.concat_tables([BASE] * 2400)  # 36,000 rows

#: `MULTIBATCH`'s size with its keys collapsed onto **one value**: same row count, same
#: columns, same types, one dominant key.
#:
#: Key *concentration* is a distinct axis from the ones above, and it was the one this matrix
#: did not have. Every other shape here spreads its keys evenly, so a partitioned or spilled
#: operator sees buckets of roughly equal size and never meets the case its bound has to
#: survive: a key that lands every one of its rows in a single bucket. Re-partitioning cannot
#: separate rows that share a key -- they re-hash together however they are salted -- so the
#: hot bucket is exactly where an out-of-core path either holds or materializes the thing it
#: spilled to avoid.
#:
#: The hot value matches on both join sides (`k=3` is in `RIGHT`, `g="a"` is in `RIGHT_STR`),
#: so the join flavors produce a large hot result rather than an empty one, and `BASE` is
#: appended so the nulls, the `-0.0`/NaN float key and the duplicates are still present in the
#: tail. `window_*` partitions by `g`, which makes this one 36,000-row window partition -- the
#: shape the window's grace path documents as the one it cannot subdivide.
_HOT_ROWS = MULTIBATCH.num_rows - BASE.num_rows
SKEWED = pa.concat_tables(
    [
        pa.table(
            {
                "k": pa.array([3] * _HOT_ROWS, pa.int64()),
                "g": pa.array(["a"] * _HOT_ROWS),
                "f": pa.array([1.5] * _HOT_ROWS, pa.float64()),
                "v": pa.array([5] * _HOT_ROWS, pa.int64()),
            }
        ),
        BASE,
    ]
)

INPUTS = {
    "base": BASE,
    "empty": BASE.slice(0, 0),
    "single": BASE.slice(0, 1),
    "multibatch": MULTIBATCH,
    "skewed": SKEWED,
}
RIGHT = pa.table(
    {"k": pa.array([1, 3, 5, 7, 9, None], pa.int64()), "w": ["p", "q", "r", "s", "u", "z"]}
)
#: A **string**-keyed build side. The integer `RIGHT` above exercises the join's `Int64` fast
#: path; nothing here reached the byte-keyed one, which hashes and compares raw bytes instead
#: of encoding both sides through arrow's row format. The values carry what could separate the
#: two encodings: a null (never matches), a duplicate key (a chain of length 2), the empty
#: string, and a key that is a strict prefix of another ("a" against "aa") — which raw bytes
#: keep distinct only because the length does.
RIGHT_STR = pa.table(
    {
        "g": pa.array(["a", "b", "c", "a", None, "", "aa"]),
        "w": ["p", "q", "r", "p2", "z", "e", "aa2"],
    }
)

ORDERINGS = list(itertools.product([False, True], [False, True]))  # (descending, nulls_first)

#: Every column of `BASE`, so ordering on it ties only between wholly identical rows.
#: `row_number` needs this: with a partial order the rank handed to each row inside a tie
#: group is a free choice, and asserting one path's choice against another's tests nothing.
TOTAL_ORDER = ["k", "g", "f", "v"]

#: operator -> (build, DuckDB SQL or None). Ordered comparisons are handled separately, since
#: an unordered assert is structurally blind to a sort bug.
UNORDERED_OPS: dict[str, tuple] = {
    "scan": (lambda d: d, "SELECT * FROM t"),
    "filter": (lambda d: d.filter(bt.col("v") > 3), "SELECT * FROM t WHERE v > 3"),
    "filter_null": (lambda d: d.filter(bt.col("k").is_null()), "SELECT * FROM t WHERE k IS NULL"),
    "project": (
        lambda d: d.select(bt.col("k"), (bt.col("v") * 2).alias("d")),
        "SELECT k, v*2 AS d FROM t",
    ),
    "with_columns": (
        lambda d: d.with_columns(z=bt.col("v") + bt.col("k")),
        "SELECT *, v+k AS z FROM t",
    ),
    "aggregate": (
        lambda d: d.group_by("g").agg(s=bt.col("v").sum()),
        "SELECT g, SUM(v) AS s FROM t GROUP BY g",
    ),
    "agg_global": (
        lambda d: d.agg(s=bt.col("v").sum(), n=bt.col("v").count()),
        "SELECT SUM(v) AS s, COUNT(v) AS n FROM t",
    ),
    "agg_float_key": (
        lambda d: d.group_by("f").agg(s=bt.col("v").sum()),
        "SELECT f, SUM(v) AS s FROM t GROUP BY f",
    ),
    "agg_null_key": (
        lambda d: d.group_by("k").agg(s=bt.col("v").sum()),
        "SELECT k, SUM(v) AS s FROM t GROUP BY k",
    ),
    "agg_multi_key": (
        lambda d: d.group_by("g", "k").agg(s=bt.col("v").sum()),
        "SELECT g, k, SUM(v) AS s FROM t GROUP BY g, k",
    ),
    "distinct": (lambda d: d.select(bt.col("g")).distinct(), "SELECT DISTINCT g FROM t"),
    "distinct_multi": (
        lambda d: d.select(bt.col("g"), bt.col("k")).distinct(),
        "SELECT DISTINCT g, k FROM t",
    ),
    "distinct_float": (lambda d: d.select(bt.col("f")).distinct(), "SELECT DISTINCT f FROM t"),
    # A *keyed* dedup: one whole row per key, payload carried. Distinct enough from the
    # whole-row form to need its own rows — it gathers rows by index rather than emitting the
    # hash table's own key columns, so a `-0.0`/NaN or null key that grouped correctly can
    # still come back attached to the wrong payload.
    #
    # The ordering is over EVERY column, for the reason `window_row_number` gives: which of
    # two rows tied on the ordering key survives is unspecified, so ordering on `k` alone
    # would compare a free choice across paths rather than the operator. With a total order
    # each tie group holds identical rows and any divergence is a real bug.
    "dedup_keyed_first": (
        lambda d: d.distinct(["g"], keep="first", order_by=TOTAL_ORDER),
        "SELECT k, g, f, v FROM (SELECT *, row_number() OVER ("
        "  PARTITION BY g ORDER BY k, g, f, v"
        ") rn FROM t) WHERE rn = 1",
    ),
    "dedup_keyed_last": (
        lambda d: d.distinct(["g"], keep="last", order_by=TOTAL_ORDER),
        "SELECT k, g, f, v FROM (SELECT *, row_number() OVER ("
        "  PARTITION BY g ORDER BY k DESC, g DESC, f DESC, v DESC"
        ") rn FROM t) WHERE rn = 1",
    ),
    # A float key, where the surviving row must follow the engine's `-0.0`/NaN identity.
    "dedup_keyed_float": (
        lambda d: d.distinct(["f"], keep="first", order_by=TOTAL_ORDER),
        None,
    ),
    # A composite key, and a key that is nearly unique (so almost nothing collapses).
    "dedup_keyed_multi": (
        lambda d: d.distinct(["g", "k"], keep="first", order_by=TOTAL_ORDER),
        None,
    ),
    # `keep="any"` picks an unspecified row, so it gets no oracle and no path comparison —
    # only the invariant that must hold however the row is picked, checked in
    # `test_keyed_dedup_any_keeps_one_real_row_per_key`.
    "union": (lambda d: d.union(d), "SELECT * FROM t UNION ALL SELECT * FROM t"),
    "limit": (lambda d: d.limit(4), None),
    "row_index": (lambda d: d.with_row_index("rid"), None),
    "join_inner": (
        lambda d: d.join(bt.from_arrow(RIGHT), left_on="k", right_on="k", how="inner"),
        None,
    ),
    "join_left": (
        lambda d: d.join(bt.from_arrow(RIGHT), left_on="k", right_on="k", how="left"),
        None,
    ),
    "join_outer": (
        lambda d: d.join(bt.from_arrow(RIGHT), left_on="k", right_on="k", how="outer"),
        None,
    ),
    "window_rank": (
        lambda d: d.with_columns(r=bt.rank().over(partition_by="g", order_by="k")),
        None,
    ),
    "window_sum": (lambda d: d.with_columns(s=bt.col("v").sum().over(partition_by="g")), None),
    # `row_number` orders on TOTAL_ORDER, not on `k` alone. `k` repeats, and which of two
    # rows tied on the ordering key receives the lower row number is unspecified in SQL, so
    # a path-vs-path assertion on `order_by="k"` compares a free choice rather than the
    # operator: the four paths agree on `BASE` by luck and diverge on `MULTIBATCH`, where
    # 12 rows tie on `k IS NULL` inside partition `g='a'`. Ordering on every column makes
    # each tie group a set of *identical* rows, so the output multiset is well defined and
    # any real divergence is a bug. The tie-order freedom itself is not untested — see
    # `test_row_number_is_a_permutation_consistent_with_its_order_key`, which pins the whole
    # contract (a permutation of 1..n per partition, monotone in the order key) on the
    # tie-heavy `order_by="k"` shape this entry used to assert too much about.
    "window_row_number": (
        lambda d: d.with_columns(rn=bt.row_number().over(partition_by="g", order_by=TOTAL_ORDER)),
        None,
    ),
    "window_global": (
        lambda d: d.with_columns(rn=bt.row_number().over(order_by=TOTAL_ORDER)),
        None,
    ),
    # Estimation-layer shapes, run through every execution path × edge-case input so a
    # spill/stream regression on any of them (not just the estimate) is caught.
    "filter_between": (
        lambda d: d.filter(bt.col("v").between(2, 6)),
        "SELECT * FROM t WHERE v BETWEEN 2 AND 6",
    ),
    "filter_in": (
        lambda d: d.filter(bt.col("k").is_in([1, 3, 5, 999])),
        "SELECT * FROM t WHERE k IN (1, 3, 5, 999)",
    ),
    "filter_not_in": (
        lambda d: d.filter(~bt.col("k").is_in([1, 3])),
        "SELECT * FROM t WHERE k NOT IN (1, 3)",
    ),
    "filter_or_eq": (
        lambda d: d.filter((bt.col("k") == 1) | (bt.col("k") == 5)),
        "SELECT * FROM t WHERE k = 1 OR k = 5",
    ),
    "filter_col_eq_col": (
        lambda d: d.filter(bt.col("k") == bt.col("v")),
        "SELECT * FROM t WHERE k = v",
    ),
    "filter_coalesce": (
        lambda d: d.filter(bt.col("k").fill_null(0) == 0),
        "SELECT * FROM t WHERE coalesce(k, 0) = 0",
    ),
    "filter_out_of_range": (
        lambda d: d.filter(bt.col("v") > 1000),
        "SELECT * FROM t WHERE v > 1000",
    ),
    "join_semi": (
        lambda d: d.join(bt.from_arrow(RIGHT), left_on="k", right_on="k", how="semi"),
        None,
    ),
    "join_anti": (
        lambda d: d.join(bt.from_arrow(RIGHT), left_on="k", right_on="k", how="anti"),
        None,
    ),
    # More aggregate measures — `count` over a null-bearing column, the min/max the metadata
    # shortcuts try to answer without executing, and a `sum`/`avg` over the NaN/-0.0 float.
    "agg_count_nullable": (
        lambda d: d.group_by("g").agg(n=bt.col("k").count()),
        "SELECT g, count(k) AS n FROM t GROUP BY g",
    ),
    "agg_min_max": (
        lambda d: d.group_by("g").agg(lo=bt.col("v").min(), hi=bt.col("v").max()),
        "SELECT g, min(v) AS lo, max(v) AS hi FROM t GROUP BY g",
    ),
    "agg_mean": (
        lambda d: d.group_by("g").agg(a=bt.col("v").mean()),
        "SELECT g, avg(v) AS a FROM t GROUP BY g",
    ),
    "agg_float_measure": (
        lambda d: d.group_by("g").agg(s=bt.col("f").sum()),
        "SELECT g, sum(f) AS s FROM t GROUP BY g",
    ),
    # Predicate shapes the estimator now reads structurally.
    "filter_is_not_null": (
        lambda d: d.filter(bt.col("k").is_not_null()),
        "SELECT * FROM t WHERE k IS NOT NULL",
    ),
    "filter_compound": (
        lambda d: d.filter((bt.col("v") > 2) & ((bt.col("k") < 5) | bt.col("g").is_null())),
        "SELECT * FROM t WHERE v > 2 AND (k < 5 OR g IS NULL)",
    ),
    "filter_negated_compound": (
        lambda d: d.filter(~((bt.col("v") > 2) & (bt.col("k") < 5))),
        "SELECT * FROM t WHERE NOT (v > 2 AND k < 5)",
    ),
    # Derived-column projections whose bounds the estimator now carries forward.
    "project_coalesce": (
        lambda d: d.select(c=bt.col("k").fill_null(-1)),
        "SELECT coalesce(k, -1) AS c FROM t",
    ),
    "project_nullif": (
        lambda d: d.select(c=bt.nullif(bt.col("v"), 5)),
        "SELECT nullif(v, 5) AS c FROM t",
    ),
    "project_greatest": (
        lambda d: d.select(c=bt.greatest(bt.col("k"), bt.col("v"))),
        "SELECT greatest(k, v) AS c FROM t",
    ),
    "project_arith": (
        lambda d: d.select(c=bt.col("v") * 2 - 3),
        "SELECT v * 2 - 3 AS c FROM t",
    ),
    "window_max": (
        lambda d: d.with_columns(m=bt.col("v").max().over(partition_by="g")),
        None,
    ),
    # A **string** join key, on all four join types. The integer entries above take the
    # join's `Int64` fast path and never reach the byte-keyed one, so a raw-byte hash or
    # comparison that disagreed with the row encoding — on the null, the duplicate, the
    # empty string, or the "a"/"aa" prefix pair in `RIGHT_STR` — was invisible here.
    "join_inner_str": (
        lambda d: d.join(bt.from_arrow(RIGHT_STR), left_on="g", right_on="g", how="inner"),
        None,
    ),
    "join_left_str": (
        lambda d: d.join(bt.from_arrow(RIGHT_STR), left_on="g", right_on="g", how="left"),
        None,
    ),
    "join_semi_str": (
        lambda d: d.join(bt.from_arrow(RIGHT_STR), left_on="g", right_on="g", how="semi"),
        None,
    ),
    "join_anti_str": (
        lambda d: d.join(bt.from_arrow(RIGHT_STR), left_on="g", right_on="g", how="anti"),
        None,
    ),
    # A group key that is (almost) unique, so the group count reaches the row count. That is
    # the shape where `assign_groups` hands its key columns back untouched instead of `take`ing
    # them at an identity permutation, and where the executor abandons pre-aggregation for the
    # partitioned shape — neither of which any low-cardinality `GROUP BY g` above exercises.
    "agg_unique_key": (
        lambda d: d.group_by("k", "g", "v").agg(s=bt.col("v").sum()),
        "SELECT k, g, v, SUM(v) AS s FROM t GROUP BY k, g, v",
    ),
    "distinct_unique_key": (
        lambda d: d.select(bt.col("k"), bt.col("g"), bt.col("v")).distinct(),
        "SELECT DISTINCT k, g, v FROM t",
    ),
}


def assert_row_number_contract(
    out: pa.Table, *, partition: str, order: str, rn: str = "rn"
) -> None:
    """Assert everything `ROW_NUMBER() OVER (PARTITION BY … ORDER BY …)` actually promises.

    Which of two rows tied on the ordering key gets the lower number is unspecified, so a
    path-vs-path comparison of the emitted rows over-asserts (see the `window_row_number`
    note in `UNORDERED_OPS`). What *is* specified, and what this pins on the tie-heavy
    input the matrix can no longer assert on, is:

    * within each partition the numbers are exactly ``1..n`` — no gap, no repeat, so a
      reducer that restarted its counter or double-counted a row is caught;
    * reading the partition in row-number order, the ordering key is non-decreasing with
      nulls last — so a window that ignored its `order_by`, or placed nulls differently on
      one path, is caught even though the tie order itself is free.

    Args:
        out: The window operator's output table.
        partition: Name of the `partition_by` column.
        order: Name of the `order_by` column.
        rn: Name of the emitted row-number column.
    """
    d = out.to_pydict()
    per: dict[object, list[tuple[int, object]]] = {}
    for part, num, key in zip(d[partition], d[rn], d[order], strict=True):
        per.setdefault(part, []).append((num, key))
    for part, rows in per.items():
        numbers = sorted(n for n, _ in rows)
        assert numbers == list(range(1, len(rows) + 1)), (
            f"partition {part!r}: row numbers {numbers} are not 1..{len(rows)}"
        )
        keys = [k for _, k in sorted(rows)]
        seen_null = False
        for i, key in enumerate(keys):
            if key is None:
                seen_null = True
                continue
            assert not seen_null, f"partition {part!r}: null sorted before {key!r} at rank {i + 1}"
            if i and keys[i - 1] is not None:
                assert keys[i - 1] <= key, (
                    f"partition {part!r}: {order} decreases {keys[i - 1]!r} -> {key!r} at "
                    f"rank {i + 1}"
                )


def _stream(ds) -> pa.Table:
    """`iter_batches()` collected back into a table (the streaming scheduling)."""
    batches = list(ds.iter_batches())
    if not batches:
        return ds.collect().slice(0, 0)
    return pa.Table.from_batches(batches, schema=batches[0].schema)


@pytest.mark.parametrize("op", sorted(UNORDERED_OPS))
@pytest.mark.parametrize("shape", sorted(INPUTS))
def test_every_path_agrees_with_the_oracle(op, shape):
    """spill and streaming are schedulings of `collect()`, so they must equal it exactly."""
    build, _ = UNORDERED_OPS[op]
    table = INPUTS[shape]
    oracle = build(bt.from_arrow(table)).collect()
    assert_tables_equal(build(bt.from_arrow(table)).collect(spill=True), oracle)
    assert_tables_equal(_stream(build(bt.from_arrow(table))), oracle)


#: Two more schedulings of the same semantics, kept beside the three above because they are
#: the same claim: only *where and when* the work happens changes.
#:
#: `spill(num_partitions=…)` forces a bucket count the data-sized default would not pick, and
#: the bucket count is what three of the four bugs in this file's header had in common — a
#: key that lands in one bucket by default and two when the count is forced is a key whose
#: grouping is being decided by the partitioner rather than by its value.
#:
#: `adaptive=True` re-plans at pipeline breakers on *measured* cardinalities, so it can pick
#: a different join side or algorithm than the one-shot plan did. It is the one path whose
#: whole purpose is to arrive at a different plan, which makes "same answer" a claim worth
#: making explicitly rather than one that follows from the operator tests.
#:
#: `repartitioned` splits the *input* before the operator runs, which is the closest thing to
#: a distributed shuffle that CI can execute: the same `partial → combine → finalize`
#: primitives run per partition and merge, with no Ray involved. The Ray-backed matrix in
#: `test_diff_distributed_operator_matrix.py` covers the real thing, and the repo contract is
#: explicit that CI installs no Ray — so without an entry here the whole operator table has no
#: partitioned coverage in the PR gate at all.
#:
#: `repartitioned_sparse` asks for far more partitions than some shapes have rows, so most
#: come back empty. An operator that mishandles an empty partial — by skipping the merge, or
#: by seeding an identity that is wrong for it — passes every dense test and fails here.
_SCHEDULINGS = {
    "spill_partitioned": lambda ds: ds.collect(spill=True, num_partitions=3),
    "adaptive": lambda ds: ds.collect(adaptive=True),
    "repartitioned": lambda ds: ds.repartition(4).collect(),
    "repartitioned_sparse": lambda ds: ds.repartition(64).collect(),
}


@pytest.mark.parametrize("scheduling", sorted(_SCHEDULINGS))
@pytest.mark.parametrize("op", sorted(UNORDERED_OPS))
@pytest.mark.parametrize("shape", sorted(INPUTS))
def test_the_replanning_and_repartitioning_paths_agree_too(scheduling, op, shape):
    build, _ = UNORDERED_OPS[op]
    table = INPUTS[shape]
    oracle = build(bt.from_arrow(table)).collect()
    got = _SCHEDULINGS[scheduling](build(bt.from_arrow(table)))
    assert_tables_equal(got, oracle)


@pytest.mark.parametrize("op", sorted(o for o, (_, sql) in UNORDERED_OPS.items() if sql))
@pytest.mark.parametrize("shape", sorted(INPUTS))
def test_every_operator_matches_duckdb(duck, op, shape):
    """...and `collect()` itself matches the external oracle, on every edge-case input."""
    build, sql = UNORDERED_OPS[op]
    table = INPUTS[shape]
    duck.register("t", table)
    assert_same(build(bt.from_arrow(table)).collect(), duck.sql(sql))


@pytest.mark.parametrize("shape", sorted(INPUTS))
@pytest.mark.parametrize("keys", [["g"], ["f"], ["g", "k"]])
def test_keyed_dedup_any_keeps_one_real_row_per_key(shape, keys):
    """`keep="any"` on all three paths, asserted on what it actually guarantees.

    Which row survives is unspecified, so there is no oracle and no path-vs-path comparison
    to make. What must hold however the row is picked is: exactly one row per distinct key,
    the same key *set* the whole-row dedup of those key columns produces, and every returned
    row is one that was really in the input — a synthesized row (one column's value paired
    with another row's) would satisfy a row-count check and fail this.

    The float key `f` carries both zeros and a NaN, so it also pins that the survivor is
    chosen under the engine's float identity rather than raw bits.
    """
    table = INPUTS[shape]
    real = {tuple(row.values()) for row in _rows(table)}
    key_set = _rows(bt.from_arrow(table).select(*keys).distinct().collect())
    for out in (
        bt.from_arrow(table).distinct(keys).collect(),
        bt.from_arrow(table).distinct(keys).collect(spill=True),
        _stream(bt.from_arrow(table).distinct(keys)),
    ):
        rows = _rows(out)
        assert len(rows) == len(key_set), f"expected one row per key, got {len(rows)}"
        assert {tuple(r[k] for k in keys) for r in rows} == {
            tuple(r[k] for k in keys) for r in key_set
        }
        for row in rows:
            assert tuple(row.values()) in real, f"row {row} was never in the input"


def _rows(table: pa.Table) -> list[dict]:
    """`table`'s rows as dicts, with float keys canonicalized the way the engine groups them.

    `-0.0` and `0.0` are one key to the engine and every NaN is one key, so a membership test
    against the raw values would call a correctly-returned row missing.
    """
    import math

    def canon(v):
        if isinstance(v, float):
            if math.isnan(v):
                return "nan"
            return 0.0 if v == 0.0 else v
        return v

    return [{k: canon(v) for k, v in row.items()} for row in table.to_pylist()]


@pytest.mark.parametrize("shape", sorted(INPUTS))
def test_row_number_is_a_permutation_consistent_with_its_order_key(shape):
    """The tie-heavy `order_by="k"` window, asserted on what it actually guarantees.

    `UNORDERED_OPS["window_row_number"]` orders on every column so its output multiset is
    well defined; this keeps the partially-ordered shape covered on all three single-node
    paths by asserting the contract instead of one path's tie order.
    """
    plan = lambda d: d.with_columns(rn=bt.row_number().over(partition_by="g", order_by="k"))  # noqa: E731
    table = INPUTS[shape]
    for out in (
        plan(bt.from_arrow(table)).collect(),
        plan(bt.from_arrow(table)).collect(spill=True),
        _stream(plan(bt.from_arrow(table))),
    ):
        assert_row_number_contract(out, partition="g", order="k")


def assert_sort_contract(
    out: pa.Table, table: pa.Table, *, key: str, descending: bool, nulls_first: bool
) -> None:
    """Assert everything ``ORDER BY <one key>`` actually promises.

    Which of two rows *tied on the key* comes first is unspecified in SQL, so a row-by-row
    comparison against another engine over-asserts on a partially-ordered key: it compares
    a free choice. On the 15-row shapes Batcher and DuckDB happened to agree; on
    `MULTIBATCH`, where each key value repeats 2,400 times, they diverge at the first tie
    group — the sort is correct and the assertion was wrong.

    What *is* specified, and what this pins instead, is:

    * the output is a permutation of the input — no row invented, dropped, or duplicated,
      which is how a spilled run that loses or re-emits a bucket is caught;
    * the non-null keys are monotone in the requested direction;
    * the nulls are contiguous, and on the requested side.

    Those three together still fail every sort bug this file was written for, including
    the spilled `descending` sort that emitted nulls mid-result.

    Args:
        out: The sort's output table.
        table: The input table it was sorted from.
        key: Name of the single sort key.
        descending: Whether the sort was descending.
        nulls_first: Whether nulls were requested first.
    """
    assert_tables_equal(out, table)  # a permutation: same multiset of rows

    keys = out.column(key).to_pylist()
    non_null = [k for k in keys if k is not None]
    nulls_at_front = keys[: len(keys) - len(non_null)]
    nulls_at_back = keys[len(non_null) :]
    if nulls_first:
        assert all(k is None for k in nulls_at_front), (
            f"nulls_first: {key} nulls are not all at the front: {keys[:20]}"
        )
        ordered_keys = keys[len(keys) - len(non_null) :]
    else:
        assert all(k is None for k in nulls_at_back), (
            f"nulls_last: {key} nulls are not all at the back: {keys[-20:]}"
        )
        ordered_keys = keys[: len(non_null)]
    assert None not in ordered_keys, f"{key} nulls are not contiguous: {keys[:20]}"
    for i in range(1, len(ordered_keys)):
        prev, cur = ordered_keys[i - 1], ordered_keys[i]
        if descending:
            assert prev >= cur, f"{key} increases {prev!r} -> {cur!r} at rank {i + 1}"
        else:
            assert prev <= cur, f"{key} decreases {prev!r} -> {cur!r} at rank {i + 1}"


@pytest.mark.parametrize(("descending", "nulls_first"), ORDERINGS)
@pytest.mark.parametrize("shape", sorted(INPUTS))
def test_sort_matches_duckdb_on_every_ordering(duck, shape, descending, nulls_first):
    """Ordered assertion — the only kind that can see a sort bug.

    Ordered on *every* column, so each tie group holds wholly identical rows and the row
    sequence is uniquely determined. Ordering on `k` alone leaves the within-tie order a
    free choice, which is not a shared contract to assert against DuckDB; that shape is
    covered by `test_sort_contract_holds_on_a_tie_heavy_key` instead.
    """
    table = INPUTS[shape]
    duck.register("t", table)
    out = (
        bt.from_arrow(table)
        .sort(*[bt.col(c) for c in TOTAL_ORDER], descending=descending, nulls_first=nulls_first)
        .collect()
    )
    d, n = ("DESC" if descending else "ASC"), ("FIRST" if nulls_first else "LAST")
    order_by = ", ".join(f"{c} {d} NULLS {n}" for c in TOTAL_ORDER)
    assert_same_ordered(out, duck.sql(f"SELECT * FROM t ORDER BY {order_by}"))


@pytest.mark.parametrize(("descending", "nulls_first"), ORDERINGS)
@pytest.mark.parametrize("shape", sorted(INPUTS))
@pytest.mark.parametrize("key", ["k", "g"])
def test_sort_contract_holds_on_a_tie_heavy_key(shape, key, descending, nulls_first):
    """The single-key sort, asserted on what it guarantees, on every path.

    This is the shape the DuckDB comparison can no longer assert a row sequence for. It is
    the interesting one — `MULTIBATCH` ties 2,400 rows per key value, spanning morsels — so
    it is checked here on all three schedulings rather than dropped.
    """
    table = INPUTS[shape]
    plan = bt.from_arrow(table).sort(bt.col(key), descending=descending, nulls_first=nulls_first)
    for out in (plan.collect(), plan.collect(spill=True), _stream(plan)):
        assert_sort_contract(out, table, key=key, descending=descending, nulls_first=nulls_first)


@pytest.mark.parametrize(("descending", "nulls_first"), ORDERINGS)
@pytest.mark.parametrize("shape", sorted(INPUTS))
@pytest.mark.parametrize("key", ["k", "g"])  # numeric key range-partitions; string key falls back
def test_sort_paths_agree_on_every_ordering(shape, key, descending, nulls_first):
    """The spilled and streamed sort equal the in-memory sort, for every ordering flag."""
    plan = bt.from_arrow(INPUTS[shape]).sort(
        bt.col(key), descending=descending, nulls_first=nulls_first
    )
    oracle = plan.collect()
    assert_tables_equal(plan.collect(spill=True), oracle, ordered=True)
    assert_tables_equal(_stream(plan), oracle, ordered=True)


# --- the assertions themselves, tested ----------------------------------------------
#
# `assert_sort_contract` replaced a row-by-row DuckDB comparison that could not survive a
# tie-heavy key. A weaker replacement would be invisible: every sort test would still be
# green, and the sort would simply stop being checked. So the helper is fed each defect it
# claims to reject, and must reject it.


def _sorted_ok(descending: bool = False, nulls_first: bool = False) -> tuple:
    """A correctly sorted (output, input) pair to mutate in the tests below."""
    table = pa.table(
        {"k": pa.array([3, 1, None, 2, None, 1], pa.int64()), "v": pa.array(list("abcdef"))}
    )
    out = (
        bt.from_arrow(table)
        .sort(bt.col("k"), descending=descending, nulls_first=nulls_first)
        .collect()
    )
    return out, table


def test_the_sort_contract_accepts_a_correct_sort():
    for descending, nulls_first in ORDERINGS:
        out, table = _sorted_ok(descending, nulls_first)
        assert_sort_contract(out, table, key="k", descending=descending, nulls_first=nulls_first)


def test_the_sort_contract_rejects_a_dropped_row():
    """The permutation half — a spilled run that loses a bucket."""
    out, table = _sorted_ok()
    with pytest.raises(AssertionError):
        assert_sort_contract(
            out.slice(0, out.num_rows - 1), table, key="k", descending=False, nulls_first=False
        )


def test_the_sort_contract_rejects_a_duplicated_row():
    """...and one that re-emits a bucket."""
    out, table = _sorted_ok()
    doubled = pa.concat_tables([out, out.slice(0, 1)])
    with pytest.raises(AssertionError):
        assert_sort_contract(doubled, table, key="k", descending=False, nulls_first=False)


def test_the_sort_contract_rejects_an_unsorted_key():
    """The monotonicity half — the same rows, in the wrong order."""
    out, table = _sorted_ok()
    scrambled = pa.concat_tables([out.slice(out.num_rows - 1, 1), out.slice(0, out.num_rows - 1)])
    with pytest.raises(AssertionError):
        assert_sort_contract(scrambled, table, key="k", descending=False, nulls_first=False)


def test_the_sort_contract_rejects_a_null_emitted_mid_result():
    """The exact historic bug: the spilled `descending` sort put nulls in the middle.

    The rows are all present and the non-null keys are still monotone, so only the
    null-placement half of the contract can catch it.
    """
    out, table = _sorted_ok(descending=True, nulls_first=False)
    rows = out.to_pylist()
    non_null = [r for r in rows if r["k"] is not None]
    nulls = [r for r in rows if r["k"] is None]
    mid = len(non_null) // 2
    interleaved = pa.Table.from_pylist(non_null[:mid] + nulls + non_null[mid:], schema=out.schema)
    with pytest.raises(AssertionError):
        assert_sort_contract(interleaved, table, key="k", descending=True, nulls_first=False)


def test_the_sort_contract_rejects_the_wrong_null_side():
    """`nulls_first=True` satisfied by a nulls-last result, and vice versa."""
    out, table = _sorted_ok(descending=False, nulls_first=False)
    with pytest.raises(AssertionError):
        assert_sort_contract(out, table, key="k", descending=False, nulls_first=True)


def test_the_sort_contract_rejects_the_wrong_direction():
    """An ascending result asserted as descending."""
    out, table = _sorted_ok(descending=False, nulls_first=False)
    with pytest.raises(AssertionError):
        assert_sort_contract(out, table, key="k", descending=True, nulls_first=False)
