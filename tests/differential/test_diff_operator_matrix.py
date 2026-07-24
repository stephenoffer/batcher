"""Every relational operator x every execution path, on one edge-case-loaded input.

`collect()`, `collect(spill=True)` and `iter_batches()` are three *schedulings* of the same
operator semantics (invariant #7), so they must agree with each other and with DuckDB — on
nulls, on empty input, on a single row, on `-0.0`/NaN float keys, and on every ordering flag.

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
INPUTS = {"base": BASE, "empty": BASE.slice(0, 0), "single": BASE.slice(0, 1)}
RIGHT = pa.table(
    {"k": pa.array([1, 3, 5, 7, 9, None], pa.int64()), "w": ["p", "q", "r", "s", "u", "z"]}
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


@pytest.mark.parametrize("op", sorted(o for o, (_, sql) in UNORDERED_OPS.items() if sql))
@pytest.mark.parametrize("shape", sorted(INPUTS))
def test_every_operator_matches_duckdb(duck, op, shape):
    """...and `collect()` itself matches the external oracle, on every edge-case input."""
    build, sql = UNORDERED_OPS[op]
    table = INPUTS[shape]
    duck.register("t", table)
    assert_same(build(bt.from_arrow(table)).collect(), duck.sql(sql))


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


@pytest.mark.parametrize(("descending", "nulls_first"), ORDERINGS)
@pytest.mark.parametrize("shape", sorted(INPUTS))
def test_sort_matches_duckdb_on_every_ordering(duck, shape, descending, nulls_first):
    """Ordered assertion — the only kind that can see a sort bug."""
    table = INPUTS[shape]
    duck.register("t", table)
    out = (
        bt.from_arrow(table)
        .sort(bt.col("k"), descending=descending, nulls_first=nulls_first)
        .collect()
    )
    d, n = ("DESC" if descending else "ASC"), ("FIRST" if nulls_first else "LAST")
    assert_same_ordered(out, duck.sql(f"SELECT * FROM t ORDER BY k {d} NULLS {n}"))


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
