"""The shuffle must agree with the group assigner about what makes two keys "the same".

Invariant #7 says one mergeable implementation serves single-node, multi-core, and
distributed: `combine_finalize(partition(partial(pₖ)))` == the single-node result. That
invariant is only as good as the *key identity* the partitioner uses — if the shuffle thinks
two keys differ but the assigner thinks they are one group, the two halves of a group land on
different reducers and the query returns two groups where the oracle returns one.

Both sides of that have been live bugs:

* a `Float64` key was row-encoded without canonicalization, so `-0.0` and `0.0` — one group to
  the assigner, one group to DuckDB — hashed to different buckets;
* an `Int64` key's NULL slots were hashed by their *arbitrary* underlying raw value (Arrow
  leaves the value under a null undefined), scattering NULLs across every bucket, so a
  distributed `GROUP BY` could emit up to `num_partitions` NULL groups instead of one.

These pin the partitioner directly, because a small in-memory query does not shuffle and so
cannot see either bug.
"""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pytest
from conftest import assert_same

pytestmark = pytest.mark.differential

bt = pytest.importorskip("batcher")
nat = pytest.importorskip("batcher._native")

from batcher.plan.expr_ir import col  # noqa: E402  (after importorskip)

PARTITIONS = 8


def _buckets_of(batch: pa.RecordBatch, key_index: int = 0) -> list[set]:
    """The set of key values landing in each shuffle bucket."""
    out: list[set] = []
    for part in nat.partition_batches([batch], [key_index], PARTITIONS):
        vals = set()
        for pb in part:
            vals.update(repr(v) for v in pb.column(key_index).to_pylist())
        out.append(vals)
    return out


def test_negative_zero_and_zero_share_a_shuffle_bucket():
    """`-0.0` and `0.0` are one SQL group, so they must reach the same reducer."""
    batch = pa.record_batch(
        {"k": pa.array([0.0, -0.0], pa.float64()), "v": pa.array([1, 1], pa.int64())}
    )
    homes = [i for i, b in enumerate(_buckets_of(batch)) if b]
    assert len(homes) == 1, f"-0.0 and 0.0 split across buckets {homes}"


def test_all_nan_payloads_share_a_shuffle_bucket():
    """Every NaN bit-pattern is one SQL group, so all NaNs must reach the same reducer."""
    nans = pa.array(
        [
            float("nan"),
            -float("nan"),
            np.float64(np.frombuffer(np.uint64(0x7FF8000000000001).tobytes(), dtype=np.float64)[0]),
        ],
        pa.float64(),
    )
    batch = pa.record_batch({"k": nans, "v": pa.array([1, 1, 1], pa.int64())})
    homes = [i for i, b in enumerate(_buckets_of(batch)) if b]
    assert len(homes) == 1, f"NaN payloads split across buckets {homes}"


def test_null_int_keys_share_a_shuffle_bucket():
    """All NULLs are one SQL group, whatever raw bytes sit under the null slots.

    Arrow leaves the value under a null undefined, so this builds an Int64 array whose null
    slots carry *distinct* leftover payloads — exactly what a parquet `pad_nulls` produces.
    """
    values = pa.py_buffer(np.arange(64, dtype=np.int64))  # distinct garbage under the nulls
    all_null = pa.py_buffer(np.zeros(8, dtype=np.uint8))  # validity: every slot null
    keys = pa.Array.from_buffers(pa.int64(), 64, [all_null, values])
    assert keys.null_count == 64
    batch = pa.record_batch({"k": keys, "v": pa.array(range(64), pa.int64())})
    homes = [i for i, b in enumerate(_buckets_of(batch)) if b]
    assert len(homes) == 1, f"NULL keys scattered across buckets {homes}"


def _mergeable_result(table: pa.Table, mappers: int = 2, reducers: int = 4) -> pa.Table:
    """Run `partial -> partition -> combine_finalize`, the distributed path, on one process."""
    group_keys = json.dumps([{"expr": col("k").to_ir(), "alias": "k"}])
    aggs = json.dumps([col("v").sum().to_ir("s")])
    chunks = np.array_split(np.arange(table.num_rows), mappers)
    per_reducer: list[list] = [[] for _ in range(reducers)]
    for idx in chunks:
        if not len(idx):
            continue
        shard = table.take(pa.array(idx)).to_batches()
        partial = nat.partial_aggregate(group_keys, aggs, shard)
        for r, bucket in enumerate(nat.partition_batches([partial], [0], reducers)):
            per_reducer[r].extend(bucket)
    out: list[pa.RecordBatch] = []
    for bucket in per_reducer:
        if not bucket:
            continue
        res = nat.combine_finalize(group_keys, aggs, bucket)
        out.extend(res if isinstance(res, list) else [res])
    return pa.Table.from_batches(out)


@pytest.mark.parametrize(
    ("name", "keys"),
    [
        ("both_zeros", pa.array([0.0] * 50 + [-0.0] * 50, pa.float64())),
        ("nan_payloads", pa.array([float("nan")] * 50 + [-float("nan")] * 50, pa.float64())),
        ("null_ints", pa.array([None] * 50 + [1] * 25 + [2] * 25, pa.int64())),
        ("mixed_floats", pa.array(([1.5, -0.0, 0.0, 2.5] * 25), pa.float64())),
    ],
)
def test_mergeable_invariant_holds_for_tricky_keys(duck, name, keys):
    """`combine_finalize(partition(partial(pₖ)))` == single-node == DuckDB (invariant #7).

    This is the end-to-end statement of the two bugs above: a shuffle that disagrees with the
    assigner about key identity breaks the invariant even though every individual primitive
    "works".
    """
    table = pa.table({"k": keys, "v": pa.array([1] * len(keys), pa.int64())})
    duck.register("t", table)
    distributed = _mergeable_result(table)
    assert_same(distributed, duck.sql("SELECT k, SUM(v) AS s FROM t GROUP BY k"))


def test_nullable_join_key_keeps_every_match(duck):
    """An inner join whose key column carries a NULL must keep every non-null match.

    Regression: the shuffle's raw-hash fast paths bailed on a nullable column to the
    `RowConverter`, while a null-free column of the same type hashed raw — the two disagreed on
    equal *non-null* keys, splitting them across buckets so the parallel/distributed join
    silently dropped matches (196,868 of 200,000 in the original repro).
    """
    n = 20_000
    left = pa.table(
        {"k": pa.array([*range(1, n + 1), None], pa.int64()), "lv": pa.array([1] * (n + 1))}
    )
    right = pa.table({"k": pa.array(list(range(1, n + 1)), pa.int64()), "rv": pa.array([2] * n)})
    got = bt.from_arrow(left).join(bt.from_arrow(right), on="k", how="inner").collect()
    duck.register("l", left)
    duck.register("r", right)
    assert_same(got, duck.sql("SELECT * FROM l JOIN r USING (k)"))
    assert got.num_rows == n


def test_composite_nullable_join_key_keeps_every_match(duck):
    """Same, for a composite `(Int64, Int64)` key with a NULL in one side."""
    n = 10_000
    left = pa.table(
        {
            "a": pa.array([*range(n), None], pa.int64()),
            "b": pa.array([1] * (n + 1), pa.int64()),
            "lv": pa.array([1] * (n + 1)),
        }
    )
    right = pa.table(
        {
            "a": pa.array(list(range(n)), pa.int64()),
            "b": pa.array([1] * n, pa.int64()),
            "rv": pa.array([2] * n),
        }
    )
    got = bt.from_arrow(left).join(bt.from_arrow(right), on=["a", "b"], how="inner").collect()
    duck.register("l2", left)
    duck.register("r2", right)
    assert_same(got, duck.sql("SELECT * FROM l2 JOIN r2 USING (a, b)"))
    assert got.num_rows == n


def test_float_join_key_matches_scalar_equality_a_documented_duckdb_divergence(duck):
    """A float join key matches iff `=` says it does: `0.0 ⋈ -0.0` joins. **DuckDB disagrees.**

    This is a deliberate, surfaced divergence from the oracle — the one case in the suite where
    Batcher is right and DuckDB is wrong, recorded here so nobody "fixes" Batcher to match it.

    SQL says `0.0 = -0.0` is TRUE, and a join matches exactly where `=` holds. DuckDB agrees on
    the scalar (`SELECT 0.0 = -0.0` → true) and in `GROUP BY` (the two zeros form one group), but
    its **hash join is bitwise**: it misses `0.0 ⋈ -0.0` under `USING (k)`, under `ON l.k = r.k`,
    and even under `FROM l, r WHERE l.k = r.k`. So DuckDB contradicts *itself* — its join
    disagrees with its own `=` operator and its own `GROUP BY`.

    Batcher is internally consistent instead: scalar `=`, `GROUP BY`, `DISTINCT`, and `JOIN` all
    fold the two zeros (and all NaN payloads), because every one of them derives key identity from
    the single `bc_runtime::keys` canonicalization. Consistency across our own operators is worth
    more than bug-compatibility with the oracle, so we take the divergence knowingly.

    NaN is *not* a divergence: DuckDB matches `NaN ⋈ NaN`, and so do we — asserted against the
    oracle below.
    """
    left = pa.table({"k": pa.array([0.0, 1.5], pa.float64()), "lv": [1, 3]})
    right = pa.table({"k": pa.array([-0.0, 1.5], pa.float64()), "rv": [4, 6]})
    got = bt.from_arrow(left).join(bt.from_arrow(right), on="k", how="inner").collect()

    # Batcher: both keys join (SQL-correct).
    assert got.num_rows == 2, f"expected 0.0 ⋈ -0.0 and 1.5 ⋈ 1.5, got {got.to_pydict()}"

    # ...and this is exactly where DuckDB differs. Pin the divergence so a DuckDB upgrade that
    # fixes their hash join turns this into a *failing* test, prompting us to drop the exception.
    duck.register("lf", left)
    duck.register("rf", right)
    duck_rows = duck.sql("SELECT * FROM lf JOIN rf USING (k)").to_arrow_table().num_rows
    assert duck_rows == 1, (
        "DuckDB now matches 0.0 with -0.0 in a hash join — it used to miss it. The divergence "
        "documented here is gone; switch this test back to a plain `assert_same` against DuckDB."
    )

    # NaN == NaN in a join: DuckDB and Batcher agree, so assert against the oracle directly.
    ln = pa.table({"k": pa.array([float("nan")], pa.float64()), "lv": [1]})
    rn = pa.table({"k": pa.array([float("nan")], pa.float64()), "rv": [2]})
    got_nan = bt.from_arrow(ln).join(bt.from_arrow(rn), on="k", how="inner").collect()
    duck.register("ln", ln)
    duck.register("rn", rn)
    assert_same(got_nan, duck.sql("SELECT * FROM ln JOIN rn USING (k)"))


def test_float_key_identity_is_the_same_across_every_operator():
    """Scalar `=`, GROUP BY, DISTINCT and JOIN must all agree on what "the same float" means.

    They agree because they all route through one `bc_runtime::keys` canonicalization. This test
    is the statement of that: if someone adds a sixth key-hashing path and forgets to canonicalize,
    the operators start disagreeing with each other, and this fails.
    """
    pair = pa.table({"a": pa.array([0.0], pa.float64()), "b": pa.array([-0.0], pa.float64())})
    scalar_eq = (
        bt.from_arrow(pair)
        .with_columns(e=(bt.col("a") == bt.col("b")))
        .collect()
        .column("e")
        .to_pylist()[0]
    )
    assert scalar_eq is True, "scalar `=` must hold 0.0 == -0.0"

    zeros = pa.table({"k": pa.array([0.0, -0.0, 1.5], pa.float64())})
    grouped = bt.from_arrow(zeros).group_by("k").agg(c=bt.col("k").count()).collect()
    assert grouped.num_rows == 2, "GROUP BY must fold the two zeros into one group"

    distinct = bt.from_arrow(zeros).select(bt.col("k")).distinct().collect()
    assert distinct.num_rows == 2, "DISTINCT must fold the two zeros into one row"

    left = pa.table({"k": pa.array([0.0, 1.5], pa.float64()), "lv": [1, 3]})
    right = pa.table({"k": pa.array([-0.0, 1.5], pa.float64()), "rv": [4, 6]})
    joined = bt.from_arrow(left).join(bt.from_arrow(right), on="k", how="inner").collect()
    assert joined.num_rows == 2, "JOIN must match 0.0 with -0.0, like `=` does"


def test_distinct_on_float_key_folds_negative_zero(duck):
    """`distinct(subset)` lowers to `row_number() OVER (PARTITION BY subset ...)`; the window
    partition grouping must fold `-0.0`/`0.0` (and all NaNs) like GROUP BY does."""
    table = pa.table(
        {
            "f": pa.array([0.0, -0.0, 0.0, float("nan"), float("nan")], pa.float64()),
            "v": [1, 2, 3, 4, 5],
        }
    )
    got = bt.from_arrow(table).distinct(["f"]).collect()
    # One row per distinct float key (two zeros fold; all NaNs fold): DuckDB's DISTINCT ON.
    duck.register("tf", table)
    assert_same(
        got.select(["f"]),
        duck.sql("SELECT DISTINCT f FROM tf"),
    )
