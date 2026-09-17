"""Differential: `group_by(maintain_order=True)` emits groups in first-appearance order.

The claim is an *order*, so every assertion here is order-sensitive (`assert_same_ordered`,
`assert_tables_equal(..., ordered=True)`); an order-independent comparison would pass whatever
order came out. DuckDB is the oracle, given an explicit input position so its answer is
defined rather than observed: ``GROUP BY g ORDER BY min(pos)``.

The order must also survive every scheduling (`collect`, spill, a forced spill bucket count,
`iter_batches`) on inputs that cross a morsel boundary, which is where a numbering that
restarted per batch or a sort that ran per bucket would show. The distributed half is
`tests/integration/test_group_by_maintain_order_distributed.py`.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from _harness import assert_same_ordered, assert_tables_equal

pytestmark = pytest.mark.differential

bt = pytest.importorskip("batcher")

#: A key whose first-appearance order (c, None, a, b) is neither sorted nor reverse-sorted,
#: with a null group, duplicates, and a float key carrying both zeros and a NaN.
BASE = pa.table(
    {
        "g": pa.array(["c", None, "a", "c", "b", "a", None, "b", "c", "a"]),
        "f": pa.array([2.5, -0.0, float("nan"), 0.0, 2.5, None, float("nan"), 1.0, -1.0, 0.0]),
        "v": pa.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], pa.int64()),
    }
)
#: Past two 16,384-row morsels, with a group ("z") that first appears only in the last morsel,
#: so a numbering that restarted per batch would sort it ahead of groups seen earlier.
MULTIBATCH = pa.concat_tables(
    [pa.concat_tables([BASE] * 3500), BASE.set_column(0, "g", pa.array(["z"] * 10))]
)
INPUTS = {
    "base": BASE,
    "multibatch": MULTIBATCH,
    "one_row": BASE.slice(0, 1),
    "empty": BASE.slice(0, 0),
}

AGGS = {
    "count_and_sum": lambda gb: gb.agg(n=bt.count(), total=bt.col("v").sum()),
    "expression_over_aggregates": lambda gb: gb.agg(mean=bt.col("v").sum() / bt.count()),
    "len_shortcut": lambda gb: gb.len(),
    "ordered_reduce": lambda gb: gb.first("v", order_by="v"),
}
DUCK = {
    "count_and_sum": "count(*) AS n, sum(v) AS total",
    "expression_over_aggregates": "sum(v) / count(*) AS mean",
    "len_shortcut": 'count(*) AS "len"',
    "ordered_reduce": "min(v) AS v",
}


def _positioned(table: pa.Table) -> pa.Table:
    return table.append_column("pos", pa.array(range(table.num_rows), pa.int64()))


def _stream(ds) -> pa.Table:
    batches = list(ds.iter_batches())
    if not batches:
        return ds.collect().slice(0, 0)
    return pa.Table.from_batches(batches, schema=batches[0].schema)


@pytest.mark.parametrize("key", ["g", "f"])
@pytest.mark.parametrize("agg", sorted(AGGS))
@pytest.mark.parametrize("shape", sorted(INPUTS))
def test_groups_come_out_in_first_appearance_order(duck, shape, agg, key):
    table = INPUTS[shape]
    columns = [key, "v"]
    got = AGGS[agg](bt.from_arrow(table.select(columns)).group_by(key, maintain_order=True))
    duck.register("t", _positioned(table.select(columns)))
    oracle = duck.sql(f"SELECT {key}, {DUCK[agg]} FROM t GROUP BY {key} ORDER BY min(pos)")
    assert_same_ordered(got.collect(), oracle)


@pytest.mark.parametrize("agg", sorted(AGGS))
@pytest.mark.parametrize("shape", sorted(INPUTS))
def test_every_scheduling_emits_the_same_order(shape, agg):
    def build():
        return AGGS[agg](bt.from_arrow(INPUTS[shape]).group_by("g", maintain_order=True))

    oracle = build().collect()
    assert_tables_equal(build().collect(spill=True), oracle, ordered=True)
    assert_tables_equal(build().collect(spill=True, num_partitions=3), oracle, ordered=True)
    assert_tables_equal(_stream(build()), oracle, ordered=True)


def test_the_fixture_order_is_not_one_a_sort_would_produce():
    """Positive control: a result sorted on the key, either way, fails the order assertion."""
    first_seen = list(dict.fromkeys(BASE.column("g").to_pylist()))
    present = [k for k in first_seen if k is not None]
    assert present not in (sorted(present), sorted(present, reverse=True))
    got = bt.from_arrow(BASE).group_by("g", maintain_order=True).len().to_pydict()["g"]
    assert got == first_seen


def test_a_derived_key_keeps_first_appearance_order(duck):
    got = (
        bt.from_arrow(BASE)
        .group_by(bucket=bt.col("v") % 3, maintain_order=True)
        .agg(total=bt.col("v").sum())
    )
    duck.register("t", _positioned(BASE))
    oracle = duck.sql(
        "SELECT v % 3 AS bucket, sum(v) AS total FROM t GROUP BY bucket ORDER BY min(pos)"
    )
    assert_same_ordered(got.collect(), oracle)


def test_the_default_is_unchanged_and_carries_no_sort():
    plain = bt.from_arrow(BASE).group_by("g").agg(n=bt.count())
    assert "Sort" not in repr(plain._plan)
    assert "Sort" in repr(bt.from_arrow(BASE).group_by("g", maintain_order=True).len()._plan)


def test_row_returning_methods_refuse_rather_than_ignore_it():
    gb = bt.from_arrow(BASE).group_by("g", maintain_order=True)
    with pytest.raises(bt.PlanError, match="maintain_order"):
        gb.head(1, order_by="v")
    with pytest.raises(bt.PlanError, match="maintain_order"):
        gb.map_groups(lambda batch: batch)


def test_a_non_bool_is_rejected():
    with pytest.raises(bt.PlanError, match="maintain_order"):
        bt.from_arrow(BASE).group_by("g", maintain_order="yes")
