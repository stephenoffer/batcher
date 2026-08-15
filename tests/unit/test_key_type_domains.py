"""A key column the row encoder cannot encode is refused at build time, not after the scan.

Grouping, `DISTINCT` and hash joins all identify rows by encoding the key columns into one
comparable byte string (`bc-runtime`'s `keys`, over arrow-rs's row format). That encoder
handles every type the engine otherwise supports -- including the nested ones people assume
it would not -- with one exception: `map`, whose entries have no canonical order and
therefore no stable encoding.

Before this the refusal arrived from Rust, after the scan, as

    RuntimeError: Not yet implemented: Row format support not yet implemented for:
    [SortField { options: SortOptions { descending: false, nulls_first: true }, data_type

naming neither the column nor the clause, and truncated mid-struct. That is the same failure
`plan.types.domains.aggregate_domain_error` was written to remove for aggregate *inputs* and
`plan.logical.join._validate_key_types` removed for *mismatched* join keys. This is the third
case and the one both left behind.

The supported list is asserted as well as the rejected one, because the risk of a rule like
this is over-rejection: `list`, `struct`, `list<struct>` and dictionary-encoded columns all
key correctly today and must keep doing so.
"""

from __future__ import annotations

import datetime
import decimal

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.unit

_MAP = pa.map_(pa.string(), pa.int64())

#: (label, arrow type, one value) for every type the row encoder accepts as a key.
_KEYABLE = [
    ("int64", pa.int64(), 1),
    ("string", pa.string(), "a"),
    ("large_string", pa.large_string(), "a"),
    ("binary", pa.binary(), b"a"),
    ("bool", pa.bool_(), True),
    ("float64", pa.float64(), 1.5),
    ("decimal", pa.decimal128(10, 2), decimal.Decimal("1.00")),
    ("date32", pa.date32(), datetime.date(2026, 1, 1)),
    ("timestamp", pa.timestamp("us"), datetime.datetime(2026, 1, 1)),
    ("list", pa.list_(pa.int64()), [1]),
    ("large_list", pa.large_list(pa.int64()), [1]),
    ("fixed_size_list", pa.list_(pa.int64(), 1), [1]),
    ("struct", pa.struct([("a", pa.int64())]), {"a": 1}),
    ("list_of_struct", pa.list_(pa.struct([("a", pa.int64())])), [{"a": 1}]),
]

#: The rejected shapes: a bare map, and a map nested at any depth.
_UNKEYABLE = [
    ("map", _MAP, [("a", 1)]),
    ("struct_of_map", pa.struct([("m", _MAP)]), {"m": [("a", 1)]}),
    ("list_of_map", pa.list_(_MAP), [[("a", 1)]]),
]


def _ds(arrow_type, value):
    schema = pa.schema([("k", arrow_type), ("v", pa.int64())])
    return bt.from_pydict({"k": [value, value], "v": [1, 2]}, schema=schema)


@pytest.mark.parametrize(("label", "dt", "value"), _UNKEYABLE, ids=[c[0] for c in _UNKEYABLE])
def test_a_map_key_is_refused_by_group_by(label, dt, value):
    with pytest.raises(PlanError, match="cannot be a key"):
        _ds(dt, value).group_by("k").agg(n=bt.count())


@pytest.mark.parametrize(("label", "dt", "value"), _UNKEYABLE, ids=[c[0] for c in _UNKEYABLE])
def test_a_map_key_is_refused_by_distinct(label, dt, value):
    """Whole-row `DISTINCT` keys on every column, so it must check the whole schema."""
    with pytest.raises(PlanError, match="cannot be a key"):
        _ds(dt, value).distinct()


@pytest.mark.parametrize(("label", "dt", "value"), _UNKEYABLE, ids=[c[0] for c in _UNKEYABLE])
def test_a_map_key_is_refused_by_a_keyed_dedup(label, dt, value):
    with pytest.raises(PlanError, match="cannot be a key"):
        _ds(dt, value).distinct(subset=["k"])


@pytest.mark.parametrize(("label", "dt", "value"), _UNKEYABLE, ids=[c[0] for c in _UNKEYABLE])
def test_a_map_key_is_refused_by_a_join(label, dt, value):
    left = _ds(dt, value)
    right = left.select(bt.col("k").alias("k2"), bt.col("v").alias("v2"))
    with pytest.raises(PlanError, match="cannot be a key"):
        left.join(right, left_on="k", right_on="k2")


def test_the_message_names_the_column_the_clause_and_a_way_forward():
    with pytest.raises(PlanError) as excinfo:
        _ds(_MAP, [("a", 1)]).group_by("k").agg(n=bt.count())
    message = str(excinfo.value)
    assert "group_by()" in message
    assert "'k'" in message
    assert "map.keys()" in message


@pytest.mark.parametrize(("label", "dt", "value"), _KEYABLE, ids=[c[0] for c in _KEYABLE])
def test_every_other_key_type_is_still_accepted(label, dt, value):
    """Over-rejection is the risk in a rule like this, so the supported list is pinned too."""
    ds = _ds(dt, value)
    assert ds.group_by("k").agg(n=bt.count()).count() >= 1
    assert ds.distinct().count() >= 1


def test_a_dictionary_key_is_still_accepted():
    """Built through `from_arrow`, since `from_pydict` cannot express a dictionary column."""
    table = pa.table(
        {"k": pa.array(["a", "b"]).dictionary_encode(), "v": pa.array([1, 2], pa.int64())}
    )
    assert bt.from_arrow(table).group_by("k").agg(n=bt.count()).count() == 2


def test_a_map_column_that_is_not_a_key_is_untouched():
    """The rule is about keys, not about carrying a map through a query."""
    ds = _ds(_MAP, [("a", 1)])
    assert ds.select(bt.col("v")).distinct().count() == 2
    assert ds.filter(bt.col("v") > 1).count() == 1
    assert ds.sort("v").limit(1).count() == 1


def test_sorting_a_map_column_is_still_allowed():
    """A sort compares values directly rather than through the row encoder, so it works —
    and a rule that rejected it would be refusing something the engine answers."""
    assert _ds(_MAP, [("a", 1)]).sort("k").count() == 2


# --- every other operator that keys through the same encoder ---------------------------
#
# `group_by`, `distinct` and `join` were the three the probe found first; sweeping the rest
# of the key-taking operators with the same map column found four more that leaked the same
# internal dump. `drop_duplicates_within_watermark`, `map_groups` and `merge` already
# refused it properly, which is what makes the list below the complete remainder rather
# than a sample.


def test_a_map_partition_key_is_refused_by_a_window():
    with pytest.raises(PlanError, match="over\\(partition_by="):
        _ds(_MAP, [("a", 1)]).with_columns(
            r=bt.col("v").rank().over(partition_by="k", order_by="v")
        )


def test_a_map_order_key_is_refused_by_a_window():
    with pytest.raises(PlanError, match="over\\(order_by="):
        _ds(_MAP, [("a", 1)]).with_columns(
            r=bt.col("v").rank().over(partition_by="v", order_by="k")
        )


def test_a_map_by_key_is_refused_by_an_asof_join():
    import datetime

    schema = pa.schema([("k", _MAP), ("ts", pa.timestamp("us"))])
    rows = {
        "k": [[("a", 1)], [("a", 2)]],
        "ts": [datetime.datetime(2026, 1, 1), datetime.datetime(2026, 1, 2)],
    }
    left = bt.from_pydict(rows, schema=schema)
    with pytest.raises(PlanError, match="join_asof"):
        left.join_asof(bt.from_pydict(rows, schema=schema), on="ts", by="k")


def test_a_map_column_is_refused_by_a_distinct_union():
    """UNION (distinct) dedups the concatenation over every column, so it keys on all."""
    ds = _ds(_MAP, [("a", 1)])
    with pytest.raises(PlanError, match="union\\(distinct=True\\)"):
        ds.union(ds, distinct=True)


def test_union_all_over_a_map_column_is_untouched():
    """UNION ALL concatenates and keys on nothing, so it must keep working."""
    ds = _ds(_MAP, [("a", 1)])
    assert ds.union(ds).count() == 4


def test_a_window_over_ordinary_keys_still_works():
    """Over-rejection guard for the window path."""
    ds = _ds(pa.int64(), 1)
    assert ds.with_columns(r=bt.col("v").rank().over(partition_by="k", order_by="v")).count() == 2
