"""`partition_by`, `match_to_schema` and `drop_nans` against DuckDB, on every path.

`partition_by` is one filter per key after an eager ``distinct``; the part a key maps to is
``WHERE k IS NOT DISTINCT FROM <key>`` in DuckDB, with NaN matching NaN, which is the key
identity `group_by` uses and the one easiest to break with a plain ``==``. `match_to_schema` is
a projection with casts decided from the schema alone. `drop_nans` is a filter whose one trap
is that ``isnan(NULL)`` is null, which a naive negation turns into "drop the null row".

Every case runs ``collect()``, a spilled ``collect(num_partitions=4)`` and ``iter_batches()``
over inputs with nulls, NaN, ``-0.0``, an empty relation, one row, duplicates, a descending
arrival order and a ``multibatch`` shape past two morsels. The results are row multisets, so
`assert_same` is the comparison; `drop_nans` tables go through `duck_materialize` because a
registered Arrow scan evaluates NaN comparisons with IEEE semantics.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, duck_materialize

pytestmark = pytest.mark.differential

PATHS = {
    "collect": lambda ds: ds.collect(),
    "spill": lambda ds: ds.collect(spill=True, num_partitions=4),
    "iter_batches": lambda ds: _stream(ds),
}


def _stream(ds: bt.Dataset) -> pa.Table:
    batches = list(ds.iter_batches())
    if not batches:
        return ds.collect().slice(0, 0)
    return pa.Table.from_batches(batches, schema=batches[0].schema)


def _keyed(n: int) -> pa.Table:
    """A nullable string key, a float key with NaN and both zeros, duplicates, a payload."""
    g = [None if i % 9 == 4 else "abc"[i % 3] for i in range(n)]
    floats = [0.0, -0.0, float("nan"), 1.5, None]
    f = [floats[i % 5] for i in range(n)]
    v = [None if i % 11 == 0 else i for i in range(n)]
    return pa.table(
        {
            "g": pa.array(g, pa.string()),
            "f": pa.array(f, pa.float64()),
            "v": pa.array(v, pa.int64()),
        }
    )


SHAPES = {
    "base": _keyed(40),
    "empty": _keyed(0),
    "single": _keyed(2).slice(1, 1),
    "descending": _keyed(40).sort_by([("v", "descending")]),
    "multibatch": _keyed(40_000),
}


def _key_sql(column: str, value) -> str:
    if value is None:
        return f"{column} IS NULL"
    if isinstance(value, float) and math.isnan(value):
        return f"isnan({column})"
    literal = f"'{value}'" if isinstance(value, str) else repr(value)
    return f"{column} = {literal}"


@pytest.mark.parametrize("path", sorted(PATHS))
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_partition_by_parts_match_duckdb(duck, shape, path):
    """One part per distinct (string, float) key, including the null and NaN keys."""
    table = SHAPES[shape]
    duck_materialize(duck, "t", table)
    parts = bt.from_arrow(table).partition_by(["g", "f"], include_key=False)
    distinct = duck.sql("SELECT DISTINCT g, f FROM t").fetchall()
    assert len(parts) == len(distinct)
    for (g, f), part in parts.items():
        where = f"{_key_sql('g', g)} AND {_key_sql('f', f)}"
        assert_same(PATHS[path](part), duck.sql(f"SELECT v FROM t WHERE {where}"))


def test_partition_by_orders_keys_with_nulls_last():
    parts = bt.from_arrow(_keyed(40)).partition_by("g")
    assert list(parts) == [("a",), ("b",), ("c",), (None,)]


@pytest.mark.parametrize("path", sorted(PATHS))
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_match_to_schema_matches_duckdb(duck, shape, path):
    """Reordered columns, an inserted typed-null column, a dropped extra, a widened int32."""
    table = SHAPES[shape]
    duck.register("t", table)
    ds = bt.from_arrow(table).match_to_schema(
        {"v": pa.int32(), "g": "string", "missing": float},
        missing_columns="insert",
        extra_columns="ignore",
    )
    out = PATHS[path](ds)
    assert out.column_names == ["v", "g", "missing"]
    assert out.schema.field("missing").type == pa.float64()
    assert_same(out, duck.sql("SELECT v, g, CAST(NULL AS DOUBLE) AS missing FROM t"))


def test_match_to_schema_computes_a_missing_column_from_an_expression(duck):
    table = _keyed(40)
    duck.register("t", table)
    ds = bt.from_arrow(table).match_to_schema(
        {"v": "int64", "g": "string", "f": "float64", "v2": "int64"},
        missing_columns={"v2": bt.col("v") * 2},
    )
    assert_same(ds.collect(), duck.sql("SELECT v, g, f, v * 2 AS v2 FROM t"))


@pytest.mark.parametrize(
    ("schema", "options", "message"),
    [
        ({"v": "int64", "g": "string"}, {}, "not in the schema"),
        ({"v": "int64", "g": "string", "f": "float64", "z": "int64"}, {}, "not in the input"),
        ({"v": "string", "g": "string", "f": "float64"}, {}, "cast it first"),
        ({"v": "int64", "g": "string", "f": "float64"}, {"extra_columns": "keep"}, "must be one"),
    ],
)
def test_match_to_schema_refuses_what_its_policies_refuse(schema, options, message):
    with pytest.raises(bt.PlanError, match=message):
        bt.from_arrow(_keyed(3)).match_to_schema(schema, **options)


@pytest.mark.parametrize("path", sorted(PATHS))
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_drop_nans_matches_duckdb(duck, shape, path):
    """NaN rows go, null rows stay, and `-0.0` is a number, over the default and a subset."""
    table = SHAPES[shape].append_column(
        "h", pa.array([float("nan") if i % 4 == 1 else 2.0 for i in range(len(SHAPES[shape]))])
    )
    duck_materialize(duck, "t", table)
    both = bt.from_arrow(table).drop_nans()
    assert_same(
        PATHS[path](both),
        duck.sql("SELECT * FROM t WHERE NOT coalesce(isnan(f), false) AND NOT isnan(h)"),
    )
    only_f = bt.from_arrow(table).drop_nans("f")
    assert_same(
        PATHS[path](only_f), duck.sql("SELECT * FROM t WHERE NOT coalesce(isnan(f), false)")
    )


def test_drop_nans_refuses_a_non_float_column():
    with pytest.raises(bt.PlanError, match="not floating-point"):
        bt.from_arrow(_keyed(3)).drop_nans(["f", "v"])
