"""Nested `.json` accessor extraction vs DuckDB — object/array leaves and key order.

Wave-2 coverage of `.json.extract_string` on non-scalar leaves. The lazy JSON scanner
must return an extracted sub-object with its *source* key order (as DuckDB's
`json_extract_string` does), not the alphabetized order `serde_json`'s default `Map`
produces when a leaf is round-tripped through a `Value`.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col
from conftest import assert_same


def test_json_extract_object_leaf_preserves_key_order(duck):
    # Source key order is b, a, c — NOT alphabetical. A reordering (a, b, c) would be a
    # silent divergence from DuckDB and from the document itself.
    tbl = pa.table(
        {
            "j": pa.array(
                [
                    '{"o":{"b":1,"a":2,"c":3}}',
                    '{"o":{ "zebra" : 10,  "apple": 20 }}',
                    '{"o":{"msg":"a  b { x }","z":9,"a":1}}',
                    '{"o":[3,1,2]}',
                    '{"o":{"nested":{"y":5,"x":6},"first":true}}',
                    None,
                ]
            ),
        }
    )
    duck.register("t", tbl)
    got = bt.from_arrow(tbl).select(v=col("j").json.extract_string("$.o")).collect()
    assert_same(got, duck.sql("SELECT json_extract_string(j, '$.o') AS v FROM t"))


def test_json_extract_big_integer_keeps_digits(duck):
    # An integer beyond i64/u64 must keep its exact digits, not degrade to `1e+20`
    # (serde_json parses it as f64). Values that fit and true floats stay canonical.
    tbl = pa.table(
        {
            "j": pa.array(
                [
                    '{"x":100000000000000000000}',
                    '{"x":9223372036854775807}',
                    '{"x":30}',
                    '{"x":1.50}',
                    '{"x":-123456789012345678901234}',
                ]
            ),
        }
    )
    duck.register("t", tbl)
    got = bt.from_arrow(tbl).select(v=col("j").json.extract_string("$.x")).collect()
    assert_same(got, duck.sql("SELECT json_extract_string(j, '$.x') AS v FROM t"))


def test_json_extract_nested_object_leaf(duck):
    tbl = pa.table(
        {"j": pa.array(['{"a":{"deep":{"d":1,"c":2,"b":3}}}', '{"a":{"deep":{}}}'])}
    )
    duck.register("t", tbl)
    got = bt.from_arrow(tbl).select(
        v=col("j").json.extract_string("$.a.deep"),
    ).collect()
    assert_same(got, duck.sql("SELECT json_extract_string(j, '$.a.deep') AS v FROM t"))


def test_json_extract_negative_array_index(duck):
    # DuckDB folds a negative subscript from the back of the array: `[-1]` is the last
    # element, `[-2]` the second-to-last, out of range -> NULL. The lazy scanner used
    # to parse the subscript as an unsigned integer, so `-1` failed to parse and the
    # step was silently DROPPED — `$.arr[-1]` returned the whole parent array instead
    # of its last element (silent wrong result). Positive indices are unaffected.
    tbl = pa.table(
        {
            "j": pa.array(
                [
                    '{"arr":[10,20,30]}',
                    '{"arr":[1,2,3,4,5]}',
                    '{"arr":[{"z":1},{"z":2}]}',
                    '{"arr":[]}',
                    "[7,8,9]",
                    None,
                ]
            ),
        }
    )
    duck.register("t", tbl)
    ds = bt.from_arrow(tbl)
    for path in ("$.arr[-1]", "$.arr[-2]", "$.arr[-3]", "$.arr[-9]", "$[-1]"):
        got = ds.select(v=col("j").json.extract_string(path)).collect()
        assert_same(got, duck.sql(f"SELECT json_extract_string(j, '{path}') AS v FROM t"))


def test_json_extract_negative_index_nested_and_int(duck):
    # A negative index that lands on an object element, extracted as an int, and the
    # positive fast path still working alongside it.
    tbl = pa.table({"j": pa.array(['{"a":[{"z":1},{"z":2},{"z":3}]}'])})
    duck.register("t", tbl)
    ds = bt.from_arrow(tbl)
    for path in ("$.a[-1].z", "$.a[0].z", "$.a[-2].z"):
        got = ds.select(v=col("j").json.extract_int(path)).collect()
        assert_same(got, duck.sql(f"SELECT json_extract(j, '{path}')::BIGINT AS v FROM t"))


def test_json_extract_negative_index_array_of_arrays(duck):
    # Chained negative subscripts on nested arrays, each folded from its own end.
    tbl = pa.table({"j": pa.array(['{"a":[[1,2],[3,4]]}'])})
    duck.register("t", tbl)
    ds = bt.from_arrow(tbl)
    for path in ("$.a[-1][-1]", "$.a[1][-2]", "$.a[-1]", "$.a[0][1]"):
        got = ds.select(v=col("j").json.extract_string(path)).collect()
        assert_same(got, duck.sql(f"SELECT json_extract_string(j, '{path}') AS v FROM t"))


def test_json_extract_negative_zero_integer(duck):
    # `-0` is an integer literal (serde parses it as f64); it is numerically zero and
    # DuckDB renders it "0", not the raw "-0". A negative-zero *float* keeps its sign.
    tbl = pa.table({"j": pa.array(['{"x":-0}', '{"x":-0.0}', '{"x":0}', '{"x":-5}'])})
    duck.register("t", tbl)
    got = bt.from_arrow(tbl).select(v=col("j").json.extract_string("$.x")).collect()
    assert_same(got, duck.sql("SELECT json_extract_string(j, '$.x') AS v FROM t"))
