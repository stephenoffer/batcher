"""Differential coverage for the nested-data API additions: JSON, list and struct editing.

Each case runs the same input through Batcher and through the DuckDB function it is
documented against, over nulls, empties, duplicates and a single row. Where Batcher
deliberately answers differently (a null list zipped with padding, the null element kept by
`list.intersect`), the difference is pinned explicitly rather than filtered away.
"""

from __future__ import annotations

import json

import duckdb
import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col, element, element_index

pytestmark = pytest.mark.differential

DOCS = [
    '{"a": 1, "b": [1, "x", 2.5], "c": {"d": "s"}, "x.y": 5}',
    '{"a": 1.5, "c": {"d": 5}}',
    '{"a": "7", "b": "notalist", "c": null}',
    '{"a": true, "b": []}',
    "{}",
    "[1, 2]",
    None,
]


def _docs(duck, rows=DOCS):
    ds = bt.from_pydict({"j": rows})
    duck.register("t", ds.collect())
    return ds


# --- JSONPath ------------------------------------------------------------------


@pytest.mark.parametrize(
    "path", ["$.a", "$.b[0]", "$.b[-1]", "$.c.d", '$."x.y"', "$.b[5]", "$", "$.missing.deeper"]
)
def test_supported_paths_match_duckdb(duck, path):
    ds = _docs(duck)
    out = ds.select(r=col("j").json.extract_string(path)).collect()
    assert_same(out, duck.sql(f"SELECT json_extract_string(j, '{path}') AS r FROM t"))


@pytest.mark.parametrize("fn", ["json_extract", "json_extract_string"])
def test_a_trailing_wildcard_lists_every_element_like_duckdb(duck, fn):
    rows = ['{"a": [1, "x", null, {"b": 2}]}', '{"a": 1}', "{}", '{"a": []}', None]
    ds = _docs(duck, rows)
    query = f"SELECT {fn}(j, '$.a[*]') AS r FROM t"
    assert_same(bt.sql(query, t=ds).collect(), duck.sql(query))


def test_a_bracket_quoted_key_reads_the_key_holding_a_dot(duck):
    ds = _docs(duck)
    out = ds.select(r=col("j").json.extract_int("$['x.y']")).collect()
    assert_same(out, duck.sql("""SELECT json_extract(j, '$."x.y"')::BIGINT AS r FROM t"""))


# --- json.decode / json.encode / json.merge_patch ------------------------------


def test_decode_matches_json_transform(duck):
    ds = _docs(duck, [*DOCS[:5], None])
    target = pa.struct(
        [("a", pa.int64()), ("b", pa.list_(pa.int64())), ("c", pa.struct([("d", pa.string())]))]
    )
    out = ds.select(r=col("j").json.decode(target)).collect()
    shape = '{"a":"BIGINT","b":["BIGINT"],"c":{"d":"VARCHAR"}}'
    assert_same(out, duck.sql(f"SELECT json_transform(j, '{shape}') AS r FROM t"))


def test_decode_of_a_list_and_a_scalar_matches_json_transform(duck):
    ds = _docs(duck, ["[1, 2.5, null]", "[]", '"7"', "true", None])
    out = ds.select(
        xs=col("j").json.decode(pa.list_(pa.float64())), n=col("j").json.decode("int64")
    ).collect()
    expect = duck.sql(
        """SELECT json_transform(j, '["DOUBLE"]') AS xs, json_transform(j, '"BIGINT"') AS n
           FROM t"""
    )
    assert_same(out, expect)


def test_strict_decode_raises_where_json_transform_strict_does(duck):
    ds = _docs(duck, ['{"a": "x"}'])
    with pytest.raises(duckdb.InvalidInputException):
        duck.sql("""SELECT json_transform_strict(j, '{"a":"BIGINT"}') FROM t""").fetchall()
    with pytest.raises(Exception, match="strict"):
        ds.select(r=col("j").json.decode(pa.struct([("a", pa.int64())]), strict=True)).collect()


def test_encode_matches_to_json_as_parsed_json(duck):
    ds = bt.from_pydict(
        {
            "s": [
                {"a": 1, "b": [1.5, None], "c": 'q"x'},
                {"a": None, "b": [], "c": ""},
                None,
            ]
        }
    )
    duck.register("t", ds.collect())
    ours = ds.select(r=col("s").json.encode()).to_pydict()["r"]
    theirs = [r[0] for r in duck.sql("SELECT to_json(s) FROM t").fetchall()]
    # Text equality would compare float spelling; the contract is the JSON value.
    assert [None if r is None else json.loads(r) for r in ours] == [
        None if r is None else json.loads(r) for r in theirs
    ]


def test_decode_reverses_encode():
    ds = bt.from_pydict({"s": [{"a": 1, "b": ["x", None]}, None]})
    t = ds.collect().schema.field("s").type
    back = ds.select(s=col("s").json.encode().json.decode(t))
    assert back.to_pydict() == ds.to_pydict()


@pytest.mark.parametrize(
    ("doc", "patch"),
    [
        ('{"a":1,"b":{"c":1,"d":2}}', '{"b":{"c":null,"e":3},"a":[1]}'),
        ('{"a":1}', "5"),
        ("[1,2]", '{"a":1}'),
        (None, '{"a":1}'),
        ('{"a":1}', None),
        ('{"a":{"x":1}}', '{"a":{"x":{"y":null}}}'),
        ('{"b":1,"a":2}', '{"c":3,"a":4}'),
        ("{}", '{"a":null}'),
    ],
)
def test_merge_patch_matches_json_merge_patch(duck, doc, patch):
    ds = bt.from_pydict({"d": [doc], "p": [patch]}).select(
        col("d").cast("string"), col("p").cast("string")
    )
    duck.register("t", ds.collect())
    out = ds.select(r=col("d").json.merge_patch(col("p"))).collect()
    assert_same(out, duck.sql("SELECT json_merge_patch(d::JSON, p::JSON)::VARCHAR AS r FROM t"))


# --- list.get with a per-row index ---------------------------------------------


def test_list_get_with_an_expression_index_matches_list_extract(duck):
    ds = bt.from_pydict(
        {"a": [[3, 1, 2], [3, 1, 2], [3, 1], None, [1], [], [4, 4]], "i": [0, 2, 5, 0, None, 0, 1]}
    )
    duck.register("t", ds.collect())
    out = ds.select(r=col("a").list.get(col("i"))).collect()
    assert_same(out, duck.sql("SELECT list_extract(a, i + 1) AS r FROM t"))


def test_a_negative_expression_index_counts_from_the_end(duck):
    ds = bt.from_pydict({"a": [[3, 1, 2], [3, 1, 2], [7]], "i": [-1, -3, -2]})
    duck.register("t", ds.collect())
    out = ds.select(r=col("a").list.get(col("i"))).collect()
    assert_same(out, duck.sql("SELECT list_extract(a, i) AS r FROM t"))


# --- list.std / list.var ddof --------------------------------------------------


@pytest.mark.parametrize(
    ("method", "ddof", "duck_fn"),
    [
        ("std", 1, "stddev_samp"),
        ("std", 0, "stddev_pop"),
        ("var", 1, "var_samp"),
        ("var", 0, "var_pop"),
    ],
)
def test_list_moments_match_list_aggregate(duck, method, ddof, duck_fn):
    ds = bt.from_pydict({"a": [[1.0, 2.0, 4.0], [5.0], [], None, [1.0, None, 3.0], [2.0, 2.0]]})
    duck.register("t", ds.collect())
    out = ds.select(r=getattr(col("a").list, method)(ddof=ddof)).collect()
    assert_same(out, duck.sql(f"SELECT list_aggregate(a, '{duck_fn}') AS r FROM t"))


# --- list.jaccard(mode="set") --------------------------------------------------


def test_set_jaccard_matches_the_duckdb_composition(duck):
    ds = bt.from_pydict(
        {
            "a": [["x", "y", "y", None], [], ["q"], ["a", "b"], None],
            "b": [["y", "z"], [], ["q", "q"], ["c"], ["a"]],
        }
    )
    duck.register("t", ds.collect())
    out = ds.select(r=col("a").list.jaccard(col("b"), mode="set")).collect()
    expect = duck.sql(
        """SELECT len(list_intersect(a, b))
                  / nullif(len(list_distinct(list_concat(a, b))), 0) AS r FROM t"""
    )
    assert_same(out, expect)


# --- list.zip ------------------------------------------------------------------

_ZIP = "list_transform(list_zip(a, b{pad}), x -> {{'left': x[1], 'right': x[2]}})"


def test_zip_of_equal_lengths_matches_list_zip(duck):
    ds = bt.from_pydict({"a": [[1, 2], [], [5]], "b": [["x", None], [], ["z"]]})
    duck.register("t", ds.collect())
    out = ds.select(r=col("a").list.zip(col("b"))).collect()
    assert_same(out, duck.sql(f"SELECT {_ZIP.format(pad='')} AS r FROM t"))


def test_padded_zip_matches_list_zip_and_strict_zip_raises(duck):
    ds = bt.from_pydict({"a": [[1, 2, 3], [1]], "b": [["x"], ["p", "q"]]})
    duck.register("t", ds.collect())
    out = ds.select(r=col("a").list.zip(col("b"), pad=True)).collect()
    assert_same(out, duck.sql(f"SELECT {_ZIP.format(pad='')} AS r FROM t"))
    with pytest.raises(Exception, match="same length"):
        ds.select(r=col("a").list.zip(col("b"))).collect()


def test_a_null_list_zips_to_null_where_duckdb_pads_it():
    # Pinned divergence: DuckDB reads a NULL list as empty and pads it.
    ds = bt.from_arrow(pa.table({"a": pa.array([None], pa.list_(pa.int64())), "b": [["x"]]}))
    out = ds.select(r=col("a").list.zip(col("b"), pad=True)).to_pydict()
    assert out == {"r": [None]}


# --- lambda scope: element_index() and outer columns ---------------------------


def test_index_lambda_matches_list_transform_with_two_parameters(duck):
    ds = bt.from_pydict({"a": [[10, 20, 30], [], None, [7, None]]})
    duck.register("t", ds.collect())
    out = ds.select(r=col("a").list.transform(element() * (element_index() + 1))).collect()
    assert_same(out, duck.sql("SELECT list_transform(a, (x, i) -> x * i) AS r FROM t"))


def test_index_filter_matches_list_filter_with_two_parameters(duck):
    ds = bt.from_pydict({"a": [[5, 6, 7, 8], [1], [], None]})
    duck.register("t", ds.collect())
    out = ds.select(r=col("a").list.filter(element_index() % 2 == 0)).collect()
    assert_same(out, duck.sql("SELECT list_filter(a, (x, i) -> i % 2 = 1) AS r FROM t"))


def test_an_outer_column_in_a_filter_matches_duckdb(duck):
    ds = bt.from_pydict({"a": [[1, 5, 9], [4, 6], None, [3, 3], []], "th": [4, 5, 1, 3, 0]})
    duck.register("t", ds.collect())
    out = ds.select(r=col("a").list.filter(element() > col("th"))).collect()
    assert_same(out, duck.sql("SELECT list_filter(a, x -> x > th) AS r FROM t"))


def test_an_outer_column_in_a_transform_matches_duckdb(duck):
    ds = bt.from_pydict({"a": [[1, 2], [3], None], "k": [10, None, 2]})
    duck.register("t", ds.collect())
    out = ds.select(r=col("a").list.transform(element() * col("k") + 1)).collect()
    assert_same(out, duck.sql("SELECT list_transform(a, x -> x * k + 1) AS r FROM t"))


def test_an_outer_column_survives_a_rename_below_the_projection():
    ds = bt.from_pydict({"a": [[1, 5, 9]], "t": [4]}).select("a", th=col("t") * 1)
    out = ds.select(r=col("a").list.filter(element() > col("th"))).to_pydict()
    assert out == {"r": [[5, 9]]}


def test_an_outer_column_reaches_a_nested_lambda():
    ds = bt.from_pydict({"a": [[[1, 2], [3]]], "k": [10]})
    body = element().list.transform(element() + col("k"))
    assert ds.select(r=col("a").list.transform(body)).to_pydict() == {"r": [[[11, 12], [13]]]}


def test_sql_lambda_over_an_outer_column_matches_duckdb(duck):
    ds = bt.from_pydict({"a": [[1, 5, 9], [4, 6]], "th": [4, 5]})
    duck.register("t", ds.collect())
    query = "SELECT list_filter(a, x -> x > th) AS r FROM t"
    assert_same(bt.sql(query, t=ds).collect(), duck.sql(query))


# --- struct.with_fields / rename_fields / drop_fields --------------------------


def _structs(duck):
    ds = bt.from_pydict({"s": [{"x": 1, "y": "a"}, {"x": None, "y": "b"}, None], "k": [5, 6, 7]})
    duck.register("t", ds.collect())
    return ds


# DuckDB's `struct_update(NULL, x := v)` builds a non-null struct `{x: v, y: NULL}`; Batcher
# keeps a null struct null, as Spark's `withField` and Polars' `with_fields` do. That is a
# deliberate, documented divergence, so the null row is compared separately below.


def test_with_fields_matches_struct_update_and_struct_insert(duck):
    ds = _structs(duck)
    s = col("s")
    edited = s.struct.with_fields(x=s.struct.field("x") * 10, z=col("k"))
    out = ds.filter(s.is_not_null()).select(r=edited).collect()
    expect = duck.sql(
        """SELECT struct_insert(struct_update(s, x := s.x * 10), z := k) AS r
           FROM t WHERE s IS NOT NULL"""
    )
    assert_same(out, expect)
    assert ds.filter(s.is_null()).select(r=edited).to_pydict() == {"r": [None]}
    duck_null = duck.sql("SELECT struct_update(s, x := 1) FROM t WHERE s IS NULL").fetchall()
    assert duck_null == [({"x": 1, "y": None},)]


def test_a_type_changing_overwrite_matches_struct_update(duck):
    ds = _structs(duck)
    out = ds.filter(col("s").is_not_null()).select(r=col("s").struct.with_fields(x="text"))
    expect = duck.sql("SELECT struct_update(s, x := 'text') AS r FROM t WHERE s IS NOT NULL")
    assert_same(out.collect(), expect)


def test_rename_and_drop_match_a_struct_pack_rebuild(duck):
    ds = _structs(duck)
    out = ds.select(
        r=col("s").struct.rename_fields({"x": "id"}), d=col("s").struct.drop_fields("x")
    ).collect()
    expect = duck.sql(
        """SELECT CASE WHEN s IS NULL THEN NULL ELSE struct_pack(id := s.x, y := s.y) END AS r,
                  CASE WHEN s IS NULL THEN NULL ELSE struct_pack(y := s.y) END AS d FROM t"""
    )
    assert_same(out, expect)


# --- set operations keep a null element, where DuckDB drops it -----------------


def test_intersect_keeps_null_where_list_intersect_drops_it(duck):
    ds = bt.from_pydict({"a": [[1, 1, None, 2]], "b": [[1, None, 3]]})
    duck.register("t", ds.collect())
    ours = ds.select(r=col("a").list.intersect(col("b"))).to_pydict()["r"]
    theirs = duck.sql("SELECT list_intersect(a, b) FROM t").fetchall()[0][0]
    assert ours == [[1, None]]
    assert theirs == [1]
    # Dropping the nulls first gives DuckDB's answer, as the docstring says.
    fixed = ds.select(r=col("a").list.drop_nulls().list.intersect(col("b").list.drop_nulls()))
    assert fixed.to_pydict()["r"] == [theirs]
