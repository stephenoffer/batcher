"""The shape-inspecting `.json` accessors, against DuckDB's JSON functions.

`array_length`, `keys`, `type_of`, and `exists` answer questions about a document's
*structure* rather than pulling one leaf out of it: how many elements, which keys, what
type, is it there at all. DuckDB has a direct equivalent for each, so each is compared
against it here rather than against a hand-written expectation.

`values` has no single DuckDB counterpart returning a list of rendered leaves, so it is
checked against `json_extract_string` element by element — the invariant that matters is
that element *i* through `values` and element *i* through `[i]` never disagree.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from _harness import assert_same
from batcher import col

# The documents both engines agree on: a populated object, an empty object, an empty
# array, a scalar where a container is asked for, a nested object, and a SQL null.
DOCS = [
    '{"z": 1, "a": [10, 20, 30], "s": "hi", "n": null}',
    "{}",
    '{"a": []}',
    '{"a": 7}',
    '{"a": {"b": 1}}',
    None,
]

# Malformed text is Batcher's documented lenient divergence: Batcher answers null, DuckDB
# raises. It is kept out of the oracle comparisons and asserted on its own below, the same
# split `test_diff_accessor_json_coerce.py` already makes.
MALFORMED = "not json at all"


def test_array_length_matches_duckdb(duck):
    t = pa.table({"j": DOCS})
    duck.register("j", t)
    out = bt.from_arrow(t).select(n=col("j").json.array_length("$.a")).collect()
    # DuckDB's json_array_length returns 0 for a non-array; Batcher returns NULL, which
    # is the honest answer (there is no array, so it has no length). Normalize DuckDB's
    # 0 to NULL for the values where the path is not an array so the oracle comparison
    # tests the counting, not that disagreement.
    assert_same(
        out,
        duck.sql(
            "SELECT CASE WHEN json_type(j, '$.a') = 'ARRAY' "
            "THEN json_array_length(j, '$.a') END n FROM j"
        ),
    )


def test_object_keys_match_duckdb_in_source_order(duck):
    t = pa.table({"j": DOCS})
    duck.register("j", t)
    out = bt.from_arrow(t).select(k=col("j").json.keys()).collect()
    assert_same(
        out,
        duck.sql("SELECT CASE WHEN json_type(j) = 'OBJECT' THEN json_keys(j) END k FROM j"),
    )


def test_type_of_matches_duckdb(duck):
    t = pa.table({"j": DOCS})
    duck.register("j", t)
    out = bt.from_arrow(t).select(ty=col("j").json.type_of("$.a")).collect()
    # DuckDB names the types in uppercase (`ARRAY`, `VARCHAR`); Batcher uses the JSON
    # spec's lowercase names, and calls a string leaf `string` rather than `VARCHAR`.
    assert_same(
        out,
        duck.sql(
            "SELECT CASE json_type(j, '$.a') "
            "WHEN 'ARRAY' THEN 'array' WHEN 'OBJECT' THEN 'object' "
            "WHEN 'VARCHAR' THEN 'string' WHEN 'BIGINT' THEN 'number' "
            "WHEN 'DOUBLE' THEN 'number' WHEN 'UBIGINT' THEN 'number' "
            "WHEN 'BOOLEAN' THEN 'boolean' WHEN 'NULL' THEN 'null' END ty FROM j"
        ),
    )


def test_exists_separates_absent_from_json_null(duck):
    t = pa.table({"j": DOCS})
    duck.register("j", t)
    out = bt.from_arrow(t).select(e=col("j").json.exists("$.n")).collect()
    # `$.n` is JSON null in the first document and absent in the rest: the case
    # `extract_string` cannot express, since both extract to SQL NULL.
    assert_same(out, duck.sql("SELECT json_exists(j, '$.n') e FROM j"))
    extracted = bt.from_arrow(t).select(x=col("j").json.extract_string("$.n")).to_pydict()
    assert extracted["x"][0] is None, "a JSON null still extracts to SQL null"


def test_values_agree_with_indexed_extraction():
    doc = '{"a": ["x", 1, {"b": 2}, null, true, [1, 2]]}'
    t = pa.table({"j": [doc]})
    ds = bt.from_arrow(t)
    whole = ds.select(v=col("j").json.values("$.a")).to_pydict()["v"][0]
    assert len(whole) == 6
    for i in range(6):
        one = ds.select(x=col("j").json.extract_string(f"$.a[{i}]")).to_pydict()["x"][0]
        assert whole[i] == one, f"element {i}: {whole[i]!r} via values, {one!r} via [{i}]"


def test_values_null_list_where_the_path_is_not_an_array():
    t = pa.table({"j": DOCS})
    got = bt.from_arrow(t).select(v=col("j").json.values("$.a")).to_pydict()["v"]
    # Populated array, empty object (no `$.a`), empty array, then a number, an object,
    # and a SQL null — everything that is not an array yields a null list, which keeps
    # "there is no array here" distinct from "the array is empty".
    assert got == [["10", "20", "30"], None, [], None, None, None]


def test_malformed_json_is_null_rather_than_an_error():
    # The documented divergence: DuckDB raises on every one of these, Batcher answers
    # null, so one bad row in a scan cannot abort the query.
    ds = bt.from_arrow(pa.table({"j": [MALFORMED]}))
    assert ds.select(x=col("j").json.array_length("$.a")).to_pydict()["x"] == [None]
    assert ds.select(x=col("j").json.keys()).to_pydict()["x"] == [None]
    assert ds.select(x=col("j").json.values("$.a")).to_pydict()["x"] == [None]
    assert ds.select(x=col("j").json.type_of("$.a")).to_pydict()["x"] == [None]
    assert ds.select(x=col("j").json.exists("$.a")).to_pydict()["x"] == [False]


def test_values_feeds_explode():
    # The point of `values`: it turns a JSON array column into a list column, so the
    # relational operators that consume lists apply to it.
    t = pa.table({"id": [1, 2], "j": ['{"xs": ["a", "b"]}', '{"xs": ["c"]}']})
    out = (
        bt.from_arrow(t)
        .with_columns(xs=col("j").json.values("$.xs"))
        .explode("xs")
        .select("id", "xs")
        .to_pydict()
    )
    assert out == {"id": [1, 1, 2], "xs": ["a", "b", "c"]}
