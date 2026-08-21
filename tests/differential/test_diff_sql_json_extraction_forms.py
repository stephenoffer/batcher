"""``json_extract`` and ``json_extract_string`` are two functions, not one spelling.

Both read the value at a path, and they differ on exactly two leaf kinds — which is why
lowering both to the unquoting accessor went unnoticed:

* a **string** leaf: ``json_extract`` keeps its quotes (``"x"``), ``json_extract_string``
  does not (``x``);
* a **JSON null**: ``json_extract`` reports the token ``null``, ``json_extract_string``
  reports SQL NULL — so under the old lowering "the key is absent" and "the key is present
  and null" were the same answer.

``json_keys`` had the mirror-image confusion: a value that exists but is not an object has
*no keys* (the empty list), not an unknown answer.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

_DOC = '{"a":"x","b":null,"c":1,"d":[1,2],"e":{"f":1},"g":true,"h":1.50}'


def _table() -> pa.Table:
    return pa.table(
        {
            "id": pa.array([1, 2, 3, 4], pa.int64()),
            "s": pa.array([_DOC, '{"a":null}', "{}", None], pa.string()),
        }
    )


@pytest.mark.parametrize("path", ["$.a", "$.b", "$.c", "$.d", "$.e", "$.g", "$.h", "$.zz"])
@pytest.mark.parametrize(
    "form", ["json_extract(s, '{p}')", "s -> '{p}'", "json_extract_string(s, '{p}')", "s ->> '{p}'"]
)
def test_each_extraction_form_renders_its_own_way(duck, form, path):
    table = _table()
    sql = f"SELECT id, {form.format(p=path)} AS r FROM j"
    duck.register("j", table)
    assert_same(bt.sql(sql, j=table).collect(), duck.sql(sql))


def test_the_two_forms_disagree_on_a_string_and_a_json_null():
    """Stated directly: a test that only checked "not an error" would pass either way."""
    table = _table()
    got = bt.sql(
        "SELECT json_extract(s, '$.a') AS a, json_extract_string(s, '$.a') AS b, "
        "json_extract(s, '$.b') AS c, json_extract_string(s, '$.b') AS d FROM j",
        j=table,
    ).collect()
    assert got.to_pydict()["a"][0] == '"x"'
    assert got.to_pydict()["b"][0] == "x"
    assert got.to_pydict()["c"][0] == "null"
    assert got.to_pydict()["d"][0] is None


def test_a_cast_over_json_extract_still_types_the_value(duck):
    """The `CAST(json_extract(...) AS BIGINT)` idiom must keep working."""
    table = _table()
    sql = "SELECT id, CAST(json_extract(s, '$.c') AS BIGINT) AS r FROM j"
    duck.register("j", table)
    assert_same(bt.sql(sql, j=table).collect(), duck.sql(sql))


def test_json_keys_of_malformed_text_stays_null():
    """The third case, and the reason the empty list is not simply "not an object".

    Malformed text did not parse, so "no keys" would claim a value that is not there.
    DuckDB raises on such a row; answering null rather than aborting the scan is this
    engine's documented divergence, and it must survive the empty-list fix.
    """
    table = pa.table({"s": pa.array(["nope", "{", '{"a":1}', "[]"], pa.string())})
    got = bt.sql("SELECT json_keys(s) AS r FROM j", j=table).collect().to_pydict()
    assert got["r"] == [None, None, ["a"], []]


def test_json_keys_of_a_non_object_is_empty_not_unknown(duck):
    table = pa.table(
        {
            "id": pa.array([1, 2, 3, 4], pa.int64()),
            "s": pa.array(['{"a":1,"b":2}', "[]", "1", None], pa.string()),
        }
    )
    sql = "SELECT id, json_keys(s) AS r FROM j"
    duck.register("j", table)
    assert_same(bt.sql(sql, j=table).collect(), duck.sql(sql))
