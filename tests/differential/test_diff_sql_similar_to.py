"""``SIMILAR TO`` — a whole-string regex match, which the translator did not recognize.

It reached the scalar path as an unhandled node and raised ``unsupported SQL expression:
SimilarTo``.

The semantics are the part worth pinning. DuckDB's ``SIMILAR TO`` is a **full** regex
match, not LIKE with regex syntax added: ``'abc' SIMILAR TO 'a%'`` is False (``%`` is not
a wildcard here) and ``'abc' SIMILAR TO 'ab'`` is False (the match must cover the whole
string). `str.regexp_matches` searches *anywhere*, so the pattern is anchored — and
wrapped in a non-capturing group, because an alternation would otherwise bind past the
anchors and ``'a|z'`` would mean "starts with a, or ends with z" rather than "is a or z".
That case is in the table below.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential


def _t() -> pa.Table:
    return pa.table({"s": pa.array(["abc", "bcd", "a", None, "zbc", "z"])})


@pytest.fixture
def tables(duck):
    duck.register("t", _t())
    return {"t": bt.from_arrow(_t())}


@pytest.mark.parametrize(
    "query",
    [
        "SELECT s, s SIMILAR TO 'a.*' AS r FROM t",
        # Not a prefix match: the pattern must cover the whole string.
        "SELECT s, s SIMILAR TO 'ab' AS r FROM t",
        # `%` is not a wildcard in SIMILAR TO.
        "SELECT s, s SIMILAR TO 'a%' AS r FROM t",
        "SELECT s, s SIMILAR TO '(a|z)bc' AS r FROM t",
        # The bare alternation: without the non-capturing group this answers wrongly.
        "SELECT s, s SIMILAR TO 'a|z' AS r FROM t",
        "SELECT s, s SIMILAR TO '[a-z]+' AS r FROM t",
        "SELECT s, s NOT SIMILAR TO 'a.*' AS r FROM t",
        "SELECT s FROM t WHERE s SIMILAR TO 'a.*'",
        "SELECT s FROM t WHERE s NOT SIMILAR TO '.*bc'",
    ],
)
def test_similar_to_matches_duckdb(tables, duck, query):
    assert_same(bt.sql(query, **tables).collect(), duck.sql(query))


def test_similar_to_is_a_whole_string_match(tables):
    got = bt.sql("SELECT s, s SIMILAR TO 'ab' AS r FROM t", **tables).collect().to_pydict()
    assert dict(zip(got["s"], got["r"], strict=True))["abc"] is False


def test_alternation_is_grouped_before_anchoring(tables):
    """`'a|z'` must mean "is a or is z", not "starts with a or ends with z"."""
    got = bt.sql("SELECT s, s SIMILAR TO 'a|z' AS r FROM t", **tables).collect().to_pydict()
    by_s = dict(zip(got["s"], got["r"], strict=True))
    assert by_s["a"] is True
    assert by_s["z"] is True
    assert by_s["abc"] is False
    assert by_s["zbc"] is False
