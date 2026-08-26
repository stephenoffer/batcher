"""JSON string extraction (`.json.extract_string`) against DuckDB, and where it diverges.

This file used to be marked "structural" and assert only hand-written expected values,
which put it in `tests/differential/` without a differential in it. `.claude/rules/testing.md`
is explicit that any relational or expression behaviour must match DuckDB on the same input,
and DuckDB has `json_extract_string`, so there was no reason for the oracle to be absent.

Checking it produced a real finding, which is why the file is now in two halves.

**On well-formed input the two agree exactly**, including the coercion that looks most like
a place they would differ: `$.age` over a JSON *number* yields the text `'30'` in both, not
an integer and not null.

**On malformed input they do not agree, and the difference is not a bug on either side.**
Batcher yields null for a row it cannot parse; DuckDB raises and fails the whole query:

    Invalid Input Error: Malformed JSON at byte 0 of input: invalid literal. Input: "not json"

Both are defensible — a column of scraped JSON usually wants the null, a strict pipeline
usually wants the error — and the contract says a legitimate divergence is "a decision to
surface explicitly, not to hide". So it is asserted from both sides below rather than worked
around by dropping the row: Batcher's null is pinned, and DuckDB's *raise* is pinned too, so
the day either engine changes its mind this file says so.

Measured against duckdb 1.5.5.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col

pytestmark = pytest.mark.differential

duckdb = pytest.importorskip("duckdb")

#: Rows both engines can parse. `None` is a null *input*, which is distinct from a row whose
#: JSON is present but broken — the two travel different paths and only one of them diverges.
_WELL_FORMED = pa.array(
    [
        '{"name": "alice", "age": 30, "addr": {"city": "NYC"}}',
        '{"name": "bob", "addr": {"city": "LA"}}',
        '{"name": "carol"}',
        None,
    ]
)

#: The same rows plus one that is not JSON at all.
_WITH_MALFORMED = pa.array([*_WELL_FORMED.to_pylist()[:3], "not json", None])

_SQL = """
    SELECT json_extract_string(j, '$.name')      AS name,
           json_extract_string(j, '$.age')       AS age,
           json_extract_string(j, '$.addr.city') AS city
    FROM t
"""


def _extract(table: pa.Table):
    return bt.from_arrow(table).select(
        name=col("j").json.extract_string("$.name"),
        age=col("j").json.extract_string("$.age"),
        city=col("j").json.extract_string("$.addr.city"),
    )


def test_extract_string_matches_duckdb_on_well_formed_json(duck):
    """A present key, a missing key, a nested path, and a null input row."""
    table = pa.table({"j": _WELL_FORMED})
    duck.register("t", table)
    assert_same(_extract(table).collect(), duck.sql(_SQL))


def test_a_json_number_extracts_as_text_in_both_engines(duck):
    """`extract_string` of a JSON *number* is the string `'30'`, not 30 and not null.

    Pinned separately because it is the coercion most likely to drift: an implementation
    that returned null for a non-string value, or that returned an integer column, would
    still satisfy a test that only looked at `$.name`.
    """
    table = pa.table({"j": _WELL_FORMED})
    duck.register("t", table)
    got = _extract(table).collect().to_pydict()
    expected = duck.sql(_SQL).to_arrow_table().to_pydict()
    assert got["age"] == expected["age"] == ["30", None, None, None]


def test_malformed_json_is_null_here_and_an_error_in_duckdb(duck):
    """The one divergence, asserted from both sides so neither can drift unnoticed.

    Batcher treats an unparseable row as missing data and yields null. DuckDB treats it as
    a malformed input and aborts the query. Dropping the row to make the comparison pass
    would hide exactly the behaviour a caller needs to know about.
    """
    table = pa.table({"j": _WITH_MALFORMED})
    duck.register("t", table)

    got = _extract(table).collect().to_pydict()
    assert got["name"] == ["alice", "bob", "carol", None, None], (
        "Batcher yields null for the unparseable row rather than failing the batch"
    )
    assert got["city"] == ["NYC", "LA", None, None, None]

    # ...and DuckDB refuses the same input outright. `pytest.raises` is the assertion: if a
    # future DuckDB starts returning null here the two engines have converged, and this test
    # failing is how anyone would find out.
    with pytest.raises(duckdb.Error, match=r"[Mm]alformed JSON"):
        duck.sql(_SQL).to_arrow_table()
