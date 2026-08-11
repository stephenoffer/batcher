"""``EXTRACT`` fields the `.dt` namespace already computed but the SQL table never listed.

`EXTRACT(isodow FROM ...)` raised *"EXTRACT field 'isodow' is not supported"* while
`col(x).dt.isodow()` sat right there — the field table simply had not been extended when
the accessors were. Seven fields were in that state: `isodow`, `isoyear`, `weekofyear`,
`dayofmonth`, `century`, `millennium` and `decade`.

Name equality is not evidence of semantic equality, which is why each is checked against
DuckDB rather than assumed: `isodow` numbers Monday as 1 where `dow` numbers Sunday as 0,
and `isoyear` is the year the ISO *week* falls in, which differs from `year` around New
Year. The dates below include 2019-12-30 — ISO week 1 of **2020** — so an `isoyear` wired
to plain `year` fails here rather than passing by luck.
"""

from __future__ import annotations

import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

_FIELDS = [
    "isodow",
    "isoyear",
    "weekofyear",
    "dayofmonth",
    "century",
    "millennium",
    "decade",
    # The fields that already worked, so the additions cannot have disturbed them.
    "year",
    "month",
    "day",
    "dow",
    "doy",
    "week",
    "quarter",
]

_INSTANTS = [
    "DATE '2021-03-15'",  # a Monday
    "DATE '2019-12-30'",  # ISO week 1 of 2020 — year and isoyear disagree
    "DATE '2000-01-01'",  # a century/millennium boundary
    "TIMESTAMP '2021-01-01 06:07:08'",
]


@pytest.mark.parametrize("field", _FIELDS)
@pytest.mark.parametrize("instant", _INSTANTS)
def test_extract_field_matches_duckdb(duck, field, instant):
    query = f"SELECT EXTRACT({field} FROM {instant}) AS r"
    assert_same(bt.sql(query).collect(), duck.sql(query))


def test_isoyear_is_not_the_calendar_year():
    """The case that separates a correct wiring from `isoyear -> year`."""
    got = bt.sql("SELECT EXTRACT(isoyear FROM DATE '2019-12-30') AS r").collect().to_pydict()
    assert got == {"r": [2020]}


def test_isodow_numbers_monday_as_one():
    """`dow` numbers Sunday as 0; `isodow` must not be wired to it."""
    got = bt.sql("SELECT EXTRACT(isodow FROM DATE '2021-03-14') AS r").collect().to_pydict()
    assert got == {"r": [7]}  # a Sunday
