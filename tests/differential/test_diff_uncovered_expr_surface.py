"""Expression functions that no differential test named, held against DuckDB.

`.claude/rules/testing.md` makes a differential test the hard gate for "new / changed
relational operator or expression". That gate applies at the moment a function is written;
it says nothing about functions written before it, and it cannot notice one that was never
covered. A sweep of the public surface against `tests/differential/` found **474 public
names that appear nowhere in that directory**.

Most of those cannot have a DuckDB oracle -- image, audio, ML and text-quality features are
Batcher capabilities rather than SQL semantics, and a differential test of `mime_type` would
be comparing against nothing. This file covers the part that *can*: functions with a real SQL
equivalent under a different spelling, which is why a name-matching sweep missed them.
`cumsum` is `SUM(...) OVER`, `is_between` is `BETWEEN`, `sec` is `1/cos`, `is_alpha` is a
POSIX class match. None of them was under the oracle before.

All of them agree today. This adds no fix; it puts a surface that happened to be right under
the check the rules require, so a rewrite of one of these kernels cannot quietly change an
answer.

Two results from building it are worth keeping, because both look like defects and are not:

**`floordiv` disagrees with DuckDB's `//` on negative operands, and both are correct.**
Batcher's `Expr.floordiv` rounds toward negative infinity -- `-7 // 2 == -4`, Python and
Polars semantics, which its docstring states. DuckDB's `//` truncates toward zero, `-3`,
which is what SQL does. The pair is only a problem if Batcher's *SQL* front-end floors, and
it does not: ``bt.sql("SELECT a // b")`` returns DuckDB's answer exactly. Each surface
follows its own convention, so this file pins both rather than picking one.

**`is_lower` is true for a string with no cased characters**, where Python's `str.islower()`
is false -- `"  "`, `""` and `"123"` differ. That is deliberate and documented on the method:
it is defined as "equals its lowercase form", explicitly "unlike pandas ``str.islower``".
Pinned here so the intent survives someone reading only the name.
"""

from __future__ import annotations

import pytest

import batcher as bt
from _harness import assert_same_ordered

pytestmark = pytest.mark.differential

#: Signed, zero, null, and both sign combinations, so a truncate-vs-floor split shows up.
_NUMBERS = {
    "a": [7, -7, 7, -7, 0, 5, None, 3],
    "b": [2, 2, -2, -2, 3, 1, 2, 4],
}
_NUMBERS_SQL = "(VALUES (7,2),(-7,2),(7,-2),(-7,-2),(0,3),(5,1),(NULL,2),(3,4)) v(a,b)"

#: Mixed case, digits, whitespace-only, empty, null, and punctuation.
_STRINGS = {"s": ["Hello World", "abc123", "  ", "ABC", "", "x", None, "a-b_c"]}
_STRINGS_SQL = "(VALUES ('Hello World'),('abc123'),('  '),('ABC'),(''),('x'),(NULL),('a-b_c')) v(s)"


@pytest.fixture
def numbers(duck):
    duck.execute(f"CREATE TABLE n AS SELECT * FROM {_NUMBERS_SQL}")
    return bt.from_pydict(_NUMBERS)


@pytest.fixture
def strings(duck):
    duck.execute(f"CREATE TABLE s AS SELECT * FROM {_STRINGS_SQL}")
    return bt.from_pydict(_STRINGS)


#: `Expr` methods with a SQL equivalent, as (name, batcher expression, SQL, table).
_NUMERIC_CASES = [
    ("cumsum", lambda c: c("a").cumsum(), "SUM(a) OVER (ORDER BY rowid)"),
    ("cummax", lambda c: c("a").cummax(), "MAX(a) OVER (ORDER BY rowid)"),
    ("cummin", lambda c: c("a").cummin(), "MIN(a) OVER (ORDER BY rowid)"),
    ("cumcount", lambda c: c("a").cumcount(), "COUNT(a) OVER (ORDER BY rowid)"),
    ("is_between", lambda c: c("a").is_between(-7, 5), "a BETWEEN -7 AND 5"),
    ("isnull", lambda c: c("a").isnull(), "a IS NULL"),
    ("notnull", lambda c: c("a").notnull(), "a IS NOT NULL"),
    ("sec", lambda c: c("b").cast("double").sec(), "1/cos(b::DOUBLE)"),
    ("csc", lambda c: c("b").cast("double").csc(), "1/sin(b::DOUBLE)"),
]

_STRING_CASES = [
    ("is_alpha", lambda c: c("s").str.is_alpha(), "regexp_matches(s,'^[[:alpha:]]+$')"),
    ("is_alnum", lambda c: c("s").str.is_alnum(), "regexp_matches(s,'^[[:alnum:]]+$')"),
    ("is_numeric", lambda c: c("s").str.is_numeric(), "regexp_matches(s,'^[[:digit:]]+$')"),
    (
        "is_space",
        lambda c: c("s").str.is_space(),
        "length(s)>0 AND regexp_matches(s,'^[[:space:]]+$')",
    ),
    ("has_digits", lambda c: c("s").str.has_digits(), "regexp_matches(s,'[[:digit:]]')"),
    (
        "digit_count",
        lambda c: c("s").str.digit_count(),
        "length(regexp_replace(s,'[^[:digit:]]','','g'))",
    ),
    ("space_count", lambda c: c("s").str.space_count(), "length(s)-length(replace(s,' ',''))"),
    (
        "removesuffix",
        lambda c: c("s").str.removesuffix("c"),
        "CASE WHEN s LIKE '%c' THEN left(s, length(s)-1) ELSE s END",
    ),
    ("line_count", lambda c: c("s").str.line_count(), "length(s)-length(replace(s,chr(10),''))+1"),
]


@pytest.mark.parametrize(
    ("name", "build", "sql"), _NUMERIC_CASES, ids=[c[0] for c in _NUMERIC_CASES]
)
def test_a_numeric_expression_matches_duckdb(name, build, sql, numbers, duck):
    """Ordered, not `assert_same`. These are per-row expressions, so a result that is right
    as a multiset and wrong per row is exactly the defect worth catching, and the
    order-independent helper cannot see it."""
    got = numbers.select(r=build(bt.col)).to_arrow()
    assert_same_ordered(got, duck.sql(f"SELECT {sql} AS r FROM n"))


@pytest.mark.parametrize(("name", "build", "sql"), _STRING_CASES, ids=[c[0] for c in _STRING_CASES])
def test_a_string_expression_matches_duckdb(name, build, sql, strings, duck):
    got = strings.select(r=build(bt.col)).to_arrow()
    assert_same_ordered(got, duck.sql(f"SELECT {sql} AS r FROM s"))


class TestTheTwoDeliberateDivergences:
    """Both look like bugs, both are intended, and both are documented on the method."""

    def test_expr_floordiv_floors_where_sql_truncates(self, numbers, duck):
        """`Expr.floordiv` is Python/Polars: toward negative infinity."""
        got = numbers.select(r=bt.col("a").floordiv(bt.col("b"))).to_pydict()["r"]
        assert got == [3, -4, -4, 3, 0, 5, None, 0]
        sql = [row[0] for row in duck.execute("SELECT a // b FROM n").fetchall()]
        assert sql == [3, -3, -3, 3, 0, 5, None, 0]
        assert got != sql, (
            "the two conventions have converged; if that is intended, this test and "
            "`floordiv`'s docstring both need updating"
        )

    def test_the_sql_front_end_truncates_like_duckdb(self, numbers, duck):
        """The half that would be a real defect. A user writing SQL must get SQL semantics,
        whatever the `Expr` spelling does."""
        got = bt.sql("SELECT a // b AS r FROM d", d=numbers).to_arrow()
        assert_same_ordered(got, duck.sql("SELECT a // b AS r FROM n"))

    def test_is_lower_is_true_for_a_string_with_no_cased_characters(self, strings):
        """Defined as "equals its lowercase form", explicitly unlike pandas `str.islower`."""
        got = strings.select(r=bt.col("s").str.is_lower()).to_pydict()["r"]
        by_value = dict(zip(_STRINGS["s"], got, strict=True))
        assert by_value["  "] is True
        assert by_value[""] is True
        assert by_value["ABC"] is False
        assert by_value["abc123"] is True
        # The control: Python disagrees on exactly the uncased cases and agrees elsewhere,
        # so this documents a real difference rather than restating the implementation.
        assert "  ".islower() is False
        assert "abc123".islower() is True
