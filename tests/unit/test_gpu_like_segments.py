"""`LIKE` with more than one wildcard, translated exactly rather than declined or approximated.

`LIKE '%special%requests%'` is TPC-H q13 and q16, and the translator declined it: the pattern
has two literal segments between wildcards, and the only shapes handled were the ones a single
`.str` call expresses. Both queries paid a full round trip to a device to discover that — 0.9 s
each on a T4, twice over for q17-shaped plans that try the fan-out and then a single worker —
before the CPU engine answered.

The reduction has one trap and it is why the obvious version is wrong: the segments must match
**in order and without overlapping**. `contains('a') & contains('b')` accepts `"ba"`, and
`LIKE '%a%b%'` does not.

It is also deliberately not a regular expression. `%a%b%` reads as `.*a.*b.*` in the engine's
Rust, in the host backend's Python and in cuDF, and the three do not agree: `.` excludes a
newline in all of them, while the engine's `LIKE` spans one. A value carrying a newline would
match on the engine and not on the device — a wrong answer, on a device, invisible to a
pandas replay.

These cases are checked against SQL's own semantics, computed independently from the pattern.
"""

from __future__ import annotations

import re

import pandas as pd
import pyarrow as pa
import pytest

from batcher.core.gpu_plan import DfBackend
from batcher.core.gpu_plan.vocab.strings import _like

pytestmark = pytest.mark.unit

#: Values chosen to separate ordered matching from unordered: `"ba"` and `"byax"` match the
#: segments of `%a%b%` in the wrong order, and `"ab\ncd"` is the newline a regex reduction
#: would silently fail.
VALUES = [
    "",
    "a",
    "ab",
    "ba",
    "aXbYc",
    "abc",
    "abcabc",
    "special requests",
    "special X requests",
    "requests special",
    "aa",
    "xayb",
    "byax",
    "ab\ncd",
    None,
]

PATTERNS = [
    "%a%b%",
    "a%b%c",
    "%special%requests%",
    "a%b",
    "%a%b",
    "a%b%",
    "ab%c%",
    "%a%a%",
    "%x%y%",
    "%ab%cd%",
    "a%",
    "%c",
    "%b%",
    "%%",
    "abc",
]


def _sql_like(value: str | None, pattern: str) -> bool | None:
    """SQL `LIKE` for a `%`-only pattern, derived from the pattern rather than from the code.

    `re.S` is the point: SQL's `%` spans a newline, and a regex `.` does not unless told to.
    That difference is exactly what the translated form must not inherit.
    """
    if value is None:
        return None
    expr = "^" + ".*".join(re.escape(part) for part in pattern.split("%")) + "$"
    return re.match(expr, value, flags=re.S) is not None


@pytest.fixture
def column():
    return pd.Series(VALUES, dtype=pd.ArrowDtype(pa.string()))


@pytest.mark.parametrize("pattern", PATTERNS)
def test_like_matches_sql_semantics(pattern, column):
    got = [None if pd.isna(v) else bool(v) for v in _like(column, pattern)]
    assert got == [_sql_like(v, pattern) for v in VALUES], pattern


def test_order_matters(column):
    """The reduction the naive version gets wrong: `"ba"` contains both segments, out of order."""
    got = dict(zip(VALUES, _like(column, "%a%b%"), strict=True))
    assert bool(got["ab"]) is True
    assert bool(got["ba"]) is False


def test_segments_may_not_overlap(column):
    """`'a%b'` on `"ab"` matches; on a value where the two would have to share a character it
    does not, which is what walking the remainder after each segment enforces."""
    one = pd.Series(["ab", "a", "b"], dtype=pd.ArrowDtype(pa.string()))
    assert [bool(v) for v in _like(one, "a%b")] == [True, False, False]


def test_a_wildcard_spans_a_newline(column):
    """The case a regex reduction fails silently: `.` excludes `\\n` in every dialect here."""
    got = dict(zip(VALUES, _like(column, "%ab%cd%"), strict=True))
    assert bool(got["ab\ncd"]) is True


def test_null_stays_null(column):
    """`LIKE` on an unknown is unknown, never False."""
    assert pd.isna(list(_like(column, "%a%b%"))[-1])


def test_three_literals_between_wildcards_decline(column):
    """The honest bound: locating a third segment needs `find` to start from a per-row index,
    which neither library expresses. Declining costs a CPU fallback; guessing costs a row."""
    from batcher.core.gpu_plan.backend import Unsupported

    with pytest.raises(Unsupported):
        _like(column, "%a%b%c%")


def test_anchors_may_not_overlap(column):
    """`LIKE 'ab%ab'` needs four characters. Without the length test, `"ab"` matches: it starts
    and ends with `ab` because it *is* `ab`, and both anchors read the same two characters."""
    short = pd.Series(["ab", "abab", "abXab"], dtype=pd.ArrowDtype(pa.string()))
    assert [bool(v) for v in _like(short, "ab%ab")] == [False, True, True]


@pytest.mark.parametrize("pattern", PATTERNS)
def test_an_empty_column_does_not_raise(pattern):
    """A shard whose filter left no rows is ordinary, not an edge case.

    `str.find` on a zero-length Arrow-backed column raises `ArrowInvalid` rather than returning
    nothing, and left to propagate that fails the shard: the fan-out then recovers it on the
    CPU engine, so the query pays a device round trip plus a recomputation to filter no rows.
    """
    empty = pd.Series([], dtype=pd.ArrowDtype(pa.string()))
    assert len(_like(empty, pattern)) == 0


def test_a_single_character_wildcard_is_still_declined(column):
    """`_` needs a regex, which is the one thing this reduction refuses to reach for."""
    from batcher.core.gpu_plan.backend import Unsupported

    with pytest.raises(Unsupported):
        _like(column, "%a_b%")


def test_it_runs_through_the_backend_the_device_uses(column):
    """Not only as a bare function: the path a translated filter actually takes."""
    be = DfBackend(pd)
    frame = pd.DataFrame({"s": column})
    ir = {
        "e": "str",
        "fn": "like",
        "input": {"e": "col", "name": "s"},
        "pattern": "%special%requests%",
    }
    from batcher.core.gpu_plan.exprs import eval_expr

    got = [None if pd.isna(v) else bool(v) for v in be.column(eval_expr(ir, frame, be), frame)]
    assert got == [_sql_like(v, "%special%requests%") for v in VALUES]
