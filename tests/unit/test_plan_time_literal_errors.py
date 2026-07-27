"""Passing a column where a plan-time constant is required fails clearly.

Several accessor methods take a value that is lowered into the JSON IR as a *constant*:
`.str.jaccard(text)`, `.map.get(key)`, `.list.contains(value)`, `.list.position(value)`.
An expression in one of those slots cannot be evaluated — but nothing checked, so the
failure surfaced far from the call and as an internal error:

* `.str.jaccard(col("t"))` reached `json.dumps` and raised
  ``TypeError: Object of type Col is not JSON serializable`` — a serializer error naming
  neither the function nor the argument;
* `.map.get(col("k"))` and the `.list` literal slots raised a bare
  ``TypeError: unsupported literal type: Col`` at plan-build time, equally far from the
  line the user wrote.

Both are internal errors escaping to someone who wrote an ordinary expression, and both
are `TypeError` rather than the project's `PlanError`. Validation now happens where the
node is constructed, so the message arrives at the call site and names the function, the
argument and the constraint.

These are unit tests, not differential ones: there is no DuckDB behaviour to compare
against, only the error contract.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher._internal.errors import PlanError

MAP_COL = pa.array([[("a", 1)]], type=pa.map_(pa.string(), pa.int64()))


@pytest.fixture
def frame():
    return pa.table({"m": MAP_COL, "l": [[1, 2]], "s": ["ab"], "k": ["a"]})


CASES = [
    ("str.jaccard", lambda: col("s").str.jaccard(col("k")), "pattern"),
    ("str.levenshtein", lambda: col("s").str.levenshtein(col("k")), "pattern"),
    ("map.get", lambda: col("m").map.get(col("k")), "key"),
    # `.map.contains` is composed over `keys().list.contains`, so it is that node's
    # literal slot that rejects — which is correct, and worth pinning as such.
    ("map.contains", lambda: col("m").map.contains(col("k")), "value"),
    ("list.contains", lambda: col("l").list.contains(col("k")), "value"),
    ("list.position", lambda: col("l").list.position(col("k")), "value"),
]


@pytest.mark.unit
@pytest.mark.parametrize(("name", "build", "argument"), CASES)
def test_a_column_in_a_constant_slot_raises_plan_error(name, build, argument):
    with pytest.raises(PlanError) as excinfo:
        build()
    message = str(excinfo.value)
    assert argument in message, "the message must name the offending argument"
    assert "plan is built" in message, "the message must say why a column cannot work"


@pytest.mark.unit
@pytest.mark.parametrize(("name", "build", "argument"), CASES)
def test_the_error_is_not_an_internal_one(name, build, argument):
    """The regression: a `TypeError` from `json` or the literal encoder means an internal
    failure reached the user."""
    with pytest.raises(Exception) as excinfo:
        build()
    assert not isinstance(excinfo.value, TypeError)
    assert "JSON serializable" not in str(excinfo.value)
    assert "unsupported literal type" not in str(excinfo.value)


@pytest.mark.unit
def test_the_error_names_the_function_the_user_called():
    """`StrFunc` is an implementation detail nobody typed; `jaccard_similarity` is not."""
    with pytest.raises(PlanError, match=r"jaccard_similarity\(\)"):
        col("s").str.jaccard(col("k"))


@pytest.mark.unit
def test_it_fails_at_the_call_not_at_collect(frame):
    """Constructing the expression is enough to raise — the point of validating at the
    edge is that the traceback points at the line the user wrote."""
    with pytest.raises(PlanError):
        col("s").str.jaccard(col("k"))


@pytest.mark.unit
def test_plain_literals_are_unaffected(frame):
    """The guard must not cost the ordinary spelling anything."""
    out = bt.from_arrow(frame).select(
        j=col("s").str.jaccard("ab"),
        c=col("l").list.contains(1),
        p=col("l").list.position(2),
        g=col("m").map.get("a"),
    )
    rows = out.to_pydict()
    assert rows["c"] == [True]
    assert rows["p"] == [2]
    assert rows["g"] == [1]


@pytest.mark.unit
def test_a_none_valued_optional_slot_is_still_allowed(frame):
    """`pattern`/`key` default to None on the functions that take no argument; the check
    must reject only an `Expr`, not everything that is not a `str`."""
    assert bt.from_arrow(frame).select(r=col("s").str.upper()).to_pydict()["r"] == ["AB"]
    assert bt.from_arrow(frame).select(r=col("m").map.keys()).to_pydict()["r"] == [["a"]]
