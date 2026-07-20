"""SQL `SEMI` / `ANTI` joins vs DuckDB — regression for a silent wrong answer.

sqlglot puts SEMI/ANTI in a join's ``kind``, not its ``side`` (a bare ``SEMI JOIN`` is
``side='' kind='SEMI'``; ``LEFT SEMI JOIN`` is ``side='LEFT' kind='SEMI'``). The
translator read only ``side``, so the kind was dropped and **every SEMI/ANTI join
silently executed as an ordinary INNER join** — ``ANTI JOIN`` returned the rows that
*matched*, carrying the right side's columns, which is the precise opposite of its
meaning. No error, just the wrong rows.

`RIGHT SEMI`/`RIGHT ANTI` were the same bug and are now run as the equivalent
left-driven join over swapped operands, which is exactly their definition.

These tests fail against the old behavior on every case below.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same


@pytest.fixture
def tu(duck):
    # `a` matches (twice on the left), `b` does not; `u` has an unmatched `c` so an
    # inner-join mistranslation is visibly different from a semi/anti result.
    t = pa.table({"k": ["a", "a", "b"], "v": [1, 2, 3]})
    u = pa.table({"k": ["a", "c"], "z": [9, 8]})
    duck.register("t", t)
    duck.register("u", u)
    return t, u


@pytest.mark.differential
@pytest.mark.parametrize("kind", ["SEMI", "ANTI"])
def test_semi_anti_join_matches_duckdb(duck, tu, kind):
    """A semi/anti join keeps left columns only and never duplicates a left row."""
    t, u = tu
    query = f"SELECT * FROM t {kind} JOIN u ON t.k = u.k"
    assert_same(bt.sql(query, t=t, u=u).collect(), duck.sql(query))


@pytest.mark.differential
@pytest.mark.parametrize("kind", ["SEMI", "ANTI"])
def test_semi_anti_join_is_not_an_inner_join(duck, tu, kind):
    """Pin the specific regression: the result must differ from the INNER join.

    The old bug made these identical. Comparing against DuckDB alone would catch it, but
    stating the inequality makes the failure mode obvious when this breaks again.
    """
    t, u = tu
    semi = bt.sql(f"SELECT * FROM t {kind} JOIN u ON t.k = u.k", t=t, u=u).collect()
    inner = bt.sql("SELECT * FROM t INNER JOIN u ON t.k = u.k", t=t, u=u).collect()
    assert semi.column_names != inner.column_names or semi.num_rows != inner.num_rows


@pytest.mark.differential
def test_anti_join_returns_the_unmatched_rows(tu):
    """`ANTI` returns rows with NO match — the exact inverse of what the bug returned."""
    t, u = tu
    out = bt.sql("SELECT * FROM t ANTI JOIN u ON t.k = u.k", t=t, u=u).collect().to_pydict()
    assert out == {"k": ["b"], "v": [3]}


@pytest.mark.parametrize("kind", ["SEMI", "ANTI"])
def test_left_semi_anti_spelling(tu, kind):
    """`LEFT SEMI/ANTI` (Spark spelling) means the same as the bare form.

    DuckDB's parser rejects this spelling, so it is asserted directly rather than
    differentially.
    """
    t, u = tu
    explicit = bt.sql(f"SELECT * FROM t LEFT {kind} JOIN u ON t.k = u.k", t=t, u=u).collect()
    bare = bt.sql(f"SELECT * FROM t {kind} JOIN u ON t.k = u.k", t=t, u=u).collect()
    assert explicit.to_pydict() == bare.to_pydict()


@pytest.mark.differential
@pytest.mark.parametrize("kind", ["SEMI", "ANTI"])
def test_right_semi_anti_equals_the_swapped_left_form(duck, tu, kind):
    """`A RIGHT SEMI JOIN B` is `B SEMI JOIN A` — the operand swap IS the semantics.

    DuckDB's parser rejects the `RIGHT SEMI` spelling, so the oracle is the swapped query,
    which DuckDB runs happily and which must produce the identical relation.
    """
    t, u = tu
    out = bt.sql(f"SELECT * FROM t RIGHT {kind} JOIN u ON t.k = u.k", t=t, u=u).collect()
    assert_same(out, duck.sql(f"SELECT * FROM u {kind} JOIN t ON u.k = t.k"))


@pytest.mark.parametrize(
    ("kind", "expected"),
    [("SEMI", {"k": ["a"], "z": [9]}), ("ANTI", {"k": ["c"], "z": [8]})],
)
def test_right_semi_anti_returns_right_side_rows(tu, kind, expected):
    """A right-semi/anti returns the RIGHT relation's rows and columns, not the left's."""
    t, u = tu
    out = bt.sql(f"SELECT * FROM t RIGHT {kind} JOIN u ON t.k = u.k", t=t, u=u).collect()
    assert out.to_pydict() == expected


def test_right_semi_join_with_differently_named_keys():
    """The swap must mirror the ON equality too, or the keys bind to the wrong side.

    `_split_join_on` reads the equality's operand *position* to decide which relation a
    key belongs to, so `ON t.a = u.b` left unmirrored would look up `a` in the swapped-in
    left relation (which only has `b`).
    """
    left = pa.table({"a": ["x", "y"], "v": [1, 2]})
    right = pa.table({"b": ["x", "z"], "w": [7, 8]})
    out = bt.sql(
        "SELECT * FROM left_t RIGHT SEMI JOIN right_t ON left_t.a = right_t.b",
        left_t=left,
        right_t=right,
    ).collect()
    assert out.to_pydict() == {"b": ["x"], "w": [7]}
