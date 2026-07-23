"""SQL join **key columns** vs DuckDB — regression for a silent wrong answer.

`Dataset.join` *coalesces* a key pair: the output carries one column, under the left
key's name, holding whichever side matched. That is correct for the DataFrame API and
for SQL's ``USING`` / ``NATURAL`` forms, which do specify a single merged key.

The SQL translator reused that one coalesced column to resolve **both** ``L.k`` and
``R.k`` under an ``ON`` form, where SQL merges nothing: the two keys stay independent
columns, each NULL-extended on its own side. So
``SELECT L.k, R.k FROM L RIGHT JOIN R ON L.k = R.k`` reported ``L.k`` as the *right*
side's key where DuckDB reports NULL — and a FULL join collapsed both keys into one
coalesced column. LEFT was wrong the same way (``R.k`` echoed the left's key). No error,
just wrong values in the key columns.

Why 4,600 differential tests missed it: `assert_same` is an order-independent multiset
over the whole row, so it *can* see this — but only if a test selects **both** sides'
key columns at once. Every existing join test selected one key, or the payload columns,
where the coalesced value is indistinguishable from the correct one. Each test below
therefore projects both keys and lets the comparison see the NULL-extended positions;
several also assert the NULL placement directly rather than trusting the multiset.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

# `1` is left-only, `2`/`3` match, `5` is right-only — so every join type has an
# unmatched row on each side to null-extend, which is exactly what the bug got wrong.
_L = pa.table({"k": [1, 2, 3], "lv": [10, 20, 30]})
_R = pa.table({"k": [2, 3, 5], "rv": [200, 300, 500]})

_OUTER = ["LEFT", "RIGHT", "FULL"]
_ALL = ["INNER", *_OUTER]


@pytest.fixture
def lr(duck):
    duck.register("l", _L)
    duck.register("r", _R)
    return _L, _R


def _both(duck, q, **tables):
    """Run `q` on both engines and assert the full rows — key columns included — match."""
    assert_same(bt.sql(q, **tables).collect(), duck.sql(q))


@pytest.mark.differential
@pytest.mark.parametrize("how", _ALL)
def test_on_form_keeps_each_side_key(duck, lr, how):
    """`ON` does not merge: each side's key keeps its own value, NULL where unmatched."""
    _both(
        duck,
        f"SELECT l.k AS lk, r.k AS rk, lv, rv FROM l {how} JOIN r ON l.k = r.k",
        l=lr[0],
        r=lr[1],
    )


@pytest.mark.differential
def test_right_join_nulls_the_left_key_of_an_unmatched_right_row(duck, lr):
    """The exact reported shape, asserting the NULL *position*, not just the multiset.

    Before the fix this returned ``lk=[2, 3, 5]`` — the right side's key echoed into the
    left's column. A multiset check over both keys catches it; pinning the row makes the
    failure legible.
    """
    q = "SELECT l.k AS lk, r.k AS rk FROM l RIGHT JOIN r ON l.k = r.k ORDER BY rk"
    out = bt.sql(q, l=lr[0], r=lr[1]).collect().to_pydict()
    assert out == {"lk": [2, 3, None], "rk": [2, 3, 5]}
    assert out == duck.sql(q).to_arrow_table().to_pydict()


@pytest.mark.differential
def test_full_join_nulls_each_key_on_its_own_side(duck, lr):
    """FULL is the worst case: before the fix both keys collapsed to one coalesced column."""
    q = "SELECT l.k AS lk, r.k AS rk FROM l FULL JOIN r ON l.k = r.k ORDER BY lk, rk"
    out = bt.sql(q, l=lr[0], r=lr[1]).collect().to_pydict()
    assert out == {"lk": [1, 2, 3, None], "rk": [None, 2, 3, 5]}
    assert out == duck.sql(q).to_arrow_table().to_pydict()


@pytest.mark.differential
def test_left_join_nulls_the_right_key_of_an_unmatched_left_row(duck, lr):
    """LEFT was wrong too — `r.k` echoed the left's key instead of NULL (a control that failed)."""
    q = "SELECT l.k AS lk, r.k AS rk FROM l LEFT JOIN r ON l.k = r.k ORDER BY lk"
    out = bt.sql(q, l=lr[0], r=lr[1]).collect().to_pydict()
    assert out == {"lk": [1, 2, 3], "rk": [None, 2, 3]}
    assert out == duck.sql(q).to_arrow_table().to_pydict()


@pytest.mark.differential
def test_inner_join_keys_agree_on_both_sides(duck, lr):
    """The control: an inner join drops every unmatched row, so the keys are equal."""
    q = "SELECT l.k AS lk, r.k AS rk FROM l INNER JOIN r ON l.k = r.k ORDER BY lk"
    out = bt.sql(q, l=lr[0], r=lr[1]).collect().to_pydict()
    assert out == {"lk": [2, 3], "rk": [2, 3]}
    assert out == duck.sql(q).to_arrow_table().to_pydict()


@pytest.mark.differential
@pytest.mark.parametrize("how", _ALL)
def test_on_form_with_differently_named_keys(duck, how):
    """`ON l.lk = r.rk` — the right key is a distinct name, and must still be projectable.

    Before the fix the join dropped `rk` outright, so this raised
    ``projection 'rk' references unknown column(s)`` rather than answering.
    """
    left = pa.table({"lk": [1, 2, 3], "lv": [10, 20, 30]})
    right = pa.table({"rk": [2, 3, 5], "rv": [200, 300, 500]})
    duck.register("ln", left)
    duck.register("rn", right)
    _both(
        duck,
        f"SELECT ln.lk, rn.rk, lv, rv FROM ln {how} JOIN rn ON ln.lk = rn.rk",
        ln=left,
        rn=right,
    )


# --- USING / NATURAL: SQL *does* specify one merged key here ------------------


@pytest.mark.differential
@pytest.mark.parametrize("how", _ALL)
def test_using_exposes_one_merged_key(duck, lr, how):
    """`USING (k)` yields a single coalesced `k` — legitimately unlike the `ON` form."""
    _both(duck, f"SELECT k, lv, rv FROM l {how} JOIN r USING (k)", l=lr[0], r=lr[1])


@pytest.mark.differential
@pytest.mark.parametrize("how", _OUTER)
def test_using_still_resolves_a_qualified_key_per_side(duck, lr, how):
    """Even under `USING`, a *qualified* `l.k` / `r.k` is that side's own value.

    DuckDB exposes the merged `k` to an unqualified reference while still answering a
    qualified one per-side, so the merged column cannot be reused for both.
    """
    _both(
        duck,
        f"SELECT l.k AS lk, r.k AS rk FROM l {how} JOIN r USING (k)",
        l=lr[0],
        r=lr[1],
    )


@pytest.mark.differential
@pytest.mark.parametrize("how", _ALL)
def test_natural_join_merges_the_shared_key(duck, lr, how):
    """NATURAL JOIN is USING over every shared column — one merged key, like USING."""
    _both(duck, f"SELECT * FROM l NATURAL {how} JOIN r", l=lr[0], r=lr[1])


# --- edge cases the key columns have to survive -------------------------------


@pytest.mark.differential
@pytest.mark.parametrize("how", _ALL)
def test_multi_column_keys_null_extend_together(duck, how):
    """Every key column of an unmatched row null-extends, not just the first."""
    left = pa.table({"a": [1, 1, 2], "b": [1, 2, 1], "lv": [10, 20, 30]})
    right = pa.table({"a": [1, 2, 3], "b": [2, 1, 1], "rv": [200, 300, 400]})
    duck.register("lm", left)
    duck.register("rm", right)
    _both(
        duck,
        f"SELECT lm.a AS la, lm.b AS lb, rm.a AS ra, rm.b AS rb, lv, rv "
        f"FROM lm {how} JOIN rm ON lm.a = rm.a AND lm.b = rm.b",
        lm=left,
        rm=right,
    )


@pytest.mark.differential
@pytest.mark.parametrize("how", _ALL)
def test_nulls_in_the_key_data_never_match(duck, how):
    """A NULL *in the data* is distinct from NULL-extension: it matches nothing, both sides.

    Without this the two NULL sources are conflated — a key column full of NULLs looks
    correct whether it came from a real value or from a coalesced echo.
    """
    left = pa.table({"k": [1, None, 3], "lv": [10, 20, 30]})
    right = pa.table({"k": [None, 3, 5], "rv": [100, 300, 500]})
    duck.register("lz", left)
    duck.register("rz", right)
    _both(
        duck,
        f"SELECT lz.k AS lk, rz.k AS rk, lv, rv FROM lz {how} JOIN rz ON lz.k = rz.k",
        lz=left,
        rz=right,
    )


@pytest.mark.differential
@pytest.mark.parametrize("how", _ALL)
def test_self_join_keeps_the_two_aliases_apart(duck, how):
    """A self-join is the sharpest case: both sides are the same table, so a coalesced
    key makes the two aliases indistinguishable."""
    t = pa.table({"k": [1, 2, 3], "nxt": [2, 3, 9], "v": [10, 20, 30]})
    duck.register("s", t)
    _both(
        duck,
        f"SELECT a.k AS ak, b.k AS bk, a.v AS av, b.v AS bv FROM s a {how} JOIN s b ON a.nxt = b.k",
        s=t,
    )


@pytest.mark.differential
@pytest.mark.parametrize("how", _ALL)
@pytest.mark.parametrize("empty", ["left", "right"])
def test_empty_input_on_either_side(duck, how, empty):
    """An empty side must still null-extend the other — not vanish, and not echo a key."""
    keys = [] if empty == "left" else [1, 2]
    left = pa.table(
        {"k": pa.array(keys, pa.int64()), "lv": pa.array([x * 10 for x in keys], pa.int64())}
    )
    rkeys = [] if empty == "right" else [2, 3]
    right = pa.table(
        {"k": pa.array(rkeys, pa.int64()), "rv": pa.array([x * 100 for x in rkeys], pa.int64())}
    )
    duck.register("le", left)
    duck.register("re", right)
    _both(
        duck,
        f"SELECT le.k AS lk, re.k AS rk, lv, rv FROM le {how} JOIN re ON le.k = re.k",
        le=left,
        re=right,
    )


@pytest.mark.differential
@pytest.mark.parametrize("how", _OUTER)
def test_duplicate_keys_on_both_sides(duck, how):
    """Many-to-many keys fan out; each output row still carries its own side's key."""
    left = pa.table({"k": [1, 1, 2, 4], "lv": [10, 11, 20, 40]})
    right = pa.table({"k": [1, 1, 3], "rv": [100, 101, 300]})
    duck.register("ld", left)
    duck.register("rd", right)
    _both(
        duck,
        f"SELECT ld.k AS lk, rd.k AS rk, lv, rv FROM ld {how} JOIN rd ON ld.k = rd.k",
        ld=left,
        rd=right,
    )


@pytest.mark.differential
def test_dataframe_join_still_coalesces_its_keys():
    """The DataFrame API's coalescing is deliberate and unchanged by the SQL-side fix.

    `bt.sql` and `Dataset.join` answer different contracts here: SQL's `ON` form keeps
    both keys, the DataFrame join merges them. This pins the second one so a future
    change to the first cannot quietly take it along.
    """
    out = bt.from_arrow(_L).join(bt.from_arrow(_R), on="k", how="full").collect().to_pydict()
    assert out["k"] == [1, 2, 3, 5]
    assert out["lv"] == [10, 20, 30, None]
    assert out["rv"] == [None, 200, 300, 500]

    named = (
        bt.from_arrow(pa.table({"lk": [1, 2, 3], "lv": [10, 20, 30]}))
        .join(
            bt.from_arrow(pa.table({"rk": [2, 3, 5], "rv": [200, 300, 500]})),
            left_on="lk",
            right_on="rk",
            how="right",
        )
        .collect()
        .to_pydict()
    )
    # The right key is merged into the left's name and dropped — the coalescing contract.
    assert named == {"lk": [2, 3, 5], "lv": [20, 30, None], "rv": [200, 300, 500]}
