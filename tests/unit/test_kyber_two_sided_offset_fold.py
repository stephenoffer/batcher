"""`fold_add_across_two_offsets` — Spark's `ReorderAssociativeOperator` shape.

`fold_add_sub_constants` already collapses a chain nested down **one** side
(`(x + c1) + c2`). It cannot see `(x + c1) + (y + c2)`: neither operand is a bare
literal, so there is nothing for it to match, and the two constants stay one addition
apart no matter how many fixpoint passes run.

The correctness argument is the ring one the whole `arith_algebra` family rests on — the
integers modulo 2**64 are commutative and associative, and the engine's `add`/`sub` wrap
rather than trapping — so the fold is exact *including across overflow*. The tests below
assert that at the `INT64_MAX` boundary rather than only on small values, because small
values cannot tell an exact reassociation from a lucky one.
"""

from __future__ import annotations

import json

import batcher as bt
from batcher import col
from batcher.kyber.optimizer import Optimizer

INT64_MAX = 2**63 - 1
INT64_MIN = -(2**63)

DS = bt.from_pydict(
    {
        "a": [1, 2, None, INT64_MAX],
        "b": [3, 4, 5, 1],
        "f": [1.0, 2.0, 3.0, 4.0],
        "g": [5.0, 6.0, 7.0, 8.0],
    }
)


def _ir(ds) -> str:
    return json.dumps(Optimizer().optimize(ds._plan).ir)


def _adds(plan: str) -> int:
    return plan.count('"op": "add"')


def test_the_two_constants_are_gathered_into_one():
    """`(a + 1) + (b + 2)` becomes `(a + b) + 3`: two adds, not three."""
    plan = _ir(DS.select(r=(col("a") + 1) + (col("b") + 2)))
    assert _adds(plan) == 2
    assert '{"int": 3}' in plan


def test_a_subtraction_on_one_side_still_folds():
    """`(a - 1) + (b + 5)` is `(a + b) + 4` — the offsets are signed."""
    plan = _ir(DS.select(r=(col("a") - 1) + (col("b") + 5)))
    assert '{"int": 4}' in plan


def test_floats_are_left_alone():
    """Float addition is not associative, so the guard must refuse it.

    `(x + 1e16) + (y + 1)` and `(x + y) + 1e16 + 1` genuinely differ, which is why this
    rule is gated on every leaf being provably signed-integer-typed.
    """
    plan = _ir(DS.select(r=(col("f") + 1) + (col("g") + 2)))
    assert _adds(plan) == 3, "the float form must keep all three additions"


def test_a_bare_column_on_one_side_is_not_this_rule():
    """`(a + 1) + b` has one constant; the existing single-side rule owns that shape."""
    plan = _ir(DS.select(r=(col("a") + 1) + col("b")))
    assert '{"int": 1}' in plan


def test_the_fold_is_exact_across_signed_overflow():
    """The claim the ring argument makes, checked where it could fail.

    At `INT64_MAX` the un-reassociated `(a + 1)` overflows first; reassociated, `(a + b)`
    does. Two's-complement addition wraps identically either way, so the answers agree —
    and both equal the wrapped reference computed here in Python.
    """
    got = DS.select(r=(col("a") + 1) + (col("b") + 2)).to_pydict()["r"]

    def wrap(v: int) -> int:
        return ((v + 2**63) % 2**64) - 2**63

    expected = [wrap(1 + 3 + 3), wrap(2 + 4 + 3), None, wrap(INT64_MAX + 1 + 3)]
    assert got == expected
    assert got[3] == INT64_MIN + 3, "the boundary row must wrap, not saturate or raise"


def test_nulls_are_preserved():
    """Every operand still appears exactly once, so a null input still nulls the row."""
    assert DS.select(r=(col("a") + 1) + (col("b") + 2)).to_pydict()["r"][2] is None


def test_re_applying_the_rule_reports_no_change():
    """A rule that keeps returning a new-but-equal node never lets the fixpoint end."""
    from batcher.kyber.rules.extra.arith_algebra import fold_add_across_two_offsets

    node = DS.select(r=(col("a") + 1) + (col("b") + 2))._plan
    once = fold_add_across_two_offsets(node, None)
    assert once is not None, "the rule must fire on this shape"
    assert fold_add_across_two_offsets(once, None) is None
