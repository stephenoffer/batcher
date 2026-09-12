"""The GPU join fan-out replicates the smaller relation, whichever side the plan put it on.

The fan-out splits the probe side and copies the build side to every device, and it used to take
the build side to be whatever the plan put on the **right**. Nothing guarantees that: Kyber
reorders a join's inputs for its own costing, and the CPU hash join it reorders for chooses its
build side at runtime, so left/right carries no promise about size.

The consequence was measured, not imagined. TPC-H q14 (`lineitem`, 60 M rows, joined to `part`,
2 M) runs at 12.2x against an empty statistics hub. Against a hub holding two of its own earlier
passes, Kyber emits the mirrored join, the fan-out tries to replicate 15.36 GB of `lineitem` to
every device, the measured-fit check refuses it, and the query silently leaves the accelerator —
so the learning loop degraded the device tier the longer the fleet ran.

Two properties are pinned here and they are different claims. **Selection**: the smaller side is
the one replicated, whichever side holds it, and only for the join type where exchanging sides
is sound. **Equivalence**: mirroring the IR produces the identical output columns, in the
identical order, so this is a scheduling change rather than a semantic one.
"""

from __future__ import annotations

import pytest

from batcher.dist.gpu.join import _mirrored_join_ir, _replicate_the_smaller_side

pytestmark = pytest.mark.unit


class _Source:
    """A relation that reports a row count and a fixed row width, as `source_bytes` reads it."""

    def __init__(self, rows: int) -> None:
        self._rows = rows

    def row_count(self) -> int:
        return self._rows

    def schema(self):
        import pyarrow as pa

        return pa.schema([pa.field("k", pa.int64()), pa.field("v", pa.int64())])


def _join_ir(join_type: str = "inner") -> dict:
    return {
        "op": "hash_join",
        "join_type": join_type,
        "left_keys": ["l_partkey"],
        "right_keys": ["p_partkey"],
        "output": [
            {"side": "left", "name": "l_extendedprice", "alias": "price"},
            {"side": "right", "name": "p_type", "alias": "kind"},
            {"side": "left", "name": "l_discount", "alias": "disc"},
        ],
    }


# --- selection ---------------------------------------------------------------


def test_a_big_right_side_is_exchanged_for_the_small_left_one():
    """The q14 shape: the plan put the 60 M-row relation on the right."""
    assert _replicate_the_smaller_side(_Source(2_000_000), _Source(60_003_620), _join_ir())


def test_a_plan_that_already_has_the_small_side_on_the_right_is_left_alone():
    """The positive control. Without it, an assertion that mirroring happens would pass
    against an implementation that mirrored unconditionally — which would be the same defect
    facing the other way."""
    assert not _replicate_the_smaller_side(_Source(60_003_620), _Source(2_000_000), _join_ir())


@pytest.mark.parametrize("join_type", ["left", "semi", "anti"])
def test_an_asymmetric_join_is_never_exchanged(join_type):
    """`left`, `semi` and `anti` are driven by left rows, so exchanging the sides would change
    the answer rather than the schedule — however lopsided the two relations are."""
    assert not _replicate_the_smaller_side(_Source(1), _Source(10**9), _join_ir(join_type))


def test_a_relation_that_will_not_size_itself_keeps_the_plan_order():
    """An unmeasurable side leaves the behaviour this path had before it measured anything."""

    class _Mute(_Source):
        def row_count(self):
            raise RuntimeError("no footer")

    assert not _replicate_the_smaller_side(_Mute(1), _Source(10**9), _join_ir())
    assert not _replicate_the_smaller_side(_Source(10**9), _Mute(1), _join_ir())


# --- equivalence -------------------------------------------------------------


def test_mirroring_swaps_the_keys():
    out = _mirrored_join_ir(_join_ir())
    assert out["left_keys"] == ["p_partkey"]
    assert out["right_keys"] == ["l_partkey"]


def test_mirroring_preserves_every_output_column_and_its_order():
    """Aliases and their order are what the chain above the join and the schema contract read."""
    original = _join_ir()
    out = _mirrored_join_ir(original)

    assert [o["alias"] for o in out["output"]] == [o["alias"] for o in original["output"]]
    assert [o["name"] for o in out["output"]] == [o["name"] for o in original["output"]]
    assert [o["side"] for o in out["output"]] == ["right", "left", "right"]


def test_mirroring_is_its_own_inverse():
    """Applied twice it is the identity, which is the cheapest statement of "this is a
    relabelling, not a rewrite"."""
    original = _join_ir()
    assert _mirrored_join_ir(_mirrored_join_ir(original)) == original


def test_mirroring_carries_every_other_field_through_untouched():
    """A field this helper does not know about must survive it — the IR is a wire contract and
    gains members without this module being edited."""
    original = {**_join_ir(), "strategy": "broadcast", "null_equal": False}
    out = _mirrored_join_ir(original)
    assert out["strategy"] == "broadcast"
    assert out["null_equal"] is False
    assert out["op"] == "hash_join"
    assert out["join_type"] == "inner"
