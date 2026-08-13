"""Every expression node with sub-expressions must be known to the optimizer's walker.

`plan/expr_rewrite/traverse.py` dispatches on exact node type through two tables:
`_EXPR_KIDS` says what a node's sub-expressions are, and `_EXPR_REBUILD` says how to put
one back together. A node type absent from `_EXPR_KIDS` is **treated as a leaf**,
silently and by design — that is what makes the dispatch O(1) for `Col` and `Lit`, which
are most of every tree.

The cost of that design is a failure mode no other gate can see. A new node with
sub-expressions that nobody adds to the table does not raise, does not fail to
deserialize, and computes the right answer on its own. What it does is make the optimizer
blind to the columns inside it, and the consequences are silent and wrong:

* `merge_projections` counts column occurrences to decide whether collapsing two
  projections would duplicate work. An invisible node counts zero, so the guard passes
  when it should not.
* The same rule then substitutes the inner projection's expressions into the outer one.
  An invisible node is not substituted, so the merged projection still references
  columns that no longer exist.

That pair is not hypothetical: it is exactly what happened when `SpatialFunc` was added,
and it surfaced as a `ColumnNotFoundError` from deep inside the optimizer on a
three-projection stack — a shape no unit test of the new functions would ever build.

This test is the guard. It enumerates every `IRNode` subclass that declares a `child`
or `children` field and asserts both tables know it.
"""

from __future__ import annotations

import dataclasses

import pytest

import batcher  # noqa: F401  (imports the whole node surface so subclasses are defined)
from batcher.plan.expr_ir import node_base
from batcher.plan.expr_ir.node_base import IRNode
from batcher.plan.expr_rewrite.traverse import _EXPR_KIDS, _EXPR_REBUILD

pytestmark = pytest.mark.unit

#: Node types deliberately absent from the tables, each with the reason. `Case` and
#: `MakeStruct` are present under their own rebuild helpers; nothing else is exempt.
KNOWN_ABSENT: dict[str, str] = {}


def _nodes_with_subexpressions() -> list[type]:
    """Every concrete `IRNode` subclass declaring a `child`/`children` field.

    Reads the field metadata directly rather than through `node_base.child_fields`,
    which caches on the class and needs an instance to do it. A contract test wants the
    declaration, not a constructed example.
    """
    seen: list[type] = []

    def walk(cls: type) -> None:
        for sub in cls.__subclasses__():
            seen.append(sub)
            walk(sub)

    walk(IRNode)
    out = []
    for cls in dict.fromkeys(seen):
        if not dataclasses.is_dataclass(cls):
            continue
        has_child = any(
            (spec := f.metadata.get(node_base._META)) is not None
            and spec.kind in (node_base._Kind.CHILD, node_base._Kind.CHILDREN)
            for f in dataclasses.fields(cls)
        )
        if has_child:
            out.append(cls)
    return sorted(out, key=lambda c: c.__name__)


def test_the_enumeration_finds_the_nodes_it_is_meant_to_guard():
    """A guard that enumerates nothing passes for the wrong reason.

    Without this, a change to the field metadata that broke `_nodes_with_subexpressions`
    would turn the two assertions below into vacuous truths, and the contract would go
    unenforced while staying green.
    """
    names = {cls.__name__ for cls in _nodes_with_subexpressions()}
    assert len(names) > 30, f"only found {len(names)}: the enumeration is broken"
    # A spread across the families, so a regression in any one of them is visible.
    for expected in ("Binary", "GeoFunc", "SpatialFunc", "StrFunc", "Coalesce"):
        assert expected in names, expected


def test_every_node_with_subexpressions_declares_its_children():
    missing = [
        cls.__name__
        for cls in _nodes_with_subexpressions()
        if cls not in _EXPR_KIDS and cls.__name__ not in KNOWN_ABSENT
    ]
    assert not missing, (
        f"{missing} declare sub-expressions but are absent from `_EXPR_KIDS`, so the "
        "optimizer treats them as leaves and cannot see the columns inside them"
    )


def test_every_node_with_subexpressions_can_be_rebuilt():
    missing = [
        cls.__name__
        for cls in _nodes_with_subexpressions()
        if cls not in _EXPR_REBUILD and cls.__name__ not in KNOWN_ABSENT
    ]
    assert not missing, (
        f"{missing} declare sub-expressions but are absent from `_EXPR_REBUILD`, so a "
        "rewrite that changes one of their children cannot put the node back together"
    )


def test_the_two_tables_cover_the_same_node_types():
    """A type in one table and not the other is a crash waiting for the right rewrite.

    `transform_expr_up` looks the node up in `_EXPR_KIDS` to find its children and then
    in `_EXPR_REBUILD` only when a child actually changed. A node in the first table but
    not the second therefore works perfectly until some rule rewrites something inside
    it, and then raises `KeyError`.
    """
    only_kids = sorted(c.__name__ for c in _EXPR_KIDS.keys() - _EXPR_REBUILD.keys())
    only_rebuild = sorted(c.__name__ for c in _EXPR_REBUILD.keys() - _EXPR_KIDS.keys())
    assert not only_kids, f"{only_kids} can be walked into but not rebuilt"
    assert not only_rebuild, f"{only_rebuild} can be rebuilt but are never walked into"


# --- The two behaviours the tables above exist to protect ---------------------------
#
# Each of these fails with a `ColumnNotFoundError` raised from inside `merge_projections`
# when its node is missing from `_EXPR_KIDS`, which is how both bugs were found.


def test_a_per_row_image_crop_survives_projection_merging():
    """`select(box).select(crop(box))` — the shape a detection pipeline is written in.

    `ImageCrop` carries its window as four sub-expressions precisely so a detector's
    predicted box can vary per row. With the node invisible to the walker, collapsing
    the two projections left the crop referencing columns the merged projection no
    longer produced.
    """
    import batcher as bt
    from batcher import col

    frames = bt.from_pydict(
        {"img": [b"not-a-real-image"], "bx": [1], "by": [2], "bw": [3], "bh": [4]}
    )
    boxes = frames.select(
        "img",
        x=col("bx") + 0,
        y=col("by") + 0,
        w=col("bw") + 0,
        h=col("bh") + 0,
    )
    cropped = boxes.select(cropped=col("img").image.crop(col("x"), col("y"), col("w"), col("h")))
    # Planning is the whole test: the crop itself would need a real image to decode.
    assert cropped.explain()
    assert cropped.columns == ["cropped"]


def test_a_rigid_transform_survives_repeated_projection_merging():
    """Three stacked projections whose output names alternate.

    Two are not enough — the first merge succeeds either way. The third is where an
    invisible node's uncounted column references let the guard in `merge_projections`
    pass when it should have refused.
    """
    import batcher as bt

    base = bt.from_pydict({"roll": [0.4], "pitch": [0.3], "yaw": [0.9]})
    to_quat = base.select(**bt.quat_from_euler("roll", "pitch", "yaw"))
    to_euler = to_quat.select(**bt.quat_to_euler(("qx", "qy", "qz", "qw")))
    again = to_euler.select(**bt.quat_from_euler("roll", "pitch", "yaw"))

    first = to_quat.to_pydict()
    last = again.to_pydict()
    for name in ("qx", "qy", "qz", "qw"):
        assert last[name][0] == pytest.approx(first[name][0], abs=1e-12), name


# --- A rebuild must carry the node's own parameters across ---------------------------


def _representative_nodes() -> dict[type, object]:
    """One constructed instance per node type, from the IR snapshot's corpus.

    The snapshot already builds a representative of every node — including, now, the
    parameterized form of each multimodal node — so the corpus is shared rather than
    restated here.
    """
    from tests.unit.test_ir_snapshot import _representatives

    by_type: dict[type, object] = {}
    for node in _representatives().values():
        # `setdefault` keeps the *first* representative of a type; the media nodes list
        # their bare `decode` form first, so ask for the richest one instead.
        current = by_type.get(type(node))
        if current is None or _parameter_count(node) > _parameter_count(current):
            by_type[type(node)] = node
    return by_type


def _parameter_count(node: object) -> int:
    """How many of `node`'s non-sub-expression fields are actually set."""
    return sum(
        1
        for name in node_base.scalar_fields_of(type(node))
        if getattr(node, name, None) not in (None, False)
    )


def test_rebuilding_a_node_preserves_every_parameter_it_carries():
    """`_EXPR_REBUILD` must return the *same node with new children*, not a partial copy.

    This is the guard the three existing tables tests did not provide. Each of them asks
    whether a node type is *present* in the tables; none asked whether the rebuilt node
    still carries its own fields. So `_EXPR_REBUILD` could — and did — hold entries like
    ``ImageFunc(e.fn, k[0], width=e.width, height=e.height)``, silently dropping
    `mean`/`std`/`channels_first`/`format`/`fill`.

    The consequences were graded from loud to invisible. `.image.encode("jpeg")` lost
    its format and the engine refused the batch outright. `.image.to_tensor_f32(...)`
    lost its per-channel normalization and `.audio.mel_spectrogram(...)` its filterbank
    sizes: right column type, right shape, quietly wrong numbers feeding a model. An
    ordinary ``select(...).select(...)`` was enough to trigger either, because that is
    what makes `merge_projections` rebuild the node.
    """
    dropped: dict[str, list[str]] = {}
    for node_type, rep in _representative_nodes().items():
        kids_of = _EXPR_KIDS.get(node_type)
        if kids_of is None:
            continue
        kids = kids_of(rep)
        if not kids:
            continue
        # Rebuild with the node's *own* children: nothing but the parameters can differ.
        rebuilt = _EXPR_REBUILD[node_type](rep, kids)
        lost = [
            name
            for name in node_base.scalar_fields_of(node_type)
            if getattr(rebuilt, name, None) != getattr(rep, name, None)
        ]
        if lost:
            dropped[node_type.__name__] = lost

    assert not dropped, (
        "rebuilding these nodes lost their own parameters, so any optimizer rewrite "
        f"silently reverts them to defaults: {dropped}"
    )


def test_every_declared_child_is_reachable_from_the_kids_table():
    """A `child()` field the tables do not descend into is invisible to column pruning.

    `VideoFunc.second` is the case that motivated this: `.video.frame_at(col("ts"), ...)`
    names the moment it wants a still from, so `second` is a sub-expression over the
    enclosing relation exactly as `input` is — but `_EXPR_KIDS` yielded only `input`, so
    the optimizer could not see that the expression reads `ts`.

    The two element-scoped bodies are the deliberate exceptions: they close over
    ``element()`` rather than over the relation, so `expr_ir.walk` stops at them too.
    """
    scoped_out = {("ListTransform", "func"), ("ListFilter", "pred")}

    unreachable: list[str] = []
    for node_type, rep in _representative_nodes().items():
        kids_of = _EXPR_KIDS.get(node_type)
        if kids_of is None:
            continue
        reached = {id(k) for k in kids_of(rep)}
        for name, is_list in node_base.child_fields_of(node_type):
            if (node_type.__name__, name) in scoped_out:
                continue
            value = getattr(rep, name, None)
            if value is None:
                continue  # an optional sub-expression this representative does not set
            declared = list(value) if is_list else [value]
            if any(id(v) not in reached for v in declared):
                unreachable.append(f"{node_type.__name__}.{name}")

    assert not unreachable, (
        "these declared sub-expressions are not yielded by `_EXPR_KIDS`, so the "
        f"optimizer cannot see the columns they read: {sorted(unreachable)}"
    )
