"""`remap_sources` shifts every Scan.source_id and rebuilds the rest unchanged.

Locks in the behavior of the generic (visitor-based) implementation: it must shift
exactly the `Scan` nodes by the offset and leave every other field of every other
node type structurally identical — across single-input, multi-input (Join/Union),
and expression-carrying nodes.
"""

from __future__ import annotations

import batcher as bt
from batcher import col, lit
from batcher.plan.logical import Scan, remap_sources
from batcher.plan.visitor import walk


def _scan_ids(plan):
    return sorted(n.source_id for n in walk(plan) if isinstance(n, Scan))


def _ir_without_source_ids(node):
    """to_ir with every scan source_id zeroed, so two plans that differ only in
    source numbering compare equal."""
    ir = node.to_ir()

    def scrub(obj):
        if isinstance(obj, dict):
            if obj.get("op") == "scan":
                obj = {**obj, "source_id": 0}
            return {k: scrub(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [scrub(v) for v in obj]
        return obj

    return scrub(ir)


def _rich_plan():
    left = bt.from_pydict({"k": [1, 2, 3], "x": [10, 20, 30]})
    right = bt.from_pydict({"k": [1, 2], "y": [100, 200]})
    joined = (
        left.filter(col("x") > lit(5))
        .with_columns(z=col("x") * lit(2))
        .join(right, on="k")
        .group_by("k")
        .agg(zsum=col("z").sum())
        .sort("k")
        .limit(10)
    )
    union = joined.union(joined)
    return union._plan


def test_remap_shifts_all_scans_by_offset():
    plan = _rich_plan()
    before = _scan_ids(plan)
    remapped = remap_sources(plan, 7)
    assert _scan_ids(remapped) == [i + 7 for i in before]


def test_remap_preserves_everything_but_source_ids():
    plan = _rich_plan()
    remapped = remap_sources(plan, 5)
    assert _ir_without_source_ids(remapped) == _ir_without_source_ids(plan)


def test_remap_zero_offset_is_identity():
    plan = _rich_plan()
    assert remap_sources(plan, 0).to_ir() == plan.to_ir()
