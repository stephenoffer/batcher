"""Key-range-aligned distributed execution: joins over tables laid out in key order.

When the tables a query joins are stored in the order of the join key, each worker can take
one key range of every such table and run the join -- and any aggregate grouped by the key
-- to completion locally, with no shuffle. See `analysis` for the rules that make that
correct, `units` for how the ranges are cut from Parquet footers, and `run` for execution.
"""

from __future__ import annotations

from batcher.dist.executors.aligned.analysis import (
    AlignedCut,
    AlignedPlan,
    KeyClass,
    find_plan,
    key_classes,
)
from batcher.dist.executors.aligned.route import aligned_route, choose_plan, try_aligned
from batcher.dist.executors.aligned.run import run_cut, run_plan, unit_plan
from batcher.dist.executors.aligned.units import Unit, plan_units, source_key_bounds

__all__ = [
    "AlignedCut",
    "AlignedPlan",
    "KeyClass",
    "Unit",
    "aligned_route",
    "choose_plan",
    "find_plan",
    "key_classes",
    "plan_units",
    "run_cut",
    "run_plan",
    "source_key_bounds",
    "try_aligned",
    "unit_plan",
]
