"""`LogicalPlan` — the declarative plan the public API builds.

Immutable node tree. Each fluent `Dataset` operation returns a new `LogicalPlan`
wrapping the previous one. Validation (column references resolve against the
input's available columns) happens here at build time so mistakes fail fast,
before the optimizer or engine ever runs. Logical plans lower to the relational
IR JSON via `to_ir()`; types of derived columns are resolved by the engine.

This package is split by node family — `base`, `relational`, `reshape`, `aggregate`,
`window`, `join`, and `transforms` — and re-exports the flat public surface here.
"""

from __future__ import annotations

from batcher.plan.logical.aggregate import (
    Aggregate,
    AggregateSpec,
    Sort,
)
from batcher.plan.logical.base import LogicalPlan, SortKeySpec
from batcher.plan.logical.join import (
    AsofJoin,
    Join,
    JoinOutputCol,
    RangeCondition,
    RangeJoin,
    WatermarkStreamJoin,
    align_join_key_types,
    asof_tolerance,
)
from batcher.plan.logical.relational import (
    Distinct,
    Filter,
    Limit,
    MapBatches,
    Project,
    Projection,
    Sample,
    Scan,
    StreamingSessionWindow,
    TransformWithState,
    Union,
    WatermarkDedup,
)
from batcher.plan.logical.reshape import RowId, Unnest, Unpivot
from batcher.plan.logical.transforms import (
    constant_column_literal,
    empty_result_schema,
    hoist_computed_keys,
    is_cartesian_key_pair,
    is_partition_independent,
    is_streamable,
    passthrough_renames,
    project_columns,
    remap_sources,
)
from batcher.plan.logical.window import Window, WindowFrame, WindowFuncSpec

__all__ = [
    "Aggregate",
    "AggregateSpec",
    "AsofJoin",
    "Distinct",
    "Filter",
    "Join",
    "JoinOutputCol",
    "Limit",
    "LogicalPlan",
    "MapBatches",
    "Project",
    "Projection",
    "RangeCondition",
    "RangeJoin",
    "RowId",
    "Sample",
    "Scan",
    "Sort",
    "SortKeySpec",
    "StreamingSessionWindow",
    "TransformWithState",
    "Union",
    "Unnest",
    "Unpivot",
    "WatermarkDedup",
    "WatermarkStreamJoin",
    "Window",
    "WindowFrame",
    "WindowFuncSpec",
    "align_join_key_types",
    "asof_tolerance",
    "constant_column_literal",
    "empty_result_schema",
    "hoist_computed_keys",
    "is_cartesian_key_pair",
    "is_partition_independent",
    "is_streamable",
    "passthrough_renames",
    "project_columns",
    "remap_sources",
]
