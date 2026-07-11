"""Metadata-first terminal resolution — the façade over the answer modules.

`_core` holds the whole-relation / filtered-count / scalar-column shortcuts; `aggregate`
holds the keyless-aggregate shortcut (split out for size). This package re-exports both
behind the stable `batcher.api.terminal.metadata_answer` import path. Layer: control-plane
`api/terminal` — decides whether a metadata answer exists and reads it from stats, never
touching a row.
"""

from __future__ import annotations

from batcher.api.terminal.metadata_answer._core import (
    global_count_plan,
    metadata_all_null,
    metadata_approx_n_unique,
    metadata_count,
    metadata_empty_table,
    metadata_has_nulls,
    metadata_is_empty,
    metadata_learned_quantile,
    metadata_max,
    metadata_min,
    metadata_n_unique,
    metadata_null_count,
)
from batcher.api.terminal.metadata_answer.aggregate import (
    is_global_aggregate,
    metadata_aggregate_table,
)

__all__ = [
    "global_count_plan",
    "is_global_aggregate",
    "metadata_aggregate_table",
    "metadata_all_null",
    "metadata_approx_n_unique",
    "metadata_count",
    "metadata_empty_table",
    "metadata_has_nulls",
    "metadata_is_empty",
    "metadata_learned_quantile",
    "metadata_max",
    "metadata_min",
    "metadata_n_unique",
    "metadata_null_count",
]
