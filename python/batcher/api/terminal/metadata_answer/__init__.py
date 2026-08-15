"""Metadata-first terminal resolution — the façade over the answer modules.

`_core` holds the whole-relation / filtered-count / scalar-column shortcuts; `aggregate`
holds the keyless-aggregate shortcut (split out for size). `pushed_count` is the one
member that is not free — it asks the source for a ``COUNT(*)`` — and is consulted only
after the free answers decline. This package re-exports them behind the stable
`batcher.api.terminal.metadata_answer` import path. Layer: control-plane `api/terminal` —
decides whether an answer exists short of running the plan, and never touches a row.
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
from batcher.api.terminal.metadata_answer.pushed_count import pushed_count

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
    "pushed_count",
]
