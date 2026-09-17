"""The shared machinery behind the batch-UDF verbs on `Dataset`.

`map_batches`, `map`, `flat_map` and the callable form of `filter` lower to one `MapBatches`
stage. `build` derives that stage, `ray_options` resolves Ray Data's resource parameters onto
what the scheduler honours, and `checks` validates the rest at the API edge.
"""

from __future__ import annotations

from batcher.api.dataset._udf.build import (
    bind_fn,
    build_filter,
    build_map_batches,
    build_rows,
)
from batcher.api.dataset._udf.ray_options import Placement, resolve_placement

__all__ = [
    "Placement",
    "bind_fn",
    "build_filter",
    "build_map_batches",
    "build_rows",
    "resolve_placement",
]
