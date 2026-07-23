"""The shared Kyber → Carbonite → Core contract loop for relational plans.

This is the single implementation of the conductor's terminal-op orchestration:
optimize the plan (full Kyber, with per-operator `ResourceBounds`), let Carbonite
govern it (admission, out-of-core spill, buffer reservation / scheduling
envelope), execute via Core with the metadata feedback sink, and record what was
measured so later plans improve. Every relational (non-UDF) terminal path —
single-node, distributed, and each adaptive stage — routes through
`run_relational`, so the contract loop is applied in exactly one place and the
paths cannot drift out of sync.

It lives in `api` because it imports all three subsystems (plus `dist`); the
independence contract forbids any of them from importing the others, so the
conductor is the one layer allowed to assemble them.

Split into a package on its seam — zero-config resolution (`autoconfig`) and the contract
loop (`run`); the import path is unchanged.
"""

from __future__ import annotations

from batcher.api.orchestration.autoconfig import (
    approx_quantile,
    resolve_auto_config,
    with_auto_config,
)
from batcher.api.orchestration.run import (
    _MAX_PARTITIONS,  # noqa: F401
    DEFAULT_PARTITIONS,
    _clamp_partitions,  # noqa: F401  (sibling modules reuse the shared partition clamp)
    partitions_from_physical,
    run_relational,
)
from batcher.api.source_stats import (
    collect_source_stats,
    invalidate_source_stats,
    persist_written_source_stats,
)
from batcher.api.tuning.decisions import auto_num_partitions

__all__ = [
    "DEFAULT_PARTITIONS",
    "approx_quantile",
    "auto_num_partitions",
    "collect_source_stats",
    "invalidate_source_stats",
    "partitions_from_physical",
    "persist_written_source_stats",
    "resolve_auto_config",
    "run_relational",
    "with_auto_config",
]
