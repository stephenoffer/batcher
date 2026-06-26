"""The execution-strategy seam: one `Executor` Protocol, one `ExecutionContext`.

Core runs queries through several execution tiers — single-node native, the
linear `map_batches`/UDF orchestrator, and (assembled by the conductor) the
distributed path — and more are planned (morsel scheduler, JIT/LLVM, GPU). Rather
than grow an if/elif/else at the dispatch site for each, every tier is an
`Executor`: a strategy with one `execute` method, selected by a registry.

This module is the neutral seam only. It defines the Protocol and the read-only
`ExecutionContext` the strategies receive. The concrete strategies and the
registry that selects between them live in `api` (the conductor), because the
local-native strategy orchestrates Kyber + Carbonite and the distributed strategy
lives in `dist` — neither of which Core may import. Core owns the contract; the
conductor wires the implementations behind it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import pyarrow as pa

if TYPE_CHECKING:
    from batcher.io.source import Source
    from batcher.metadata import MetadataHub
    from batcher.plan.logical import LogicalPlan
    from batcher.plan.profile import ProfileCollector

__all__ = ["ExecutionContext", "Executor"]


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Inputs an `Executor` needs, beyond the plan and its sources.

    Carries exactly what the existing execution functions take as side inputs:
    the output column names (to build an empty-result schema), the process
    MetadataHub (the feedback sink the local path records into and Kyber learns
    from), and the distributed knobs (`num_workers`, `transport`). A strategy uses
    only the fields it needs; the rest are inert.

    The dataclass is frozen — but `profile`, when present, is a *mutable output sink*
    the execution writes its planned/measured facts into (the reference is fixed; its
    contents are not). Everything else is read-only input.
    """

    columns: list[str]
    hub: MetadataHub
    num_workers: int | None = None
    transport: str = "auto"
    # Whether the user opted this result into the process result cache
    # (`Dataset.cache()`). Honored only by the single-node relational path; the
    # conductor keys the cache by plan signature + input identity.
    cache: bool = False
    # Per-source `SourceStatistics` already collected by the conductor (e.g. the
    # metadata-answer attempt for a `count()`/`is_empty()` that missed). When set, the
    # relational path reuses it instead of re-reading every source's footer/manifest —
    # so a single terminal op reads source statistics once, not per optimization pass.
    # `None` means "not collected yet"; the path collects its own.
    source_stats: list | None = None
    # An optional profiling sink. When set, the relational path records its planned
    # estimates, admission verdict, and measured per-operator metrics into it, so the
    # conductor can assemble a `QueryProfile` (for `explain(analyze=True)`, `stats()`,
    # and the per-query event log). `None` for an ordinary run — zero overhead.
    profile: ProfileCollector | None = None


class Executor(Protocol):
    """A single execution strategy: run a plan over its sources to a table.

    Implementations are thin wrappers over the existing execution functions
    (`execute_local`, `execute_with_udfs`, `execute_distributed`). The selection
    of *which* strategy runs a given plan is the registry's job, not the
    strategy's — an `Executor.execute` assumes it was chosen correctly.
    """

    def execute(
        self,
        plan: LogicalPlan,
        sources: list[Source],
        ctx: ExecutionContext,
    ) -> pa.Table: ...
