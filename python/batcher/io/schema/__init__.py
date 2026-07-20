"""`io.schema` — reconciling schemas that disagree across the files of one source.

A source is often many files written at different times, so their schemas drift:
a column is added, a type widens, field order changes. `evolution` is where that
drift is detected (`schema_drift` → `SchemaDrift`) and resolved into the one schema
a scan must present (`unify_schemas`, `reconcile_batches`, `normalize_batch`) — so
that every operator above sees a single stable schema and no format has to solve
this for itself.

Read-time only, and deliberately so: this reconciles what was *already written*.
Deciding what a sink writes is the format's job, and the neutral type vocabulary the
unified schema is expressed in belongs to `plan.types`.
"""

from __future__ import annotations

from batcher.io.schema.evolution import (
    SchemaDrift,
    normalize_batch,
    reconcile_batches,
    schema_drift,
    unify_schemas,
)

__all__ = [
    "SchemaDrift",
    "normalize_batch",
    "reconcile_batches",
    "schema_drift",
    "unify_schemas",
]
