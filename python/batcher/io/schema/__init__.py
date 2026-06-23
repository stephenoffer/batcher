"""Read-time schema handling — reconciliation across multiple files (`evolution`)."""

from __future__ import annotations

from batcher.io.schema.evolution import (
    SchemaDrift,
    normalize_batch,
    schema_drift,
    unify_schemas,
)

__all__ = ["SchemaDrift", "normalize_batch", "schema_drift", "unify_schemas"]
