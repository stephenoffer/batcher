"""Governance — who may read which rows and columns, and through what mask.

The fourth subsystem, independent of `kyber`, `carbonite`, and `core`, and imported
only by `api`. It decides (like Kyber) and never executes (like Kyber), but its
rewrites are *mandatory*, not optimizations: they run before the optimizer, on every
read, and no configuration turns them off.

`ResidencyCatalog` answers a different question of the same shape: not who may read a
dataset but *where it may be computed on*, which is the half of a sovereignty obligation a
scheduler can break silently by placing a stage in another region.

`SecurityCatalog` holds the declared policy; `Principal` is the identity; `enforce`
rewrites a plan so the principal can read only what the catalog allows. The column
masks themselves are ordinary expressions (`batcher.mask`, `batcher.hmac_sha256`,
`batcher.aes_encrypt`), so masking runs in the Rust data plane at full speed.
"""

from __future__ import annotations

from batcher.governance.audit import GovernanceEvent
from batcher.governance.catalog import SecurityCatalog
from batcher.governance.enforce import enforce
from batcher.governance.filters import AttributeIn, MatchesAttribute
from batcher.governance.lineage import Origin, column_lineage
from batcher.governance.masks import Encrypt, Nullify, Pseudonymize, Redact
from batcher.governance.policy import ColumnMask, Grant, RowFilter, TagMask
from batcher.governance.principal import Principal
from batcher.governance.residency import (
    RESIDENCY_MODES,
    DataResidency,
    ResidencyCatalog,
    ResidencyVerdict,
    active_residency,
    set_residency,
)

__all__ = [
    "RESIDENCY_MODES",
    "AttributeIn",
    "ColumnMask",
    "DataResidency",
    "Encrypt",
    "GovernanceEvent",
    "Grant",
    "MatchesAttribute",
    "Nullify",
    "Origin",
    "Principal",
    "Pseudonymize",
    "Redact",
    "ResidencyCatalog",
    "ResidencyVerdict",
    "RowFilter",
    "SecurityCatalog",
    "TagMask",
    "active_residency",
    "column_lineage",
    "enforce",
    "set_residency",
]
