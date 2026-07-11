"""Governance — who may read which rows and columns, and through what mask.

The fourth subsystem, independent of `kyber`, `carbonite`, and `core`, and imported
only by `api`. It decides (like Kyber) and never executes (like Kyber), but its
rewrites are *mandatory*, not optimizations: they run before the optimizer, on every
read, and no configuration turns them off.

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

__all__ = [
    "AttributeIn",
    "ColumnMask",
    "Encrypt",
    "GovernanceEvent",
    "Grant",
    "MatchesAttribute",
    "Nullify",
    "Origin",
    "Principal",
    "Pseudonymize",
    "Redact",
    "RowFilter",
    "SecurityCatalog",
    "TagMask",
    "column_lineage",
    "enforce",
]
