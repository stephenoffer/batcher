"""The conductor's half of governance: install a policy, and apply it at every read.

`api` is the only layer allowed to import a subsystem, so this is where
`batcher.governance` meets the rest of the engine. `security()` installs a catalog and
principal for a scope; `govern_scan` applies them to each scan `api.session._scan`
builds, which is the single place a source becomes a plan — and therefore the single
place enforcement has to happen for it to be unbypassable.
"""

from __future__ import annotations

from batcher.api.security._authn import authenticate, current_verifier, set_verifier
from batcher.api.security._binding import govern_scan, table_name
from batcher.api.security._context import SecurityContext, current_security, security

__all__ = [
    "SecurityContext",
    "authenticate",
    "current_security",
    "current_verifier",
    "govern_scan",
    "security",
    "set_verifier",
    "table_name",
]
