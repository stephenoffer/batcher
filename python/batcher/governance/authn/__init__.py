"""Credential verification: turning a presented credential into a verified `Principal`.

`base` holds the `CredentialVerifier` contract and the honest statement of what it does
and does not buy; `verifiers` holds the three implementations Batcher ships.

Batcher authorizes; it does not authenticate in the sense a network service does. See
`base` for exactly where the line is.
"""

from __future__ import annotations

from batcher.governance.authn.base import AuthenticationError, CredentialVerifier
from batcher.governance.authn.verifiers import (
    HmacTokenVerifier,
    JwtVerifier,
    ProcessIdentityVerifier,
)

__all__ = [
    "AuthenticationError",
    "CredentialVerifier",
    "HmacTokenVerifier",
    "JwtVerifier",
    "ProcessIdentityVerifier",
]
