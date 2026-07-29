"""The contract a credential verifier implements, and what it may not promise.

Batcher **authorizes**: given an identity, it decides which rows and columns that identity
may read. It does not **authenticate** in the sense a network service does, and this
package does not change that — read `verified_principal` below for exactly what it buys.

# The honest scope

Batcher is a library imported into the caller's process. Code already inside that process
can construct any `Principal` it likes, including one holding every role, and no in-process
check can prevent that. Adding a token check does not move the trust boundary; the boundary
is the process, and it always will be.

What a verifier *does* buy is the difference between two very different situations:

- **Asserted**: `bt.Principal("root", roles=["admin"])`. A name someone typed. Nothing
  behind it. Perfectly fine for a single-user notebook, and worthless as a control.
- **Verified**: a `Principal` reconstructed from a credential this process checked against
  a key or a public key it holds — so the *claims* (who, which roles, until when) came from
  an issuer rather than from the caller's imagination.

The second is what lets `governance.require_verified_principal` mean something: a
deployment can refuse to run a query whose identity nobody vouched for. It does not stop
someone with code execution in the process from bypassing it. Run one process per trust
domain; that is the control, and this is the thing that makes the control expressible.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from batcher._internal.errors import AccessDeniedError
from batcher.governance.principal import Principal

__all__ = ["AuthenticationError", "CredentialVerifier"]


class AuthenticationError(AccessDeniedError):
    """A credential was presented and rejected.

    Distinct from a plain `AccessDeniedError`, which means "this identity is real and is
    not allowed to read that". This means "I do not believe you are who you say you are",
    which is a different thing to log, alert on, and rate-limit.

    Inherits `AccessDeniedError` (and so `PermissionError`) so a caller that broadly
    catches authorization failures still catches this one.
    """


@runtime_checkable
class CredentialVerifier(Protocol):
    """Turns a credential into a `Principal` whose claims this process has checked.

    Implementations must fail **closed**: any credential they cannot fully validate — bad
    signature, expired, malformed, unknown issuer — raises `AuthenticationError`. Returning
    an unverified `Principal` on a doubtful credential is worse than having no verifier,
    because the deployment then believes an identity that was never established.

    The returned `Principal` must carry a non-empty `issuer`, which is what makes
    `Principal.verified` true and what `require_verified_principal` gates on.
    """

    def verify(self, credential: str) -> Principal:
        """Validate `credential` and return the identity it establishes.

        Args:
            credential: The token, assertion, or handle presented by the caller.

        Returns:
            The verified `Principal`, carrying a non-empty `issuer`.

        Raises:
            AuthenticationError: If the credential is absent, malformed, expired, or
                fails its signature check.
        """
        ...
