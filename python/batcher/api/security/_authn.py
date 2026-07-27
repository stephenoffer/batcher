"""Installing a credential verifier, and using it to establish an identity.

The seam between the layer that authenticates and the layer that authorizes. A host with a
network edge — a notebook server, an API gateway, a Ray job submitter — installs a verifier
once at startup; everything downstream turns a credential into a `Principal` by asking
here, instead of each call site inventing its own trust decision.

The verifier is a module-level global rather than a `ContextVar`, and deliberately: it is
deployment configuration, fixed at startup, not something a query scope should vary. The
*identity* is per-scope (`bt.security(...)`); how identities are established is not.
"""

from __future__ import annotations

import threading

from batcher.governance.authn import AuthenticationError, CredentialVerifier
from batcher.governance.principal import Principal

__all__ = ["authenticate", "current_verifier", "set_verifier"]

_LOCK = threading.Lock()
_VERIFIER: CredentialVerifier | None = None


def set_verifier(verifier: CredentialVerifier | None) -> None:
    """Install the process-wide credential verifier, or clear it with None.

    Call once at startup, from the layer that owns the network edge.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.governance.authn import ProcessIdentityVerifier
            >>> bt.set_verifier(ProcessIdentityVerifier(roles={"analyst"}))
            >>> bt.authenticate("").verified
            True
            >>> bt.set_verifier(None)

    Args:
        verifier: The verifier to install, or None to remove the current one.
    """
    global _VERIFIER
    with _LOCK:
        _VERIFIER = verifier


def current_verifier() -> CredentialVerifier | None:
    """The installed verifier, or None when no identity provider is configured.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.set_verifier(None)
            >>> bt.current_verifier() is None
            True

    Returns:
        The installed `CredentialVerifier`, or None.
    """
    return _VERIFIER


def authenticate(credential: str = "") -> Principal:
    """Turn `credential` into a `Principal` whose claims have been checked.

    The one way to obtain a *verified* principal. Constructing `Principal(...)` directly is
    still supported and still correct for a single-user session — it just produces an
    asserted identity, which `governance.require_verified_principal` can refuse.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.governance.authn import HmacTokenVerifier
            >>> verifier = HmacTokenVerifier(key="s3cret", issuer="gateway")
            >>> bt.set_verifier(verifier)
            >>> principal = bt.authenticate(verifier.mint("ana", roles=["analyst"]))
            >>> principal.name, sorted(principal.roles), principal.issuer
            ('ana', ['analyst'], 'gateway')
            >>> bt.set_verifier(None)

    Args:
        credential: The token or assertion to verify. Ignored by
            `ProcessIdentityVerifier`, which reads the OS identity instead.

    Returns:
        The verified `Principal`.

    Raises:
        AuthenticationError: If no verifier is installed, or the credential is rejected.
    """
    verifier = _VERIFIER
    if verifier is None:
        raise AuthenticationError(
            "no credential verifier is installed, so no identity can be verified.",
            hint=(
                "Install one at startup — bt.set_verifier(ProcessIdentityVerifier()) for "
                "a single-node deployment, or an HmacTokenVerifier / JwtVerifier."
            ),
        )
    return verifier.verify(credential)
