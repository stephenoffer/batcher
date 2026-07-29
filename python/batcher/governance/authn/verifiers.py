"""The credential verifiers Batcher ships.

Three, deliberately, and two of them depend on nothing outside the standard library:

- `ProcessIdentityVerifier` — the OS user. The single-node default, and the honest one:
  on a box where each user runs their own process, the OS already answered "who is this".
- `HmacTokenVerifier` — a signed token, verified against a shared key. For a host that
  mints tokens for its own workers (a job submitter handing a token to each Ray task).
- `JwtVerifier` — RS256/ES256 against a JWKS endpoint. The OIDC integration, and the only
  one with a dependency, so it is **optional**: `pyjwt` is imported lazily and its absence
  raises a `MissingDependencyError` naming the extra rather than failing at import.

Three implementations is not padding. The `CredentialVerifier` Protocol would be an empty
framework with one, and the anti-speculation rule says so; with three the seam is carrying
real weight, and each covers a deployment shape the others cannot.

Everything here fails **closed**. A credential that cannot be fully validated raises
`AuthenticationError` — never a `Principal` with the claims taken on trust, which would be
worse than having no verifier because the deployment would then believe an identity nobody
established.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field

from batcher.governance.authn.base import AuthenticationError
from batcher.governance.principal import Principal

__all__ = ["HmacTokenVerifier", "JwtVerifier", "ProcessIdentityVerifier"]

#: Leeway applied to expiry checks, in seconds. Clocks between a token issuer and the
#: engine drift; without a small allowance a perfectly good token is rejected for a second
#: of skew, which reads to an operator as a flaky auth system.
_CLOCK_SKEW_S = 30.0


def _b64url_decode(segment: str) -> bytes:
    """Decode a base64url segment, restoring the padding JWT-style encoders strip."""
    padding = "=" * (-len(segment) % 4)
    try:
        return base64.urlsafe_b64decode(segment + padding)
    except (binascii.Error, ValueError) as exc:
        raise AuthenticationError("credential is not valid base64url.") from exc


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _principal_from_claims(claims: dict, issuer: str) -> Principal:
    """Build a verified `Principal` from already-validated claims.

    Raises:
        AuthenticationError: If the claims carry no subject, or roles of the wrong shape.
    """
    name = claims.get("sub") or claims.get("name")
    if not isinstance(name, str) or not name:
        raise AuthenticationError(
            "credential carries no subject.",
            hint="The token needs a `sub` (or `name`) claim identifying the principal.",
        )
    roles = claims.get("roles", [])
    if isinstance(roles, str):
        # A bare string would iterate into one role per character — the same trap
        # `Principal.__post_init__` rejects, caught here with a message about the token.
        raise AuthenticationError(
            f"credential's `roles` claim is a string ({roles!r}), not a list.",
            hint='Encode roles as a JSON array, e.g. "roles": ["analyst"].',
        )
    attrs = {k: v for k, v in (claims.get("attrs") or {}).items() if isinstance(v, str)}
    expires_at = claims.get("exp")
    return Principal(
        name=name,
        roles=frozenset(str(r) for r in roles),
        attrs=attrs,
        issuer=issuer,
        expires_at=float(expires_at) if isinstance(expires_at, (int, float)) else None,
    )


@dataclass(frozen=True, slots=True)
class ProcessIdentityVerifier:
    """The OS user running this process, as a verified identity.

    The default worth reaching for first, because on the deployment Batcher actually
    recommends — one process per trust domain — the operating system has *already* done the
    authentication, and re-doing it in Python adds nothing. The credential argument is
    ignored: there is nothing to present, the answer is who the kernel says you are.

    `roles` maps the OS user to application roles, so a deployment can say "the `etl` unix
    account holds the `writer` role" without inventing a token infrastructure.

    Examples:
        .. doctest::

            >>> from batcher.governance.authn import ProcessIdentityVerifier
            >>> verifier = ProcessIdentityVerifier(roles={"analyst"})
            >>> principal = verifier.verify("")
            >>> principal.verified, principal.issuer
            (True, 'os')
    """

    #: Roles granted to the OS user. Empty means the principal holds none.
    roles: frozenset[str] = field(default_factory=frozenset)

    def verify(self, credential: str) -> Principal:  # noqa: ARG002  (nothing to present)
        """Return the OS user as a verified principal.

        Args:
            credential: Ignored — the OS identity is not presented, it is read.

        Returns:
            A `Principal` named for the OS user, with `issuer="os"`.

        Raises:
            AuthenticationError: If the OS user cannot be determined at all.
        """
        import getpass

        try:
            user = getpass.getuser()
        except Exception as exc:  # pragma: no cover - no passwd entry and no env
            raise AuthenticationError(
                "cannot determine the OS user of this process.",
                hint="Set USER/LOGNAME, or use an explicit verifier.",
            ) from exc
        return Principal(name=user, roles=frozenset(self.roles), issuer="os")


@dataclass(frozen=True, slots=True)
class HmacTokenVerifier:
    """A compact signed token, verified against a shared secret. Standard library only.

    The token is ``<base64url(claims_json)>.<base64url(hmac_sha256)>``. It is deliberately
    not a JWT: there is no algorithm field, so there is no "alg: none" attack and no
    algorithm-confusion attack — this verifier does exactly one thing and cannot be talked
    into doing another.

    For a host that mints its own tokens: a job submitter that authenticates a user, then
    hands each Ray task a token carrying that identity.

    The key may be a secret *reference* (``env:NAME``, ``file:PATH``), resolved at verify
    time through the same machinery the connectors use, so the key never has to sit in a
    config file or a plan.

    Examples:
        .. doctest::

            >>> from batcher.governance.authn import HmacTokenVerifier
            >>> verifier = HmacTokenVerifier(key="s3cret", issuer="acme-submitter")
            >>> token = verifier.mint("ana", roles=["analyst"], ttl_seconds=60)
            >>> principal = verifier.verify(token)
            >>> principal.name, sorted(principal.roles), principal.verified
            ('ana', ['analyst'], True)
    """

    #: The shared secret, or an ``env:``/``file:`` reference to it.
    key: str
    #: Recorded as the principal's `issuer`, so an audit log says who vouched.
    issuer: str = "hmac"

    def _resolved_key(self) -> bytes:
        from batcher.io.credentials import resolve_secret

        resolved = resolve_secret(self.key, what="HMAC token key")
        if not resolved:
            raise AuthenticationError(
                "the HMAC verifier has no key.",
                hint="Pass key='env:BATCHER_TOKEN_KEY' or a literal secret.",
            )
        return resolved.encode("utf-8")

    def mint(self, subject: str, *, roles=(), attrs=None, ttl_seconds: float = 3600) -> str:
        """Mint a token for `subject`. For the host that issues them, and for tests.

        Args:
            subject: The principal's name.
            roles: Roles to grant.
            attrs: Attribute-based-access-control attributes.
            ttl_seconds: How long the token is valid.

        Returns:
            The encoded token.
        """
        claims = {
            "sub": subject,
            "roles": list(roles),
            "attrs": dict(attrs or {}),
            "exp": time.time() + ttl_seconds,
        }
        payload = _b64url_encode(json.dumps(claims, sort_keys=True).encode("utf-8"))
        signature = hmac.new(self._resolved_key(), payload.encode("ascii"), hashlib.sha256)
        return f"{payload}.{_b64url_encode(signature.digest())}"

    def verify(self, credential: str) -> Principal:
        """Check the signature and expiry, then return the identity.

        Args:
            credential: The token.

        Returns:
            The verified `Principal`.

        Raises:
            AuthenticationError: If the token is malformed, mis-signed, or expired.
        """
        if not credential or "." not in credential:
            raise AuthenticationError("credential is not a signed token.")
        payload, _, presented = credential.rpartition(".")

        expected = hmac.new(self._resolved_key(), payload.encode("ascii"), hashlib.sha256)
        # Constant-time: a short-circuiting `==` leaks the signature one byte at a time to
        # anyone who can time the call, which is the classic way a MAC check is defeated.
        if not hmac.compare_digest(_b64url_encode(expected.digest()), presented):
            raise AuthenticationError("credential signature does not verify.")

        try:
            claims = json.loads(_b64url_decode(payload))
        except (ValueError, UnicodeDecodeError) as exc:
            raise AuthenticationError("credential payload is not valid JSON.") from exc
        if not isinstance(claims, dict):
            raise AuthenticationError("credential payload is not a claims object.")

        principal = _principal_from_claims(claims, self.issuer)
        if principal.expired(time.time() - _CLOCK_SKEW_S):
            raise AuthenticationError(
                f"credential for {principal.name!r} expired.",
                hint="Mint a fresh token; expiry is checked with 30s of clock leeway.",
            )
        return principal


@dataclass(frozen=True, slots=True)
class JwtVerifier:
    """An OIDC ID token, verified against the issuer's published keys.

    The integration for a deployment that already has an identity provider: the layer with
    the network edge authenticates the user, and the resulting token flows down to Batcher,
    which checks it against the provider's JWKS rather than trusting the caller.

    `pyjwt` is imported lazily, so this class costs nothing until used and its absence is a
    clear `MissingDependencyError` rather than an import failure for everybody. Batcher's
    core dependencies are deliberately four packages; an OIDC library is not one of them.

    Signature algorithms are pinned by `algorithms`, which defaults to asymmetric ones
    only. That default is load-bearing: allowing `HS256` alongside `RS256` is the
    algorithm-confusion attack, where an attacker signs a token with the *public* key as an
    HMAC secret and the verifier accepts it.

    Examples:
        .. doctest::

            >>> from batcher.governance.authn import JwtVerifier
            >>> JwtVerifier(jwks_url="https://idp/.well-known/jwks.json").algorithms
            ('RS256', 'ES256')
    """

    #: Where to fetch the issuer's public keys.
    jwks_url: str
    #: Expected `iss` claim; empty accepts whatever the token says (not recommended).
    issuer: str = ""
    #: Expected `aud` claim; empty skips the audience check.
    audience: str = ""
    #: Permitted signature algorithms. Asymmetric only, by default and on purpose.
    algorithms: tuple[str, ...] = ("RS256", "ES256")

    def verify(self, credential: str) -> Principal:
        """Validate the JWT's signature, issuer, audience, and expiry.

        Args:
            credential: The encoded JWT.

        Returns:
            The verified `Principal`.

        Raises:
            MissingDependencyError: If `pyjwt` is not installed.
            AuthenticationError: If the token fails any check.
        """
        try:
            import jwt
            from jwt import PyJWKClient
        except ImportError as exc:  # pragma: no cover - exercised by the extras matrix
            from batcher._internal.errors import MissingDependencyError

            raise MissingDependencyError(
                "JwtVerifier needs PyJWT, which is not installed.",
                hint="pip install 'batcher[oidc]' (or pip install pyjwt[crypto]).",
            ) from exc

        try:
            signing_key = PyJWKClient(self.jwks_url).get_signing_key_from_jwt(credential)
            claims = jwt.decode(
                credential,
                signing_key.key,
                algorithms=list(self.algorithms),
                issuer=self.issuer or None,
                audience=self.audience or None,
                options={"verify_aud": bool(self.audience)},
            )
        except Exception as exc:
            # Everything PyJWT raises — bad signature, expired, wrong audience,
            # unreachable JWKS — is one thing to a caller: this credential is not good.
            raise AuthenticationError(f"credential rejected: {exc}") from exc

        return _principal_from_claims(claims, self.issuer or str(claims.get("iss") or "jwt"))
