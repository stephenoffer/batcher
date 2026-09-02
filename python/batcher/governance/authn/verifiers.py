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
import threading
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

from batcher._internal.errors import SecurityWarning
from batcher._internal.optional import require
from batcher.governance.authn.base import AuthenticationError
from batcher.governance.principal import Principal

__all__ = ["HmacTokenVerifier", "JwtVerifier", "ProcessIdentityVerifier"]

#: Leeway applied to expiry checks, in seconds. Clocks between a token issuer and the
#: engine drift; without a small allowance a perfectly good token is rejected for a second
#: of skew, which reads to an operator as a flaky auth system.
_CLOCK_SKEW_S = 30.0

#: JWKS clients, keyed by the URL they fetch. `PyJWKClient` caches the key set it fetched,
#: but that cache lives on the *instance*, so constructing one inside `verify` threw the
#: cache away after every credential: a hundred queries meant a hundred fetches, from every
#: worker on a distributed run. Keyed on the URL rather than held on the verifier, because
#: a serving layer that builds a verifier per request is the ordinary shape and would
#: otherwise defeat the cache just as completely.
#:
#: Process-local, and deliberately so. Each worker must fetch and validate the issuer's
#: keys itself; a client shipped from the driver would be the driver vouching for the
#: issuer, which is exactly the trust hop this verifier exists to avoid.
_jwks_clients: dict[tuple[str, int], Any] = {}
_jwks_lock = threading.Lock()

#: How long a fetched key set is reused before `PyJWKClient` refetches it. Bounds how long
#: a revoked signing key stays accepted, and is the same default the library uses.
_JWKS_LIFESPAN_S = 300

#: Discovered ``jwks_uri`` per issuer. An issuer's discovery document is static
#: configuration that changes when the provider is reconfigured, not per request, so it is
#: fetched once per process. Same reasoning, and same process-local scope, as the JWKS
#: cache above: each worker discovers for itself rather than trusting the driver.
_discovered: dict[str, str] = {}


def _jwks_client(jwks_url: str, *, lifespan: int = _JWKS_LIFESPAN_S) -> Any:
    """The shared `PyJWKClient` for `jwks_url`, building it on first use.

    An unknown `kid` still triggers a refetch inside the client, so a rotated signing key
    is picked up without waiting out the lifespan.
    """
    jwt = require("jwt", feature="JwtVerifier", provides="PyJWT", extra="oidc")
    key = (jwks_url, lifespan)
    with _jwks_lock:
        client = _jwks_clients.get(key)
        if client is None:
            client = jwt.PyJWKClient(jwks_url, cache_jwk_set=True, lifespan=lifespan)
            _jwks_clients[key] = client
        return client


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

    **Set `issuer` and `audience`.** Both default to empty, both are then skipped, and both
    skips are accepted attacks rather than merely loose configuration. A valid signature
    proves the token came from the key set at `jwks_url`; it proves nothing about *who it
    was minted for*. The large identity providers publish one key set across many tenants
    and many applications, so without `iss` a token from another tenant of the same provider
    verifies, and without `aud` a token minted for a different application of the same tenant
    verifies -- in both cases a real token, correctly signed, issued to somebody else and
    replayed here. Leaving either empty is legal and warns (`SecurityWarning`), because a
    deployment mid-migration may genuinely not know its audience yet; it is not a
    configuration to run on.

    Examples:
        .. doctest::

            >>> from batcher.governance.authn import JwtVerifier
            >>> verifier = JwtVerifier(
            ...     jwks_url="https://idp/.well-known/jwks.json",
            ...     issuer="https://idp/",
            ...     audience="batcher",
            ... )
            >>> verifier.algorithms
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

    def __post_init__(self) -> None:
        """Warn about the two checks that are skipped when left unset.

        At construction rather than at `verify`, so the warning names the line that
        configured the verifier rather than a line in the middle of a pipeline, and so it is
        emitted once per verifier instead of once per credential.
        """
        skipped = [
            f"{what} unset, so a token issued to {who} verifies"
            for what, value, who in (
                ("issuer (`iss`)", self.issuer, "another tenant of the same identity provider"),
                ("audience (`aud`)", self.audience, "another application of the same tenant"),
            )
            if not value
        ]
        if not skipped:
            return
        detail = "; ".join(skipped)
        warnings.warn(
            f"JwtVerifier(jwks_url={self.jwks_url!r}) skips a claim check: {detail}. "
            "A valid signature proves which key set signed the token, never who it was "
            "minted for.",
            SecurityWarning,
            stacklevel=3,
        )

    @classmethod
    def from_issuer(cls, issuer: str, *, audience: str = "", timeout: float = 10.0) -> JwtVerifier:
        """Build a verifier by discovering the issuer's keys from its OIDC metadata.

        An operator knows their issuer URL, which is what their identity provider calls
        itself and what every other service in the estate is already configured with. The
        JWKS URL is an implementation detail of that provider, published at a well-known
        path precisely so nobody has to look it up and paste it into another config file.

        The discovery document is fetched once per process and cached, then the key set is
        cached separately by `verify`. On a distributed query each worker discovers and
        fetches for itself, against its own network path to the provider.

        Examples:
            .. doctest::

                >>> from batcher.governance.authn import JwtVerifier
                >>> v = JwtVerifier.from_issuer(  # doctest: +SKIP
                ...     "https://login.microsoftonline.com/tenant/v2.0", audience="batcher"
                ... )
                >>> v.issuer  # doctest: +SKIP
                'https://login.microsoftonline.com/tenant/v2.0'

        Args:
            issuer: The provider's issuer URL, as it appears in the token's ``iss`` claim.
            audience: Expected ``aud`` claim; empty skips the audience check.
            timeout: Seconds to wait for the discovery request.

        Returns:
            A `JwtVerifier` bound to the discovered JWKS endpoint and to `issuer`.

        Raises:
            AuthenticationError: If the discovery document cannot be read or names no
                ``jwks_uri``. Raised here, at configuration time, rather than on the first
                credential, so a misconfigured issuer fails where it is set rather than
                where it is used.
        """
        return cls(
            jwks_url=_discover_jwks(issuer, timeout=timeout), issuer=issuer, audience=audience
        )

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
        # Through the one guard: the hand-rolled hint named `batcher[oidc]`, which is neither
        # a real extra nor this distribution (it is `batcher-engine`), so the single actionable
        # line in the error was a command that fails.
        jwt = require("jwt", feature="JwtVerifier", provides="PyJWT", extra="oidc")

        try:
            signing_key = _jwks_client(self.jwks_url).get_signing_key_from_jwt(credential)
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


def _discover_jwks(issuer: str, *, timeout: float) -> str:
    """The ``jwks_uri`` an issuer publishes, fetched once per process and cached."""
    import urllib.error
    import urllib.request

    base = issuer.rstrip("/")
    cached = _discovered.get(base)
    if cached is not None:
        return cached
    url = f"{base}/.well-known/openid-configuration"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            document = json.loads(response.read())
    except (OSError, ValueError) as exc:
        raise AuthenticationError(
            f"cannot read OIDC metadata for issuer {issuer!r}: {exc}",
            hint=f"Check that {url} is reachable from this machine.",
        ) from exc
    jwks_uri = document.get("jwks_uri")
    if not jwks_uri:
        raise AuthenticationError(
            f"OIDC metadata for issuer {issuer!r} names no 'jwks_uri'",
            hint="Pass jwks_url= explicitly if the provider does not publish one.",
        )
    with _jwks_lock:
        _discovered[base] = str(jwks_uri)
    return str(jwks_uri)
