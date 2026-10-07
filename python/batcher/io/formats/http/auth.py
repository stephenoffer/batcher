"""Auth providers for the HTTP sources: a static bearer token and OAuth2 client credentials.

Both hold **references**, never secrets. A token or client secret is a literal or an
``env:``/``file:``/``cmd:``/key-store reference (`io.credentials.resolve_secret`), resolved
at request time on the machine sending the request, so neither the provider nor a pickled
split carrying it ever holds the plaintext. Neither prints its secret field in a repr.

An OAuth2 access token is the one piece of secret material that has to be *kept*, because
fetching one per request would multiply the API traffic. It is cached at module level, keyed
by the token endpoint, client and scope, so it never rides on an object that gets pickled or
logged, and so every source in the process using the same client shares one token. It is
refreshed ``_EXPIRY_SKEW_S`` before it expires and, once, when the API answers 401 -- the
case of a token revoked or rotated server-side before its stated expiry.

The auth provider protocol is two methods: ``headers()`` returns the headers to add, and
``invalidate()`` drops any cached credential so the next ``headers()`` fetches a fresh one.
"""

from __future__ import annotations

import base64
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlencode

from batcher._internal.errors import BackendError

__all__ = ["AuthProvider", "BearerToken", "OAuth2ClientCredentials"]

#: Refresh a cached token this many seconds before the expiry the server stated, so a
#: request sent just before expiry does not arrive just after it.
_EXPIRY_SKEW_S = 60.0

#: Cached OAuth2 tokens: key -> (access_token, expires_at monotonic seconds).
_TOKENS: dict[tuple[str, ...], tuple[str, float]] = {}
_TOKENS_LOCK = threading.Lock()


class AuthProvider(Protocol):
    """What an HTTP source asks of its ``auth=``: headers to add, and a way to refresh."""

    def headers(self) -> dict[str, str]: ...

    def invalidate(self) -> None: ...


@dataclass(frozen=True, slots=True)
class BearerToken:
    """Send ``Authorization: Bearer <token>``, with the token given by reference.

    Pass ``"env:GITHUB_TOKEN"``, ``"file:/run/secrets/token"``, a ``cmd:`` or key-store
    reference, or a literal. The reference is resolved for each request, so a rotated
    secret file is picked up without rebuilding the read, and the token never appears in
    the plan, a repr, or a log line.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> auth = bt.io.BearerToken("env:API_TOKEN")
            >>> "API_TOKEN" in repr(auth)
            False

    Args:
        token: The token, as a secret reference or a literal.
        scheme: The authorization scheme word, ``"Bearer"`` unless the API wants another.
    """

    token: str = field(repr=False)
    scheme: str = "Bearer"

    def headers(self) -> dict[str, str]:
        """The ``Authorization`` header, with the token resolved now.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.io.BearerToken("abc").headers()
                {'Authorization': 'Bearer abc'}

        Returns:
            The headers to add to a request.
        """
        from batcher.io.credentials import resolve_secret

        token = resolve_secret(self.token, what="HTTP bearer token")
        if not token:
            raise BackendError("HTTP bearer token resolved to an empty value")
        return {"Authorization": f"{self.scheme} {token}"}

    def invalidate(self) -> None:
        """Nothing is cached, so there is nothing to drop.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.io.BearerToken("abc").invalidate()
        """


@dataclass(frozen=True, slots=True)
class OAuth2ClientCredentials:
    """Fetch and refresh an access token with the OAuth2 client-credentials grant.

    A token is requested from `token_url` (RFC 6749 section 4.4), cached in-process until
    shortly before the ``expires_in`` it came with, and re-fetched once when the API
    answers 401. The client secret is a secret reference resolved only when a token is
    fetched. This is the grant Microsoft Graph, Salesforce and most machine-to-machine APIs
    accept.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> auth = bt.io.OAuth2ClientCredentials(
            ...     token_url="https://login.example.com/oauth2/token",
            ...     client_id="my-app",
            ...     client_secret="env:CLIENT_SECRET",
            ...     scope="api.read",
            ... )
            >>> auth.client_id
            'my-app'

    Args:
        token_url: The authorization server's token endpoint.
        client_id: The client identifier.
        client_secret: The client secret, as a secret reference or a literal.
        scope: The space-separated scopes to request, or None for the client's default.
        audience: An ``audience`` parameter, for servers that require one (Auth0).
        client_auth: ``"body"`` sends the client credentials as form fields;
            ``"basic"`` sends them as HTTP Basic authentication (RFC 6749 section 2.3.1).
        timeout: Seconds to wait for the token endpoint.
    """

    token_url: str
    client_id: str
    client_secret: str = field(repr=False)
    scope: str | None = None
    audience: str | None = None
    client_auth: str = "body"
    timeout: float = 30.0

    def __post_init__(self) -> None:
        if self.client_auth not in ("body", "basic"):
            from batcher._internal.errors import PlanError

            raise PlanError(
                f"OAuth2ClientCredentials.client_auth must be 'body' or 'basic', "
                f"got {self.client_auth!r}"
            )

    def _key(self) -> tuple[str, ...]:
        return (self.token_url, self.client_id, self.scope or "", self.audience or "")

    def headers(self) -> dict[str, str]:
        """The ``Authorization`` header, fetching or refreshing the token as needed.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> auth = bt.io.OAuth2ClientCredentials(
                ...     token_url="https://idp.example/token", client_id="c", client_secret="s"
                ... )
                >>> auth.headers()  # doctest: +SKIP
                {'Authorization': 'Bearer eyJ...'}

        Returns:
            The headers to add to a request.
        """
        key = self._key()
        with _TOKENS_LOCK:
            cached = _TOKENS.get(key)
            if cached is None or time.monotonic() >= cached[1]:
                cached = self._fetch()
                _TOKENS[key] = cached
        return {"Authorization": f"Bearer {cached[0]}"}

    def invalidate(self) -> None:
        """Drop the cached token so the next request fetches a fresh one.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.io.OAuth2ClientCredentials(
                ...     token_url="https://idp.example/token", client_id="c", client_secret="s"
                ... ).invalidate()
        """
        with _TOKENS_LOCK:
            _TOKENS.pop(self._key(), None)

    def _fetch(self) -> tuple[str, float]:
        """POST the client-credentials grant and return ``(token, refresh_at)``."""
        import urllib.error
        import urllib.request

        from batcher.io.credentials import resolve_secret

        secret = resolve_secret(self.client_secret, what="OAuth2 client secret") or ""
        form: dict[str, str] = {"grant_type": "client_credentials"}
        if self.scope:
            form["scope"] = self.scope
        if self.audience:
            form["audience"] = self.audience
        request = urllib.request.Request(self.token_url, method="POST")
        request.add_header("Content-Type", "application/x-www-form-urlencoded")
        request.add_header("Accept", "application/json")
        if self.client_auth == "basic":
            pair = f"{self.client_id}:{secret}".encode()
            request.add_unredirected_header(
                "Authorization", f"Basic {base64.b64encode(pair).decode()}"
            )
        else:
            form["client_id"] = self.client_id
            form["client_secret"] = secret
        try:
            with urllib.request.urlopen(
                request, data=urlencode(form).encode(), timeout=self.timeout
            ) as response:
                document: Any = json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as exc:
            # The token endpoint's error body is the useful part (`invalid_client`,
            # `invalid_scope`), and it never echoes the secret back.
            detail = exc.read()[:300].decode("utf-8", "replace")
            raise BackendError(
                f"OAuth2 token request to {self.token_url} failed with HTTP {exc.code}: {detail}"
            ) from None
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise BackendError(
                f"OAuth2 token request to {self.token_url} failed: {type(exc).__name__}: {exc}"
            ) from None
        token = document.get("access_token") if isinstance(document, dict) else None
        if not token:
            raise BackendError(f"OAuth2 token response from {self.token_url} has no access_token")
        expires_in = float(document.get("expires_in") or 3600)
        return str(token), time.monotonic() + max(expires_in - _EXPIRY_SKEW_S, 0.0)
