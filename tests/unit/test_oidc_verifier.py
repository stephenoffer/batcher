"""`JwtVerifier` must not refetch the issuer's keys on every credential it checks.

A verifier is called once per request in a serving deployment and once per worker on a
distributed query, so a JWKS fetch per verification turns the identity provider into a
hard dependency of every query and a rate-limit target proportional to cluster size.

The fetch count is measured against a **real JWKS server** rather than reasoned about,
because the caching lives inside `PyJWKClient` and the question is whether Batcher's use of
it preserves the cache, not whether the cache exists.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from batcher.governance.authn import AuthenticationError, JwtVerifier

jwt = pytest.importorskip("jwt", reason="pyjwt not installed")
pytest.importorskip("cryptography", reason="pyjwt[crypto] not installed")

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def keypair():
    """One RSA keypair, and the JWKS document that publishes its public half."""
    from cryptography.hazmat.primitives.asymmetric import rsa

    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    numbers = private.public_key().public_numbers()

    def b64(value: int) -> str:
        import base64

        raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    jwks = {
        "keys": [
            {
                "kty": "RSA",
                "kid": "test-key",
                "use": "sig",
                "alg": "RS256",
                "n": b64(numbers.n),
                "e": b64(numbers.e),
            }
        ]
    }
    return private, jwks


class _Jwks(BaseHTTPRequestHandler):
    """Serves the JWKS and counts how many times it was asked for."""

    def do_GET(self) -> None:
        self.server.fetches += 1
        body = json.dumps(self.server.jwks).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        """Silence the stdlib access log."""


@pytest.fixture()
def jwks_server(keypair):
    """A localhost JWKS endpoint that records its fetch count."""
    _, jwks = keypair
    server = HTTPServer(("127.0.0.1", 0), _Jwks)
    server.jwks = jwks
    server.fetches = 0
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def _url(server) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}/jwks.json"


def _token(private, **claims) -> str:
    payload = {
        "iss": "https://idp.test",
        "aud": "batcher",
        "sub": "alice",
        "exp": int(time.time()) + 300,
        "iat": int(time.time()),
        **claims,
    }
    return jwt.encode(payload, private, algorithm="RS256", headers={"kid": "test-key"})


def test_a_valid_token_becomes_a_verified_principal(keypair, jwks_server):
    """The happy path, so the caching assertions below are about a working verifier."""
    private, _ = keypair
    verifier = JwtVerifier(
        jwks_url=_url(jwks_server), issuer="https://idp.test", audience="batcher"
    )
    principal = verifier.verify(_token(private, roles=["analyst"]))

    assert principal.name == "alice"
    assert "analyst" in principal.roles


def test_the_keys_are_fetched_once_across_many_verifications(keypair, jwks_server):
    """One verifier checking many tokens must hit the identity provider once.

    Constructing the JWKS client inside `verify` discarded its cache after every call, so
    a hundred queries meant a hundred fetches, from every worker.
    """
    private, _ = keypair
    verifier = JwtVerifier(
        jwks_url=_url(jwks_server), issuer="https://idp.test", audience="batcher"
    )
    for index in range(5):
        verifier.verify(_token(private, sub=f"user-{index}"))

    assert jwks_server.fetches == 1, (
        f"fetched the JWKS {jwks_server.fetches} times for 5 verifications"
    )


def test_two_verifiers_for_the_same_issuer_share_the_fetch(keypair, jwks_server):
    """Caching keyed on the URL, not on the object, so a per-request verifier still caches.

    A serving layer that builds a verifier per request is the ordinary shape, and it would
    otherwise defeat the cache entirely.
    """
    private, _ = keypair
    url = _url(jwks_server)
    for index in range(3):
        JwtVerifier(jwks_url=url, issuer="https://idp.test", audience="batcher").verify(
            _token(private, sub=f"user-{index}")
        )

    assert jwks_server.fetches == 1, f"fetched {jwks_server.fetches} times for 3 verifiers"


def test_a_bad_signature_is_still_rejected(keypair, jwks_server):
    """Caching must not turn into accepting. The control for the two tests above."""
    from cryptography.hazmat.primitives.asymmetric import rsa

    _, _ = keypair
    attacker = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    forged = jwt.encode(
        {
            "iss": "https://idp.test",
            "aud": "batcher",
            "sub": "mallory",
            "exp": int(time.time()) + 300,
        },
        attacker,
        algorithm="RS256",
        headers={"kid": "test-key"},
    )
    verifier = JwtVerifier(
        jwks_url=_url(jwks_server), issuer="https://idp.test", audience="batcher"
    )
    with pytest.raises(AuthenticationError):
        verifier.verify(forged)


def test_an_expired_token_is_rejected(keypair, jwks_server):
    """The other control: a well-signed token past its expiry must not pass."""
    private, _ = keypair
    stale = _token(private, exp=int(time.time()) - 10)
    verifier = JwtVerifier(
        jwks_url=_url(jwks_server), issuer="https://idp.test", audience="batcher"
    )
    with pytest.raises(AuthenticationError):
        verifier.verify(stale)


class _Discovery(BaseHTTPRequestHandler):
    """Serves an OIDC discovery document pointing at a JWKS URL, and counts fetches."""

    def do_GET(self) -> None:
        if self.path == "/.well-known/openid-configuration":
            self.server.discoveries += 1
            self._json({"issuer": self.server.issuer, "jwks_uri": self.server.jwks_uri})
        else:
            self.send_response(404)
            self.end_headers()

    def _json(self, document: dict) -> None:
        body = json.dumps(document).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        """Silence the stdlib access log."""


@pytest.fixture()
def discovery(jwks_server):
    """A localhost OIDC issuer publishing a discovery document."""
    server = HTTPServer(("127.0.0.1", 0), _Discovery)
    server.discoveries = 0
    server.jwks_uri = _url(jwks_server)
    server.issuer = ""
    threading.Thread(target=server.serve_forever, daemon=True).start()
    server.issuer = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def test_discovery_finds_the_keys_from_the_issuer_alone(keypair, discovery, jwks_server):
    """An operator names the issuer; the JWKS endpoint is looked up, not pasted in."""
    from batcher.governance.authn.verifiers import _discovered

    _discovered.clear()
    private, _ = keypair
    verifier = JwtVerifier.from_issuer(discovery.issuer, audience="batcher")

    assert verifier.jwks_url == _url(jwks_server)
    principal = verifier.verify(_token(private, iss=discovery.issuer))
    assert principal.name == "alice"


def test_discovery_happens_once_per_process(keypair, discovery):
    """The discovery document is static configuration, not a per-request lookup."""
    from batcher.governance.authn.verifiers import _discovered

    _discovered.clear()
    for _ in range(3):
        JwtVerifier.from_issuer(discovery.issuer, audience="batcher")

    assert discovery.discoveries == 1, f"discovered {discovery.discoveries} times"


def test_a_bad_issuer_fails_where_it_is_configured():
    """A misconfigured issuer must fail at construction, not on the first credential."""
    from batcher.governance.authn.verifiers import _discovered

    _discovered.clear()
    with pytest.raises(AuthenticationError, match="OIDC metadata"):
        JwtVerifier.from_issuer("http://127.0.0.1:1/nope")
