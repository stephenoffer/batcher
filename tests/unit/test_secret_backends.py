"""Key-store secret references resolve against the store, on the machine that needs them.

Vault is exercised against a **real HTTP server** on localhost, because the parts worth
testing are the ones a mock removes: the KV v2 nesting, the token header, the ``#key``
selection, and the Kubernetes login exchange. The cloud stores are asserted on their
failure guidance instead, since standing up three vendor SDKs in CI would test the SDKs.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from batcher._internal.errors import BackendError
from batcher.io.credentials import is_secret_ref, resolve_secret

pytestmark = pytest.mark.unit


class _Vault(BaseHTTPRequestHandler):
    """A Vault KV v2 stand-in: one path with two keys, plus the Kubernetes login route."""

    def do_GET(self) -> None:
        self.server.seen.append((self.path, self.headers.get("X-Vault-Token")))
        if self.path == "/v1/secret/data/db":
            self._json({"data": {"data": {"password": "hunter2", "username": "svc"}}})
        elif self.path == "/v1/secret/data/solo":
            self._json({"data": {"data": {"only": "the-one"}}})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        self.server.logins.append(json.loads(self.rfile.read(length)))
        self._json({"auth": {"client_token": "exchanged-token"}})

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
def vault(monkeypatch):
    """A localhost Vault, with VAULT_ADDR pointed at it and no ambient token."""
    server = HTTPServer(("127.0.0.1", 0), _Vault)
    server.seen = []
    server.logins = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("VAULT_ADDR", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.delenv("VAULT_TOKEN", raising=False)
    monkeypatch.delenv("VAULT_K8S_ROLE", raising=False)
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def test_key_store_references_are_recognized_as_references():
    """A key-store reference must not be passed through to a connector as a password."""
    for reference in (
        "vault:secret/data/db#password",
        "aws-sm:prod/warehouse",
        "aws-ssm:/prod/pw",
        "gcp-sm:projects/p/secrets/s/versions/latest",
        "azure-kv:https://v.vault.azure.net/secrets/db",
    ):
        assert is_secret_ref(reference), reference
    # The control: something that merely contains a colon is a literal, not a reference.
    assert not is_secret_ref("postgres://user:pw@host/db")
    assert not is_secret_ref("hunter2")


def test_vault_reads_a_kv_v2_key(vault, monkeypatch):
    """The nested KV v2 shape is unwrapped and the named key selected."""
    monkeypatch.setenv("VAULT_TOKEN", "root-token")
    assert resolve_secret("vault:secret/data/db#password") == "hunter2"
    assert resolve_secret("vault:secret/data/db#username") == "svc"

    path, token = vault.seen[0]
    assert path == "/v1/secret/data/db"
    assert token == "root-token", "the token must travel as the X-Vault-Token header"


def test_vault_needs_a_key_when_the_path_holds_several(vault, monkeypatch):
    """Guessing which of two keys was meant would be worse than saying so."""
    monkeypatch.setenv("VAULT_TOKEN", "root-token")
    with pytest.raises(BackendError, match="add '#<key>'"):
        resolve_secret("vault:secret/data/db")
    # A single-key path is unambiguous, so it needs no suffix. This is the positive control
    # for the rule above: without it, the error could be unconditional and look correct.
    assert resolve_secret("vault:secret/data/solo") == "the-one"


def test_vault_names_the_key_it_could_not_find(vault, monkeypatch):
    """The error names the reference and the available keys, never the secret."""
    monkeypatch.setenv("VAULT_TOKEN", "root-token")
    with pytest.raises(BackendError) as caught:
        resolve_secret("vault:secret/data/db#nope")
    message = str(caught.value)
    assert "nope" in message and "password" in message
    assert "hunter2" not in message, "an error message must never carry the secret"


def test_vault_exchanges_a_kubernetes_token_when_there_is_none(vault, monkeypatch, tmp_path):
    """A worker with no long-lived token logs in with its projected service account token.

    This is the per-worker, credential-free path: each machine authenticates as itself
    rather than inheriting a token the driver shipped.
    """
    jwt = tmp_path / "token"
    jwt.write_text("projected-jwt")
    monkeypatch.setenv("VAULT_K8S_ROLE", "batcher-worker")
    monkeypatch.setenv("VAULT_K8S_TOKEN_PATH", str(jwt))

    assert resolve_secret("vault:secret/data/db#password") == "hunter2"
    assert vault.logins == [{"role": "batcher-worker", "jwt": "projected-jwt"}]
    assert vault.seen[-1][1] == "exchanged-token", "the exchanged token must be used"


def test_vault_says_what_is_missing_when_it_cannot_authenticate(vault):
    """No token and no Kubernetes role is a configuration error, not a traceback."""
    with pytest.raises(BackendError, match="VAULT_TOKEN"):
        resolve_secret("vault:secret/data/db#password")


def test_a_missing_vault_address_is_named():
    """Without VAULT_ADDR the reference cannot resolve, and the message says which knob."""
    import os

    prior = os.environ.pop("VAULT_ADDR", None)
    try:
        with pytest.raises(BackendError, match="VAULT_ADDR"):
            resolve_secret("vault:secret/data/db#password")
    finally:
        if prior is not None:
            os.environ["VAULT_ADDR"] = prior


def test_a_malformed_reference_is_diagnosed_before_the_sdk_is_reached():
    """A malformed reference is malformed whether or not the vendor SDK is installed.

    Checking the SDK first sent the reader to `pip install` something when the actual
    problem was the reference they wrote.
    """
    with pytest.raises(BackendError, match=r"vault\.azure\.net"):
        resolve_secret("azure-kv:just-a-name")
    with pytest.raises(BackendError, match="leading slash"):
        resolve_secret("aws-ssm:prod/warehouse/password")
    with pytest.raises(BackendError, match="Secret Manager resource"):
        resolve_secret("gcp-sm:my-secret")


def test_literals_and_none_still_pass_straight_through():
    """Adding schemes must not change what a plain password or a None does."""
    assert resolve_secret("hunter2") == "hunter2"
    assert resolve_secret(None) is None
