"""Shuffle-TLS configuration, validation, and PEM loading (no Ray, no engine).

The transport-level TLS/mTLS handshakes are proven in the Rust `bc-transport` tests;
these cover the control-plane half: a half-configured deployment must fail at config
time, and the loader must read the mounted PEM files (or fail loudly on a missing one)
rather than deferring the error to the first cross-node fetch.
"""

from __future__ import annotations

import dataclasses

import pytest

from batcher._internal.errors import ConfigError
from batcher.carbonite.transfer.tls import load_shuffle_tls
from batcher.config import Config, ShuffleTlsConfig
from batcher.config.validation import validate_config

pytestmark = pytest.mark.unit


def _with_tls(tls: ShuffleTlsConfig) -> Config:
    base = Config()
    return dataclasses.replace(base, distributed=dataclasses.replace(base.distributed, tls=tls))


@pytest.fixture
def certs(tmp_path):
    """Minimal PEM-shaped files (content need not be a real cert for the loader)."""
    files = {}
    for name in ("ca", "server_cert", "server_key", "client_cert", "client_key"):
        p = tmp_path / f"{name}.pem"
        p.write_text(f"-----BEGIN CERTIFICATE-----\n{name}\n-----END CERTIFICATE-----\n")
        files[name] = str(p)
    return files


def test_tls_is_off_by_default():
    cfg = ShuffleTlsConfig()
    assert not cfg.enabled
    assert load_shuffle_tls(cfg) is None


def test_a_default_config_validates():
    validate_config(Config())  # TLS off → no TLS checks run


def test_enabling_tls_without_a_ca_is_rejected():
    with pytest.raises(ConfigError, match="ca_cert_path"):
        validate_config(_with_tls(ShuffleTlsConfig(enabled=True)))


def test_enabling_tls_without_server_identity_is_rejected(certs):
    with pytest.raises(ConfigError, match="server_cert_path"):
        validate_config(_with_tls(ShuffleTlsConfig(enabled=True, ca_cert_path=certs["ca"])))


def test_a_client_cert_without_its_key_is_rejected(certs):
    tls = ShuffleTlsConfig(
        enabled=True,
        ca_cert_path=certs["ca"],
        server_cert_path=certs["server_cert"],
        server_key_path=certs["server_key"],
        client_cert_path=certs["client_cert"],  # key missing
    )
    with pytest.raises(ConfigError, match="client_cert_path and client_key_path"):
        validate_config(_with_tls(tls))


def test_a_complete_config_validates(certs):
    tls = ShuffleTlsConfig(
        enabled=True,
        ca_cert_path=certs["ca"],
        server_cert_path=certs["server_cert"],
        server_key_path=certs["server_key"],
        require_client_auth=True,
        server_name="batcher-shuffle",
    )
    validate_config(_with_tls(tls))  # no raise


def test_loader_reads_the_mounted_pem(certs):
    tls = ShuffleTlsConfig(
        enabled=True,
        ca_cert_path=certs["ca"],
        server_cert_path=certs["server_cert"],
        server_key_path=certs["server_key"],
        client_cert_path=certs["client_cert"],
        client_key_path=certs["client_key"],
        require_client_auth=True,
        server_name="host.internal",
    )
    mat = load_shuffle_tls(tls)
    assert mat is not None
    assert "server_cert" in mat.server_cert_pem
    assert "ca" in mat.ca_pem
    assert mat.client_ca_pem is not None  # mTLS → server verifies clients against the CA
    assert mat.client_cert_pem is not None
    assert mat.server_name == "host.internal"


def test_server_auth_only_leaves_client_ca_unset(certs):
    """Without require_client_auth the server does not demand a client certificate."""
    tls = ShuffleTlsConfig(
        enabled=True,
        ca_cert_path=certs["ca"],
        server_cert_path=certs["server_cert"],
        server_key_path=certs["server_key"],
        require_client_auth=False,
    )
    mat = load_shuffle_tls(tls)
    assert mat.client_ca_pem is None


def test_a_missing_pem_file_fails_loudly(tmp_path):
    tls = ShuffleTlsConfig(
        enabled=True,
        ca_cert_path=str(tmp_path / "nope.pem"),
        server_cert_path=str(tmp_path / "nope.pem"),
        server_key_path=str(tmp_path / "nope.pem"),
    )
    with pytest.raises(ConfigError, match="cannot read"):
        load_shuffle_tls(tls)


def test_a_non_pem_file_is_rejected(tmp_path):
    junk = tmp_path / "junk.pem"
    junk.write_text("this is not a certificate")
    tls = ShuffleTlsConfig(
        enabled=True,
        ca_cert_path=str(junk),
        server_cert_path=str(junk),
        server_key_path=str(junk),
    )
    with pytest.raises(ConfigError, match="not PEM-encoded"):
        load_shuffle_tls(tls)


def test_env_overlay_reaches_the_nested_tls_section(monkeypatch, certs):
    """A platform configures the shuffle TLS entirely through env vars (12-factor).

    The nested `distributed.tls.*` path is reachable, and the fully-specified env config
    passes validation — so a deployment can enable mTLS with only environment variables.
    """
    monkeypatch.setenv("BATCHER_DISTRIBUTED_TLS_ENABLED", "true")
    monkeypatch.setenv("BATCHER_DISTRIBUTED_TLS_CA_CERT_PATH", certs["ca"])
    monkeypatch.setenv("BATCHER_DISTRIBUTED_TLS_SERVER_CERT_PATH", certs["server_cert"])
    monkeypatch.setenv("BATCHER_DISTRIBUTED_TLS_SERVER_KEY_PATH", certs["server_key"])
    monkeypatch.setenv("BATCHER_DISTRIBUTED_TLS_REQUIRE_CLIENT_AUTH", "true")
    monkeypatch.setenv("BATCHER_DISTRIBUTED_TLS_SERVER_NAME", "shuffle.svc")
    cfg = Config.from_env()  # runs validation
    assert cfg.distributed.tls.enabled
    assert cfg.distributed.tls.require_client_auth
    assert cfg.distributed.tls.server_name == "shuffle.svc"
    assert cfg.distributed.tls.ca_cert_path == certs["ca"]
