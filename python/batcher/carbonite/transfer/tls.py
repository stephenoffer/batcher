"""Load the shuffle TLS material a worker presents and trusts.

`config.ShuffleTlsConfig` holds file *paths* — the enterprise pattern is to mount
certificates as secrets on every node (a Kubernetes secret volume, cert-manager, a
cloud private-CA), never to carry PEM in the config or the plan IR. This module reads
those files at worker start into the PEM strings the native transport wants.

Carbonite owns the data-plane transport (`batcher._native` / `bc-transport`), so the
loader lives here rather than in the neutral config layer, which must not do file I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from batcher._internal.errors import ConfigError
from batcher.config.config import ShuffleTlsConfig

__all__ = ["ShuffleTlsMaterial", "load_shuffle_tls"]


@dataclass(frozen=True, slots=True)
class ShuffleTlsMaterial:
    """The PEM strings a worker's Flight server presents and its client trusts.

    `server_cert_pem`/`server_key_pem` are this node's server identity;
    `client_ca_pem` (server side) is set only under mTLS, so the server requires and
    verifies a peer's certificate. `ca_pem` is the trust root this node's *client* uses
    to verify peers, and `client_cert_pem`/`client_key_pem` (optional) are the identity
    it presents outbound under mTLS. `server_name` is the SAN verified against peers.
    """

    server_cert_pem: str
    server_key_pem: str
    client_ca_pem: str | None
    ca_pem: str
    client_cert_pem: str | None
    client_key_pem: str | None
    server_name: str


def _read(label: str, path: str) -> str:
    """Read a PEM file, raising a clear `ConfigError` if it is missing or unreadable."""
    try:
        text = Path(path).read_text()
    except OSError as exc:
        raise ConfigError(f"shuffle TLS {label}: cannot read {path!r}: {exc}") from exc
    if "-----BEGIN " not in text:
        raise ConfigError(f"shuffle TLS {label}: {path!r} is not PEM-encoded")
    return text


def load_shuffle_tls(cfg: ShuffleTlsConfig) -> ShuffleTlsMaterial | None:
    """Read the PEM files named by `cfg`, or return None when TLS is disabled.

    Called once per worker process. Raises `ConfigError` on a missing or malformed file
    so a misconfigured deployment fails loudly at startup rather than at the first fetch.

    Args:
        cfg: The distributed shuffle TLS configuration (paths + flags).

    Returns:
        The loaded PEM material, or None when `cfg.enabled` is False.

    Raises:
        ConfigError: If a referenced PEM file is missing or not PEM-encoded.
    """
    if not cfg.enabled:
        return None
    ca_pem = _read("ca_cert_path", cfg.ca_cert_path)
    client_cert = client_key = None
    if cfg.client_cert_path:
        client_cert = _read("client_cert_path", cfg.client_cert_path)
        client_key = _read("client_key_path", cfg.client_key_path)
    return ShuffleTlsMaterial(
        server_cert_pem=_read("server_cert_path", cfg.server_cert_path),
        server_key_pem=_read("server_key_path", cfg.server_key_path),
        # The same cluster CA signs both directions, so it is the client-verification
        # root on the server side under mTLS.
        client_ca_pem=ca_pem if cfg.require_client_auth else None,
        ca_pem=ca_pem,
        client_cert_pem=client_cert,
        client_key_pem=client_key,
        server_name=cfg.server_name,
    )
