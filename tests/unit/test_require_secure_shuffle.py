"""`distributed.require_secure_shuffle` refuses a shuffle that is unauthenticated or plaintext.

The token and TLS both default off, which is right for a trusted network and is also how a
deployment carrying regulated data ends up serving it to anything that reaches a worker's
port. The switch makes "this fleet must be secured" a setting that fails closed: TLS is
checked where the fleet is spawned, before any worker starts, because the token can arrive
by environment variable or as a secret reference, and because a hardened config also runs
single-node, where there is no shuffle to refuse.

No Ray here: the spawn-time check is the function `spawn_flight_workers` calls, and it is
exercised directly.
"""

from __future__ import annotations

import dataclasses

import pytest

from batcher._internal.errors import ConfigError
from batcher.config import Config, ShuffleTlsConfig
from batcher.config.validation import validate_config
from batcher.config.validation.distributed import require_secure_shuffle

pytestmark = pytest.mark.unit

TLS = ShuffleTlsConfig(
    enabled=True,
    ca_cert_path="/etc/batcher/ca.pem",
    server_cert_path="/etc/batcher/server.pem",
    server_key_path="/etc/batcher/server.key",
)


PLAINTEXT = ShuffleTlsConfig()


def _cfg(*, require: bool, tls: ShuffleTlsConfig = PLAINTEXT) -> Config:
    base = Config()
    return dataclasses.replace(
        base,
        distributed=dataclasses.replace(base.distributed, require_secure_shuffle=require, tls=tls),
    )


def test_it_is_off_by_default():
    assert Config().distributed.require_secure_shuffle is False
    require_secure_shuffle(Config().distributed, token="")


def test_config_validation_accepts_it_without_tls():
    """A hardened config also runs single-node, where there is no shuffle to secure."""
    validate_config(_cfg(require=True))


def test_the_spawn_check_refuses_a_missing_token():
    with pytest.raises(ConfigError, match="shuffle token"):
        require_secure_shuffle(_cfg(require=True, tls=TLS).distributed, token="")


def test_the_spawn_check_refuses_missing_tls():
    with pytest.raises(ConfigError, match=r"shuffle token .* and TLS"):
        require_secure_shuffle(_cfg(require=True).distributed, token="")


def test_the_spawn_check_passes_a_secured_fleet():
    require_secure_shuffle(_cfg(require=True, tls=TLS).distributed, token="s3cret")


def test_the_spawn_path_calls_the_check():
    """A guard nothing calls guards nothing; pin the call site in the fleet spawner."""
    import inspect

    from batcher.dist import flight_worker

    source = inspect.getsource(flight_worker.spawn_flight_workers)
    assert "require_secure_shuffle(dc, token)" in source


def test_hardened_turns_every_switch_on():
    cfg = Config().hardened(audit_path="/tmp/audit.jsonl")
    assert cfg.governance.mode == "strict"
    assert cfg.governance.require_verified_principal is True
    assert cfg.governance.audit_path == "/tmp/audit.jsonl"
    assert cfg.execution.udf_isolation == "strict"
    assert cfg.distributed.require_secure_shuffle is True
    validate_config(cfg)


def test_hardened_keeps_unrelated_settings():
    base = Config()
    base = base.replace(execution=dataclasses.replace(base.execution, morsel_rows=4096))
    assert base.hardened(audit_path="/tmp/a.jsonl").execution.morsel_rows == 4096
