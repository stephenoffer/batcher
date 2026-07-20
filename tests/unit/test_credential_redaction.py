"""Connector credentials must never reach a `repr()`.

Every SQL/NoSQL connector is a frozen dataclass holding its credentials as ordinary
fields, and their docstrings promise the secret is "never logged". The generated
dataclass `__repr__` broke that promise: it printed passwords, tokens, and full
connection URIs verbatim. The leak is not hypothetical — split objects are pickled and
sent to Ray workers, so any exception inside a worker renders a frame whose locals
include the split, putting a live database password in the driver traceback and in
whatever aggregates those logs.

This scans the connector surface rather than listing known offenders, so a *new*
connector cannot reintroduce the leak by simply not being in a hand-written list.
"""

from __future__ import annotations

import dataclasses
import importlib
import pkgutil
import warnings

import pytest

pytestmark = pytest.mark.unit

# Substrings that mark a field as carrying authentication material. `*_kwargs` counts:
# Snowflake and ClickHouse pass the password inside their connect-kwargs dict.
_SECRET_HINTS = (
    "password",
    "token",
    "api_key",
    "secret",
    "passphrase",
    "conn_uri",
    "credential",
    "connection_kwargs",
    "client_kwargs",
    # ADBC carries the DSN/URI *and* the password in `db_kwargs`, and any driver
    # option in `conn_kwargs`. These matched no hint above, so the connector with the
    # richest credential dict was the one the scan did not cover.
    "db_kwargs",
    "conn_kwargs",
    # An ODBC connection string is `SERVER=…;UID=…;PWD=…` — it carries the password in
    # its body rather than in a field named for it, so nothing above matched it. `dsn`
    # accepts the same syntax. This one was leaking into `identity()` too, which is
    # *persisted* to the stats store rather than merely printed.
    "connection_string",
    "dsn",
)

# Fields whose name matches a hint but which hold no secret. `resume_token` is a stream
# position and `BrokerMessage.key` is a message key — both are data, and redacting them
# would cost real debuggability for no security gain.
_NOT_SECRETS = {
    ("BrokerMessage", "resume_token"),
}


def _connector_dataclasses():
    """Every dataclass defined under `batcher.io.formats`, with its module name."""
    import batcher.io.formats as formats

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for info in pkgutil.walk_packages(formats.__path__, formats.__name__ + "."):
            try:
                module = importlib.import_module(info.name)
            except Exception:  # pragma: no cover - optional driver absent
                continue
            for name in dir(module):
                obj = getattr(module, name)
                if dataclasses.is_dataclass(obj) and getattr(obj, "__module__", None) == info.name:
                    yield info.name, obj


def test_no_credential_field_is_rendered_in_repr():
    leaking = [
        f"{module}:{cls.__name__}.{f.name}"
        for module, cls in _connector_dataclasses()
        for f in dataclasses.fields(cls)
        if f.repr
        and any(h in f.name.lower() for h in _SECRET_HINTS)
        and (cls.__name__, f.name) not in _NOT_SECRETS
    ]
    assert not leaking, (
        "these credential fields are rendered by the generated dataclass __repr__ and "
        "will appear in tracebacks and logs — mark each `field(repr=False)`: "
        f"{sorted(leaking)}"
    )


@pytest.mark.parametrize(
    ("build", "secret"),
    [
        (
            lambda: importlib.import_module("batcher.io.formats.sql.connectorx").ConnectorXSource(
                query="select 1", conn_uri="postgres://u:PW_SECRET@h/db"
            ),
            "PW_SECRET",
        ),
        (
            lambda: importlib.import_module("batcher.io.formats.sql.clickhouse").ClickHouseSource(
                query="q", host="h", database="d", password="PW_SECRET"
            ),
            "PW_SECRET",
        ),
        (
            lambda: importlib.import_module("batcher.io.formats.sql.snowflake").SnowflakeSource(
                query="q", connection_kwargs={"password": "PW_SECRET"}
            ),
            "PW_SECRET",
        ),
        (
            lambda: importlib.import_module("batcher.io.formats.sql.databricks").DatabricksSource(
                table="t", workspace="w", token="PW_SECRET"
            ),
            "PW_SECRET",
        ),
    ],
)
def test_secret_value_is_absent_from_rendered_repr(build, secret):
    """The end-to-end check: the actual secret string must not appear in the output."""
    assert secret not in repr(build())
