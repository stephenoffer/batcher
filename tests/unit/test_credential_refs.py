"""Connection credentials by reference: ``env:NAME`` / ``file:PATH``.

The crypto-key functions have had reference indirection for a while; connection
credentials did not, even though they are the larger secret surface — most deployments
care more about a database password than a column-encryption key.

The property under test is *where the secret exists*. A reference must stay a reference on
the source object and in the pickled split, and become the secret only on the machine that
opens the connection. That is what keeps a password out of the driver's memory, off the
wire to workers, and out of any traceback or log line that renders a split.

Resolution is therefore lazy by construction: a connector resolves in `_client` /
`_driver` / `_cluster` / `_connect`, never in `__init__`. A test that only checked
"resolve_secret returns the right string" would pass even if a connector resolved eagerly
and reintroduced the leak, so these assert on the pickled bytes.
"""

from __future__ import annotations

import pickle

import pytest

from batcher._internal.errors import BackendError
from batcher.io.credentials import is_secret_ref, resolve_secret

pytestmark = pytest.mark.unit

_SECRET = "s3cret-value"


def test_literals_and_none_pass_through():
    """A raw password must keep working — this is additive, not a migration."""
    assert resolve_secret("plain-password") == "plain-password"
    assert resolve_secret(None) is None
    assert not is_secret_ref("plain-password")
    assert not is_secret_ref(None)


def test_env_reference_resolves(monkeypatch):
    monkeypatch.setenv("BATCHER_TEST_PW", _SECRET)
    assert is_secret_ref("env:BATCHER_TEST_PW")
    assert resolve_secret("env:BATCHER_TEST_PW") == _SECRET


def test_file_reference_resolves(tmp_path):
    # Trailing whitespace is stripped: a mounted secret file usually ends with a newline,
    # and sending that newline as part of the password is a baffling auth failure.
    path = tmp_path / "pw"
    path.write_text(f"{_SECRET}\n", encoding="utf-8")
    assert resolve_secret(f"file:{path}") == _SECRET


@pytest.mark.parametrize(
    ("ref", "needle"),
    [
        ("env:BATCHER_TEST_ABSENT", "BATCHER_TEST_ABSENT"),
        ("file:/nonexistent/pw", "/nonexistent/pw"),
    ],
)
def test_a_missing_reference_fails_loudly_naming_the_reference(monkeypatch, ref, needle):
    """Failing closed matters: silently resolving to empty would attempt an unauthenticated
    connection, and the error must name the *reference*, never a secret."""
    monkeypatch.delenv("BATCHER_TEST_ABSENT", raising=False)
    with pytest.raises(BackendError) as caught:
        resolve_secret(ref, what="Test credential")
    assert needle in str(caught.value)
    assert _SECRET not in str(caught.value)


def _sources():
    """One source per connector family, each holding a credential reference."""
    from batcher.io.formats.nosql.mongo import MongoSource
    from batcher.io.formats.nosql.neo4j import Neo4jSource
    from batcher.io.formats.nosql.redis import RedisSource
    from batcher.io.formats.sql.clickhouse import ClickHouseSource
    from batcher.io.formats.sql.connectorx import ConnectorXSource

    return [
        ("redis", RedisSource(host="h", password="env:BATCHER_TEST_PW")),
        (
            "neo4j",
            Neo4jSource(
                uri="bolt://h", username="u", password="env:BATCHER_TEST_PW", cypher="MATCH (n)"
            ),
        ),
        ("mongo", MongoSource(uri="env:BATCHER_TEST_PW", database="d", collection="c")),
        ("connectorx", ConnectorXSource(query="q", conn_uri="env:BATCHER_TEST_PW")),
        (
            "clickhouse",
            ClickHouseSource(query="q", host="h", database="d", password="env:BATCHER_TEST_PW"),
        ),
    ]


@pytest.mark.parametrize("name", [n for n, _ in _sources()])
def test_the_secret_never_enters_the_pickled_source(monkeypatch, name):
    """Splits are pickled to workers. The secret must not be in those bytes."""
    monkeypatch.setenv("BATCHER_TEST_PW", _SECRET)
    source = dict(_sources())[name]
    blob = pickle.dumps(source)
    assert _SECRET.encode() not in blob, f"{name} pickled its resolved secret"
    assert b"env:BATCHER_TEST_PW" in blob, f"{name} did not carry the reference"


def test_the_clickhouse_split_carries_the_reference_not_the_secret(monkeypatch):
    """ClickHouse builds its connection params on the driver and ships them on the split —
    the shape most likely to resolve eagerly by accident."""
    monkeypatch.setenv("BATCHER_TEST_PW", _SECRET)
    from batcher.io.formats.sql.clickhouse import ClickHouseSource

    split = ClickHouseSource(
        query="q", host="h", database="d", password="env:BATCHER_TEST_PW"
    )._split()
    blob = pickle.dumps(split)
    assert _SECRET.encode() not in blob
    assert b"env:BATCHER_TEST_PW" in blob


def test_nosql_secret_accessor_resolves_on_demand(monkeypatch):
    monkeypatch.setenv("BATCHER_TEST_PW", _SECRET)
    from batcher.io.formats.nosql.redis import RedisSource

    assert RedisSource(host="h", password="env:BATCHER_TEST_PW")._secret("password") == _SECRET
