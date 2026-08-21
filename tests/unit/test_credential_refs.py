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


# --- `cmd:`, the external key store -----------------------------------------------------------


def test_a_cmd_reference_takes_the_helpers_stdout(monkeypatch, tmp_path):
    # One knob reaches `vault kv get`, `aws secretsmanager get-secret-value`, or a bespoke
    # fetcher, without this package taking a dependency on any of them.
    helper = tmp_path / "fetch.sh"
    helper.write_text('#!/bin/sh\nprintf "%s" "secret-for-$1"\n')
    helper.chmod(0o755)
    monkeypatch.setenv("BATCHER_SECRET_COMMAND", str(helper))
    assert resolve_secret("cmd:db-password") == "secret-for-db-password"


def test_cmd_is_inert_until_an_operator_configures_the_program(monkeypatch):
    # The security story: a plan is data and may arrive from somewhere less trusted than the
    # cluster, so the reference names the *secret*, never the program.
    monkeypatch.delenv("BATCHER_SECRET_COMMAND", raising=False)
    with pytest.raises(BackendError) as exc:
        resolve_secret("cmd:db-password")
    assert "BATCHER_SECRET_COMMAND" in str(exc.value)


def test_a_cmd_reference_cannot_inject_a_second_command(monkeypatch, tmp_path):
    # The helper is run as an argv list, never through a shell, so a metacharacter in the
    # reference is an ordinary character in the argument.
    seen = tmp_path / "argument"
    helper = tmp_path / "fetch.sh"
    helper.write_text(f'#!/bin/sh\nprintf "%s" "$1" > {seen}\nprintf "ok"\n')
    helper.chmod(0o755)
    monkeypatch.setenv("BATCHER_SECRET_COMMAND", str(helper))
    assert resolve_secret("cmd:name; touch /tmp/pwned") == "ok"
    assert seen.read_text() == "name; touch /tmp/pwned"


def test_a_failing_helper_names_the_reference_and_its_stderr_not_its_stdout(monkeypatch, tmp_path):
    # stdout is the secret, so it must never reach an error message even on failure.
    helper = tmp_path / "fetch.sh"
    helper.write_text('#!/bin/sh\nprintf "%s" "leaked-secret"\necho "no such key" >&2\nexit 3\n')
    helper.chmod(0o755)
    monkeypatch.setenv("BATCHER_SECRET_COMMAND", str(helper))
    with pytest.raises(BackendError) as exc:
        resolve_secret("cmd:absent", what="ClickHouse password")
    message = str(exc.value)
    assert "ClickHouse password" in message and "cmd:absent" in message
    assert "no such key" in message
    assert "leaked-secret" not in message


def test_an_empty_answer_is_an_error_not_an_empty_password(monkeypatch, tmp_path):
    # A helper that silently returns nothing would otherwise become an empty credential, and
    # the connection failure that follows names the database rather than the secret.
    helper = tmp_path / "fetch.sh"
    helper.write_text("#!/bin/sh\nexit 0\n")
    helper.chmod(0o755)
    monkeypatch.setenv("BATCHER_SECRET_COMMAND", str(helper))
    with pytest.raises(BackendError):
        resolve_secret("cmd:blank")


def test_a_missing_helper_program_is_reported_not_raised_as_oserror(monkeypatch, tmp_path):
    monkeypatch.setenv("BATCHER_SECRET_COMMAND", str(tmp_path / "absent"))
    with pytest.raises(BackendError):
        resolve_secret("cmd:anything")


def test_a_cmd_reference_is_recognized_as_a_reference():
    assert is_secret_ref("cmd:name") is True
