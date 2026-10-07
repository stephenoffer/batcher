"""The Airbyte bridge: protocol-level tests over a fake message stream, then a fake connector.

The protocol tests pin the two things AP-467 asks for -- records keep their order, and a
STATE is accepted only after every record before it was taken -- without running anything.
The end-to-end tests run a small Python program that speaks the Airbyte protocol on stdout
(the shape of any connector executable), so ``discover``, the configured catalog, ``--state``
on resume, and the exit-status check run for real.
"""

from __future__ import annotations

import json
import sys
import textwrap

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import BackendError, PlanError
from batcher.io.formats.saas.airbyte import (
    AirbyteMessages,
    AirbyteSource,
    json_schema_to_arrow,
    parse_messages,
)

SCHEMA = pa.schema([("id", pa.int64()), ("tags", pa.string())])


def _record(i: int, stream: str = "users") -> dict:
    return {"type": "RECORD", "record": {"stream": stream, "data": {"id": i, "tags": [i]}}}


def _state(cursor: int) -> dict:
    return {
        "type": "STATE",
        "state": {
            "type": "STREAM",
            "stream": {"stream_descriptor": {"name": "users"}, "stream_state": {"cursor": cursor}},
        },
    }


def test_records_keep_order_and_state_is_accepted_only_after_its_records_are_taken():
    messages = [
        _record(1),
        _record(2),
        _record(99, stream="other"),
        _state(2),
        {"type": "LOG", "log": {"level": "INFO", "message": "hi"}},
        _record(3),
        _state(3),
        _record(4),
    ]
    reader = AirbyteMessages("users", SCHEMA, json_fields=frozenset({"tags"}))
    batches = reader.batches(messages)
    first = next(batches)
    assert first.column("id").to_pylist() == [1, 2]
    assert first.column("tags").to_pylist() == ["[1]", "[2]"]
    # The batch before STATE(2) has been handed out but not yet taken past: no state yet.
    assert reader.accepted == {}
    second = next(batches)
    assert reader.state_document()["messages"][0]["stream"]["stream_state"] == {"cursor": 2}
    assert second.column("id").to_pylist() == [3]
    third = next(batches)
    assert reader.accepted["stream::users"]["stream"]["stream_state"] == {"cursor": 3}
    # Record 4 comes after the last STATE: it is delivered, and covered by no checkpoint.
    assert third.column("id").to_pylist() == [4]
    assert list(batches) == []
    assert reader.accepted["stream::users"]["stream"]["stream_state"] == {"cursor": 3}


def test_global_and_legacy_state_and_batch_size():
    reader = AirbyteMessages("users", SCHEMA, json_fields=frozenset({"tags"}), batch_rows=2)
    msgs = [_record(1), _record(2), _record(3), {"type": "STATE", "state": {"data": {"x": 1}}}]
    assert [b.num_rows for b in reader.batches(msgs)] == [2, 1]
    assert reader.state_document() == {"legacy": {"x": 1}}
    glob = AirbyteMessages("users", SCHEMA)
    list(glob.batches([{"type": "STATE", "state": {"type": "GLOBAL", "global": {"s": 1}}}]))
    assert glob.state_document() == {"messages": [{"type": "GLOBAL", "global": {"s": 1}}]}


def test_trace_error_fails_the_read():
    reader = AirbyteMessages("users", SCHEMA)
    msgs = [
        _record(1),
        {"type": "TRACE", "trace": {"type": "ERROR", "error": {"message": "bad key"}}},
    ]
    with pytest.raises(BackendError, match="bad key"):
        list(reader.batches(msgs))


def test_parse_messages_and_type_mapping():
    lines = [b'{"type": "RECORD"}\n', b"junk\n", b"\n", b"[1]\n", '{"type": "STATE"}']
    assert [m["type"] for m in parse_messages(lines)] == ["RECORD", "STATE"]
    schema = json_schema_to_arrow(
        {
            "properties": {
                "a": {"type": ["null", "integer"]},
                "b": {"type": "number", "airbyte_type": "integer"},
                "c": {"type": "number"},
                "d": {"type": "boolean"},
                "e": {"type": "object"},
            }
        }
    )
    assert schema.types == [pa.int64(), pa.int64(), pa.float64(), pa.bool_(), pa.string()]


_CONNECTOR = textwrap.dedent(
    """
    import json, sys
    args = sys.argv[1:]
    verb = args[0]
    opts = dict(zip(args[1::2], args[2::2]))
    config = json.load(open(opts["--config"]))
    assert config["api_key"] == "k-123", config
    emit = lambda m: print(json.dumps(m), flush=True)
    schema = {"properties": {"id": {"type": "integer"}, "meta": {"type": "object"}}}
    if verb == "discover":
        emit({"type": "CATALOG", "catalog": {"streams": [{
            "name": "users", "json_schema": schema,
            "supported_sync_modes": ["full_refresh", "incremental"],
            "default_cursor_field": ["id"]}]}})
        sys.exit(0)
    catalog = json.load(open(opts["--catalog"]))
    log = open(config["log"], "a")
    prior = opts.get("--state") and json.load(open(opts["--state"]))
    log.write(json.dumps({"catalog": catalog, "state": prior}) + "\\n")
    start = 0
    if "--state" in opts:
        start = json.load(open(opts["--state"]))[0]["stream"]["stream_state"]["cursor"]
    for i in range(start + 1, start + 4):
        emit({"type": "RECORD", "record": {"stream": "users", "data": {"id": i, "meta": {"i": i}}}})
        if i == start + 2:
            emit({"type": "STATE", "state": {"type": "STREAM", "stream": {
                "stream_descriptor": {"name": "users"}, "stream_state": {"cursor": i}}}})
    if config.get("crash"):
        print("boom", file=sys.stderr)
        sys.exit(3)
    """
)


@pytest.fixture
def connector(tmp_path):
    script = tmp_path / "source_fake.py"
    script.write_text(_CONNECTOR)
    return [sys.executable, str(script)]


def test_connector_end_to_end_resumes_from_the_accepted_state(connector, tmp_path, monkeypatch):
    monkeypatch.setenv("AIRBYTE_FAKE_KEY", "k-123")
    log = tmp_path / "calls.jsonl"
    config = {"api_key": "env:AIRBYTE_FAKE_KEY", "log": str(log)}
    state = str(tmp_path / "ab.json")
    ds = bt.read.airbyte("users", command=connector, config=config, state=state)
    assert ds.schema.types == [pa.int64(), pa.string()]
    assert sorted(ds.to_pydict()["id"]) == [1, 2, 3]
    saved = bt.io.Incremental(state=state).load()
    assert saved["messages"][0]["stream"]["stream_state"] == {"cursor": 2}
    # Second read: the accepted state goes back with --state; record 3 (after the last
    # STATE) is read again, which is Airbyte's at-least-once contract.
    src = AirbyteSource("users", command=connector, config=config, state=state)
    assert [i for b in src.read() for i in b.column("id").to_pylist()] == [3, 4, 5]
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls[0]["state"] is None
    assert calls[0]["catalog"]["streams"][0]["sync_mode"] == "incremental"
    assert calls[-1]["state"][0]["stream"]["stream_state"] == {"cursor": 2}


def test_connector_crash_fails_the_read_and_keeps_the_state(connector, tmp_path, monkeypatch):
    monkeypatch.setenv("AIRBYTE_FAKE_KEY", "k-123")
    state = str(tmp_path / "ab.json")
    config = {"api_key": "env:AIRBYTE_FAKE_KEY", "log": str(tmp_path / "l"), "crash": True}
    src = AirbyteSource("users", command=connector, config=config, state=state)
    with pytest.raises(BackendError, match="status 3: boom"):
        src.read()
    assert bt.io.Incremental(state=state).load() is None


def test_unknown_stream_and_bad_arguments(connector, tmp_path, monkeypatch):
    monkeypatch.setenv("AIRBYTE_FAKE_KEY", "k-123")
    config = {"api_key": "env:AIRBYTE_FAKE_KEY", "log": str(tmp_path / "l")}
    with pytest.raises(BackendError, match="offers \\['users'\\]"):
        AirbyteSource("orders", command=connector, config=config).schema()
    with pytest.raises(PlanError):
        AirbyteSource("users")
    with pytest.raises(PlanError):
        AirbyteSource("users", command=connector, sync_mode="cdc")
    image = AirbyteSource("users", image="airbyte/source-faker:6")
    argv = image._argv("/tmp/w", "read")
    assert argv[:4] == ["docker", "run", "--rm", "-i"] and argv[-2:] == [
        "airbyte/source-faker:6",
        "read",
    ]
