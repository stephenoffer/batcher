"""A table sink must be reachable from the distributed write path, not only from `write`.

`dist.executors.write._write_plan_shard` had three exits and every one of them spoke the
**file**-shard protocol: `resume` to skip a ``part-{idx}`` already present,
`max_rows_per_file` to roll one shard into several, `write_stream_shard` to encode a shard
without materializing it. A table sink implements none of those, because a database table
has no part file to skip, no file to roll over, and no encoder to stream into — it takes
rows and applies them.

So every distributed write to a table sink died inside a Ray worker with ``TypeError:
write_partitioned() got an unexpected keyword argument 'resume'``. ADBC, Snowflake and
MongoDB had shipped that way; the DB-API and operational-store sinks inherited it.

It was invisible to the gate **twice over**, which is why this file exists rather than more
cases in the suite that missed it:

* CI installs no Ray, so nothing in the PR gate ever executes `_write_plan_shard`.
* `tests/io/test_sql_distributed_write.py` — the suite whose entire subject is
  distributed-write safety — calls `sink.write_partitioned(...)` **directly**, with the two
  keywords a table sink accepts. It passed, and went on passing, against a method the real
  path could not call at all.

The lesson generalizes past this bug: a test that reproduces the *shape* of a call is not a
test of the call. So what is pinned here is the **protocol** — mechanically, over every
registered sink — plus the dispatch that routes each kind to the exit it can survive.
"""

from __future__ import annotations

import inspect

import pyarrow as pa
import pytest

from batcher.io.formats import SINKS

pytestmark = pytest.mark.unit

#: The keywords `_write_plan_shard` passes on its file-shard exits.
_FILE_SHARD_KEYWORDS = frozenset({"partition_by", "file_index", "resume", "max_rows_per_file"})

#: The keywords it passes on the table exit (`_write_table_shard`).
_TABLE_SHARD_KEYWORDS = frozenset({"file_index"})


def _sink_classes() -> list[tuple[str, type]]:
    """Every registered sink class, by name."""
    return [(name, SINKS.get(name)) for name in SINKS.names() if isinstance(SINKS.get(name), type)]


def _accepted(fn) -> tuple[set[str], bool]:
    """The keyword names `fn` accepts, and whether it absorbs any others via ``**kwargs``."""
    params = inspect.signature(fn).parameters
    names = {
        n for n, p in params.items() if n != "self" and p.kind is not inspect.Parameter.VAR_KEYWORD
    }
    absorbs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    return names, absorbs


def _is_table_sink(cls: type) -> bool:
    """The discriminator `_write_plan_shard` itself uses."""
    return not hasattr(cls, "write_stream_shard")


@pytest.mark.parametrize(
    ("name", "cls"), _sink_classes(), ids=lambda v: v if isinstance(v, str) else ""
)
def test_every_sink_can_take_the_shard_call_the_distributed_path_makes(name, cls) -> None:
    """A sink must accept exactly the keywords its own exit passes.

    Not "some" of them: a missing one is a `TypeError` from inside a Ray worker, on a path
    no local run reaches.
    """
    write_partitioned = getattr(cls, "write_partitioned", None)
    if write_partitioned is None:
        # A sink with no `write_partitioned` at all is only acceptable when it is a
        # *refusal* — `HudiSink` exists to raise a typed error naming what Batcher cannot
        # do. Anything else is a sink that would fail with an AttributeError on the first
        # shard instead.
        with pytest.raises(Exception, match=r"requires|not supported|read"):
            cls()
        return
    accepted, absorbs = _accepted(write_partitioned)
    required = _TABLE_SHARD_KEYWORDS if _is_table_sink(cls) else _FILE_SHARD_KEYWORDS
    missing = required - accepted
    assert absorbs or not missing, (
        f"{name}.write_partitioned does not accept {sorted(missing)}, which the "
        f"{'table' if _is_table_sink(cls) else 'file'} exit of "
        f"dist.executors.write._write_plan_shard passes. A distributed write to it raises "
        "TypeError inside a Ray worker."
    )


@pytest.mark.parametrize(
    ("name", "cls"), _sink_classes(), ids=lambda v: v if isinstance(v, str) else ""
)
def test_a_file_sink_also_offers_the_streaming_shard_exits(name, cls) -> None:
    """The two exits are a pair: a sink that streams a shard must also split one."""
    if _is_table_sink(cls):
        pytest.skip(f"{name} is a table sink; it has no file layout to stream or split")
    assert hasattr(cls, "write_stream_parts"), (
        f"{name} has write_stream_shard but not write_stream_parts, so a shard with a "
        "row cap has no exit at all"
    )


def test_a_table_sink_is_routed_to_the_exit_it_survives(monkeypatch) -> None:
    """The dispatch itself, not a reconstruction of it.

    This is the assertion the old suite could not make: it drives `_write_plan_shard`, so a
    keyword the real path adds and a sink does not take fails here rather than on a cluster.
    """
    from batcher.dist.executors import write as dist_write

    calls: list[dict] = []

    class _TableSink:
        """The narrow protocol a table sink implements, and nothing more."""

        def write_partitioned(self, table, path, *, partition_by=None, file_index=0):
            calls.append({"rows": table.num_rows, "path": path, "file_index": file_index})
            return []

    monkeypatch.setitem(SINKS._items, "_probe_table_sink", lambda **_kw: _TableSink())
    monkeypatch.setattr(dist_write, "engine", lambda: object())
    monkeypatch.setattr(
        "batcher.dist.executors.partition_io.read_partition_descriptor",
        lambda _p: [pa.record_batch({"id": [1, 2]})],
    )
    monkeypatch.setattr(
        dist_write, "_write_table_shard", dist_write._write_table_shard, raising=True
    )

    class _Nat:
        def execute_plan(self, _ir, batches, _cfg):
            return list(batches[0])

    monkeypatch.setattr(dist_write, "engine", lambda: _Nat())
    dist_write._write_plan_shard(
        "{}", {}, "_probe_table_sink", {}, "orders", None, 3, "{}", None, False
    )
    assert calls == [{"rows": 2, "path": "orders", "file_index": 3}]


def test_a_shard_with_no_rows_writes_nothing_to_a_table(monkeypatch) -> None:
    """An empty shard must not open a connection, let alone apply an empty overwrite."""
    from batcher.dist.executors import write as dist_write

    class _Boom:
        def write_partitioned(self, *a, **k):
            raise AssertionError("an empty shard reached the sink")

    monkeypatch.setitem(SINKS._items, "_probe_empty_sink", lambda **_kw: _Boom())
    monkeypatch.setattr(dist_write, "engine", lambda: object())
    monkeypatch.setattr(
        "batcher.dist.executors.partition_io.read_partition_descriptor", lambda _p: []
    )
    assert (
        dist_write._write_plan_shard(
            "{}", {}, "_probe_empty_sink", {}, "orders", None, 0, "{}", None, False
        )
        == []
    )


# --- the single-node exit had the same shape, discovered by catching ------------------


def test_a_sinks_own_type_error_is_not_mistaken_for_a_missing_keyword() -> None:
    """`_sink_write` used to decide "does this sink take `resume`?" by calling and catching.

    A `TypeError` has two meanings at a call site — "no such parameter" and "the callee
    raised" — and catching cannot tell them apart. A sink that applied half its rows and
    *then* raised a `TypeError` was retried without the keyword, so a database append wrote
    those rows a second time, reported success, and left duplicates nothing would explain.

    Now the capability is read off the signature. This test drives the shape that used to
    double-write: a sink whose `write` accepts `resume` and raises `TypeError` from inside.
    """
    from batcher.api.terminal.core import _sink_write

    calls: list[int] = []

    class _RaisesTypeErrorWhileWriting:
        def write(self, table, path, *, resume=False):
            calls.append(1)
            raise TypeError("something inside the sink went wrong")

    with pytest.raises(TypeError):
        _sink_write(_RaisesTypeErrorWhileWriting(), pa.table({"x": [1]}), "t", resume=False)
    assert calls == [1], f"the sink was called {len(calls)} times; a retry would duplicate rows"


def test_a_sink_without_resume_is_still_called_without_it() -> None:
    from batcher.api.terminal.core import _sink_write

    seen: dict = {}

    class _NarrowSink:
        def write(self, table, path):
            seen["path"] = path
            return "written"

    assert _sink_write(_NarrowSink(), pa.table({"x": [1]}), "orders", resume=False) == "written"
    assert seen["path"] == "orders"


def test_a_requested_resume_a_sink_cannot_honor_is_refused() -> None:
    """Silently dropping it risks duplicate ingest on a re-run, which is the whole point."""
    from batcher._internal.errors import PlanError
    from batcher.api.terminal.core import _sink_write

    class _NarrowSink:
        def write(self, table, path):
            raise AssertionError("the sink must not be called at all")

    with pytest.raises(PlanError, match=r"resume=True\) is not supported"):
        _sink_write(_NarrowSink(), pa.table({"x": [1]}), "orders", resume=True)


def test_a_sink_that_absorbs_kwargs_is_treated_as_accepting_resume() -> None:
    from batcher.api.terminal.core import _sink_write

    seen: dict = {}

    class _OpenSink:
        def write(self, table, path, **kwargs):
            seen.update(kwargs)
            return "written"

    _sink_write(_OpenSink(), pa.table({"x": [1]}), "orders", resume=True)
    assert seen == {"resume": True}
