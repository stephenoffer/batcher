"""A destructive write disposition must not be applied once per shard.

A file sink gives every shard of a distributed write its own ``part-N`` file, so shards
cannot collide. A database sink has no such luxury: all shards ingest into **one** table,
and a disposition that replaces that table is applied by each shard independently. Shard 2
discards what shard 1 loaded, and the table keeps only whichever shard finished last.

Everything about this failure is the shape `CLAUDE.md` warns about. It is invisible
single-node, because there is only ever one shard. It raises nothing. And it produces a
wrong answer rather than an error, only at cluster scale. Measured before the fix: three
shards of two rows each left **two rows of six** in the table.

It bit Snowflake hardest because ``overwrite`` is its *default* — `Writer.__call__` passes
``mode="overwrite"`` and Snowflake is in `_MODE_AWARE_SINKS`, so a plain
``ds.write.snowflake("ORDERS")`` that happened to run distributed silently kept a fraction
of the rows.

The fix is a refusal rather than a repair, and the reason is worth recording: making it
work needs the destructive step to happen exactly once *before* any shard writes. Shards
run concurrently on separate workers, so electing shard 0 to do it does not work either —
an append that lands before the replace is destroyed by it. The `Sink` protocol has no
driver-side prepare hook to hang that on, so the honest outcome is a typed error naming
the safe spelling.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher._internal.errors import BackendError
from batcher.io.formats.sql.adbc import ADBCSink
from batcher.io.formats.sql.snowflake import SnowflakeSink

pytestmark = pytest.mark.unit


class _Cursor:
    def __init__(self, log: list[tuple[str, int, str | None]]) -> None:
        self._log = log

    def adbc_ingest(self, path: str, table: pa.Table, mode: str | None = None) -> None:
        self._log.append((path, table.num_rows, mode))


class _Conn:
    def __init__(self, log: list[tuple[str, int, str | None]]) -> None:
        self._log = log

    def cursor(self) -> _Cursor:
        return _Cursor(self._log)

    def commit(self) -> None:
        return None

    def close(self) -> None:
        return None


@pytest.fixture
def ingest_log(monkeypatch):
    """Records every `adbc_ingest` the sink issues, with its disposition."""
    log: list[tuple[str, int, str | None]] = []
    monkeypatch.setattr(
        "batcher.io.formats.sql.adbc.source._connect",
        lambda driver, db_kwargs, conn_kwargs: _Conn(log),
    )
    return log


def _shard(index: int) -> pa.Table:
    return pa.table({"id": [index * 10, index * 10 + 1]})


@pytest.mark.parametrize("mode", ["replace", "create"])
def test_adbc_refuses_a_destructive_mode_across_shards(ingest_log, mode) -> None:
    sink = ADBCSink(driver="d", db_kwargs={}, mode=mode)
    sink.write_partitioned(_shard(0), "orders", file_index=0)

    with pytest.raises(BackendError, match="distributed write"):
        sink.write_partitioned(_shard(1), "orders", file_index=1)


def test_adbc_single_shard_write_may_still_replace(ingest_log) -> None:
    """One shard is an ordinary write — the disposition is only unsafe when it repeats."""
    ADBCSink(driver="d", db_kwargs={}, mode="replace").write_partitioned(
        _shard(0), "orders", file_index=0
    )
    assert ingest_log == [("orders", 2, "replace")]


def test_adbc_append_is_safe_across_shards(ingest_log) -> None:
    sink = ADBCSink(driver="d", db_kwargs={}, mode="append")
    for index in range(3):
        sink.write_partitioned(_shard(index), "orders", file_index=index)
    assert [rows for _, rows, _ in ingest_log] == [2, 2, 2]
    assert {mode for *_, mode in ingest_log} == {"append"}


def test_adbc_default_mode_is_safe_across_shards(ingest_log) -> None:
    """The default must not be one that a distributed write has to refuse."""
    sink = ADBCSink(driver="d", db_kwargs={})
    for index in range(3):
        sink.write_partitioned(_shard(index), "orders", file_index=index)
    assert len(ingest_log) == 3


def test_snowflake_refuses_overwrite_across_shards() -> None:
    """Snowflake's default disposition, and the one that lost rows in practice."""
    sink = SnowflakeSink(connection_kwargs={"account": "a"}, mode="overwrite")
    with pytest.raises(BackendError, match="overwrite"):
        sink.write_partitioned(_shard(1), "ORDERS", file_index=1)


def test_snowflake_error_names_the_safe_spelling() -> None:
    """A refusal that does not say what to do instead is only half a fix."""
    sink = SnowflakeSink(connection_kwargs={"account": "a"}, mode="overwrite")
    with pytest.raises(BackendError, match="append"):
        sink.write_partitioned(_shard(2), "ORDERS", file_index=2)
