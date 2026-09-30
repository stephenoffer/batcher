"""A staged SQL write publishes every shard in one transaction, or none of them.

Unstaged, a distributed write is one transaction per shard: a failure on the last shard
leaves the earlier ones published, and ``overwrite`` is refused past the first shard
because each would empty the table the others wrote. Staged, each shard writes into its
own staging table and the driver's `commit` is the single point at which the target
changes.

The shards are driven by hand through `write_partitioned(..., file_index=n)` and `commit`,
the exact calls the distributed executor makes, so the multi-shard protocol is exercised
without a cluster. What this cannot show is a real Ray run; that is recorded, not claimed.
"""

from __future__ import annotations

import sqlite3

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import BackendError
from batcher.io.formats.sql.dbapi import DBAPISink
from batcher.io.formats.sql.dbapi._staged import STAGE_MARKER
from batcher.io.manifest import WriteManifest

pytestmark = pytest.mark.io


@pytest.fixture
def uri(tmp_path):
    return f"sqlite:///{tmp_path / 'app.db'}"


def _conn(uri_: str) -> sqlite3.Connection:
    return sqlite3.connect(uri_.removeprefix("sqlite:///"))


def rows(uri_: str) -> list[tuple]:
    conn = _conn(uri_)
    try:
        return sorted(conn.execute('SELECT * FROM "orders"').fetchall())
    finally:
        conn.close()


def tables(uri_: str) -> list[str]:
    conn = _conn(uri_)
    try:
        return sorted(r[0] for r in conn.execute("SELECT name FROM sqlite_master"))
    finally:
        conn.close()


def _seed(uri_: str) -> None:
    bt.from_pydict({"id": [100], "amt": [1.0]}).write.sql("orders", uri=uri_, mode="append")


SHARDS = [pa.table({"id": [1, 2], "amt": [1.0, 2.0]}), pa.table({"id": [3], "amt": [3.0]})]


def _run_shards(sink: DBAPISink) -> WriteManifest:
    files = []
    for i, shard in enumerate(SHARDS):
        files += sink.write_partitioned(shard, "orders", file_index=i)
    return WriteManifest(tuple(files), schema=SHARDS[0].schema)


class TestAStagedOverwrite:
    def test_unstaged_it_is_refused_past_the_first_shard(self, uri):
        """The control: the refusal staging exists to lift."""
        sink = DBAPISink(uri=uri, mode="overwrite")
        with pytest.raises(BackendError, match="discard the shards before it"):
            _run_shards(sink)

    def test_nothing_changes_until_the_commit(self, uri):
        _seed(uri)
        _run_shards(DBAPISink(uri=uri, mode="overwrite", staged=True))
        assert rows(uri) == [(100, 1.0)]
        assert sum(STAGE_MARKER in t for t in tables(uri)) == 2

    def test_the_commit_publishes_every_shard_and_drops_the_stages(self, uri):
        _seed(uri)
        sink = DBAPISink(uri=uri, mode="overwrite", staged=True)
        sink.commit(_run_shards(sink), "orders")
        assert rows(uri) == [(1, 1.0), (2, 2.0), (3, 3.0)]
        assert tables(uri) == ["orders"]

    def test_an_empty_overwrite_empties_the_table(self, uri):
        _seed(uri)
        sink = DBAPISink(uri=uri, mode="overwrite", staged=True)
        written = sink.write_partitioned(SHARDS[0].slice(0, 0), "orders", file_index=0)
        sink.commit(WriteManifest(tuple(written), schema=SHARDS[0].schema), "orders")
        assert rows(uri) == []

    def test_a_failed_publish_leaves_the_target_unchanged(self, uri):
        _seed(uri)
        sink = DBAPISink(uri=uri, mode="overwrite", staged=True)
        manifest = _run_shards(sink)
        # Drop one stage behind the sink's back, so the second INSERT ... SELECT fails
        # after the DELETE and the first copy already ran inside the transaction.
        stage = manifest.files[1].stats["stage_table"]
        conn = _conn(uri)
        conn.execute(f'DROP TABLE "{stage}"')
        conn.commit()
        conn.close()
        with pytest.raises(BackendError, match="rolled back"):
            sink.commit(manifest, "orders")
        assert rows(uri) == [(100, 1.0)]
        assert tables(uri) == ["orders"]


class TestAStagedAppend:
    def test_it_appends_every_shard_at_commit(self, uri):
        _seed(uri)
        sink = DBAPISink(uri=uri, mode="append", staged=True)
        manifest = _run_shards(sink)
        assert rows(uri) == [(100, 1.0)]
        sink.commit(manifest, "orders")
        assert rows(uri) == [(1, 1.0), (2, 2.0), (3, 3.0), (100, 1.0)]

    def test_it_creates_a_missing_target(self, uri):
        sink = DBAPISink(uri=uri, mode="append", staged=True)
        sink.commit(_run_shards(sink), "orders")
        assert rows(uri) == [(1, 1.0), (2, 2.0), (3, 3.0)]


class TestThroughTheWriter:
    def test_a_staged_overwrite_through_ds_write_sql(self, uri):
        _seed(uri)
        bt.from_pydict({"id": [7], "amt": [7.0]}).write.sql(
            "orders", uri=uri, mode="overwrite", staged=True
        )
        assert rows(uri) == [(7, 7.0)]
        assert tables(uri) == ["orders"]


class TestRefusals:
    def test_a_keyed_mode_is_refused(self, uri):
        with pytest.raises(BackendError, match="staged"):
            DBAPISink(uri=uri, mode="upsert", key_columns=("id",), staged=True)

    def test_a_borrowed_connection_is_refused(self):
        conn = sqlite3.connect(":memory:")
        with pytest.raises(BackendError, match="borrowed"):
            DBAPISink(connection=conn, mode="append", staged=True)


def test_a_staged_or_sequenced_write_never_routes_to_bulk_ingest() -> None:
    from batcher.io.formats.sql.routing import write_backend

    assert write_backend("append", {"driver": "adbc_driver_sqlite"}) == "adbc"
    assert write_backend("append", {"driver": "adbc_driver_sqlite", "staged": True}) == "dbapi"
    assert write_backend("append", {"driver": "x", "sequence_by": ("s",)}) == "dbapi"
