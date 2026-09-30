"""A repeated key in a keyed SQL write gets a winner chosen by value, not by frame order.

A keyed DML write binds one statement per row, so without `sequence_by` the row that runs
last wins. That is frame order, which a distributed write does not fix: two shards holding
one key each commit their own winner, and which commits last decides the table. With
`sequence_by` the greatest row per key is written -- across the whole frame for a batch
write (a mergeable `distinct(keep="last")` before the sink), and per micro-batch for a
stream, which is its unit of commit.

Frame order is made adversarial here on purpose: the winning row comes *first*, so an
implementation that fell back to "last row wins" would write the loser.
"""

from __future__ import annotations

import logging
import sqlite3

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import BackendError
from batcher.io.formats.sql.dbapi import DBAPISink
from batcher.io.formats.sql.dbapi._bind import duplicate_key_rows, latest_per_key

pytestmark = pytest.mark.io


@pytest.fixture
def uri(tmp_path):
    return f"sqlite:///{tmp_path / 'app.db'}"


def rows(uri_: str) -> list[tuple]:
    conn = sqlite3.connect(uri_.removeprefix("sqlite:///"))
    try:
        return sorted(conn.execute('SELECT * FROM "orders"').fetchall())
    finally:
        conn.close()


FRAME = {"id": [1, 1, 2, 2], "seq": [9, 3, 1, 5], "amt": [90.0, 30.0, 10.0, 50.0]}


class TestSequenceBy:
    def test_the_greatest_sequence_wins_however_the_frame_is_ordered(self, uri):
        bt.from_pydict(FRAME).write.sql(
            "orders", uri=uri, mode="upsert", key_columns="id", sequence_by="seq"
        )
        assert rows(uri) == [(1, 9, 90.0), (2, 5, 50.0)]

    def test_without_it_frame_order_decides(self, uri):
        """The control: the same frame, no `sequence_by`, keeps the last row per key."""
        bt.from_pydict(FRAME).write.sql("orders", uri=uri, mode="upsert", key_columns="id")
        assert rows(uri) == [(1, 3, 30.0), (2, 5, 50.0)]

    def test_the_winner_holds_across_several_input_files(self, uri, tmp_path):
        """Each file is its own partition, so the loser of key 1 is last in scan order."""
        base = tmp_path / "src"
        base.mkdir()
        bt.from_pydict({"id": [1], "seq": [9], "amt": [90.0]}).write(str(base / "a.parquet"))
        bt.from_pydict({"id": [1], "seq": [3], "amt": [30.0]}).write(str(base / "b.parquet"))
        bt.read.parquet(str(base)).write.sql(
            "orders", uri=uri, mode="upsert", key_columns="id", sequence_by="seq"
        )
        assert rows(uri) == [(1, 9, 90.0)]

    def test_a_delete_insert_with_it_writes_one_row_per_key(self, uri):
        bt.from_pydict(FRAME).write.sql(
            "orders", uri=uri, mode="delete_insert", key_columns="id", sequence_by="seq"
        )
        assert rows(uri) == [(1, 9, 90.0), (2, 5, 50.0)]

    def test_a_null_sequence_never_beats_a_non_null_one(self, uri):
        frame = {"id": [1, 1, 2, 2], "seq": [2, None, None, 1], "amt": [2.0, 0.0, 0.0, 1.0]}
        bt.from_pydict(frame).write.sql(
            "orders", uri=uri, mode="upsert", key_columns="id", sequence_by="seq"
        )
        assert rows(uri) == [(1, 2, 2.0), (2, 1, 1.0)]

    def test_the_sink_and_the_dataset_dedupe_agree_on_nulls(self):
        frame = {"id": [1, 1, 2, 2], "seq": [2, None, None, 1], "amt": [2.0, 0.0, 0.0, 1.0]}
        engine = bt.from_pydict(frame).distinct(["id"], keep="last", order_by="seq")
        sink = latest_per_key(pa.table(frame), ("id",), ("seq",))
        assert engine.sort("id").to_pydict() == sink.sort_by("id").to_pydict()

    def test_an_unkeyed_mode_is_refused(self, uri):
        with pytest.raises(BackendError, match="keyed mode"):
            bt.from_pydict(FRAME).write.sql("orders", uri=uri, mode="append", sequence_by="seq")

    def test_the_sink_alone_applies_it_to_one_shard(self, uri):
        """The per-micro-batch path: the sink, with no dataset-level dedupe in front of it."""
        sink = DBAPISink(uri=uri, mode="upsert", key_columns=("id",), sequence_by="seq")
        written = sink.write(pa.table(FRAME), "orders")
        assert written.rows == 2
        assert rows(uri) == [(1, 9, 90.0), (2, 5, 50.0)]


class TestTheHelpers:
    def test_a_tie_on_the_sequence_falls_back_to_frame_order(self):
        t = pa.table({"id": [1, 1], "seq": [1, 1], "v": ["a", "b"]})
        assert latest_per_key(t, ("id",), ("seq",)).to_pydict()["v"] == ["b"]

    def test_a_composite_key_and_sequence(self):
        t = pa.table({"a": [1, 1, 1], "b": [1, 1, 2], "s1": [1, 1, 0], "s2": [5, 7, 0]})
        out = latest_per_key(t, ("a", "b"), ("s1", "s2")).to_pydict()
        assert out == {"a": [1, 1], "b": [1, 2], "s1": [1, 0], "s2": [7, 0]}

    def test_survivors_keep_their_frame_order(self):
        t = pa.table({"id": [3, 1, 2, 1], "seq": [0, 0, 0, 1]})
        assert latest_per_key(t, ("id",), ("seq",)).column("id").to_pylist() == [3, 2, 1]

    def test_duplicate_count(self):
        assert duplicate_key_rows(pa.table({"id": [1, 1, 1, 2]}), ("id",)) == 2
        assert duplicate_key_rows(pa.table({"id": [1, 2]}), ("id",)) == 0


def test_a_repeated_key_without_sequence_by_is_reported(uri, caplog):
    with caplog.at_level(logging.WARNING):
        bt.from_pydict(FRAME).write.sql("orders", uri=uri, mode="upsert", key_columns="id")
    assert any("sequence_by" in r.getMessage() for r in caplog.records)


def test_unique_keys_are_not_reported(uri, caplog):
    with caplog.at_level(logging.WARNING):
        bt.from_pydict({"id": [1, 2], "seq": [1, 1], "amt": [1.0, 2.0]}).write.sql(
            "orders", uri=uri, mode="upsert", key_columns="id"
        )
    assert not any("sequence_by" in r.getMessage() for r in caplog.records)
