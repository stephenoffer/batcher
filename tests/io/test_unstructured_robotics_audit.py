"""Audit pins for the unstructured + robotics IO readers.

Each test here fails against the pre-fix code and passes after. The theme is
**identity collisions**: a source restricted to a subset of a path's data (a topic /
signal / suffix / encoding) is a *different relation* — different rows, different row
count — yet the pre-fix `identity()` handed every restriction the same stats key, so
Kyber would serve one restriction's cardinalities for another (the same silent bug the
SQL sources fold their connection into `connection_fingerprint` to avoid).

The MCAP tests round-trip a real file (the `mcap` package is a light, pure-Python dep);
the text/binary tests use real files and need no driver. Identity is a pure function, so
the collision pins need no file at all.
"""

from __future__ import annotations

import importlib.util

import pytest

from batcher.io.formats.robotics.mcap import MCAP_SCHEMA, MCAPSource
from batcher.io.formats.robotics.mdf import MDFSource
from batcher.io.formats.unstructured.binary import BinarySource
from batcher.io.formats.unstructured.text import TextSource

pytestmark = pytest.mark.unit

_HAS_MCAP = importlib.util.find_spec("mcap") is not None


# --------------------------------------------------------------------------- #
# Identity collisions (pure functions — no file, no driver)                    #
# --------------------------------------------------------------------------- #
def test_mcap_identity_distinct_per_topic_set() -> None:
    """Two topic restrictions on one log are different relations → different keys."""
    gps = MCAPSource("drive.mcap", topics=["/gps"])
    lidar = MCAPSource("drive.mcap", topics=["/lidar"])
    assert gps.identity() != lidar.identity()  # fails pre-fix: both were "mcap:drive.mcap"


def test_mcap_identity_topic_order_independent() -> None:
    """Topic order does not change which rows are read, so it must not change the key."""
    a = MCAPSource("drive.mcap", topics=["/a", "/b"])
    b = MCAPSource("drive.mcap", topics=["/b", "/a"])
    assert a.identity() == b.identity()


def test_mcap_identity_unrestricted_unchanged() -> None:
    """The whole-log key stays the plain ``format:path``."""
    assert MCAPSource("drive.mcap").identity() == "mcap:drive.mcap"


def test_mcap_identity_topic_vs_whole_distinct() -> None:
    """A one-topic read is a different relation from the whole log."""
    restricted = MCAPSource("drive.mcap", topics=["/gps"]).identity()
    assert restricted != MCAPSource("drive.mcap").identity()


def test_mdf_identity_distinct_per_signal_set() -> None:
    """Two signal restrictions on one measurement are different relations → different keys."""
    a = MDFSource("run.mf4", signals=["EngineRPM"])
    b = MDFSource("run.mf4", signals=["VehicleSpeed"])
    assert a.identity() != b.identity()  # fails pre-fix: identical inherited FileSource key


def test_mdf_identity_signal_order_independent() -> None:
    """Signal order does not change the read, so it must not change the key."""
    a = MDFSource("run.mf4", signals=["A", "B"])
    b = MDFSource("run.mf4", signals=["B", "A"])
    assert a.identity() == b.identity()


def test_binary_identity_distinct_per_suffix() -> None:
    """Different suffixes expand a path to different files → different relations → keys."""
    jpg = BinarySource("corpus/", suffix=".jpg")
    png = BinarySource("corpus/", suffix=".png")
    assert jpg.identity() != png.identity()  # fails pre-fix: both were "binary:corpus/"


def test_binary_identity_default_unchanged() -> None:
    """The match-all (no suffix) key stays the plain ``binary:path``."""
    assert BinarySource("corpus/").identity() == "binary:corpus/"


def test_text_identity_distinct_per_encoding() -> None:
    """The same path decoded under different encodings is different text → different key."""
    u = TextSource("log.txt", encoding="utf-8")
    latin = TextSource("log.txt", encoding="latin-1")
    assert u.identity() != latin.identity()  # fails pre-fix: encoding omitted from key


def test_text_identity_utf8_aliases_unchanged() -> None:
    """The utf-8 default (however spelled) keeps the common ``text:mode:path`` key."""
    assert TextSource("log.txt", encoding="UTF-8").identity() == TextSource("log.txt").identity()
    assert TextSource("log.txt", encoding="utf8").identity() == TextSource("log.txt").identity()


# --------------------------------------------------------------------------- #
# Restrictions really are different relations (real reads back the above)      #
# --------------------------------------------------------------------------- #
def _write_mcap(path: str) -> None:
    from mcap.writer import Writer

    with open(path, "wb") as buf:
        w = Writer(buf)
        w.start()
        sid = w.register_schema(name="std_msgs/String", encoding="jsonschema", data=b"{}")
        gps = w.register_channel(topic="/gps", message_encoding="json", schema_id=sid)
        lidar = w.register_channel(topic="/lidar", message_encoding="json", schema_id=sid)
        for i in range(5):
            w.add_message(
                channel_id=gps,
                log_time=i * 1000,
                data=b"GPS" + str(i).encode(),
                publish_time=i * 1000,
                sequence=i,
            )
        for i in range(3):
            # A non-UTF-8 binary payload: proves the reader keeps it raw, never decodes it.
            w.add_message(
                channel_id=lidar,
                log_time=i * 1000,
                data=b"\x00\x01\x02LIDAR",
                publish_time=i * 1000,
                sequence=i,
            )
        w.finish()


@pytest.mark.skipif(not _HAS_MCAP, reason="mcap package not installed")
def test_mcap_topic_restriction_changes_rows_and_stats(tmp_path) -> None:
    """A topic-restricted read yields fewer rows AND a smaller `statistics()` count.

    This is what makes the shared identity a real bug: the two sources genuinely differ
    in row count, so caching stats under one key corrupts the other's plan.
    """
    import pyarrow as pa

    p = str(tmp_path / "drive.mcap")
    _write_mcap(p)

    whole = pa.Table.from_batches(MCAPSource(p).read())
    gps = pa.Table.from_batches(MCAPSource(p, topics=["/gps"]).read())

    assert whole.num_rows == 8
    assert gps.num_rows == 5
    assert set(gps.column("topic").to_pylist()) == {"/gps"}

    assert MCAPSource(p).statistics().row_count == 8
    assert MCAPSource(p, topics=["/gps"]).statistics().row_count == 5


@pytest.mark.skipif(not _HAS_MCAP, reason="mcap package not installed")
def test_mcap_keeps_payload_raw_and_binary(tmp_path) -> None:
    """The `data` column is the raw payload, `large_binary`, never decoded to a message."""
    import pyarrow as pa

    p = str(tmp_path / "drive.mcap")
    _write_mcap(p)
    table = pa.Table.from_batches(MCAPSource(p).read())

    assert table.schema.field("data").type == pa.large_binary()
    assert MCAP_SCHEMA.field("data").type == pa.large_binary()
    lidar = table.filter(pa.compute.equal(table.column("topic"), "/lidar"))
    assert lidar.column("data").to_pylist()[0] == b"\x00\x01\x02LIDAR"


def test_binary_suffix_selects_different_files(tmp_path) -> None:
    """Different suffixes read disjoint file sets, so they are different relations."""
    import pyarrow as pa

    (tmp_path / "a.jpg").write_bytes(b"jpeg-bytes")
    (tmp_path / "b.png").write_bytes(b"png-bytes")

    jpg = pa.Table.from_batches(BinarySource(str(tmp_path), suffix=".jpg").read())
    png = pa.Table.from_batches(BinarySource(str(tmp_path), suffix=".png").read())

    assert [u.rsplit("/", 1)[-1] for u in jpg.column("uri").to_pylist()] == ["a.jpg"]
    assert [u.rsplit("/", 1)[-1] for u in png.column("uri").to_pylist()] == ["b.png"]


def test_text_encoding_changes_decoded_content(tmp_path) -> None:
    """The same bytes decoded under different encodings produce different text."""
    import pyarrow as pa

    # 0xE9 is 'é' in latin-1 but an invalid lone byte / different char under utf-8.
    (tmp_path / "note.txt").write_bytes(b"caf\xe9\n")

    latin = pa.Table.from_batches(TextSource(str(tmp_path), encoding="latin-1").read())
    assert latin.column("text").to_pylist() == ["café"]
    # Same path, different encoding -> different decoded text; hence a different relation
    # and (post-fix) a different identity.
    assert (
        TextSource(str(tmp_path), encoding="latin-1").identity()
        != TextSource(str(tmp_path), encoding="utf-8").identity()
    )
