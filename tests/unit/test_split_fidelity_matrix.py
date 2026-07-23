"""Across every format: the splits reproduce the whole-source read, exactly.

A worker never sees the source object. It receives a **pickled split** and rebuilds a
reader from its fields alone. So any constructor argument that decides *which rows* or
*which columns* a read produces, and is not carried on the split, is applied single-node
and silently dropped distributed. The same query then returns different data depending on
how it ran, with no error — the extra or missing rows are real rows from real files.

This has now been found three times, in three unrelated connectors: Iceberg's
`row_filter` (single-node 10 rows, distributed 100), Hudi's `as_of_instant` (time travel
returning the current table), and CSV's declared `schema` (byte ranges disagreeing with
each other on a column's type). It is a *class* of bug, not three incidents, and it is
invisible to every single-node test.

So this file tests the class rather than the instances: for each format, read the source
whole, then read it through pickled splits, and require the two to agree. A new connector
that forgets `_reader_kwargs` fails here on the day it is added.
"""

from __future__ import annotations

import pickle

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytestmark = pytest.mark.unit

_ROWS = 300


def _data() -> pa.Table:
    return pa.table(
        {
            "k": pa.array(list(range(_ROWS)), pa.int64()),
            "g": pa.array([f"g{i % 7}" for i in range(_ROWS)]),
        }
    )


# Each builder writes a file and returns a constructed source. Formats whose optional
# arguments change what is read are exercised *with* those arguments set, since a default
# value cannot reveal a dropped one.
def _parquet(tmp_path):
    from batcher.io.formats.structured.parquet.source import ParquetSource

    pq.write_table(_data(), str(tmp_path / "a.parquet"), row_group_size=64)
    return ParquetSource(str(tmp_path))


def _csv_declared(tmp_path):
    """CSV *with a declared schema* — the argument that was being dropped."""
    import pyarrow.csv as pacsv

    from batcher.io.formats.structured.csv import CSVSource

    pacsv.write_csv(_data(), str(tmp_path / "a.csv"))
    declared = pa.schema([("k", pa.int64()), ("g", pa.string())])
    return CSVSource(str(tmp_path), schema=declared)


def _orc(tmp_path):
    import pyarrow.orc as porc

    from batcher.io.formats.structured.orc import ORCSource

    porc.write_table(_data(), str(tmp_path / "a.orc"))
    return ORCSource(str(tmp_path))


def _arrow_ipc(tmp_path):
    import pyarrow.ipc as ipc

    from batcher.io.formats.structured.arrow_ipc import ArrowIPCSource

    table = _data()
    with ipc.new_file(str(tmp_path / "a.arrow"), table.schema) as writer:
        for batch in table.to_batches(max_chunksize=64):
            writer.write_batch(batch)
    return ArrowIPCSource(str(tmp_path))


def _avro(tmp_path):
    pytest.importorskip("fastavro")
    from batcher.io.formats.structured.avro import AvroSink, AvroSource

    AvroSink().write(_data(), str(tmp_path / "a.avro"))
    return AvroSource(str(tmp_path))


def _json(tmp_path):
    from batcher.io.formats.semistructured.json import JSONSource

    with open(tmp_path / "a.json", "w") as fh:
        for row in _data().to_pylist():
            fh.write(f'{{"k": {row["k"]}, "g": "{row["g"]}"}}\n')
    return JSONSource(str(tmp_path))


def _mcap_with_topics(tmp_path):
    """MCAP *restricted to one topic* — dropping `topics` would return every topic."""
    pytest.importorskip("mcap")
    from mcap.writer import Writer

    from batcher.io.formats.robotics import MCAPSource

    with open(tmp_path / "a.mcap", "wb") as fh:
        writer = Writer(fh)
        writer.start()
        schema_id = writer.register_schema(name="S", encoding="ros2msg", data=b"x")
        channels = {
            t: writer.register_channel(topic=t, message_encoding="cdr", schema_id=schema_id)
            for t in ("/imu", "/lidar")
        }
        for i in range(100):
            for channel in channels.values():
                writer.add_message(
                    channel_id=channel, log_time=i, publish_time=i, sequence=i, data=b"z"
                )
        writer.finish()
    return MCAPSource(str(tmp_path), topics=["/imu"])


def _tfrecord(tmp_path):
    import struct

    from batcher.io.formats.ml.tfrecord import TFRecordSource

    with open(tmp_path / "a.tfrecord", "wb") as fh:
        for i in range(_ROWS):
            payload = f"rec{i}".encode()
            fh.write(struct.pack("<Q", len(payload)))
            fh.write(struct.pack("<I", 0))
            fh.write(payload)
            fh.write(struct.pack("<I", 0))
    return TFRecordSource(str(tmp_path))


def _point_cloud(tmp_path):
    """Point cloud with a non-default column layout — `columns` decides the schema."""
    import numpy as np

    from batcher.io.formats.ml.point_cloud import PointCloudSource

    np.arange(400, dtype=np.float32).reshape(-1, 4).tofile(str(tmp_path / "a.bin"))
    return PointCloudSource(str(tmp_path), columns=("px", "py", "pz", "refl"), frame_column=None)


_BUILDERS = {
    "parquet": _parquet,
    "csv_declared_schema": _csv_declared,
    "orc": _orc,
    "arrow_ipc": _arrow_ipc,
    "avro": _avro,
    "json": _json,
    "mcap_topic_restricted": _mcap_with_topics,
    "tfrecord": _tfrecord,
    "point_cloud": _point_cloud,
}


def _rows_and_columns(batches) -> tuple[int, list[str]]:
    batches = list(batches)
    rows = sum(b.num_rows for b in batches)
    names = batches[0].schema.names if batches else []
    return rows, names


@pytest.mark.parametrize("fmt", sorted(_BUILDERS))
def test_pickled_splits_reproduce_the_whole_source_read(fmt: str, tmp_path) -> None:
    """The class of bug: a constructor argument the split forgot to carry.

    Pickling is not incidental — it is how the worker receives the split, and it is what
    proves the argument travelled rather than being reachable through a shared object.
    """
    source = _BUILDERS[fmt](tmp_path)

    whole_rows, whole_columns = _rows_and_columns(source.read())
    shipped = [pickle.loads(pickle.dumps(s)) for s in source.splits()]
    split_batches = [b for s in shipped for b in s.read()]
    split_rows, split_columns = _rows_and_columns(split_batches)

    assert split_rows == whole_rows, f"{fmt}: splits yielded {split_rows} of {whole_rows} rows"
    if whole_columns and split_columns:
        assert split_columns == whole_columns, f"{fmt}: splits produced a different schema"


@pytest.mark.parametrize("fmt", sorted(_BUILDERS))
def test_splits_are_a_cover_not_a_sample(fmt: str, tmp_path) -> None:
    """Every split must contribute, and none may duplicate the others' rows.

    A source that silently collapses to a single whole-source split still passes the
    row-count check above, so assert the shape too.
    """
    source = _BUILDERS[fmt](tmp_path)
    splits = source.splits()

    assert splits, f"{fmt}: produced no splits at all"
    per_split = [sum(b.num_rows for b in s.read()) for s in splits]
    assert sum(per_split) == sum(b.num_rows for b in source.read())


@pytest.mark.parametrize("fmt", sorted(_BUILDERS))
def test_read_and_iter_batches_agree(fmt: str, tmp_path) -> None:
    """The third path, and a third instance of the same class.

    `read()` and `iter_batches()` are two implementations of one contract, and they have
    drifted apart twice already: CSV re-inferred types in `read()` and returned a column
    the streaming path refused, and the PDF reader labelled every row with the *directory*
    in one path and the document in the other. Both are invisible unless the two are
    compared directly.
    """
    source = _BUILDERS[fmt](tmp_path)

    from_read = pa.Table.from_batches(list(source.read()))
    from_iter = pa.Table.from_batches(list(source.iter_batches()))

    assert from_read.schema == from_iter.schema, f"{fmt}: the two read paths disagree on schema"
    assert from_read.equals(from_iter), f"{fmt}: the two read paths returned different data"


@pytest.mark.parametrize("fmt", sorted(_BUILDERS))
def test_the_advertised_schema_is_what_a_read_produces(fmt: str, tmp_path) -> None:
    """`schema()` is a contract: the engine types its operators from it before reading.

    A source that advertises one type and delivers another has broken that contract, and
    the breakage surfaces far from here — as a cast error deep in an operator, or as a
    silently wrong result. CSV did exactly this (advertised `int64`, delivered `string`).
    """
    source = _BUILDERS[fmt](tmp_path)
    advertised = source.schema()

    batches = list(source.read())
    if not batches:
        return
    produced = batches[0].schema

    assert produced.names == advertised.names, f"{fmt}: column names differ from schema()"
    for name in advertised.names:
        assert produced.field(name).type == advertised.field(name).type, (
            f"{fmt}: column {name!r} is {produced.field(name).type}, "
            f"but schema() advertised {advertised.field(name).type}"
        )


def test_every_file_source_forwards_the_base_reader_options() -> None:
    """A fourth instance of the same shape: options accepted and silently ignored.

    `FileSource` gives every file format `on_error`, `schema_mode` and `files`. A subclass
    whose `__init__` calls `super().__init__(path)` — dropping them — still *accepts*
    `on_error="skip"` at the public reader, because it lands in `**opts`, and then does
    nothing with it. The user gets the default all-or-nothing behaviour while believing
    corrupt files are being skipped, and a single bad file in a directory of thousands
    fails the read.

    Excel, logs, point_cloud and protobuf each did this. Checked structurally rather than
    per format, so a new connector cannot reintroduce it.
    """
    import inspect
    import re

    import batcher  # noqa: F401 — registers the formats
    from batcher.io.base import FileSource
    from batcher.io.formats.base import SOURCES

    dropping = []
    for name in sorted(SOURCES.names()):
        cls = SOURCES.get(name)
        if not (isinstance(cls, type) and issubclass(cls, FileSource)):
            continue
        if "__init__" not in cls.__dict__:
            continue  # inherits the base signature wholesale
        call = re.search(r"super\(\)\.__init__\(([^)]*)\)", inspect.getsource(cls.__init__))
        forwarded = call.group(1) if call else ""
        if "kwargs" not in forwarded and "on_error" not in forwarded:
            dropping.append(name)

    assert not dropping, (
        f"these sources drop the base reader options (on_error/schema_mode/files) in "
        f"super().__init__: {dropping}"
    )
