"""`FileSink.write_stream_parts` — a row-capped write that never materializes the result.

A `max_rows_per_file` write used to require the whole table on the driver first, which is
backwards: the caller capping file size is usually the caller whose result does not fit.
These tests pin the rollover behavior at the sink level, where no engine is involved:
the cap is honored, every row survives in order, and a resumed write keeps each row in
the part file it was already in.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from batcher.io import ParquetSink

pytestmark = pytest.mark.integration


def _batches(total: int, per_batch: int) -> list[pa.RecordBatch]:
    table = pa.table({"v": list(range(total))})
    return list(table.to_batches(max_chunksize=per_batch))


def _read_dir(path) -> list[int]:
    out: list[int] = []
    for f in sorted(path.rglob("*.parquet")):
        out.extend(pq.read_table(str(f)).column("v").to_pylist())
    return out


def test_the_cap_is_honored_and_every_row_survives(tmp_path):
    out = tmp_path / "out"
    written = ParquetSink().write_stream_parts(
        iter(_batches(1000, 128)), str(out), max_rows_per_file=300
    )
    assert [w.rows for w in written] == [300, 300, 300, 100]
    files = sorted(out.rglob("*.parquet"))
    assert len(files) == 4
    assert max(pq.read_table(str(f)).num_rows for f in files) <= 300
    assert _read_dir(out) == list(range(1000))


def test_a_batch_straddling_the_cap_is_split_not_overflowed(tmp_path):
    # 400-row batches against a 300-row cap: every boundary falls mid-batch.
    out = tmp_path / "out"
    written = ParquetSink().write_stream_parts(
        iter(_batches(1200, 400)), str(out), max_rows_per_file=300
    )
    assert [w.rows for w in written] == [300] * 4
    assert _read_dir(out) == list(range(1200))


def test_a_cap_larger_than_the_stream_writes_one_file(tmp_path):
    out = tmp_path / "out"
    written = ParquetSink().write_stream_parts(
        iter(_batches(10, 4)), str(out), max_rows_per_file=1000
    )
    assert len(written) == 1
    assert written[0].rows == 10
    assert _read_dir(out) == list(range(10))


def test_an_empty_stream_still_writes_a_readable_empty_part(tmp_path):
    out = tmp_path / "out"
    schema = pa.schema([("v", pa.int64())])
    written = ParquetSink().write_stream_parts(
        iter([]), str(out), max_rows_per_file=100, schema=schema
    )
    assert len(written) == 1
    files = sorted(out.rglob("*.parquet"))
    assert len(files) == 1
    table = pq.read_table(str(files[0]))
    assert table.num_rows == 0
    assert table.schema.names == ["v"]


def test_empty_batches_in_the_stream_do_not_produce_empty_files(tmp_path):
    out = tmp_path / "out"
    schema = pa.schema([("v", pa.int64())])
    empty = pa.RecordBatch.from_pylist([], schema=schema)
    stream = [empty, *_batches(50, 25), empty]
    written = ParquetSink().write_stream_parts(iter(stream), str(out), max_rows_per_file=100)
    assert [w.rows for w in written] == [50]
    assert _read_dir(out) == list(range(50))


def test_resume_keeps_finished_parts_and_leaves_rows_in_the_same_file(tmp_path):
    out = tmp_path / "out"
    sink = ParquetSink()
    sink.write_stream_parts(iter(_batches(1000, 100)), str(out), max_rows_per_file=300)
    first = {f.name: pq.read_table(str(f)).column("v").to_pylist() for f in out.rglob("*.parquet")}

    # Re-run with resume: the already-present parts are skipped, and their rows are
    # drained from the stream so nothing slides into a later file.
    written = sink.write_stream_parts(
        iter(_batches(1000, 100)), str(out), max_rows_per_file=300, resume=True
    )
    assert [w.rows for w in written] == [300, 300, 300, 100]
    second = {f.name: pq.read_table(str(f)).column("v").to_pylist() for f in out.rglob("*.parquet")}
    assert second == first
    assert _read_dir(out) == list(range(1000))


def test_resume_after_a_partial_run_completes_the_remaining_parts(tmp_path):
    out = tmp_path / "out"
    sink = ParquetSink()
    # A run that died after two parts: write only the first 600 rows.
    sink.write_stream_parts(iter(_batches(600, 100)), str(out), max_rows_per_file=300)
    assert len(sorted(out.rglob("*.parquet"))) == 2

    sink.write_stream_parts(iter(_batches(1000, 100)), str(out), max_rows_per_file=300, resume=True)
    assert len(sorted(out.rglob("*.parquet"))) == 4
    assert _read_dir(out) == list(range(1000))


@pytest.mark.parametrize("fmt", ["parquet", "csv", "json", "arrow", "orc", "avro"])
def test_every_file_sink_rolls_over_at_the_cap(tmp_path, fmt):
    """The rollover is the base class's, so a format that overrides `write_stream`
    (NDJSON straight-through, the CSV window) must still honor the cap."""
    from batcher.io.formats.base import SOURCES
    from batcher.io.sink import SINKS

    sink = SINKS.get(fmt)()
    out = tmp_path / fmt
    written = sink.write_stream_parts(iter(_batches(1000, 128)), str(out), max_rows_per_file=300)
    assert [w.rows for w in written] == [300, 300, 300, 100]
    assert len(sorted(out.iterdir())) == 4
    # Read back through the matching source (no engine involved), so every part is
    # confirmed to be a valid, complete file of that format rather than merely present.
    got = sorted(
        v
        for w in written
        for batch in SOURCES.get(fmt)(w.path).read()
        for v in batch.column("v").to_pylist()
    )
    assert got == list(range(1000))


def _refuse_collect(monkeypatch, why: str) -> None:
    """Make `_collect` fatal, so any materialization on the driver fails the test loudly."""
    import batcher.api.terminal.core as terminal

    def _no_collect(*_args, **_kwargs):
        raise AssertionError(why)

    monkeypatch.setattr(terminal, "_collect", _no_collect)


@pytest.mark.parametrize(
    ("shape", "expect_rows"),
    [("sort", 900), ("group_by", 9)],
)
def test_a_capped_write_of_a_breaker_plan_also_streams(tmp_path, monkeypatch, shape, expect_rows):
    """A breaker under a row cap streams too, not only a breaker-free pipeline.

    `iter_batches` yields a top-level aggregate by folding one running state, and a
    top-level sort from the out-of-core bucket pipeline — both in bounded memory. The write
    path nonetheless asked only whether the plan was breaker-*free*, so `sort().write(...)`
    and `group_by().agg().write(...)` collected the whole result onto the driver even under
    a cap that says, in as many words, that it does not fit there.
    """
    import batcher as bt

    src = str(tmp_path / "src.parquet")
    pq.write_table(
        pa.table({"v": list(range(1000)), "g": [i % 9 for i in range(1000)]}),
        src,
        row_group_size=100,
    )
    _refuse_collect(monkeypatch, f"the capped {shape} write materialized on the driver")

    ds = bt.read.parquet(src).filter(bt.col("v") < 900)
    ds = ds.sort("v") if shape == "sort" else ds.group_by("g").agg(n=bt.count())
    out = str(tmp_path / "out")
    manifest = ds.write.parquet(out, max_rows_per_file=250)
    # The guard has done its job; reading the result back is an ordinary collect.
    monkeypatch.undo()

    assert manifest.total_rows == expect_rows
    back = bt.read.parquet(out)
    assert back.count() == expect_rows
    if shape == "sort":
        # The sort's answer survives the rollover. Compared as a multiset, because which
        # rows land in which part file is the sink's business and the glob order is not the
        # write order — what this asserts is that no row was lost or duplicated.
        assert sorted(back.to_pydict()["v"]) == list(range(900))
    else:
        assert sum(back.to_pydict()["n"]) == 900


def test_a_capped_write_over_a_file_source_never_materializes_the_result(tmp_path, monkeypatch):
    """The routing, not just the rollover.

    A `max_rows_per_file` write used to be excluded from the streaming path, so asking for
    a file size *caused* the whole result to be collected onto the driver first. Proving
    that is now the streaming path is a matter of making `_collect` unusable and watching
    the write succeed anyway.
    """
    import pyarrow.parquet as pq

    import batcher as bt
    import batcher.api.terminal.core as terminal

    src = str(tmp_path / "src.parquet")
    pq.write_table(pa.table({"v": list(range(1000))}), src, row_group_size=100)

    def _no_collect(*_args, **_kwargs):
        raise AssertionError("the capped write materialized the result on the driver")

    monkeypatch.setattr(terminal, "_collect", _no_collect)
    out = str(tmp_path / "out")
    manifest = (
        bt.read.parquet(src).filter(bt.col("v") < 900).write.parquet(out, max_rows_per_file=250)
    )
    assert manifest.num_files == 4
    assert manifest.total_rows == 900
    assert bt.read.parquet(out).count() == 900
    assert max(pq.read_table(str(f)).num_rows for f in (tmp_path / "out").glob("*.parquet")) == 250
