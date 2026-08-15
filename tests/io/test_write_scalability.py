"""The write path's scalability changes, held to the behavior they replaced.

Each group here covers one change that made a write faster, and each asserts the thing the
speed-up could plausibly have broken rather than the speed itself:

* the vectorized NDJSON encoder must agree with the two encoders it fronts — same values,
  and the same column *types* when the file is read back;
* the concurrent streaming part-writer must produce the same files, in the same order,
  with the same rows, as the serial roll-over did;
* `hive_partition_run_starts` must find the same run boundaries as the per-row loop it
  replaced, including on the single-row table whose Arrow kernel segfaults;
* a distributed write shard must write the same rows whether it streamed or materialized;
* a partitioned Delta write must index each file with *its* partition's rows, derived in
  one pass instead of one full-table filter per file.
"""

from __future__ import annotations

import io
import json

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.json as pajson
import pytest

from batcher.io.base._hive import hive_partition_run_starts
from batcher.io.formats.base import SINKS
from batcher.io.formats.semistructured.json_encoding import (
    _table_to_ndjson,
    _table_to_ndjson_exact,
)
from batcher.io.formats.semistructured.json_vector import ndjson_vectorized

pytestmark = pytest.mark.unit


# --- the vectorized NDJSON encoder ------------------------------------------------


def _reference(table: pa.Table) -> bytes:
    """What the writer would have produced before the vectorized path existed."""
    try:
        return _table_to_ndjson_exact(table)
    except (TypeError, ValueError):
        return _table_to_ndjson(table)


def _rows(blob: bytes) -> list[dict]:
    return [json.loads(line) for line in blob.decode().splitlines() if line.strip()]


VECTORIZED_CASES = {
    "ints": pa.table({"a": pa.array([1, -2, 0, None], pa.int64())}),
    "int beyond float53": pa.table({"a": pa.array([9007199254740993], pa.int64())}),
    "bools": pa.table({"a": pa.array([True, False, None])}),
    "floats": pa.table({"a": pa.array([3.141592653589793, 1 / 3, 1e308, 5e-324, None])}),
    # Whole floats must keep a `.0`, or the reader infers int64 and the column changes type.
    "whole floats": pa.table({"a": pa.array([0.0, -0.0, 1.0, 100.0, 1e17])}),
    "nonfinite": pa.table({"a": pa.array([float("nan"), float("inf"), -1.5])}),
    "float32": pa.table({"a": pa.array([1.5, 2.25], pa.float32())}),
    "strings": pa.table({"s": ["a", "", None, "hello world"]}),
    "quotes and backslash": pa.table({"s": ['he said "hi"', "back\\slash", 'both"and\\']}),
    "short escapes": pa.table({"s": ["a\nb", "a\tb", "a\rb", "a\bb", "a\fb"]}),
    "non-ascii": pa.table({"s": ["café", "日本語", "emoji 🎉"]}),
    "struct": pa.table({"s": pa.array([{"a": 1.5, "b": "x"}, None, {"a": None, "b": None}])}),
    "nested struct": pa.table({"s": pa.array([{"o": {"i": 3.141592653589793}}, {"o": None}])}),
    "all-null column": pa.table({"a": pa.array([None, None], pa.null())}),
    "wide": pa.table(
        {
            "i": pa.array(range(40), pa.int64()),
            "f": pa.array([v / 7 for v in range(40)], pa.float64()),
            "s": pa.array([f"v{v}" for v in range(40)]),
        }
    ),
}


@pytest.mark.parametrize("name", sorted(VECTORIZED_CASES))
def test_vectorized_ndjson_matches_the_encoder_it_replaces(name):
    table = VECTORIZED_CASES[name]
    got = ndjson_vectorized(table)
    assert got is not None, f"{name} should take the vectorized path"
    assert _rows(got) == _rows(_reference(table))


@pytest.mark.parametrize("name", sorted(VECTORIZED_CASES))
def test_vectorized_ndjson_reads_back_with_the_same_types(name):
    # The characteristic defect of a faster encoder is a *type* change on readback (a
    # whole float rendered `1` comes back int64), which comparing values cannot see.
    table = VECTORIZED_CASES[name]
    mine = pajson.read_json(io.BytesIO(ndjson_vectorized(table)))
    theirs = pajson.read_json(io.BytesIO(_reference(table)))
    assert mine.schema == theirs.schema
    assert mine.to_pylist() == theirs.to_pylist()


def test_vectorized_ndjson_declines_rather_than_guessing():
    # Types with no Arrow rendering, and a control character with no short escape. Each
    # must return None so the caller falls back — never a partial or invalid document.
    #
    # Timestamps used to be on this list and are deliberately not any more: declining them
    # sent every temporal column to the pandas encoder, which reads a timestamp column's
    # raw integers as nanoseconds whatever its unit is, so a `timestamp[us]` was written
    # divided by a million. They now render as ISO-8601 here — pinned by
    # `test_io_json_temporal_precision.py`, which is a separate module precisely because
    # the reference encoder these cases compare against is the one that was wrong.
    for table in (
        pa.table({"a": pa.array([[1, 2], None])}),
        pa.table({"b": pa.array([b"x"], pa.binary())}),
        pa.table({"s": ["bell\x07here"]}),
    ):
        assert ndjson_vectorized(table) is None


def test_vectorized_ndjson_now_renders_timestamps_itself():
    # The other half of the change above: it accepts them rather than declining.
    blob = ndjson_vectorized(pa.table({"t": pa.array([1, 2], pa.timestamp("ms"))}))
    assert blob is not None
    assert blob == b'{"t":"1970-01-01 00:00:00.001"}\n{"t":"1970-01-01 00:00:00.002"}\n'


def test_vectorized_ndjson_empty_table_is_still_readable():
    # A 0-byte file is rejected by `pyarrow.json.read_json` as "Empty JSON file".
    blob = ndjson_vectorized(pa.table({"x": pa.array([], pa.float64())}))
    assert blob == b"\n"


def test_ndjson_chunking_is_invisible(monkeypatch):
    # The encoder works in row chunks to bound memory; the seams must not show.
    import batcher.io.formats.semistructured.json_vector as jv

    table = pa.table({"i": pa.array(range(1000), pa.int64()), "f": [v / 3 for v in range(1000)]})
    whole = ndjson_vectorized(table)
    monkeypatch.setattr(jv, "_CHUNK_ROWS", 7)
    assert ndjson_vectorized(table) == whole


# --- concurrent streaming part writes ---------------------------------------------


def _batches(rows: int, chunk: int) -> list[pa.RecordBatch]:
    return pa.table({"x": pa.array(range(rows), pa.int64())}).to_batches(max_chunksize=chunk)


@pytest.mark.parametrize("fmt", ["parquet", "csv", "json", "arrow"])
@pytest.mark.parametrize(("rows", "cap"), [(0, 10), (1, 10), (10, 10), (1000, 100), (997, 100)])
def test_write_stream_parts_rows_and_order(tmp_path, fmt, rows, cap):
    sink = SINKS.get(fmt)()
    schema = pa.schema([("x", pa.int64())])
    written = sink.write_stream_parts(
        iter(_batches(rows, 7)), str(tmp_path / fmt), max_rows_per_file=cap, schema=schema
    )
    assert sum(f.rows for f in written) == rows
    # An empty stream still leaves one (empty) readable file, as it always did.
    assert len(written) == max(1, -(-rows // cap))
    # Names are positional, and `resume` maps rows onto them — so they must come back in
    # submission order however the concurrent encodes happened to finish.
    names = [f.path.rsplit("/", 1)[1] for f in written]
    assert names == sorted(names)


def test_write_stream_parts_content_survives_concurrency(tmp_path):
    import batcher as bt

    sink = SINKS.get("parquet")()
    sink.write_stream_parts(iter(_batches(5000, 13)), str(tmp_path / "out"), max_rows_per_file=100)
    assert sorted(bt.read.parquet(str(tmp_path / "out")).to_pydict()["x"]) == list(range(5000))


def test_write_stream_parts_resume_skips_without_shifting_rows(tmp_path):
    import batcher as bt

    out = str(tmp_path / "out")
    sink = SINKS.get("parquet")()
    first = sink.write_stream_parts(iter(_batches(1000, 13)), out, max_rows_per_file=100)
    again = sink.write_stream_parts(
        iter(_batches(1000, 13)), out, max_rows_per_file=100, resume=True
    )
    assert [f.path for f in again] == [f.path for f in first]
    assert sum(f.rows for f in again) == 1000
    # The resumed run must not have slid rows into the wrong part file.
    assert sorted(bt.read.parquet(out).to_pydict()["x"]) == list(range(1000))


def test_write_stream_shard_names_its_file_like_the_materializing_path(tmp_path):
    sink = SINKS.get("parquet")()
    written = sink.write_stream_shard(iter(_batches(50, 7)), str(tmp_path / "d"), file_index=3)
    assert written.path.endswith("part-00003.parquet")
    assert written.rows == 50


# --- hive run-start detection ------------------------------------------------------


def _run_starts_by_loop(ordered: pa.Table, cols: list[str]) -> list[int]:
    """The per-row Python loop `hive_partition_run_starts` replaced."""
    n = ordered.num_rows
    if n == 0:
        return []
    changed = None
    for name in cols:
        column = ordered.column(name)
        previous, current = column.slice(0, n - 1), column.slice(1, n - 1)
        same = pc.fill_null(pc.equal(previous, current), False)
        same = pc.or_(same, pc.and_(pc.is_null(previous), pc.is_null(current)))
        if pa.types.is_floating(column.type):
            same = pc.or_(
                same,
                pc.and_(
                    pc.fill_null(pc.is_nan(previous), False),
                    pc.fill_null(pc.is_nan(current), False),
                ),
            )
        differs = pc.invert(pc.fill_null(same, False))
        changed = differs if changed is None else pc.or_(changed, differs)
    return [0, *(i + 1 for i, flag in enumerate(changed.to_pylist()) if flag)]


RUN_CASES = {
    "empty": (pa.table({"k": pa.array([], pa.int64())}), ["k"]),
    # One row: the vectorized kernel is handed a zero-chunk array here, which segfaults.
    "single row": (pa.table({"k": [5]}), ["k"]),
    "two same": (pa.table({"k": [5, 5]}), ["k"]),
    "two different": (pa.table({"k": [5, 6]}), ["k"]),
    "nulls group together": (pa.table({"k": [None, None, 1, 1, None]}), ["k"]),
    "nans group together": (pa.table({"k": [float("nan"), float("nan"), 1.0]}), ["k"]),
    "all distinct": (pa.table({"k": list(range(50))}), ["k"]),
    "all identical": (pa.table({"k": [7] * 50}), ["k"]),
    "multi column": (pa.table({"a": [1, 1, 2], "b": ["x", "y", "y"]}), ["a", "b"]),
    "strings": (pa.table({"k": ["a", "a", "b", "c", "c"]}), ["k"]),
}


@pytest.mark.parametrize("name", sorted(RUN_CASES))
def test_hive_run_starts_match_the_row_loop(name):
    table, cols = RUN_CASES[name]
    assert hive_partition_run_starts(table, cols, pc) == _run_starts_by_loop(table, cols)


def test_single_row_partitioned_write_round_trips(tmp_path):
    # The end-to-end form of the segfault above: one row, partitioned.
    import batcher as bt

    out = str(tmp_path / "one")
    bt.from_pydict({"k": ["a"], "v": [1]}).write(
        out, "parquet", partition_by=["k"], distributed=False
    )
    assert bt.read.parquet(out).count() == 1


# --- distributed write shard: streamed == materialized -----------------------------


def test_distributed_shard_streams_the_same_rows_it_would_have_materialized(tmp_path):
    """`_write_plan_shard`'s two branches must write the same rows to the same files.

    `num_files` is what selects the materializing branch (it names a total across the
    write, so it cannot be resolved before the rows are counted); the default layout
    streams. Running both over one partition descriptor compares them directly, with no
    cluster involved.
    """
    import batcher as bt
    from batcher.dist.executors.partition_io import partition_descriptors
    from batcher.dist.executors.plan_analysis import _relabel_single_source
    from batcher.dist.executors.ray_runtime import engine_config_json
    from batcher.dist.executors.write import _write_plan_shard
    from batcher.io.base._layout import FileLayout

    src = str(tmp_path / "src")
    # `distributed=False` on the fixture write, deliberately: `tmp_path` is driver-local, so
    # a run that happens to see a Ray cluster would write the fixture on worker nodes that
    # this process cannot then read. The shard call below is the thing under test.
    bt.range(20_000).select(i=bt.col("value"), x=(bt.col("value") % 7).cast("float64")).write(
        src, "parquet", max_rows_per_file=2_000, distributed=False
    )
    ds = bt.read.parquet(src).filter(bt.col("i") % 3 != 0)
    map_plan, sid = _relabel_single_source(ds._plan)
    map_ir = json.dumps(map_plan.to_ir())
    part = partition_descriptors(ds._sources[sid], 1)[0]
    cfg = engine_config_json()

    results = {}
    for label, layout in (("stream", FileLayout()), ("materialize", FileLayout(num_files=1))):
        out = str(tmp_path / label)
        files = _write_plan_shard(map_ir, part, "parquet", {}, out, None, 0, cfg, layout, False)
        results[label] = (
            sum(f.rows for f in files),
            sorted(bt.read.parquet(out).to_pydict()["i"]),
        )
    assert results["stream"][0] == results["materialize"][0]
    assert results["stream"][1] == results["materialize"][1]
    assert results["stream"][1] == [v for v in range(20_000) if v % 3 != 0]


# --- Delta: per-partition statistics, derived in one pass ---------------------------


def _stats_by_mask(table: pa.Table, written, partition_by: list[str]) -> dict:
    """The mask-per-file derivation the one-pass grouping replaced."""
    from batcher.io.formats.lakehouse.delta._commit import collect_file_stats

    if not partition_by or not written.partition_values:
        return collect_file_stats(table)
    mask = None
    for column, value in written.partition_values.items():
        col = table.column(column)
        eq = (
            pc.is_null(col)
            if value is None
            else pc.equal(col, pa.scalar(value, table.schema.field(column).type))
        )
        mask = eq if mask is None else pc.and_(mask, eq)
    rows = table.filter(mask) if mask is not None else table
    return collect_file_stats(rows.drop_columns(list(written.partition_values)))


DELTA_STAT_CASES = {
    "plain": (pa.table({"k": [1, 1, 2, 2, 3], "v": [10, 20, 30, 40, 50]}), ["k"]),
    "nulls in key": (pa.table({"k": [None, None, 1, 1, 2], "v": [1, 2, 3, 4, 5]}), ["k"]),
    "nulls in value": (pa.table({"k": [1, 1, 2], "v": [None, 5, None]}), ["k"]),
    "multi-column key": (
        pa.table({"a": [1, 1, 2], "b": ["x", "y", "y"], "v": [7, 8, 9]}),
        ["a", "b"],
    ),
    "single row": (pa.table({"k": [1], "v": [42]}), ["k"]),
    "one key only": (pa.table({"k": [3] * 50, "v": list(range(50))}), ["k"]),
    "string key": (pa.table({"k": ["a", "a", "b"], "v": [1, 2, 3]}), ["k"]),
}


@pytest.mark.parametrize("name", sorted(DELTA_STAT_CASES))
def test_delta_partition_stats_match_the_mask_derivation(name):
    from batcher.io.base import FileSink
    from batcher.io.formats.lakehouse.delta.sink import DeltaSink
    from batcher.io.manifest import WrittenFile

    table, cols = DELTA_STAT_CASES[name]
    grouped = DeltaSink._rows_per_partition(table, cols)
    for key_values, sub in FileSink._hive_partition(table, cols):
        written = WrittenFile(
            path="p", rows=sub.num_rows, bytes=0, partition_values=dict(key_values)
        )
        assert DeltaSink._stats_for(table, written, cols, grouped) == _stats_by_mask(
            table, written, cols
        )


def test_a_nan_partition_key_is_indexed_rather_than_recorded_as_empty(tmp_path):
    """The mask form recorded a NaN partition's file as holding **zero** rows.

    `col == NaN` is False for every row, so the mask selected nothing and the file was
    committed with `num_records: 0` and no bounds. That is not a slow statistic, it is a
    wrong one: `count()` is answered from the log, and a predicate prunes the file on its
    bounds — so the rows were physically present and unreachable. Measured on the previous
    code, this table read back as **3 rows of 5**, and the filter below returned 0.
    """
    import batcher as bt

    out = str(tmp_path / "t")
    bt.from_pydict({"k": [float("nan"), float("nan"), 1.0, 1.0, 2.0], "v": [1, 2, 3, 4, 5]}).write(
        out, "delta", partition_by=["k"], distributed=False
    )
    back = bt.read(out, format="delta")
    assert back.count() == 5
    assert back.filter(bt.col("v") <= 2).count() == 2
    assert sorted(back.to_pydict()["v"]) == [1, 2, 3, 4, 5]


def test_delta_partition_stats_are_one_pass_not_one_filter_per_file(monkeypatch, tmp_path):
    """The grouping must be built once per shard, not once per output file.

    Asserted structurally because the cost is what the change is about: at 1,000
    partitions the mask form ran 1,000 full-table filters, and the write went from 1.5x
    the equivalent plain-Parquet write to 6.4x purely as the partition count grew.
    """
    from batcher.io.formats.lakehouse.delta.sink import DeltaSink

    calls = []
    original = DeltaSink._rows_per_partition

    def counted(table, partition_by):
        calls.append(table.num_rows)
        return original(table, partition_by)

    monkeypatch.setattr(DeltaSink, "_rows_per_partition", staticmethod(counted))
    table = pa.table({"k": [i % 40 for i in range(4000)], "v": list(range(4000))})
    written = DeltaSink(partition_by=["k"]).write_partitioned(
        table, str(tmp_path / "t"), partition_by=["k"]
    )
    assert len(written) == 40  # forty files...
    assert calls == [4000]  # ...from one pass over the shard


# --- ORC: the stripe size is the reader's parallelism ceiling ----------------------


def test_orc_stripes_bound_how_many_workers_can_read_the_file(tmp_path):
    """A stripe is what `ORCSource.splits()` cuts a file into, so the write picks it.

    pyarrow's 64 MiB default gave an 8M-row file two stripes, which caps a distributed
    read of that file at two workers however large the cluster is. Both write paths (the
    collect path's `_write_file` and the streaming path's `ORCWriter`) must apply the
    sink's stripe size, or the layout depends on which route the write happened to take.
    """
    import numpy as np

    import batcher as bt

    orc = pytest.importorskip("pyarrow.orc")
    # Random doubles, because a stripe is cut on *encoded* size and a compressible column
    # would fit a whole test-sized table in one stripe whatever the setting.
    rows = 400_000
    table = pa.table({"x": pa.array(np.random.default_rng(0).random(rows))})
    stripe = 256 << 10  # small, so the test stays fast rather than depending on the default

    collected = str(tmp_path / "collected.orc")
    bt.from_arrow(table).write(
        collected, "orc", single_file=True, distributed=False, stripe_size=stripe
    )

    src = str(tmp_path / "src.parquet")
    bt.from_arrow(table).write(src, "parquet", single_file=True, distributed=False)
    streamed = str(tmp_path / "streamed.orc")
    bt.read.parquet(src).write(
        streamed, "orc", single_file=True, distributed=False, stripe_size=stripe
    )

    stripes = {p: orc.ORCFile(p).nstripes for p in (collected, streamed)}
    # The layout must not depend on which route the write took — the collect path goes
    # through `_write_file`, the streaming path through `ORCWriter`, and both must apply
    # the sink's stripe size.
    assert stripes[collected] == stripes[streamed] > 1
    for path in (collected, streamed):
        assert len(bt.read(path, format="orc")._sources[0].splits()) == stripes[path]
        assert bt.read(path, format="orc").count() == rows


def test_orc_default_stripe_size_is_finer_than_the_ecosystem_default():
    """The default has to be well under pyarrow's 64 MiB, or splits stay in single digits.

    Measured on an 8M-row / 68 MB file: 64 MiB gave **2** stripes and so 2 splits, which
    is the ceiling on how many workers can read that file. 8 MiB gives 8, for the same
    write time (1,172-1,245 ms across the range) and the same file size (67.6-67.7 MB).
    """
    from batcher.io.formats.structured.orc import _STRIPE_BYTES

    assert _STRIPE_BYTES <= 16 << 20


def test_orc_stripe_size_is_configurable(tmp_path):
    import numpy as np

    orc = pytest.importorskip("pyarrow.orc")

    table = pa.table({"x": pa.array(np.random.default_rng(0).random(400_000))})
    coarse, fine = str(tmp_path / "coarse.orc"), str(tmp_path / "fine.orc")
    SINKS.get("orc")(stripe_size=64 << 20).write(table, coarse)
    SINKS.get("orc")(stripe_size=256 << 10).write(table, fine)
    assert orc.ORCFile(fine).nstripes > orc.ORCFile(coarse).nstripes == 1
