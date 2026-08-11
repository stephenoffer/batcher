"""Audit of the structured / semistructured file formats against the IO contract.

These are the defect classes that shipped in the `sql/` connectors and were fixed there:
fake streaming, a schema read that scans data, footer metadata left on the floor, pushdown
that never reaches the reader, and split kwargs that silently revert to their defaults on a
worker. The file formats are the most-used readers in the engine, so the same audit is
pinned here — every test below fails against the pre-fix reader.

Real files and real round trips throughout: these formats can be exercised for real, and a
spy on a reader that is never called proves nothing.
"""

from __future__ import annotations

import json
import os

import pyarrow as pa
import pyarrow.ipc as paipc
import pyarrow.json as pajson
import pyarrow.orc as paorc
import pytest

from batcher.io.formats.semistructured.json import JSONSource
from batcher.io.formats.structured.arrow_ipc import ArrowIPCSource
from batcher.io.formats.structured.csv import CSVRangeSplit, CSVSource
from batcher.io.formats.structured.orc import ORCSource

pytestmark = pytest.mark.unit


def _write_ndjson(path: str, rows: list[dict]) -> str:
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def _rows(n: int) -> list[dict]:
    return [{"i": i, "s": "y" * 60} for i in range(n)]


# --------------------------------------------------------------------------------------
# Class 1 — fake streaming
# --------------------------------------------------------------------------------------


def test_json_iter_batches_decodes_in_bounded_windows(tmp_path, monkeypatch):
    """`iter_batches` must not hand the whole file to one `read_json` call.

    Before the `_iter_file` override, `iter_batches` yielded batches out of a table that
    already held the complete decoded file — streaming in name only, and an OOM on a file
    larger than the worker's envelope.
    """
    path = _write_ndjson(str(tmp_path / "a.json"), _rows(20_000))
    size = os.path.getsize(path)

    seen: list[int] = []
    real = pajson.read_json

    def spy(source, *args, **kwargs):
        # The reader may be a `BytesIO` or a zero-copy `pa.BufferReader`; both answer how
        # many bytes this decode was given. Anything else falls back to the whole file, so
        # an unrecognized reader fails the assertion rather than passing it by default.
        if hasattr(source, "getvalue"):
            seen.append(len(source.getvalue()))
        elif hasattr(source, "size"):
            seen.append(source.size())
        else:
            seen.append(size)
        return real(source, *args, **kwargs)

    monkeypatch.setattr("batcher.io.formats.semistructured.json._JSON_STREAM_CHUNK_BYTES", 1 << 16)
    src = JSONSource(path)
    # Schema inference is a separate (still whole-file) read — see `_read_schema`. Warm and
    # cache it first so this test measures the *data* path it is about.
    src.schema()
    monkeypatch.setattr(pajson, "read_json", spy)

    batches = list(src.iter_batches())

    assert sum(b.num_rows for b in batches) == 20_000
    assert seen, "read_json was never called"
    # The load-bearing assertion: no single decode saw the whole file.
    assert max(seen) < size
    assert len(seen) > 1


def test_json_streaming_equals_materializing_read(tmp_path, monkeypatch):
    """The windowed stream must return exactly what the whole-file read returns."""
    monkeypatch.setattr("batcher.io.formats.semistructured.json._JSON_STREAM_CHUNK_BYTES", 1 << 12)
    rows = [
        {"i": 0, "f": 1.5, "s": "a", "n": None},
        {"i": 1, "f": -0.0, "s": "", "n": None},
        {"i": 2, "f": 2.0, "s": "z" * 300, "n": None},
    ] * 400
    path = _write_ndjson(str(tmp_path / "b.json"), rows)
    src = JSONSource(path)

    streamed = pa.Table.from_batches(list(src.iter_batches()), schema=src.schema())
    whole = pa.Table.from_batches(src.read(), schema=src.schema())

    assert streamed.schema == whole.schema
    assert streamed.to_pylist() == whole.to_pylist()


def test_json_streaming_record_wider_than_the_window(tmp_path, monkeypatch):
    """A single record longer than the window is read whole, not cut in half."""
    monkeypatch.setattr("batcher.io.formats.semistructured.json._JSON_STREAM_CHUNK_BYTES", 64)
    rows = [{"i": 0, "s": "x" * 5000}, {"i": 1, "s": "y" * 5000}]
    path = _write_ndjson(str(tmp_path / "wide.json"), rows)

    out = pa.Table.from_batches(list(JSONSource(path).iter_batches()))

    assert out.to_pylist() == rows


def test_json_streaming_projection_reaches_the_parser(tmp_path, monkeypatch):
    """A projected stream parses only the projected columns (class 5, on the stream path)."""
    monkeypatch.setattr("batcher.io.formats.semistructured.json._JSON_STREAM_CHUNK_BYTES", 1 << 12)
    path = _write_ndjson(str(tmp_path / "p.json"), _rows(3_000))

    batches = list(JSONSource(path).iter_batches(["i"]))

    assert batches
    assert all(b.schema.names == ["i"] for b in batches)
    assert sum(b.num_rows for b in batches) == 3_000


# --------------------------------------------------------------------------------------
# Class 6 — correctness edges of the streamed read
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rows",
    [
        pytest.param([{"i": 1}], id="single_row"),
        pytest.param([{"i": None}, {"i": None}], id="all_null_column"),
        pytest.param([{"i": 1}, {"i": 1}, {"i": 1}], id="duplicate_values"),
        pytest.param([{"f": -0.0}, {"f": 0.0}], id="negative_zero"),
        pytest.param([{"i": -(2**63) + 1}, {"i": 2**63 - 1}], id="int64_bounds"),
    ],
)
def test_json_stream_edges_match_the_whole_file_read(tmp_path, monkeypatch, rows):
    monkeypatch.setattr("batcher.io.formats.semistructured.json._JSON_STREAM_CHUNK_BYTES", 32)
    path = _write_ndjson(str(tmp_path / "edge.json"), rows)
    src = JSONSource(path)

    streamed = [b.to_pydict() for b in src.iter_batches()]
    whole = [b.to_pydict() for b in src.read()]

    def flat(batches):
        out: dict[str, list] = {}
        for batch in batches:
            for key, values in batch.items():
                out.setdefault(key, []).extend(values)
        return out

    assert flat(streamed) == flat(whole)


def test_json_empty_file_streams_the_same_error_as_read(tmp_path):
    """An empty file must keep raising, not quietly stream as zero rows."""
    path = str(tmp_path / "empty.json")
    open(path, "w").close()
    src = JSONSource(path)

    with pytest.raises(Exception) as read_exc:
        src.read()
    with pytest.raises(Exception) as iter_exc:
        list(src.iter_batches())

    assert type(iter_exc.value) is type(read_exc.value)


# --------------------------------------------------------------------------------------
# Class 6 — split kwargs silently reverting to constructor defaults on a worker
# --------------------------------------------------------------------------------------


def test_json_splits_carry_reader_kwargs(tmp_path):
    """`on_error`/credentials must ride the split, or a worker rebuilds a fail-fast reader."""
    path = _write_ndjson(str(tmp_path / "a.json"), _rows(5))
    src = JSONSource(path, on_error="skip")

    splits = src.splits()

    assert splits
    assert all(getattr(s, "kwargs", {}).get("on_error") == "skip" for s in splits)


def test_json_split_rebuilds_a_reader_that_is_still_tolerant(tmp_path):
    """The reader a worker rebuilds from the split must carry the declared tolerance.

    This is the mechanism the kwargs exist for: a worker does
    ``SOURCES.get("json")(path, **split.kwargs)``, so a dropped `on_error` turns an
    explicitly tolerated read back into a fail-fast one, off the driver where nobody sees it.
    """
    _write_ndjson(str(tmp_path / "good.json"), _rows(3))
    bad = str(tmp_path / "bad.json")
    with open(bad, "w") as fh:
        fh.write("{not json at all\n")

    splits = JSONSource(str(tmp_path), on_error="skip").splits()
    bad_split = next(s for s in splits if s.path == bad)
    reader = bad_split._reader()

    # Rebuilt with the policy, so the corrupt file is skipped rather than fatal.
    assert reader.read() == []
    assert reader.corrupt_files() == [bad]


def test_csv_splits_carry_reader_kwargs_when_size_is_unavailable(tmp_path, monkeypatch):
    """The unsized branch dropped the kwargs the sized branch carried."""
    path = str(tmp_path / "a.csv")
    with open(path, "w") as fh:
        fh.write("k,v\n1,2\n")
    src = CSVSource(path, on_error="skip")

    class _NoSize:
        """A filesystem whose `size` fails — the object-store case that most needs the
        credentials and policy the unsized branch was dropping."""

        def __init__(self, inner):
            self._inner = inner

        def size(self, _path):
            raise OSError("size unavailable")

        def __getattr__(self, name):
            return getattr(self._inner, name)

    monkeypatch.setattr(src, "_fs", _NoSize(src._fs))

    splits = src.splits()

    assert splits
    assert all(getattr(s, "kwargs", {}).get("on_error") == "skip" for s in splits)


def test_arrow_ipc_splits_fall_back_when_credentials_cannot_ride(tmp_path):
    """`ArrowBlockSplit` carries no kwargs, so a BYO/tolerant read must use `FileSplit`."""
    path = str(tmp_path / "a.arrow")
    table = pa.table({"x": list(range(100))})
    with paipc.new_file(path, table.schema) as writer:
        for batch in table.to_batches(max_chunksize=10):
            writer.write_batch(batch)

    plain = ArrowIPCSource(path).splits()
    tolerant = ArrowIPCSource(path, on_error="skip").splits()

    # Unchanged for the ordinary read: still one split per block.
    assert len(plain) == 10
    assert all(type(s).__name__ == "ArrowBlockSplit" for s in plain)
    # Tolerant read: a single FileSplit that actually carries the policy.
    assert len(tolerant) == 1
    assert getattr(tolerant[0], "kwargs", {}).get("on_error") == "skip"
    assert sum(b.num_rows for b in tolerant[0].read()) == 100


# --------------------------------------------------------------------------------------
# Class 5 — projection pushdown on the distributed CSV path
# --------------------------------------------------------------------------------------


def test_csv_range_split_pushes_projection_into_the_parse(tmp_path, monkeypatch):
    """The range split converted every column and then discarded most of them."""
    path = str(tmp_path / "wide.csv")
    with open(path, "w") as fh:
        fh.write("a,b,c\n")
        for i in range(200):
            fh.write(f"{i},{i * 2},{i * 3}\n")
    schema = CSVSource(path).schema()
    size = os.path.getsize(path)

    seen: dict = {}
    import pyarrow.csv as pacsv

    real = pacsv.read_csv

    def spy(source, *args, **kwargs):
        convert = kwargs.get("convert_options")
        seen["include_columns"] = list(getattr(convert, "include_columns", None) or [])
        return real(source, *args, **kwargs)

    monkeypatch.setattr(pacsv, "read_csv", spy)

    batches = CSVRangeSplit(path, 0, size, schema).read(["c"])

    assert seen["include_columns"] == ["c"]
    assert [b.schema.names for b in batches] == [["c"]] * len(batches)


def test_csv_range_split_projection_result_is_unchanged(tmp_path):
    """Pushdown must change only the cost — the rows and their order stay identical."""
    path = str(tmp_path / "w.csv")
    with open(path, "w") as fh:
        fh.write("a,b,c\n")
        for i in range(50):
            fh.write(f"{i},{i * 2},{i * 3}\n")
    schema = CSVSource(path).schema()
    size = os.path.getsize(path)

    # Projection order must be honored as an ordered list, not file order.
    batches = CSVRangeSplit(path, 0, size, schema).read(["c", "a"])
    out = pa.Table.from_batches(batches)

    assert out.schema.names == ["c", "a"]
    assert out.column("c").to_pylist() == [i * 3 for i in range(50)]
    assert out.column("a").to_pylist() == list(range(50))


def test_csv_range_splits_cover_every_row_exactly_once(tmp_path):
    """The byte ranges still tile the file after the parse change."""
    path = str(tmp_path / "t.csv")
    with open(path, "w") as fh:
        fh.write("k,v\n")
        for i in range(2_000):
            fh.write(f"{i},{i}\n")
    src = CSVSource(path)

    splits = src.splits(target_size=4096)
    assert len(splits) > 1
    seen = [
        value
        for split in splits
        for batch in split.read(["k"])
        for value in batch.column("k").to_pylist()
    ]

    assert sorted(seen) == list(range(2_000))


# --------------------------------------------------------------------------------------
# Class 3 — footer metadata the format states but did not report
# --------------------------------------------------------------------------------------


def test_orc_statistics_reports_byte_size_and_stripe_count(tmp_path):
    """The ORC footer states both; `statistics()` was dropping them."""
    path = str(tmp_path / "t.orc")
    table = pa.table({"a": list(range(50_000)), "b": ["x" * 20] * 50_000})
    paorc.write_table(table, path, stripe_size=1 << 16)
    footer_stripes = paorc.ORCFile(path).nstripes

    stats = ORCSource(path).statistics()

    assert stats is not None
    assert stats.row_count == 50_000
    assert stats.exact_rows is True
    assert stats.byte_size == os.path.getsize(path)
    assert stats.row_group_count == footer_stripes
    assert footer_stripes > 1


def test_orc_statistics_sums_across_files(tmp_path):
    for i in range(3):
        paorc.write_table(pa.table({"a": list(range(1_000))}), str(tmp_path / f"p{i}.orc"))

    stats = ORCSource(str(tmp_path)).statistics()

    assert stats is not None
    assert stats.row_count == 3_000
    assert stats.byte_size == sum(os.path.getsize(str(tmp_path / f"p{i}.orc")) for i in range(3))
    assert stats.row_group_count == 3


def test_orc_statistics_fabricates_no_column_bounds(tmp_path):
    """pyarrow exposes no ORC stripe statistics, so no min/max may be claimed.

    An exact-looking bound is worse than none: a terminal answers `min()`/`max()` from an
    EXACT one without executing the query.
    """
    path = str(tmp_path / "n.orc")
    paorc.write_table(pa.table({"f": [1.0, float("nan"), 3.0]}), path)

    stats = ORCSource(path).statistics()

    assert stats is not None
    assert dict(stats.columns) == {}
    assert stats.bounds_include_nan is False


def test_orc_statistics_on_an_empty_file(tmp_path):
    path = str(tmp_path / "e.orc")
    paorc.write_table(pa.table({"a": pa.array([], pa.int64())}), path)

    stats = ORCSource(path).statistics()

    assert stats is not None
    assert stats.row_count == 0
    assert stats.is_empty() is True
