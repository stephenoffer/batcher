"""The five silent-failure classes, audited across the lakehouse connectors.

Every bug pinned here passes review and every existing test while being wrong. They are
the same five that were already found and fixed in the `sql/` connector family, re-run
against `io/formats/lakehouse/` — plus a sixth that is specific to this family and is the
highest-value of the set.

**1 · Fake streaming.** `iter_batches` that materializes the whole relation and then
re-chunks it. The signature streams; the memory does not. Hudi's source and its file-slice
split both did ``pa.Table.from_batches(...).to_batches()``.

**2 · Materializing `schema()`.** Reading *data* to learn column names that the transaction
log, the manifest, or — for a shared table — the Parquet footer already states. Delta
Sharing read an entire pre-signed Parquet file, over the network, to answer `schema()`.

**3 · Identity collision / 4 · credential leaks.** `identity()` is *persisted* as the key a
source's learned statistics are filed under, so it must name everything that distinguishes
the relation and nothing that authenticates to it. Iceberg's `_catalog_key` did the exact
opposite: it serialized the catalog property mapping verbatim, so the secret access key was
written into the stats store in clear text and a rotated key orphaned the statistics.

**5 · Distributed-write data loss.** Checked and found correct: both lakehouse sinks commit
once, on the driver, in one transaction. What was *not* correct was a writer left unclosed
when a staged write raised mid-stream.

**6 · Metadata exploitation.** These formats carry exact row counts and per-column bounds in
their manifests, and the `SourceStatistics` contract exists to consume them. Delta Sharing
already parsed those statistics for file skipping and then threw them away rather than
declaring them — the one connector in the family whose metadata arrives *free over the
wire*, and the only one that did not use it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------------------
# 1 · Fake streaming — Hudi
# --------------------------------------------------------------------------------------


class _CountingBatches:
    """A batch source that records how many batches have actually been pulled from it.

    This is what tells a real stream from a fake one. A generator handed to
    `pa.Table.from_batches` is drained *completely* before the first row can be yielded,
    so a connector that materializes shows `pulled == len(batches)` after a single
    `next()`; one that streams shows `pulled == 1`.
    """

    def __init__(self, batches: list[pa.RecordBatch]) -> None:
        self._batches = batches
        self.pulled = 0

    def __iter__(self):
        for batch in self._batches:
            self.pulled += 1
            yield batch


def _three_batches() -> list[pa.RecordBatch]:
    return [
        pa.RecordBatch.from_pydict({"id": [i], "v": [f"r{i}"]}, schema=_SCHEMA) for i in range(3)
    ]


_SCHEMA = pa.schema([("id", pa.int64()), ("v", pa.string())])


def _hudi_source(monkeypatch, batches: _CountingBatches):
    """A `HudiSource` whose backend hands back `batches`, with no Hudi table on disk."""
    from batcher.io.formats.lakehouse.hudi import HudiSource

    source = HudiSource("/nonexistent/table")
    monkeypatch.setattr(
        HudiSource,
        "_table",
        lambda self: SimpleNamespace(
            read_snapshot=lambda _filters: batches,
            get_schema=lambda: _SCHEMA,
        ),
    )
    return source


def test_hudi_source_iter_batches_does_not_materialize_the_table(monkeypatch):
    """Taking one batch off a Hudi scan must not pull the whole snapshot into memory."""
    batches = _CountingBatches(_three_batches())
    source = _hudi_source(monkeypatch, batches)

    stream = source.iter_batches()
    first = next(stream)

    assert first.num_rows == 1
    assert batches.pulled == 1, (
        f"iter_batches drained {batches.pulled}/3 batches to yield the first — it is "
        "materializing the snapshot and re-chunking it, so peak memory is the whole table."
    )


def test_hudi_source_iter_batches_still_returns_every_row(monkeypatch):
    """Streaming must not change *what* is returned — same rows, same order."""
    batches = _CountingBatches(_three_batches())
    source = _hudi_source(monkeypatch, batches)

    out = pa.Table.from_batches(list(source.iter_batches()), schema=_SCHEMA)

    assert out.column("id").to_pylist() == [0, 1, 2]
    assert out.column("v").to_pylist() == ["r0", "r1", "r2"]


def test_hudi_source_iter_batches_applies_projection_per_batch(monkeypatch):
    """A projection has to survive the move from whole-table `.select()` to per-batch."""
    batches = _CountingBatches(_three_batches())
    source = _hudi_source(monkeypatch, batches)

    out = list(source.iter_batches(["v"]))

    assert [b.schema.names for b in out] == [["v"]] * 3


def test_hudi_file_slice_split_iter_batches_streams(monkeypatch):
    """The worker-side split streams its file slice rather than building it whole."""
    from batcher.io.formats.lakehouse import hudi as hudi_mod

    batches = _CountingBatches(_three_batches())
    split = hudi_mod.HudiFileSliceSplit("/nonexistent/table", "base.parquet", {})
    monkeypatch.setattr(
        hudi_mod,
        "_require_hudi",
        lambda: (
            lambda _uri, options=None: SimpleNamespace(
                create_file_group_reader_with_options=lambda: SimpleNamespace(
                    read_file_slice_by_base_file_path=lambda _p: batches
                )
            )
        ),
    )

    first = next(split.iter_batches())

    assert first.num_rows == 1
    assert batches.pulled == 1, (
        f"the split drained {batches.pulled}/3 batches to yield one — a worker holds the "
        "whole data file, which is exactly what a per-file split exists to avoid."
    )


# --------------------------------------------------------------------------------------
# 2 · Materializing schema() — Delta Sharing
# --------------------------------------------------------------------------------------


@pytest.fixture
def shared_files(tmp_path):
    """Two local Parquet files standing in for a shared table's pre-signed URLs."""
    paths = []
    for index, ids in enumerate(([1, 2, 3], [10, 11])):
        path = tmp_path / f"part-{index}.parquet"
        pq.write_table(
            pa.table({"id": ids, "v": [f"r{i}" for i in ids]}, schema=_SCHEMA),
            path,
        )
        paths.append(str(path))
    return paths


def _add_file(url: str, *, rows: int, lo: int, hi: int, nulls: int = 0):
    """One Delta Sharing `AddFile`, carrying the Delta `stats` JSON the server sends."""
    return SimpleNamespace(
        url=url,
        stats=json.dumps(
            {
                "numRecords": rows,
                "minValues": {"id": lo},
                "maxValues": {"id": hi},
                "nullCount": {"id": nulls},
            }
        ),
    )


def _sharing_source(shared_files):
    from batcher.io.formats.lakehouse.delta_sharing import DeltaSharingSource

    source = DeltaSharingSource("profile#share.schema.table")
    source._files_cache = [
        _add_file(shared_files[0], rows=3, lo=1, hi=3),
        _add_file(shared_files[1], rows=2, lo=10, hi=11),
    ]
    return source


def test_delta_sharing_schema_reads_the_footer_not_the_data(monkeypatch, shared_files):
    """`schema()` must come from Parquet metadata, never from a full pre-signed fetch.

    A shared file is fetched over the network from object storage. Reading one end-to-end
    to learn its column names bills the whole file for a question its 8-byte footer answers
    — and `schema()` is called at plan time on *every* read.
    """
    from batcher.io.formats.lakehouse import delta_sharing as ds_mod

    def _refuse(*args, **kwargs):
        raise AssertionError(
            "schema() fetched a whole shared Parquet file; the footer states the schema."
        )

    monkeypatch.setattr(ds_mod, "_read_presigned", _refuse)

    assert _sharing_source(shared_files).schema().names == ["id", "v"]


def test_delta_sharing_split_schema_reads_the_footer_not_the_data(monkeypatch, shared_files):
    """The worker-side split answers `schema()` from the footer too."""
    from batcher.io.formats.lakehouse import delta_sharing as ds_mod

    monkeypatch.setattr(
        ds_mod,
        "_read_presigned",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("fetched the whole file")),
    )
    split = ds_mod.DeltaSharingFileSplit(file_url=shared_files[0])

    assert split.schema().names == ["id", "v"]


def test_delta_sharing_schema_matches_the_data_it_reads(shared_files):
    """The cheap schema must be the *same* schema the read produces."""
    source = _sharing_source(shared_files)

    assert source.schema().equals(pa.Table.from_batches(source.read(), schema=_SCHEMA).schema)


# --------------------------------------------------------------------------------------
# 6 · Metadata exploitation — Delta Sharing
# --------------------------------------------------------------------------------------


def test_delta_sharing_declares_exact_row_count_from_server_stats(shared_files):
    """The sharing server sends `numRecords` per file; that is an exact count, free."""
    stats = _sharing_source(shared_files).statistics()

    assert stats is not None, "statistics() returned nothing though the server sent numRecords"
    assert stats.row_count == 5
    assert stats.exact_rows is True


def test_delta_sharing_declares_column_bounds_from_server_stats(shared_files):
    """`minValues`/`maxValues` are a zone map — without them Kyber cannot prune the scan."""
    stats = _sharing_source(shared_files).statistics()

    assert stats is not None
    assert "id" in stats.columns, "no column bounds, though the server sent min/max per file"
    assert (stats.columns["id"].min, stats.columns["id"].max) == (1, 11)


def test_delta_sharing_row_count_uses_the_declared_stats(shared_files):
    """`row_count()` is answered from metadata rather than declared unknowable."""
    assert _sharing_source(shared_files).row_count() == 5


def test_delta_sharing_reports_nothing_when_the_server_sent_no_stats(shared_files):
    """A server that sends no statistics must yield no statistics — never a fabricated one.

    This is the direction that matters. An invented row count is not a slow plan, it is a
    wrong answer: `count()` is served straight from an exact statistic without executing.
    """
    from batcher.io.formats.lakehouse.delta_sharing import DeltaSharingSource

    source = DeltaSharingSource("profile#share.schema.table")
    source._files_cache = [SimpleNamespace(url=shared_files[0], stats=None)]

    stats = source.statistics()

    assert stats is None or stats.row_count is None


def test_delta_sharing_bounds_never_claim_to_rank_nan(shared_files):
    """`bounds_include_nan` stays False: Delta stats omit NaN, so they cannot answer max()."""
    stats = _sharing_source(shared_files).statistics()

    assert stats is not None
    assert stats.bounds_include_nan is False


# --------------------------------------------------------------------------------------
# 3 · Identity collision / 4 · credential leaks
# --------------------------------------------------------------------------------------

_SECRET = "wJalrXUtnFEMI-SUPER-SECRET-KEY"

_CATALOG = {
    "type": "rest",
    "uri": "https://unity.example.com/api/2.1/unity-catalog/iceberg",
    "warehouse": "prod",
    "s3.access-key-id": "AKIAIOSFODNN7EXAMPLE",
    "s3.secret-access-key": _SECRET,
    "credential": "oauth-client:oauth-secret",
}


def test_iceberg_identity_does_not_persist_catalog_credentials():
    """`identity()` is written to the stats store — a secret in it is a secret at rest."""
    from batcher.io.formats.lakehouse.iceberg.source import IcebergSource

    source = IcebergSource("db.orders", catalog=dict(_CATALOG))

    identity = source.identity()

    assert _SECRET not in identity, f"the secret access key is persisted in {identity!r}"
    assert "oauth-secret" not in identity
    assert "AKIAIOSFODNN7EXAMPLE" not in identity


def test_iceberg_identity_still_separates_two_catalogs():
    """Hiding the credentials must not collapse two warehouses onto one stats key."""
    from batcher.io.formats.lakehouse.iceberg.source import IcebergSource

    other = {**_CATALOG, "warehouse": "staging"}

    assert IcebergSource("db.orders", catalog=dict(_CATALOG)).identity() != IcebergSource(
        "db.orders", catalog=other
    ).identity()


def test_iceberg_identity_is_stable_across_processes():
    """A `hash()`-based key would be salted per process and reuse nothing, silently."""
    import subprocess
    import sys

    code = (
        "from batcher.io.formats.lakehouse.iceberg.source import _catalog_key;"
        f"print(_catalog_key({_CATALOG!r}))"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(2)
    }

    assert len(runs) == 1, f"the catalog key differs between processes: {runs}"


def test_iceberg_identity_survives_a_credential_rotation():
    """Rotating a key must not orphan the statistics learned under the old one."""
    from batcher.io.formats.lakehouse.iceberg.source import IcebergSource

    rotated = {**_CATALOG, "s3.secret-access-key": "ROTATED", "credential": "oauth:rotated"}

    assert (
        IcebergSource("db.orders", catalog=dict(_CATALOG)).identity()
        == IcebergSource("db.orders", catalog=rotated).identity()
    )


def test_delta_file_split_repr_hides_storage_options():
    """A split's `repr` lands in tracebacks and task logs; its storage options are secrets."""
    from batcher.io.formats.lakehouse.delta.source import DeltaFileSplit

    split = DeltaFileSplit(
        "s3://bucket/table", "part-0.parquet", {"aws_secret_access_key": _SECRET}, 3
    )

    assert _SECRET not in repr(split)


def test_hudi_file_slice_split_repr_hides_options():
    """Same for Hudi: `options` carries the cloud storage credentials."""
    from batcher.io.formats.lakehouse.hudi import HudiFileSliceSplit

    split = HudiFileSliceSplit("s3://bucket/t", "base.parquet", {"aws.secret.key": _SECRET})

    assert _SECRET not in repr(split)


def test_split_reprs_still_identify_the_file():
    """Redaction must not blind a debugger — the locators stay visible."""
    from batcher.io.formats.lakehouse.delta.source import DeltaFileSplit

    text = repr(DeltaFileSplit("s3://bucket/table", "part-0.parquet", {"k": _SECRET}, 3))

    assert "part-0.parquet" in text and "s3://bucket/table" in text


# --------------------------------------------------------------------------------------
# 5 · Resource leaks on the write path
# --------------------------------------------------------------------------------------


def test_stage_stream_closes_its_writer_when_a_batch_raises(tmp_path):
    """A staged write that fails mid-stream must not leak the open `ParquetWriter`.

    `stage_stream` wrote batches through a `ParquetWriter` and closed it only on the
    success path. A shard that raised partway — an OOM, a preempted worker, a bad batch —
    left the writer open, holding its buffers and its file handle, and the atomic writer's
    context then exited around a file still being written to.
    """
    from batcher.io.formats.lakehouse._staging import stage_stream

    opened: list[object] = []
    real_writer = pq.ParquetWriter

    class _TrackingWriter(real_writer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            opened.append(self)

    def _batches():
        yield pa.RecordBatch.from_pydict({"id": [1], "v": ["a"]}, schema=_SCHEMA)
        raise RuntimeError("worker preempted mid-shard")

    pq.ParquetWriter = _TrackingWriter
    try:
        with pytest.raises(RuntimeError, match="preempted"):
            stage_stream(_batches(), str(tmp_path / "staging"))
    finally:
        pq.ParquetWriter = real_writer

    assert opened, "test did not exercise the streaming writer path"
    assert all(w.is_open is False for w in opened), (
        "the ParquetWriter was left open after the write raised — the shard's buffers and "
        "file handle leak, once per failed shard."
    )
