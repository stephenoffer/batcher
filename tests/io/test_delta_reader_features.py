"""A Delta reader feature Batcher does not implement is refused, never read past.

Every table here is a real one written by delta-rs into ``tmp_path``. The failure these
guard against returns the right number of rows with the wrong values in them and raises
nothing, so each test asserts a refusal *and* a control that reads normally.
"""

from __future__ import annotations

from types import SimpleNamespace

import pyarrow as pa
import pytest

from batcher._internal.errors import BackendError
from batcher.io.formats.lakehouse.delta import DeltaSource
from batcher.io.formats.lakehouse.delta._snapshot import _unreadable_reason, open_snapshot

pytestmark = pytest.mark.io

deltalake = pytest.importorskip("deltalake")


def _column_mapped(path: str) -> None:
    deltalake.write_deltalake(
        path,
        pa.table({"a": [1, 2, 3]}),
        configuration={
            "delta.columnMapping.mode": "name",
            "delta.minReaderVersion": "2",
            "delta.minWriterVersion": "5",
        },
    )


def test_a_column_mapped_table_is_refused_rather_than_read_as_nulls(tmp_path) -> None:
    """Its data files name the column ``col-<uuid>``, so a by-name read is all NULL."""
    path = str(tmp_path / "mapped")
    _column_mapped(path)
    # The silent failure this replaces, shown on the table itself: delta-rs's own reader.
    assert deltalake.DeltaTable(path).to_pyarrow_table().column("a").null_count == 3

    with pytest.raises(BackendError, match="column mapping"):
        list(DeltaSource(path).iter_batches())
    with pytest.raises(BackendError, match="column mapping"):
        DeltaSource(path).schema()


def test_metadata_only_questions_still_answer_on_a_refused_table(tmp_path) -> None:
    """Counting files from the log reads no column, so it has nothing to get wrong."""
    path = str(tmp_path / "mapped")
    _column_mapped(path)
    assert open_snapshot(path).add_actions().num_rows == 1


def test_an_unknown_reader_feature_is_refused_by_name() -> None:
    def table(features: list[str] | None, version: int = 3) -> SimpleNamespace:
        protocol = SimpleNamespace(min_reader_version=version, reader_features=features)
        return SimpleNamespace(
            protocol=lambda: protocol,
            metadata=lambda: SimpleNamespace(configuration={}),
        )

    assert "futureFeature" in (_unreadable_reason(table(["futureFeature"])) or "")
    assert "typeWidening" in (_unreadable_reason(table(["deletionVectors", "typeWidening"])) or "")
    assert "reader version 4" in (_unreadable_reason(table(None, version=4)) or "")
    # What delta-rs turns on for an ordinary DV-enabled table must stay readable.
    assert _unreadable_reason(table(["deletionVectors", "variantType"])) is None
    assert _unreadable_reason(table(None, version=1)) is None


def _dv_enabled(path: str) -> None:
    deltalake.write_deltalake(
        path,
        pa.table({"id": list(range(6))}),
        configuration={"delta.enableDeletionVectors": "true"},
    )


def _deletion_vectors_fail(monkeypatch) -> None:
    def boom(self):
        raise RuntimeError("deletion vector file unreadable")

    monkeypatch.setattr(deltalake.DeltaTable, "deletion_vectors", boom)


def test_unreadable_deletion_vectors_fail_closed(tmp_path, monkeypatch) -> None:
    """Reading on without the vectors would hand every deleted row back."""
    path = str(tmp_path / "dv")
    _dv_enabled(path)
    _deletion_vectors_fail(monkeypatch)

    source = DeltaSource(path)
    with pytest.raises(BackendError, match="deletion vectors"):
        list(source.iter_batches())
    with pytest.raises(BackendError, match="deletion vectors"):
        source.row_count()


def test_a_table_that_cannot_have_deletion_vectors_reads_through_the_same_failure(
    tmp_path, monkeypatch
) -> None:
    """Without the reader feature, "no vectors" is true whatever the API says."""
    path = str(tmp_path / "plain")
    deltalake.write_deltalake(path, pa.table({"id": list(range(6))}))
    _deletion_vectors_fail(monkeypatch)

    out = pa.Table.from_batches(list(DeltaSource(path).iter_batches()))
    assert sorted(out.column("id").to_pylist()) == list(range(6))


def test_a_dv_enabled_table_with_readable_vectors_still_reads(tmp_path) -> None:
    path = str(tmp_path / "dv")
    _dv_enabled(path)
    assert "deletionVectors" in (deltalake.DeltaTable(path).protocol().reader_features or [])

    out = pa.Table.from_batches(list(DeltaSource(path).iter_batches()))
    assert sorted(out.column("id").to_pylist()) == list(range(6))
