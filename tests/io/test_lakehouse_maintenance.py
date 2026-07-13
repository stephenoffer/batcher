"""Table maintenance: compaction rewrites, vacuum deletes, and never the other way round.

The invariant every test here defends: **compaction must not delete a data file an older
version still references.** A lakehouse table is a log over files, so a rewrite that also
removes what it replaced destroys time travel — and destroys it *silently*, because a
`count()` is answered from the log and keeps reporting rows whose files are gone. That is
exactly what `bt.compact` did to a Delta table before it learned the difference between a
file directory and a transactional table.

So the split is: compaction rewrites and retires files *from the log*, leaving them on
storage; `vacuum` is the only operation that deletes, and only past a retention window.
"""

from __future__ import annotations

import glob
import os

import pytest

import batcher as bt
from batcher._internal.errors import PlanError

deltalake = pytest.importorskip("deltalake", reason="deltalake not installed")

pytestmark = pytest.mark.integration


def _table(uri: str, appends: int = 6) -> str:
    """A Delta table built by `appends` small writes — one data file each."""
    for i in range(appends):
        bt.from_pydict({"x": [i], "g": [i % 2]}).write.delta(uri, mode="append")
    return uri


def _live_files(uri: str) -> int:
    return len(deltalake.DeltaTable(uri).file_uris())


def _physical_files(uri: str) -> int:
    return len(glob.glob(os.path.join(uri, "**", "*.parquet"), recursive=True))


# --- compaction ------------------------------------------------------------


def test_compact_bin_packs_a_delta_table(tmp_path) -> None:
    uri = _table(str(tmp_path / "t"))
    assert _live_files(uri) == 6

    bt.compact(uri)

    assert _live_files(uri) == 1
    assert sorted(bt.read.delta(uri).collect().column("x").to_pylist()) == [0, 1, 2, 3, 4, 5]


def test_compaction_does_not_delete_the_files_older_versions_reference(tmp_path) -> None:
    """The regression this module exists for.

    `bt.compact` used to read the table, overwrite it, and then delete the replaced part
    files — which versions 0..n-1 still point at. Time travel died, and died invisibly:
    `count()` kept answering from the log long after the data was gone, so the corruption
    only surfaced when someone actually read an old version.
    """
    uri = _table(str(tmp_path / "t"))
    before = _physical_files(uri)

    bt.compact(uri)

    # the compacted output is *added*; nothing that an older version needs is removed
    assert _physical_files(uri) > before
    assert bt.read.delta(uri, version=0).collect().to_pydict() == {"x": [0], "g": [0]}
    # version N is the state after N+1 appends, so it must still read all N+1 rows
    assert bt.read.delta(uri, version=2).collect().num_rows == 3
    assert sorted(bt.read.delta(uri, version=4).collect().column("x").to_pylist()) == [
        0,
        1,
        2,
        3,
        4,
    ]


def test_compact_detects_a_delta_table_without_being_told(tmp_path) -> None:
    """A Delta table announces itself with `_delta_log`; a caller must not have to name it.

    Being unable to infer it is how the unsafe path got taken — the user passes
    ``format="parquet"``, the table is treated as a plain directory, and the rewrite
    deletes referenced files.
    """
    from batcher.io.detect import detect_format

    uri = _table(str(tmp_path / "t"), appends=2)
    assert detect_format(uri) == "delta"


def test_z_order_clusters_and_keeps_the_rows(tmp_path) -> None:
    """Z-ordering is a compaction that sorts, so the data must be untouched by it."""
    uri = _table(str(tmp_path / "t"))

    metrics = bt.compact(uri, z_order=["x"])

    assert metrics["numFilesAdded"] >= 1
    assert sorted(bt.read.delta(uri).collect().column("x").to_pylist()) == [0, 1, 2, 3, 4, 5]


def test_z_order_on_a_plain_directory_is_refused(tmp_path) -> None:
    """A file directory has no log, so there is no transaction to cluster within."""
    out = str(tmp_path / "d")
    bt.from_pydict({"x": [1, 2]}).write(out, format="parquet")
    with pytest.raises(PlanError, match="transactional table"):
        bt.compact(out, z_order=["x"], format="parquet")


# --- vacuum ----------------------------------------------------------------


def test_vacuum_defaults_to_a_dry_run(tmp_path) -> None:
    """The one operation that destroys data must not do so unless asked."""
    uri = _table(str(tmp_path / "t"))
    bt.compact(uri)
    before = _physical_files(uri)

    would_delete = bt.vacuum(uri, retention_hours=0, enforce_retention_duration=False)

    assert would_delete, "the compacted-away files should be reclaimable"
    assert _physical_files(uri) == before, "a dry run must delete nothing"


def test_vacuum_reclaims_the_files_compaction_retired(tmp_path) -> None:
    uri = _table(str(tmp_path / "t"))
    bt.compact(uri)

    deleted = bt.vacuum(uri, retention_hours=0, dry_run=False, enforce_retention_duration=False)

    assert deleted
    assert _live_files(uri) == 1
    assert sorted(bt.read.delta(uri).collect().column("x").to_pylist()) == [0, 1, 2, 3, 4, 5]


def test_vacuum_refuses_a_plain_directory(tmp_path) -> None:
    """Nothing references a file in a plain directory, so there is nothing to reclaim."""
    out = str(tmp_path / "d")
    bt.from_pydict({"x": [1, 2]}).write(out, format="parquet")
    with pytest.raises(PlanError, match="transactional table"):
        bt.vacuum(out, format="parquet")


# --- auto-compaction -------------------------------------------------------


def test_auto_compact_fires_once_small_files_have_piled_up(tmp_path) -> None:
    """The trigger is the table's *standing* small files, not this write's output.

    A write that adds one small file to a table already holding a thousand is exactly the
    case that needs compacting — a per-write trigger would never fire on it.
    """
    uri = str(tmp_path / "t")
    for i in range(60):
        bt.from_pydict({"x": [i]}).write.delta(uri, mode="append", auto_compact=(i == 59))

    assert _live_files(uri) == 1
    assert len(bt.read.delta(uri).collect().column("x").to_pylist()) == 60
    assert bt.read.delta(uri, version=0).collect().num_rows == 1  # time travel intact


def test_auto_compact_does_not_fire_below_the_threshold(tmp_path) -> None:
    """Compacting a three-file table would cost more than it saves."""
    uri = str(tmp_path / "t")
    for i in range(3):
        bt.from_pydict({"x": [i]}).write.delta(uri, mode="append", auto_compact=True)

    assert _live_files(uri) == 3


def test_small_file_count_reads_the_log_not_the_files(tmp_path) -> None:
    """The auto-compaction check must be a metadata read — it runs after every write."""
    from batcher.io.formats.lakehouse import table_maintenance

    uri = _table(str(tmp_path / "t"))
    backend = table_maintenance("delta")

    assert backend.small_file_count(uri, below_bytes=1 << 30) == 6
    assert backend.small_file_count(uri, below_bytes=1) == 0


# --- the registry ----------------------------------------------------------


def test_only_transactional_formats_have_maintenance() -> None:
    from batcher.io.formats.lakehouse import table_maintenance

    assert table_maintenance("delta") is not None
    assert table_maintenance("iceberg") is not None
    assert table_maintenance("parquet") is None  # a file directory has no log


def test_iceberg_compaction_refuses_rather_than_pretending() -> None:
    """pyiceberg cannot rewrite data files, and saying so beats silently doing nothing.

    A maintenance call that quietly no-ops is worse than one that fails: the user comes
    away believing the table was compacted while the small-file problem is still there.
    """
    from batcher._internal.errors import BackendError
    from batcher.io.formats.lakehouse import table_maintenance

    pytest.importorskip("pyiceberg", reason="pyiceberg not installed")
    with pytest.raises(BackendError, match="rewrite_data_files"):
        table_maintenance("iceberg").compact("db.t")
