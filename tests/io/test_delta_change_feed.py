"""The `delta_cdf` source, which had no test at all.

`DeltaChangeFeedSource` is registered as `delta_cdf` in the `SOURCES` registry and nothing
exercised it -- not by name, not through `read_changes`, not through a `cdf=` argument.
Sixty-nine source formats are registered and this was the one no test reached, found by
reconciling the registry against the suite. It works; it was simply unverified, which is a
different thing from correct and stops being true silently.

The sinks already have that reconciliation: `tests/io/test_format_fidelity_matrix.py` holds
every registry sink in `LOCAL_FILE_SINKS` or in `NOT_LOCALLY_WRITABLE` with a reason, and
fails on a sink in neither. The sources have no equivalent, which is why this gap could sit
there.

What is pinned here is the contract a CDF consumer depends on, not merely that a read
returns rows: the row-level change *type*, the commit version each change came from, and
that the version window is genuinely half-bounded. An incremental ETL step reads this with a
watermark and merges the result into a target, so a window that silently ignored
`starting_version` would re-apply every change from the beginning of the table on every run.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

pytest.importorskip("deltalake", reason="deltalake not installed")

from deltalake import DeltaTable, write_deltalake

from batcher.io.formats.lakehouse.delta.stream import DeltaChangeFeedSource

pytestmark = pytest.mark.integration

#: The table property the feed requires. Without it delta-rs records no change data and the
#: source has nothing to read, which is a configuration error rather than an empty result.
_CDF_ENABLED = {"delta.enableChangeDataFeed": "true"}


def _changes(uri: str, **window) -> pa.Table | None:
    batches = list(DeltaChangeFeedSource(uri, **window).read())
    return pa.Table.from_batches(batches) if batches else None


@pytest.fixture
def table_with_an_insert_and_a_delete(tmp_path):
    """Two commits: version 0 inserts two rows, version 1 deletes one."""
    uri = str(tmp_path / "cdf")
    write_deltalake(uri, pa.table({"k": [1, 2], "v": ["a", "b"]}), configuration=_CDF_ENABLED)
    DeltaTable(uri).delete("k = 1")
    assert DeltaTable(uri).version() == 1
    return uri


def test_the_feed_carries_the_change_type_beside_the_data(table_with_an_insert_and_a_delete):
    """`_change_type` is the column the whole feed exists for."""
    changes = _changes(table_with_an_insert_and_a_delete, starting_version=0)
    assert changes is not None, "an enabled feed over two commits produced no rows"

    assert set(changes.column_names) >= {
        "k",
        "v",
        "_change_type",
        "_commit_version",
        "_commit_timestamp",
    }
    assert sorted(set(changes.column("_change_type").to_pylist())) == ["delete", "insert"]
    assert changes.num_rows == 3, "two inserts at v0 and one delete at v1"


def test_each_change_carries_the_commit_it_came_from(table_with_an_insert_and_a_delete):
    """`_commit_version` is what a consumer stores as its watermark."""
    changes = _changes(table_with_an_insert_and_a_delete, starting_version=0)
    by_version = dict(
        zip(
            changes.column("_change_type").to_pylist(),
            changes.column("_commit_version").to_pylist(),
            strict=True,
        )
    )
    assert by_version["insert"] == 0
    assert by_version["delete"] == 1


def test_starting_version_bounds_the_window(table_with_an_insert_and_a_delete):
    """The property an incremental read depends on.

    A window that ignored `starting_version` would re-apply every change from the beginning
    of the table on every run -- correct-looking rows, and a consumer that double-counts
    forever. Asserted against the full window rather than a bare count, so it fails if the
    bound is dropped *or* if it swallows everything.
    """
    uri = table_with_an_insert_and_a_delete
    full = _changes(uri, starting_version=0)
    later = _changes(uri, starting_version=1)

    assert later is not None
    assert later.num_rows < full.num_rows, "`starting_version=1` returned the whole feed"
    assert sorted(set(later.column("_change_type").to_pylist())) == ["delete"]
    assert set(later.column("_commit_version").to_pylist()) == {1}


def test_a_table_without_the_feed_enabled_is_an_error_not_an_empty_read(tmp_path):
    """The control, and the reason an empty result is not good enough.

    Without this, every assertion above would still pass against a source that returned
    nothing at all for any input -- `_changes` would be `None` and the tests would need to
    tolerate it. A table with the property switched off must be distinguishable from a table
    with no changes.
    """
    uri = str(tmp_path / "plain")
    write_deltalake(uri, pa.table({"k": [1], "v": ["a"]}))
    with pytest.raises(Exception, match=r"(?i)change data|cdf|not enabled|configuration"):
        _changes(uri, starting_version=0)
