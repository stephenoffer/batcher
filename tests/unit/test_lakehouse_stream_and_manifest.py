"""A change feed must not report data consumed before the consumer has it.

`snapshot_position()` / `seek()` exist so a streaming consumer can checkpoint where it got
to and resume there. `DeltaStreamSource.iter_batches` advanced the cursor to the latest
version **before the first `yield`** — so the position said "consumed" while the batches
were still sitting inside an unstarted generator.

A consumer that checkpoints and then dies partway through the drain resumes at
`latest + 1` and never sees the rest. Measured on a two-commit table: the consumer took
one batch of 3 rows, and resuming from its own checkpoint returned **0 of the remaining
3**. Silent, unrecoverable, and worse the larger the window.

Advancing after the drain makes the stream at-least-once — replay a window on failure,
which is the correct failure mode for a change feed and what every checkpointing consumer
already handles.

Also covered here: two statistics paths that failed on shapes the *server* controls rather
than the user — an add-action log without `num_records`, and a shared-file manifest whose
first file has no stats.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

pytestmark = pytest.mark.unit

deltalake = pytest.importorskip("deltalake")


@pytest.fixture
def cdf_table(tmp_path):
    """A change-data-feed table with two commits, three rows each."""
    from deltalake import write_deltalake

    path = str(tmp_path / "t")
    write_deltalake(
        path,
        pa.table({"x": [1, 2, 3]}),
        mode="overwrite",
        configuration={"delta.enableChangeDataFeed": "true"},
    )
    write_deltalake(path, pa.table({"x": [4, 5, 6]}), mode="append")
    return path


def _source(path):
    from batcher.io.formats.lakehouse.delta.stream import DeltaStreamSource

    return DeltaStreamSource(path)


def test_a_partial_consumer_does_not_lose_the_rest(cdf_table) -> None:
    """The bug, end to end: checkpoint after one batch, resume, and the rest is gone."""
    source = _source(cdf_table)
    stream = source.iter_batches()
    next(stream)  # the consumer takes one batch, then fails

    resumed = _source(cdf_table)
    resumed.seek(source.snapshot_position())

    assert sum(b.num_rows for b in resumed.iter_batches()) == 6


def test_the_cursor_does_not_move_until_the_drain_completes(cdf_table) -> None:
    """The mechanism under the bug, asserted directly."""
    source = _source(cdf_table)
    before = source.snapshot_position()
    stream = source.iter_batches()
    next(stream)

    assert source.snapshot_position() == before, "the cursor advanced mid-drain"


def test_a_full_drain_does_advance_the_cursor(cdf_table) -> None:
    """At-least-once must not become always-replay: a completed pass has to commit."""
    source = _source(cdf_table)

    first = sum(b.num_rows for b in source.iter_batches())
    second = sum(b.num_rows for b in source.iter_batches())

    assert first == 6
    assert second == 0, "a completed window was replayed"


def test_an_abandoned_generator_leaves_the_cursor_alone(cdf_table) -> None:
    """Abandoning the iterator is the ordinary shape of a consumer giving up."""
    source = _source(cdf_table)
    before = source.snapshot_position()

    for _ in source.iter_batches():
        break

    assert source.snapshot_position() == before


def test_an_empty_window_still_advances(cdf_table) -> None:
    """Nothing to deliver means nothing to lose — the window must not be re-read forever."""
    source = _source(cdf_table)
    list(source.iter_batches())
    position = source.snapshot_position()

    list(source.iter_batches())

    assert source.snapshot_position() == position


# ---- statistics that must degrade, not raise ---------------------------------


def test_row_count_survives_a_log_without_num_records(tmp_path, monkeypatch) -> None:
    """A writer is not obliged to record `num_records`, and the planner can do without it.

    Reading the column unguarded raised `KeyError` out of a best-effort statistic, failing
    the whole query for a number that was only ever an optimization. `_snapshot` already
    guards the same column for its zone maps.
    """
    from deltalake import write_deltalake

    from batcher.io.formats.lakehouse.delta.source import DeltaSource

    path = str(tmp_path / "t")
    write_deltalake(path, pa.table({"x": [1, 2, 3]}), mode="overwrite")
    source = DeltaSource(path)
    monkeypatch.setattr(
        type(source), "_add_actions", lambda self: pa.table({"path": ["a.parquet"]})
    )

    assert source.row_count() is None


def test_the_sharing_manifest_keeps_stats_when_the_first_file_has_none() -> None:
    """`from_pylist` takes its columns from the first dict alone.

    A shared table whose first file carries no `stats` lost every `min.`/`max.` column —
    for the whole table, silently — so the zone maps the server sent were discarded and
    nothing was ever pruned.
    """
    from batcher.io.formats.lakehouse.delta_sharing import _stats_manifest

    class File:
        def __init__(self, url, stats):
            self.url = url
            self.stats = stats

    manifest = _stats_manifest(
        [
            File("a", None),  # no stats — this used to define the column set
            File("b", '{"numRecords": 10, "minValues": {"x": 1}, "maxValues": {"x": 5}}'),
        ]
    )

    assert manifest is not None
    assert "min.x" in manifest.column_names
    assert manifest.column("max.x").to_pylist() == [None, 5]


def test_the_sharing_split_carries_the_declared_row_count() -> None:
    """Otherwise `row_count()` opens a pre-signed URL per split to re-read a footer the
    manifest already parsed."""
    from batcher.io.formats.lakehouse.delta_sharing import DeltaSharingFileSplit

    assert DeltaSharingFileSplit(file_url="https://x/a.parquet", rows=42).row_count() == 42
