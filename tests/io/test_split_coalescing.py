"""A small-file corpus must plan as its bytes, not as its object count.

One split per file is the right unit while files are large and few. Past that it is the
whole cost of the read: every split is a scheduled task, a pickled locator, and a worker
round trip, so a directory of a million 4 KB objects becomes a million tasks to move four
gigabytes and no size of cluster helps, because the cost is per file rather than per byte.

These tests pin the two halves of the fix that can silently go wrong. The packing must
group *only* adjacent files and cover the input exactly once, or a read loses or duplicates
rows with nothing to reveal it. And the grouping must never make a read narrower than the
machine reading it, which is the failure mode a byte target alone walks straight into.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher.io.splits import FileSplit, MultiFileSplit, pack_files

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# The packing itself.
# --------------------------------------------------------------------------
def test_adjacent_files_are_grouped_up_to_the_target() -> None:
    assert pack_files([10, 10, 10, 10], target_bytes=20, min_runs=1) == [(0, 2), (2, 4)]


def test_a_file_larger_than_the_target_is_its_own_run_never_divided() -> None:
    """This packing groups; it cannot split a file. A run must therefore never be empty."""
    assert pack_files([100, 1, 1], target_bytes=10, min_runs=1) == [(0, 1), (1, 3)]


def test_the_runs_cover_the_input_exactly_once_and_in_order() -> None:
    sizes = [3, 1, 4, 1, 5, 9, 2, 6]
    runs = pack_files(sizes, target_bytes=7, min_runs=1)
    covered = [i for start, stop in runs for i in range(start, stop)]
    assert covered == list(range(len(sizes)))


def test_packing_never_leaves_fewer_runs_than_the_parallelism_floor() -> None:
    """Eight 10 MB files under a 128 MiB target is one task and seven idle cores. The floor
    re-packs against ``total / floor`` instead of accepting the collapse."""
    sizes = [10] * 8
    assert pack_files(sizes, target_bytes=10_000, min_runs=1) == [(0, 8)]
    assert len(pack_files(sizes, target_bytes=10_000, min_runs=4)) >= 4


def test_fewer_files_than_the_floor_are_never_grouped_at_all() -> None:
    """The floor binds at the file count, so grouping engages only once files outnumber the
    parallelism available to read them. Two tiny files stay two splits — halving a two-file
    read buys nothing and costs half its parallelism."""
    assert pack_files([10, 10], target_bytes=1_000, min_runs=64) == [(0, 1), (1, 2)]
    assert len(pack_files([10] * 5, target_bytes=1_000, min_runs=64)) == 5


def test_no_files_packs_to_no_runs() -> None:
    assert pack_files([], target_bytes=100, min_runs=8) == []


# --------------------------------------------------------------------------
# End to end through a real source.
# --------------------------------------------------------------------------
def _many_csvs(root, n: int) -> str:
    directory = root / "many"
    directory.mkdir()
    for i in range(n):
        (directory / f"c{i:05d}.csv").write_text(f"a,b\n{i},{i}\n")
    return str(directory)


@pytest.fixture
def small_box(monkeypatch):
    """Pin the parallelism floor, so these tests measure the packing and not the host.

    `_MIN_SPLITS` is ``max(64, cores * 8)``, which is the right production figure and the
    wrong thing for a test to depend on: on a 96-core machine the floor is 768, so 2,000 tiny
    files pack to ~1,000 runs and a 400-file corpus produces no `MultiFileSplit` at all. The
    assertions below then failed for a property of the box rather than of the code, and would
    equally have *passed* on a small box with the packing broken. 64 is the floor's own lower
    bound, so this pins it to a value production also uses.
    """
    from batcher.io.base import source as source_mod

    monkeypatch.setattr(source_mod, "_MIN_SPLITS", 64)


def _split_paths(splits) -> list[str]:
    """Every file each split covers, in split order."""
    out: list[str] = []
    for split in splits:
        out.extend(split.paths if isinstance(split, MultiFileSplit) else [split.path])
    return out


def test_a_directory_of_tiny_files_plans_far_fewer_splits_than_files(tmp_path, small_box) -> None:
    from batcher.io.formats.structured.csv import CSVSource

    source = CSVSource(_many_csvs(tmp_path, 2_000))
    splits = source.splits()

    assert len(splits) < len(source._files()) / 4
    assert any(isinstance(s, MultiFileSplit) for s in splits)


def test_grouped_splits_cover_every_file_exactly_once(tmp_path, small_box) -> None:
    """The correctness half. A packing bug here loses or duplicates rows silently."""
    from batcher.io.formats.structured.csv import CSVSource

    source = CSVSource(_many_csvs(tmp_path, 2_000))

    assert _split_paths(source.splits()) == source._files()


def test_a_grouped_read_returns_exactly_the_rows_of_an_ungrouped_one(tmp_path) -> None:
    directory = _many_csvs(tmp_path, 500)

    rows = sorted(r["a"] for r in bt.read.csv(directory).collect().to_pylist())

    assert rows == list(range(500))


def test_a_multi_file_split_streams_the_same_rows_it_reads(tmp_path, small_box) -> None:
    from batcher.io.formats.structured.csv import CSVSource

    source = CSVSource(_many_csvs(tmp_path, 400))
    grouped = next(s for s in source.splits() if isinstance(s, MultiFileSplit))

    read = pa.Table.from_batches(grouped.read()).column("a").to_pylist()
    streamed = pa.Table.from_batches(list(grouped.iter_batches())).column("a").to_pylist()

    assert read == streamed
    assert len(read) == len(grouped.paths)


def test_a_multi_file_split_honors_a_projection(tmp_path, small_box) -> None:
    from batcher.io.formats.structured.csv import CSVSource

    source = CSVSource(_many_csvs(tmp_path, 400))
    grouped = next(s for s in source.splits() if isinstance(s, MultiFileSplit))

    assert pa.Table.from_batches(grouped.read(["b"])).column_names == ["b"]


def test_grouped_splits_have_distinct_identities(tmp_path, small_box) -> None:
    """Identity keys a split's learned statistics; two groups sharing one would have them
    describe each other's files."""
    from batcher.io.formats.structured.csv import CSVSource

    splits = CSVSource(_many_csvs(tmp_path, 2_000)).splits()
    identities = [s.identity() for s in splits]

    assert len(set(identities)) == len(identities)


def test_a_few_large_files_are_left_one_split_each(tmp_path) -> None:
    """Grouping is for the small-file case. Files already worth a task apiece keep one, or
    coalescing would trade a real read's parallelism away for nothing."""
    from batcher.io.formats.structured.parquet.source import ParquetSource

    directory = tmp_path / "big"
    directory.mkdir()
    payload = pa.table({"x": list(range(200_000))})
    for i in range(4):
        pq.write_table(payload, directory / f"p{i}.parquet")

    splits = ParquetSource(str(directory)).splits()

    assert not any(isinstance(s, MultiFileSplit) for s in splits)


def test_a_pinned_file_list_still_plans_one_split_per_file(tmp_path) -> None:
    """Sizes come from the directory listing, and a pinned subset never had one. Rather than
    pay an O(files) HEAD sweep to plan a read whose cost is what grouping reduces, this
    degrades to the previous behavior."""
    from batcher.io.formats.structured.csv import CSVSource

    directory = _many_csvs(tmp_path, 20)
    files = CSVSource(directory)._files()

    splits = CSVSource(directory, files=files).splits()

    assert all(isinstance(s, FileSplit) for s in splits)
    assert len(splits) == len(files)
