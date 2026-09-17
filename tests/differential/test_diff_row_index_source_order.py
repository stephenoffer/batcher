"""`with_row_index` numbers rows in source order, under every scheduling that could reorder them.

The explicit row-order policy (every order-dependent expression needs ``order_by``) tells a
user with no natural key to write ``.with_row_index("_row")`` right after the read and order
by ``"_row"``. That advice is only sound if the index really is the *source* order: the
first file's rows first, each file's rows in file order, and nothing reshuffled by the
parallel scan. At 200 rows nothing shards, so the fixture is sized past the engine's
``MIN_ROWS_TO_SHARD`` (4 morsels = 65,536 rows) and split over several Parquet files, each
spanning a morsel boundary.

The oracle is a ``seq`` column written into the files themselves, monotonically increasing
across files in path order. DuckDB confirms that reading the same glob in file order gives
``seq`` back as ``0..n-1``, so the property checked is the source's order and not an
assumption about it.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytestmark = pytest.mark.differential

bt = pytest.importorskip("batcher")

_FILES = 5
_ROWS_PER_FILE = 20_011  # 100,055 rows: past the sharding threshold, prime-ish per file.
_TOTAL = _FILES * _ROWS_PER_FILE


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("row_index_source")
    for f in range(_FILES):
        start = f * _ROWS_PER_FILE
        seq = list(range(start, start + _ROWS_PER_FILE))
        # `v` is deliberately not monotone, so an engine that sorted by it would be caught.
        table = pa.table({"seq": seq, "v": [(s * 7919) % 1009 for s in seq]})
        pq.write_table(table, root / f"part-{f:03d}.parquet", row_group_size=7_000)
    return root


def _check(indexed: pa.Table) -> None:
    assert indexed.num_rows == _TOTAL
    rows = indexed.column("_row").to_pylist()
    seqs = indexed.column("seq").to_pylist()
    # Positional: the index must equal the source sequence on every row, which fails for a
    # reordered scan even when the multiset of indexes is right.
    mismatched = [i for i, (r, s) in enumerate(zip(rows, seqs, strict=True)) if r != s]
    assert not mismatched, f"{len(mismatched)} rows off source order, first at {mismatched[0]}"


def test_the_source_order_is_the_path_order(source: Path, duck) -> None:
    """Positive control: the files really do carry ``seq`` as ``0..n-1`` in path order."""
    glob = str(source / "*.parquet")
    got = duck.sql(
        f"SELECT seq FROM read_parquet('{glob}', filename=true, file_row_number=true) "
        "ORDER BY filename, file_row_number"
    )
    seqs = [row[0] for row in got.fetchall()]
    assert seqs == list(range(_TOTAL))


def test_collect_assigns_the_index_in_source_order(source: Path) -> None:
    ds = bt.read.parquet(str(source)).with_row_index("_row")
    _check(ds.collect())


def test_iter_batches_assigns_the_index_in_source_order(source: Path) -> None:
    ds = bt.read.parquet(str(source)).with_row_index("_row")
    batches = list(ds.iter_batches())
    assert len(batches) > 1, "the fixture must stream in several batches to test anything"
    _check(pa.Table.from_batches(batches))


def test_partitioned_collect_assigns_the_index_in_source_order(source: Path) -> None:
    ds = bt.read.parquet(str(source)).with_row_index("_row")
    _check(ds.collect(num_partitions=4))


def test_the_index_survives_a_downstream_order_dependent_window(source: Path) -> None:
    """The policy's own recipe: index, then a running value ordered by the index."""
    ds = (
        bt.read.parquet(str(source))
        .with_row_index("_row")
        .with_columns(prev=bt.col("seq").shift(1).over(order_by="_row"))
        .sort("_row")
    )
    out = ds.collect()
    prev = out.column("prev").to_pylist()
    assert prev[0] is None
    assert prev[1:] == list(range(_TOTAL - 1))
