"""Page-level pruning skips pages, and never changes the answer.

Row-group pruning is coarse: a row group is ~1M rows, so a selective predicate still decoded
one in full. The writer has always emitted the ColumnIndex/OffsetIndex (per-page min/max plus
each page's first row index) precisely so a reader could do better; nothing read it back. On
TPC-H sf1 `lineitem`, `l_orderkey < 100` matches 105 rows and decoded 122,880 — 1,170x.

The reader now turns a pushed predicate into a `RowSelection` over the surviving pages.

**The correctness bar is the only thing that matters here.** Pruning is an optimization, so
it is allowed to select a *superset* of the matching rows (the engine's `Filter` re-checks
every row regardless) and never a subset. A subset is silent data loss — no error, just
missing rows — so most of this file is adversarial: predicates that must not prune, types the
index cannot decide, and `OR`ed columns where a wrong lattice would narrow the selection.

These run through `bt.read.parquet(...).filter(...)`, i.e. the real path, and compare against
the unfiltered read filtered in Python. If the two ever disagree, pruning dropped a row.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.unit

# Small pages so one row group holds many of them — that is what makes page pruning
# observable at a test-friendly size.
_PAGE_ROWS = 1_000
_ROWS = 50_000


@pytest.fixture
def sorted_file(tmp_path):
    """A clustered column, the shape page pruning exists for."""
    path = str(tmp_path / "sorted.parquet")
    table = pa.table(
        {
            "k": pa.array(range(_ROWS), pa.int64()),
            "g": pa.array([f"g{i % 100}" for i in range(_ROWS)]),
            "f": pa.array([float(i) for i in range(_ROWS)], pa.float64()),
        }
    )
    pq.write_table(
        table,
        path,
        row_group_size=_ROWS,  # ONE row group, so any win is page-level
        data_page_size=_PAGE_ROWS * 8,
        write_page_index=True,
    )
    return path


def _expected(path, mask) -> list:
    """The oracle: read everything, filter in Python."""
    table = pq.read_table(path)
    return [row for row in table.to_pylist() if mask(row)]


def _actual(dataset) -> list:
    return dataset.collect().to_pylist()


def test_a_selective_predicate_returns_exactly_the_matching_rows(sorted_file) -> None:
    """The headline case, and the one the amplification number comes from."""
    result = _actual(bt.read.parquet(sorted_file).filter(col("k") < 100))

    assert result == _expected(sorted_file, lambda r: r["k"] < 100)


@pytest.mark.parametrize(
    ("predicate", "mask"),
    [
        (col("k") < 100, lambda r: r["k"] < 100),
        (col("k") > 49_900, lambda r: r["k"] > 49_900),
        (col("k") == 25_000, lambda r: r["k"] == 25_000),
        (col("k") != 25_000, lambda r: r["k"] != 25_000),
        (col("k") <= 0, lambda r: r["k"] <= 0),
        (col("k") >= _ROWS - 1, lambda r: r["k"] >= _ROWS - 1),
        (col("f") < 100.0, lambda r: r["f"] < 100.0),
        (col("g") == "g7", lambda r: r["g"] == "g7"),
        (col("k") > _ROWS, lambda r: False),  # matches nothing at all
        (col("k") >= 0, lambda r: True),  # matches everything
    ],
)
def test_every_predicate_shape_agrees_with_the_oracle(sorted_file, predicate, mask) -> None:
    """Each op, each type, and both degenerate ends of the range."""
    assert _actual(bt.read.parquet(sorted_file).filter(predicate)) == _expected(sorted_file, mask)


def test_conjunctions_and_disjunctions_agree_with_the_oracle(sorted_file) -> None:
    """`And` intersects selections and `Or` unions them — a swapped lattice loses rows.

    `Or` is the dangerous one: if an undecidable side were treated as "select nothing"
    rather than "cannot decide", the union would narrow and drop real matches.
    """
    both = bt.read.parquet(sorted_file).filter((col("k") > 100) & (col("k") < 200))
    either = bt.read.parquet(sorted_file).filter((col("k") < 100) | (col("k") > 49_900))

    assert _actual(both) == _expected(sorted_file, lambda r: 100 < r["k"] < 200)
    assert _actual(either) == _expected(sorted_file, lambda r: r["k"] < 100 or r["k"] > 49_900)


def test_a_predicate_on_one_column_does_not_prune_by_another(sorted_file) -> None:
    """`g` is round-robin, so no page can be pruned by it — every row of `k < 100` must
    still appear. A column mix-up (matching a leaf name rather than the full path) is
    exactly how the row-group pruner once dropped rows."""
    result = _actual(bt.read.parquet(sorted_file).filter((col("k") < 100) & (col("g") == "g7")))

    assert result == _expected(sorted_file, lambda r: r["k"] < 100 and r["g"] == "g7")


def test_an_unclustered_column_still_returns_every_match(tmp_path) -> None:
    """Pruning must be *sound*, not just effective. On shuffled data almost no page can be
    dropped, and the answer must be complete anyway."""
    import numpy as np

    path = str(tmp_path / "shuffled.parquet")
    keys = np.random.default_rng(0).permutation(_ROWS)
    pq.write_table(
        pa.table({"k": pa.array(keys, pa.int64())}),
        path,
        row_group_size=_ROWS,
        data_page_size=_PAGE_ROWS * 8,
        write_page_index=True,
    )

    result = _actual(bt.read.parquet(path).filter(col("k") < 100))

    assert sorted(r["k"] for r in result) == list(range(100))


def test_a_file_without_a_page_index_is_unaffected(tmp_path) -> None:
    """The degradation path: no index means read the row group whole, as before."""
    path = str(tmp_path / "noindex.parquet")
    pq.write_table(
        pa.table({"k": pa.array(range(_ROWS), pa.int64())}),
        path,
        row_group_size=_ROWS,
        write_page_index=False,
    )

    result = _actual(bt.read.parquet(path).filter(col("k") < 100))

    assert sorted(r["k"] for r in result) == list(range(100))


def test_nulls_are_not_pruned_away(tmp_path) -> None:
    """A page whose bounds exclude the literal may still hold nulls, and `IS NULL` must
    find them. Null handling is where bounds-based pruning most easily goes wrong."""
    path = str(tmp_path / "nulls.parquet")
    values = [None if i % 1000 == 0 else i for i in range(_ROWS)]
    pq.write_table(
        pa.table({"k": pa.array(values, pa.int64())}),
        path,
        row_group_size=_ROWS,
        data_page_size=_PAGE_ROWS * 8,
        write_page_index=True,
    )

    nulls = _actual(bt.read.parquet(path).filter(col("k").is_null()))
    small = _actual(bt.read.parquet(path).filter(col("k") < 100))

    assert len(nulls) == _ROWS // 1000
    assert sorted(r["k"] for r in small) == list(range(1, 100))


def test_projection_and_pruning_compose(sorted_file) -> None:
    """Both push into the same decode; a row-selection bug shows up as misaligned columns."""
    result = _actual(bt.read.parquet(sorted_file).filter(col("k") < 100).select("k", "g"))

    assert result == [
        {"k": r["k"], "g": r["g"]} for r in _expected(sorted_file, lambda r: r["k"] < 100)
    ]


def test_count_after_pruning_is_exact(sorted_file) -> None:
    """A count is where an over- or under-selection is most visible."""
    assert bt.read.parquet(sorted_file).filter(col("k") < 100).count() == 100


# ---- that it actually prunes -------------------------------------------------


def test_pruning_actually_engages(tmp_path) -> None:
    """Every test above passes just as happily when NOTHING is pruned.

    That is not hypothetical — it is what this feature did on its first working build. The
    selection was computed correctly and then never applied, because
    `ParquetObjectReader::get_metadata` ignores `ArrowReaderOptions::with_page_index` and
    consults its own `preload_*` flags instead. No index was loaded, every page survived,
    and the results were perfectly correct. A green suite, a no-op feature.

    So this asserts the *decoder* did less work: the native reader returns only the rows it
    decoded, so a filtered read that returns fewer rows than the row group holds is direct
    evidence that pages were skipped.
    """
    import json

    from batcher._internal.native import engine

    rows = 200_000
    path = str(tmp_path / "engages.parquet")
    pq.write_table(
        pa.table({"k": pa.array(range(rows), pa.int64())}),
        path,
        row_group_size=rows,
        data_page_size=64 * 1024,
        write_page_index=True,
    )
    native = engine()
    predicate = json.dumps({"node": "cmp", "col": "k", "op": "lt", "lit": 100})

    whole = sum(b.num_rows for b in native.read_parquet(path, [0], None, 8192))
    pruned = sum(b.num_rows for b in native.read_parquet_filtered(path, [0], None, 8192, predicate))

    assert whole == rows
    assert pruned < whole, "page pruning did not engage — the whole row group was decoded"
    assert pruned >= 100, "pruning dropped rows that match the predicate"


def test_pruning_is_a_superset_never_a_subset(tmp_path) -> None:
    """The safety property stated as an invariant rather than per predicate.

    Selecting extra rows costs time; selecting too few loses data silently. Sweeping the
    literal across the whole key range exercises page boundaries, which is exactly where an
    off-by-one in the page→row-range mapping would show up.
    """
    import json

    from batcher._internal.native import engine

    rows = 100_000
    path = str(tmp_path / "superset.parquet")
    pq.write_table(
        pa.table({"k": pa.array(range(rows), pa.int64())}),
        path,
        row_group_size=rows,
        data_page_size=32 * 1024,
        write_page_index=True,
    )
    native = engine()

    for literal in (0, 1, 4095, 4096, 4097, 50_000, 99_999, 100_000):
        predicate = json.dumps({"node": "cmp", "col": "k", "op": "lt", "lit": literal})
        batches = native.read_parquet_filtered(path, [0], None, 8192, predicate)
        returned = {v for b in batches for v in b.column("k").to_pylist()}

        assert set(range(literal)) <= returned, f"k < {literal} lost matching rows"
