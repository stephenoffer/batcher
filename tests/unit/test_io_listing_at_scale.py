"""Source discovery must survive a corpus of millions of files.

Every test here pins a property that decides whether a large multimodal or lakehouse
corpus can be *planned* at all, as opposed to one that decides whether a query is fast.
They are separable, and the distinction matters: a listing change silently alters which
files a query reads, so most of these assert a **result**, and only the explicitly-named
ones assert a call count.

The call-counting tests monkeypatch the listing/footer entry points and count invocations
rather than timing anything, so they are deterministic and need no object store: the thing
that makes a million-file read impossible is the *number* of driver-side round trips, and
that is exactly what is counted.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from batcher.io.formats.structured.parquet.source import ParquetSource
from batcher.io.splits import FileSplit, NormalizedFileSplit, RowGroupSplit, WholeSourceSplit

pytestmark = pytest.mark.unit


def _write(path, table: pa.Table) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))
    return str(path)


def _t(**cols) -> pa.Table:
    return pa.table(cols)


# ---------------------------------------------------------------------------
# Globbing: a wildcard in a DIRECTORY component (audit item 2)
# ---------------------------------------------------------------------------


@pytest.fixture
def hive_tree(tmp_path):
    """``data/date=<d>/part-0.parquet`` — the layout every partitioned writer produces."""
    for day in ("2024-01-01", "2024-01-02", "2024-01-03"):
        _write(tmp_path / "data" / f"date={day}" / "part-0.parquet", _t(x=[1, 2]))
    return tmp_path / "data"


def test_a_wildcard_in_a_directory_component_matches_its_files(hive_tree) -> None:
    """``data/*/part-0.parquet`` raised "matched no files" instead of matching.

    The old `_glob` listed the parent of the first wildcard NON-recursively and then
    `fnmatch`ed full paths, so it saw only ``data``'s direct children — the partition
    directories, never the files inside them — and concluded the glob matched nothing.
    That is the most common layout there is, and the failure mode was an exception on a
    pattern every other engine accepts.
    """
    source = ParquetSource(f"{hive_tree}/*/part-0.parquet")

    assert len(source._files()) == 3
    assert source.read()[0].num_rows == 2


def test_a_wildcard_directory_component_pairs_with_a_wildcard_filename(hive_tree) -> None:
    """The same fault, with both components globbed (``data/date=*/*.parquet``)."""
    source = ParquetSource(f"{hive_tree}/date=*/*.parquet")

    assert len(source._files()) == 3


def test_a_directory_wildcard_selects_only_the_matching_partitions(hive_tree) -> None:
    """Partition pruning by glob must actually prune — this is a correctness surface.

    If the directory component were ignored (or matched too broadly) the query would
    silently read partitions the user excluded, which is a wrong answer rather than a
    slow one.
    """
    source = ParquetSource(f"{hive_tree}/date=2024-01-0[12]/part-0.parquet")

    assert sorted(p.rsplit("/", 2)[1] for p in source._files()) == [
        "date=2024-01-01",
        "date=2024-01-02",
    ]


def test_a_single_star_does_not_cross_a_directory_boundary(tmp_path) -> None:
    """`*` matches within one path segment; only `**` descends.

    `fnmatch`'s `*` crosses `/`, so matching full paths with it conflates
    ``dir/*.parquet`` with ``dir/**/*.parquet`` — a read that quietly pulls in an entire
    nested corpus when the user asked for one directory.
    """
    _write(tmp_path / "d" / "top.parquet", _t(x=[1]))
    _write(tmp_path / "d" / "nested" / "deep.parquet", _t(x=[1]))

    shallow = ParquetSource(f"{tmp_path}/d/*.parquet")._files()
    deep = ParquetSource(f"{tmp_path}/d/**/*.parquet")._files()

    assert [p.rsplit("/", 1)[-1] for p in shallow] == ["top.parquet"]
    assert [p.rsplit("/", 1)[-1] for p in deep] == ["deep.parquet"]


def test_a_directory_wildcard_lists_only_the_matching_partitions(tmp_path, monkeypatch) -> None:
    """The walk must LIST per matching component, not list the whole subtree and filter.

    This is the scale property: on a corpus with millions of objects beneath ``data``, a
    flat recursive listing materializes every path on the driver before matching. Walking
    component by component touches only the partitions the pattern selects.
    """
    for day in range(6):
        for part in range(4):
            _write(tmp_path / "data" / f"date=d{day}" / f"part-{part}.parquet", _t(x=[1]))

    from batcher.io.filesystem import resolve_filesystem

    fs = resolve_filesystem(str(tmp_path))
    listed_dirs: list[str] = []
    original = type(fs)._list_dir

    def _spy(self, in_path: str):
        listed_dirs.append(in_path)
        return original(self, in_path)

    # A `**` glob would instead go through `_recursive_glob`, which lists the whole subtree
    # in one shot — assert that path is never taken for a single-star pattern.
    recursive_calls = []
    monkeypatch.setattr(
        type(fs),
        "_recursive_glob",
        lambda self, p: recursive_calls.append(p) or [],
    )
    monkeypatch.setattr(type(fs), "_list_dir", _spy)

    files = fs.expand(f"{tmp_path}/data/date=d1/*.parquet", suffix=".parquet")

    assert len(files) == 4
    assert recursive_calls == [], "a non-`**` glob must never list the whole subtree"
    # Only `data` (to match `date=d1`) and the one matching partition are listed — never
    # the five sibling partitions the pattern excludes.
    assert not any("date=d0" in d or "date=d2" in d for d in listed_dirs), (
        "the walk must list only the partitions the pattern selects, not every sibling"
    )


# ---------------------------------------------------------------------------
# The glob stat-storm: listing info must be recorded (audit item 3)
# ---------------------------------------------------------------------------


def test_a_glob_records_listing_info_so_identity_needs_no_stat(tmp_path) -> None:
    """A glob dropped the size/mtime its own listing returned, forcing a stat per file.

    `file_identity` then stat-ed every matched file — three times per query, each its own
    thread-pool task. On a 2,000-file read that storm outweighed the Parquet read itself;
    at a million files it is the whole query. The listing already carries both fields.
    """
    for i in range(5):
        _write(tmp_path / f"part-{i}.parquet", _t(x=[1]))

    from batcher.io.filesystem import resolve_filesystem

    fs = resolve_filesystem(str(tmp_path))
    files = fs.expand(f"{tmp_path}/*.parquet", suffix=".parquet")

    assert len(files) == 5
    assert all(fs.listing_info(f) is not None for f in files), (
        "every globbed file's (size, mtime) must come from the listing, not a stat"
    )


def test_the_prefix_scoped_remote_glob_also_records_listing_info(monkeypatch) -> None:
    """The fsspec fast path recorded nothing, so the stat storm survived on remote globs.

    The test above passes on a *local* path, and `_glob_prefix_scoped` declines local
    schemes outright — so the one glob shape that skipped the recording was the one nothing
    covered. It is also the shape that needs it most: the fast path is taken exactly for
    ``dir/PREFIX*.ext``, the many-small-files layout it exists to speed up. Measured on a
    1,024-file S3 corpus before the fix: 3,072 `_stat` calls costing 1.58 s against a 340 ms
    read of the same bytes.

    fsspec is faked rather than reached, so this asserts the contract (list once, record what
    came back) with no network and no credentials.
    """
    import sys
    import types

    from batcher.io._backend import _ArrowFileSystem

    listed = {
        "bucket/dir/part-0.parquet": {"size": 11, "LastModified": 1_700_000_000.0},
        "bucket/dir/part-1.parquet": {"size": 22, "LastModified": 1_700_000_001.0},
        "bucket/dir/_SUCCESS": {"size": 0, "LastModified": 1_700_000_002.0},
    }
    calls = {"n": 0}

    class _FakeBackend:
        def glob(self, pattern, detail=False):
            calls["n"] += 1
            assert detail, "the listing must be requested with its detail, not re-stat-ed"
            return dict(listed)

    fake = types.ModuleType("fsspec")
    fake.filesystem = lambda _scheme: _FakeBackend()
    monkeypatch.setitem(sys.modules, "fsspec", fake)

    import pyarrow.fs as pafs

    fs = _ArrowFileSystem(pafs.LocalFileSystem(), "s3://", atomic_rename=False)
    files = fs._glob_prefix_scoped("s3://bucket/dir/part-*.parquet", "bucket/dir/part-*.parquet")

    assert calls["n"] == 1, "one LIST, not one per file"
    assert files == ["s3://bucket/dir/part-0.parquet", "s3://bucket/dir/part-1.parquet"], (
        "the marker file must still be filtered out of the returned set"
    )
    for path, size in (("part-0.parquet", 11), ("part-1.parquet", 22)):
        info = fs.listing_info(f"s3://bucket/dir/{path}")
        assert info is not None, f"{path} must answer from the listing, not a stat"
        assert info[0] == size
        assert info[1] > 0, "the listing's timestamp must survive into the identity"


def test_glob_listing_info_does_not_survive_this_filesystem_overwriting_the_file(
    tmp_path,
) -> None:
    """Recording listing info is only safe because a write invalidates it.

    Filesystem objects are cached and long-lived, so without this a source that listed a
    directory and then overwrote a file in it would keep reporting the pre-overwrite
    ``(size, mtime)``. Every metadata cache keyed on that identity would then serve the
    PREVIOUS file's footer and row count for the new bytes — stale metadata about new
    data, which is worse than a stale file: row-group offsets index into the middle of it.
    """
    target = _write(tmp_path / "part-0.parquet", _t(x=[1]))

    from batcher.io.filesystem import resolve_filesystem

    fs = resolve_filesystem(str(tmp_path))
    fs.expand(f"{tmp_path}/*.parquet", suffix=".parquet")
    assert fs.listing_info(target) is not None

    with fs.atomic_writer(target) as fh:
        pq.write_table(_t(x=[1, 2, 3]), fh)

    assert fs.listing_info(target) is None, (
        "a path this filesystem just wrote must not answer from the pre-write listing"
    )


# ---------------------------------------------------------------------------
# The driver footer storm (audit item 6)
# ---------------------------------------------------------------------------


def _count_footer_reads(monkeypatch) -> list[str]:
    """Record every Parquet footer read, however it is reached."""
    from batcher.io.splits import parquet as parquet_splits

    seen: list[str] = []
    original = parquet_splits._read_footer

    def _spy(path: str):
        seen.append(path)
        return original(path)

    monkeypatch.setattr(parquet_splits, "_read_footer", _spy)
    parquet_splits._FOOTERS.clear()
    return seen


def test_planning_a_wide_corpus_does_not_read_a_footer_per_file(tmp_path, monkeypatch) -> None:
    """`splits()` fanned 64 threads over EVERY file on the driver, before any task ran.

    At a million files that is a million metadata GETs while the whole cluster idles, to
    subdivide files already far more numerous than the workers. Past the threshold the
    plan is one whole-file split per file, which reads no footer at all.
    """
    for i in range(12):
        _write(tmp_path / f"part-{i}.parquet", _t(x=[1, 2]))
    monkeypatch.setenv("BATCHER_MAX_FOOTER_PLAN_FILES", "4")
    import importlib

    from batcher.io.base import source as source_module

    importlib.reload(source_module)
    try:
        seen = _count_footer_reads(monkeypatch)
        splits = source_module.FileSource.splits(ParquetSource(str(tmp_path)))

        assert len(splits) == 12
        assert all(isinstance(s, FileSplit) for s in splits)
        assert seen == [], "planning a wide corpus must not read a footer per file"
    finally:
        importlib.reload(source_module)


def test_a_narrow_corpus_still_gets_row_group_splits(tmp_path) -> None:
    """The threshold must not cost sub-file parallelism where it is the point.

    A handful of large files is exactly the shape row-group splits exist for, so the
    footer sweep stays on below the threshold.
    """
    _write(tmp_path / "part-0.parquet", _t(x=list(range(100))))

    splits = ParquetSource(str(tmp_path)).splits()

    assert splits and all(isinstance(s, RowGroupSplit) for s in splits)


def test_a_predicate_keeps_the_footer_sweep_even_on_a_wide_corpus(tmp_path, monkeypatch) -> None:
    """Skipping footers must not silently disable plan-time file skipping.

    Footer statistics let `_file_splits` drop row-groups — often whole files — before they
    become tasks. On a large selective scan that pruning is worth far more than the sweep
    costs, so a pushed predicate suspends the threshold.
    """
    for i in range(12):
        _write(tmp_path / f"part-{i}.parquet", _t(x=[i, i + 1]))
    monkeypatch.setenv("BATCHER_MAX_FOOTER_PLAN_FILES", "4")
    import importlib

    from batcher.io.base import source as source_module

    importlib.reload(source_module)
    try:
        predicate = {
            "e": "binary",
            "op": "gt",
            "left": {"e": "col", "name": "x"},
            "right": {"e": "lit", "value": 100},
        }
        splits = source_module.FileSource.splits(ParquetSource(str(tmp_path)), None, predicate)

        assert not any(isinstance(s, FileSplit) for s in splits), (
            "a pushed predicate must still reach the footers that prune with it"
        )
    finally:
        importlib.reload(source_module)


# ---------------------------------------------------------------------------
# Schema evolution must not destroy read parallelism (audit item 9)
# ---------------------------------------------------------------------------


@pytest.fixture
def evolving(tmp_path):
    """Three files whose schemas differ — a column added, then a type promoted."""
    _write(tmp_path / "p0.parquet", _t(id=[1, 2]))
    _write(tmp_path / "p1.parquet", _t(id=[3, 4], tag=["a", "b"]))
    _write(tmp_path / "p2.parquet", _t(id=[5.5, 6.5], tag=["c", "d"]))
    return tmp_path


def test_a_schema_evolving_source_splits_per_file(evolving) -> None:
    """A schema-evolving read collapsed to ONE `WholeSourceSplit` — one task, one worker.

    That is correct and unscalable: a petabyte-scale dataset whose producers disagree
    about the schema is precisely the one that must fan out, and it was the only shape
    guaranteed not to.
    """
    splits = ParquetSource(str(evolving), schema_mode="union").splits()

    assert len(splits) == 3
    assert not any(isinstance(s, WholeSourceSplit) for s in splits)
    assert all(isinstance(s, NormalizedFileSplit) for s in splits)


def test_per_file_evolving_splits_reproduce_the_whole_source_read(evolving) -> None:
    """The parallel plan must return exactly what the single-task plan returned.

    This is the load-bearing assertion of the change: per-file splits are only admissible
    if each one reshapes its file to the SAME unified schema the whole-source read
    unifies to. A mismatch here is batches that will not concatenate, or worse, silently
    mistyped columns.
    """
    source = ParquetSource(str(evolving), schema_mode="union")
    whole = pa.Table.from_batches(source.read(), schema=source.schema())

    splits = source.splits()
    per_file = pa.Table.from_batches([b for s in splits for b in s.read()], schema=source.schema())

    assert per_file.schema == whole.schema
    assert per_file.sort_by("id").equals(whole.sort_by("id"))


def test_an_evolving_split_fills_a_missing_column_with_nulls(evolving) -> None:
    """A file predating a column addition must still produce that column.

    The worker cannot ask its reader for a column the file does not have, so the request
    is trimmed and `normalize_batch` fills it back in. Getting this wrong raises on the
    worker rather than on the driver, which is where it would be noticed.
    """
    source = ParquetSource(str(evolving), schema_mode="union")
    first = next(s for s in source.splits() if s.path.endswith("p0.parquet"))

    batches = first.read()

    assert "tag" in batches[0].schema.names
    assert batches[0].column("tag").null_count == batches[0].num_rows


def test_an_evolving_split_honors_a_projection(evolving) -> None:
    """Projection must narrow the unified schema, not the file's own."""
    source = ParquetSource(str(evolving), schema_mode="union")
    first = next(s for s in source.splits() if s.path.endswith("p0.parquet"))

    batches = first.read(["tag"])

    assert batches[0].schema.names == ["tag"]


def test_an_evolving_split_is_picklable(evolving) -> None:
    """Splits ship to a worker, so a carried `pa.Schema` must survive the trip."""
    import pickle

    split = ParquetSource(str(evolving), schema_mode="union").splits()[0]

    assert pickle.loads(pickle.dumps(split)).schema() == split.schema()


def test_a_capped_source_still_reads_as_one_split(evolving) -> None:
    """`n_rows` caps the SOURCE, so it can never be distributed across splits.

    Each split would honor the cap independently and the union would return
    ``n_rows x len(splits)`` rows. The per-file change above must not reach this case.
    """
    splits = ParquetSource(str(evolving), schema_mode="union", n_rows=3).splits()

    assert len(splits) == 1
    assert isinstance(splits[0], WholeSourceSplit)
