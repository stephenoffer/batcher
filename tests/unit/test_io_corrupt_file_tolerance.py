"""`FileSource(on_error=...)` — one unreadable file must not cost the whole corpus.

Before this existed the IO layer had no `try`/`except` anywhere in the read spine, so a
single truncated shard aborted a ten-thousand-file read. These tests pin both halves of
the contract: `raise` (the default) still aborts, and `skip` returns the readable rows
*and* names what it dropped — because a silently-partial read is worse than a loud
failure.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from batcher._internal.errors import ConfigError
from batcher.io.formats.structured.parquet.source import ParquetSource

pytestmark = pytest.mark.unit


@pytest.fixture
def corpus(tmp_path):
    """Four readable Parquet files plus one that is not Parquet at all."""
    for i in range(4):
        pq.write_table(
            pa.table({"id": list(range(i * 100, (i + 1) * 100))}), tmp_path / f"p{i}.parquet"
        )
    # Sorts last, so the readable files are already in flight when it fails.
    (tmp_path / "zbad.parquet").write_bytes(b"not a parquet file")
    return str(tmp_path)


def test_default_still_raises(corpus) -> None:
    with pytest.raises(Exception):  # noqa: B017 — the backend's own decode error
        ParquetSource(corpus).read()


def test_skip_drops_only_the_bad_file_on_read(corpus) -> None:
    src = ParquetSource(corpus, on_error="skip")
    rows = sum(b.num_rows for b in src.read())

    assert rows == 400
    assert [p.rsplit("/", 1)[-1] for p in src.corrupt_files()] == ["zbad.parquet"]


def test_skip_drops_only_the_bad_file_on_iter_batches(corpus) -> None:
    """The streaming path has its own error handling and must agree with `read`."""
    src = ParquetSource(corpus, on_error="skip")
    rows = sum(b.num_rows for b in src.iter_batches())

    assert rows == 400
    assert [p.rsplit("/", 1)[-1] for p in src.corrupt_files()] == ["zbad.parquet"]


def test_skip_preserves_values_not_merely_the_count(corpus) -> None:
    src = ParquetSource(corpus, on_error="skip")
    ids = pa.Table.from_batches(src.read()).column("id").to_pylist()

    assert sorted(ids) == list(range(400))


def test_a_clean_corpus_reports_nothing_skipped(tmp_path) -> None:
    pq.write_table(pa.table({"id": [1, 2, 3]}), tmp_path / "a.parquet")
    src = ParquetSource(str(tmp_path), on_error="skip")

    assert sum(b.num_rows for b in src.read()) == 3
    assert src.corrupt_files() == []


def test_unknown_mode_is_rejected_at_construction(corpus) -> None:
    with pytest.raises(ConfigError, match="on_error"):
        ParquetSource(corpus, on_error="permissive")


def test_split_kwargs_carry_on_error(corpus) -> None:
    """`on_error` must ride the split, or it is void wherever a worker rebuilds the reader.

    A worker reconstructs the reader as `SOURCES.get(fmt)(path, **split.kwargs)`, so a
    policy missing from those kwargs silently reverts to the `"raise"` default. This
    asserts the kwargs themselves, because that dict is the whole contract with the
    distributed executor, the streaming reader, and the GPU backend.
    """
    splits = ParquetSource(corpus, on_error="skip").splits()

    assert splits, "expected per-file splits to exercise the rebuild path"
    assert all(s.kwargs.get("on_error") == "skip" for s in splits)


def test_clean_read_keeps_row_group_splits(tmp_path) -> None:
    """A default read must keep the sub-file fast path — tolerance is what costs it.

    Uses a clean corpus: with a corrupt file present the default `"raise"` correctly
    aborts split planning, which is a different contract (asserted below).
    """
    for i in range(3):
        pq.write_table(pa.table({"id": list(range(100))}), tmp_path / f"p{i}.parquet")

    splits = ParquetSource(str(tmp_path)).splits()

    assert splits
    assert not any(getattr(s, "kwargs", {}).get("on_error") for s in splits)


def test_planning_raises_on_a_corrupt_file_by_default(corpus) -> None:
    """Split planning reads footers, so it must fail fast unless tolerance was asked for."""
    with pytest.raises(Exception):  # noqa: B017 — the backend's own decode error
        ParquetSource(corpus).splits()


def test_planning_survives_a_corrupt_file_under_skip(corpus) -> None:
    """Split planning must not abort the query before a worker ever runs.

    This is the driver-side half of the contract. Planning reads file metadata (a Parquet
    footer), so it trips on exactly the corruption a tolerated read exists to survive —
    and it trips on the *driver*. Before this, `on_error="skip"` covered the read but not
    the plan, so one truncated shard still killed the whole distributed query.

    Parquet resolves this by planning a whole-file split under `skip` (it carries the
    policy; a `RowGroupSplit` cannot), which defers the failure to the worker rather than
    skipping here — so `corrupt_files()` stays empty on the driver and the drop is
    reported by the read, asserted in `test_rebuilt_split_reader_tolerates_its_bad_file`.
    """
    splits = ParquetSource(corpus, on_error="skip").splits()

    assert len(splits) == 5, "one whole-file split per file, bad one included"
    assert all(s.kwargs.get("on_error") == "skip" for s in splits)


def test_rebuilt_split_reader_tolerates_its_bad_file(corpus) -> None:
    """The end of the contract that matters: a rebuilt reader survives the corrupt file.

    Reads every split the way a worker does. Under `skip` the bad split yields nothing
    and the healthy ones still yield all 400 rows; before `on_error` was threaded this
    raised out of the corrupt split instead.
    """
    splits = ParquetSource(corpus, on_error="skip").splits()

    rows = 0
    for split in splits:
        reader = split._reader()
        rows += sum(b.num_rows for b in reader.read())

    assert rows == 400


def test_rebuilt_split_reader_still_raises_by_default(corpus) -> None:
    """The tolerance must come from the policy, not from splitting papering over it."""
    with pytest.raises(Exception):  # noqa: B017 — the backend's own decode error
        for split in ParquetSource(corpus).splits():
            split._reader().read()
