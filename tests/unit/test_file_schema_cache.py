"""A file's schema is read once per identity, and a rewritten file re-reads it.

`schema()` memoizes on the source object, which is rebuilt every time a user writes
`bt.read.parquet(uri)`, so a repeated query re-opened files to re-read schemas that had not
changed -- two opens per query on the strict path (file 0 for the schema, the last file for the
dropped-column warning), 58-71 ms each over S3 (`benchmarks/BENCHMARK_RESULTS.md`).

The correctness half is what these pin. A schema served after the file was rewritten would
type the read against columns it no longer has, so identity must invalidate.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher.io.base import source as source_mod

pytestmark = pytest.mark.unit


def _write(path, names):
    pq.write_table(pa.table({n: pa.array(np.arange(4), pa.int64()) for n in names}), path)


@pytest.fixture
def fresh_cache(monkeypatch):
    """An empty schema cache, and a count of the reads that actually reached a file."""
    monkeypatch.setattr(source_mod, "_FILE_SCHEMAS", source_mod.FileMetaCache(64))
    reads = []
    # `_read_schema` is overridden per format, so counting it on the ABC intercepts nothing --
    # the first version of this fixture did that and its positive control caught it. Count the
    # pyarrow call the Parquet reader actually makes, which is the real "a file was opened".
    original = pq.read_schema

    def counted(*a, **k):
        reads.append(1)
        return original(*a, **k)

    monkeypatch.setattr(pq, "read_schema", counted)
    return reads


def test_a_second_read_of_the_same_files_reads_no_schema(tmp_path, fresh_cache):
    """The point: two S3 opens per query, paid once per file instead of per query."""
    _write(tmp_path / "a.parquet", ["x", "y"])
    _write(tmp_path / "b.parquet", ["x", "y"])
    glob = f"{tmp_path}/*.parquet"

    first = bt.read.parquet(glob).schema
    reads_after_first = len(fresh_cache)
    second = bt.read.parquet(glob).schema

    assert reads_after_first > 0, "the first read must actually open a file"
    assert len(fresh_cache) == reads_after_first, "the second must be served from the cache"
    assert [f.name for f in first] == [f.name for f in second] == ["x", "y"]


def test_a_rewritten_file_is_re_read(tmp_path, fresh_cache):
    """The correctness half: a stale schema would type the read against columns that are gone."""
    path = tmp_path / "a.parquet"
    _write(path, ["x", "y"])
    glob = f"{tmp_path}/*.parquet"

    assert [f.name for f in bt.read.parquet(glob).schema] == ["x", "y"]
    before = len(fresh_cache)

    _write(path, ["x", "y", "z"])  # same path, different content
    again = [f.name for f in bt.read.parquet(glob).schema]

    assert len(fresh_cache) > before, "a changed file must be re-read, not served"
    assert again == ["x", "y", "z"]


def test_two_sources_over_the_same_file_share_the_read(tmp_path, fresh_cache):
    """The cache is keyed by file, not by `Dataset`, which is the whole point."""
    _write(tmp_path / "a.parquet", ["x"])
    glob = f"{tmp_path}/*.parquet"

    seen = [bt.read.parquet(glob).schema]
    n = len(fresh_cache)
    seen.extend(bt.read.parquet(glob).schema for _ in range(3))

    assert len(fresh_cache) == n
    assert all([f.name for f in s] == ["x"] for s in seen)
