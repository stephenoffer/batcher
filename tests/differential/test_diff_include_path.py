"""`include_path=` names each row's source file the way DuckDB's ``filename=true`` does (AP-427).

The Spark ``input_file_name()`` gap recorded in the migration ledger. Compared as a row
multiset against DuckDB over the same files, through `collect`, `iter_batches`, and the
split pickle round trip a distributed read takes.
"""

from __future__ import annotations

import pickle

import pyarrow as pa
import pytest
from tests._harness import assert_same

import batcher as bt
from batcher._internal.errors import FormatError
from batcher.io.source import plan_splits

pytestmark = pytest.mark.differential

_PARTS = {
    "a": {"k": [1, 2, None], "v": ["x", "y", "y"]},  # nulls and duplicates
    "b": {"k": [3], "v": [None]},  # one row
    "c": {"k": [], "v": []},  # empty
}


@pytest.fixture(scope="module", params=["parquet", "csv"])
def corpus(request, tmp_path_factory):
    fmt = request.param
    root = tmp_path_factory.mktemp(f"files_{fmt}")
    schema = pa.schema([("k", pa.int64()), ("v", pa.string())])
    for name, cols in _PARTS.items():
        ds = bt.from_arrow(pa.table(cols, schema=schema))
        getattr(ds.write, fmt)(str(root / f"{name}.{fmt}"), single_file=True)
    return fmt, str(root)


def _duck(fmt: str, root: str, column: str = "path"):
    duckdb = pytest.importorskip("duckdb")
    reader = "read_parquet" if fmt == "parquet" else "read_csv"
    return duckdb.connect().sql(
        f"SELECT k, v, filename AS {column} FROM {reader}('{root}/*.{fmt}', filename=true)"
    )


def _read(fmt: str, root: str, **opts) -> bt.Dataset:
    return getattr(bt.read, fmt)(f"{root}/*.{fmt}", **opts)


def test_collect_matches_duckdb_filename(corpus):
    fmt, root = corpus
    assert_same(_read(fmt, root, include_path=True).to_arrow(), _duck(fmt, root))


def test_a_named_column(corpus):
    fmt, root = corpus
    got = _read(fmt, root, include_path="src").to_arrow()
    assert got.column_names == ["k", "v", "src"]
    assert_same(got, _duck(fmt, root, "src"))


def test_iter_batches_matches_collect(corpus):
    fmt, root = corpus
    ds = _read(fmt, root, include_path=True)
    streamed = pa.Table.from_batches(list(ds.iter_batches()))
    assert_same(streamed, _duck(fmt, root))


def test_splits_survive_the_pickle_a_distributed_read_takes(corpus):
    fmt, root = corpus
    source = _read(fmt, root, include_path=True)._sources[0]
    splits = [pickle.loads(pickle.dumps(s)) for s in plan_splits(source)]
    assert len(splits) == len(_PARTS)  # one per file: the attribution is per split
    batches = [b for s in splits for b in s.read()]
    got = pa.Table.from_batches(batches, schema=splits[0].schema())
    assert_same(
        got.cast(pa.schema([("k", pa.int64()), ("v", pa.string()), ("path", pa.string())])),
        _duck(fmt, root),
    )


def test_projecting_only_the_path_keeps_every_row(corpus):
    fmt, root = corpus
    got = _read(fmt, root, include_path=True).select("path").to_arrow()
    assert_same(got, _duck(fmt, root).select("path"))


def test_filter_and_group_by_the_path(corpus):
    fmt, root = corpus
    ds = _read(fmt, root, include_path=True)
    got = ds.filter(bt.col("path").str.ends_with(f"a.{fmt}")).group_by("path").agg(n=bt.count())
    duck = _duck(fmt, root).filter(f"path LIKE '%a.{fmt}'").aggregate("path, count(*) AS n")
    assert_same(got.to_arrow(), duck)


def test_without_the_option_nothing_changes(corpus):
    fmt, root = corpus
    assert _read(fmt, root).columns == ["k", "v"]


def test_a_colliding_name_is_refused(corpus):
    fmt, root = corpus
    with pytest.raises(FormatError, match="already has a column named 'v'"):
        _read(fmt, root, include_path="v")


def test_a_non_file_source_is_refused(corpus):
    _, root = corpus
    with pytest.raises(FormatError, match="file readers only"):
        bt.read(root, format="parquet_dataset", include_path=True)


def test_union_schema_mode_null_fills_and_tags(tmp_path):
    bt.from_pydict({"k": [1]}).write.parquet(str(tmp_path / "a.parquet"), single_file=True)
    bt.from_pydict({"k": [2], "extra": ["e"]}).write.parquet(
        str(tmp_path / "b.parquet"), single_file=True
    )
    got = bt.read.parquet(str(tmp_path), schema_mode="union", include_path=True).to_arrow()
    duckdb = pytest.importorskip("duckdb")
    duck = duckdb.connect().sql(
        f"SELECT k, extra, filename AS path FROM read_parquet('{tmp_path}/*.parquet', "
        "union_by_name=true, filename=true)"
    )
    assert_same(got, duck)


def test_a_row_cap_is_a_whole_source_cap(tmp_path):
    for name in ("a", "b"):
        bt.from_pydict({"k": [1, 2, 3]}).write.json(
            str(tmp_path / f"{name}.json"), single_file=True
        )
    ds = bt.read.json(str(tmp_path), n_rows=4, include_path=True)
    got = ds.to_pydict()
    assert len(got["k"]) == 4
    assert got["path"] == [str(tmp_path / "a.json")] * 3 + [str(tmp_path / "b.json")]
