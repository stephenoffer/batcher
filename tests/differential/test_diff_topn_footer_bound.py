"""A top-N seeded from Parquet row-group statistics on its *first* run still matches DuckDB.

`kyber.learned_tuning.topn_footer` proves a bound from the footers, so unlike the learned bound
it rewrites the very first run, and on every scheduling -- a streamed run included, which has
no row count to check the bound against. So each case runs once per scheduling and compares
**positionally** against DuckDB reading the same files: `assert_same` is order-independent and
cannot see the one thing this optimization could get wrong, which is *which* rows a sort keeps.

The data is built to attack the proof. The key is clustered across row groups, so the bound
really does prune, and it carries dense ties that straddle row-group boundaries, so the k-th
value is shared by rows in groups the bound keeps and groups it could be tempted to drop. Nulls
are scattered through it, because the proof counts only non-null rows toward `k`.

Each shape also carries a control that the bound engaged -- rows returned by the reader -- so a
change that silently stopped seeding cannot pass as "still correct", and the declined shapes
assert the reader was *not* narrowed, so a change that seeded them cannot either.
"""

from __future__ import annotations

import datetime

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _harness import assert_same_ordered
from batcher.api.orchestration import topn_seeding
from batcher.io.formats.structured.parquet.source import ParquetSource

pytestmark = pytest.mark.differential

_ROWS = 24_000
_ROW_GROUP = 1_000


@pytest.fixture(autouse=True)
def _seed_small_relations(monkeypatch):
    """Let a test-sized relation seed; the production floor is a cost guard, not semantics."""
    monkeypatch.setattr(topn_seeding, "_MIN_FOOTER_SEED_ROWS", 0)


@pytest.fixture
def rows_read(monkeypatch):
    """Total rows every `ParquetSource.read` / `iter_batches` returned during the test."""
    seen = {"rows": 0}
    read, stream = ParquetSource.read, ParquetSource.iter_batches

    def _read(self, projection=None, predicate=None):
        out = read(self, projection, predicate)
        seen["rows"] += sum(b.num_rows for b in out)
        return out

    def _stream(self, projection=None, predicate=None):
        for batch in stream(self, projection, predicate):
            seen["rows"] += batch.num_rows
            yield batch

    monkeypatch.setattr(ParquetSource, "read", _read)
    monkeypatch.setattr(ParquetSource, "iter_batches", _stream)
    return seen


def _key_values(kind: str, ints: np.ndarray) -> pa.Array:
    if kind == "int64":
        return pa.array(ints, pa.int64())
    if kind == "int32":
        return pa.array(ints.astype("int32"), pa.int32())
    if kind == "date32":
        return pa.array(ints.astype("int32"), pa.date32())
    if kind == "timestamp_ms":
        return pa.array(ints * 1000, pa.timestamp("ms"))
    if kind == "timestamp_tz":
        return pa.array(ints * 1_000_000, pa.timestamp("us", tz="UTC"))
    raise AssertionError(kind)


def _write(tmp_path, kind: str, *, reverse: bool = False, float_key: bool = False) -> str:
    """Two files of clustered, tie-dense, null-scattered keys, many row groups each."""
    rng = np.random.default_rng(7)
    # Blocks of 700 equal values: every block straddles a 1,000-row group boundary somewhere.
    ints = (np.arange(_ROWS) // 700).astype("int64")
    if reverse:
        ints = ints[::-1].copy()
    nulls = rng.random(_ROWS) < 0.05
    key = _key_values(kind, ints) if not float_key else pa.array(ints.astype("float64"))
    key = pa.array(
        [None if n else v for v, n in zip(key.to_pylist(), nulls, strict=True)], key.type
    )
    table = pa.table({"x": key, "p": np.arange(_ROWS, dtype="int64"), "v": rng.random(_ROWS)})
    half = _ROWS // 2
    out = tmp_path / kind
    out.mkdir()
    pq.write_table(table.slice(0, half), out / "a.parquet", row_group_size=_ROW_GROUP)
    pq.write_table(table.slice(half), out / "b.parquet", row_group_size=_ROW_GROUP)
    return str(out)


def _stream(ds) -> pa.Table:
    batches = list(ds.iter_batches())
    return pa.Table.from_batches(batches) if batches else ds.collect().slice(0, 0)


_SCHEDULINGS = {
    "collect": lambda ds: ds.collect(),
    "spill": lambda ds: ds.collect(spill=True),
    "iter_batches": _stream,
}


@pytest.mark.parametrize("kind", ["int64", "int32", "date32", "timestamp_ms", "timestamp_tz"])
@pytest.mark.parametrize("descending", [True, False])
@pytest.mark.parametrize("k", [1, 7, 1_500])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("scheduling", sorted(_SCHEDULINGS))
def test_first_run_matches_duckdb_and_prunes(
    duck, tmp_path, rows_read, kind, descending, k, reverse, scheduling
):
    path = _write(tmp_path, kind, reverse=reverse)
    direction = "DESC" if descending else "ASC"
    sql = (
        f"SELECT p, x IS NULL AS x_null FROM read_parquet('{path}/*.parquet') "
        f"ORDER BY x {direction} NULLS LAST, p {direction} LIMIT {k}"
    )

    query = bt.read.parquet(f"{path}/*.parquet").sort("x", "p", descending=descending).limit(k)
    got = _SCHEDULINGS[scheduling](query)

    # `p` is unique, so comparing it in order pins exactly which rows came back and where.
    # The key itself is compared through `p` rather than directly because DuckDB renders a
    # zoned timestamp in the session's zone, which is a presentation difference, not a result.
    result = pa.table({"p": got.column("p"), "x_null": pa.compute.is_null(got.column("x"))})
    assert_same_ordered(result, duck.sql(sql))
    assert rows_read["rows"] < _ROWS // 2, "the footer bound did not narrow the read"


def test_a_renamed_projection_still_seeds_and_matches(duck, tmp_path, rows_read):
    path = _write(tmp_path, "int64")
    query = (
        bt.read.parquet(f"{path}/*.parquet")
        .select(bt.col("x").alias("key"), "p")
        .sort("key", "p", descending=True)
        .limit(10)
    )
    sql = (
        f"SELECT x AS key, p FROM read_parquet('{path}/*.parquet') "
        "ORDER BY key DESC NULLS LAST, p DESC LIMIT 10"
    )
    assert_same_ordered(query.collect(), duck.sql(sql))
    assert rows_read["rows"] < _ROWS // 2


@pytest.mark.parametrize(
    "shape",
    ["nulls_first", "float_key", "filter_below_sort", "computed_key", "offset"],
)
def test_declined_shapes_match_duckdb_and_read_everything(duck, tmp_path, rows_read, shape):
    """Each of these would be unsound or unprovable to seed, so the reader must see it all."""
    path = _write(tmp_path, "int64", float_key=shape == "float_key")
    ds = bt.read.parquet(f"{path}/*.parquet")
    src = f"read_parquet('{path}/*.parquet')"
    if shape == "nulls_first":
        query = ds.sort("x", "p", descending=True, nulls_first=True).limit(5)
        sql = f"SELECT p FROM {src} ORDER BY x DESC NULLS FIRST, p DESC LIMIT 5"
    elif shape == "float_key":
        query = ds.sort("x", "p", descending=True).limit(5)
        sql = f"SELECT p FROM {src} ORDER BY x DESC NULLS LAST, p DESC LIMIT 5"
    elif shape == "filter_below_sort":
        query = ds.filter(bt.col("v") > 0.5).sort("x", "p", descending=True).limit(5)
        sql = f"SELECT p FROM {src} WHERE v > 0.5 ORDER BY x DESC NULLS LAST, p DESC LIMIT 5"
    elif shape == "computed_key":
        query = (
            ds.with_columns((bt.col("x") * -1).alias("neg"))
            .sort("neg", "p", descending=True)
            .limit(5)
        )
        sql = f"SELECT p FROM {src} ORDER BY -x DESC NULLS LAST, p DESC LIMIT 5"
    else:
        query = ds.sort("x", "p", descending=True).limit(5, offset=3)
        sql = f"SELECT p FROM {src} ORDER BY x DESC NULLS LAST, p DESC LIMIT 5 OFFSET 3"

    got = query.collect()
    assert_same_ordered(got.select(["p"]), duck.sql(sql))
    # A declined shape is read in full. The filtered shape may prune on `v` itself, which is
    # its own pushdown and not this bound, so it is held to "not narrowed on the key".
    if shape != "filter_below_sort":
        assert rows_read["rows"] == _ROWS


def test_the_production_floor_keeps_small_relations_out(tmp_path, monkeypatch, rows_read):
    """Positive control for the fixture that lowers the floor: at its real value, no seed."""
    monkeypatch.setattr(topn_seeding, "_MIN_FOOTER_SEED_ROWS", 1_000_000)
    path = _write(tmp_path, "int64")
    got = bt.read.parquet(f"{path}/*.parquet").sort("x", "p", descending=True).limit(3).collect()
    assert got.num_rows == 3
    assert rows_read["rows"] == _ROWS


def test_a_date_bound_filters_on_the_date_not_a_timestamp(duck, tmp_path):
    """A date column's footer bound is a `date`, and the seeded filter must compare as one."""
    path = _write(tmp_path, "date32")
    got = bt.read.parquet(f"{path}/*.parquet").sort("x", "p", descending=True).limit(3).collect()
    last = max(v for v in got.column("x").to_pylist() if v is not None)
    assert isinstance(last, datetime.date)
    sql = (
        f"SELECT x, p FROM read_parquet('{path}/*.parquet') "
        "ORDER BY x DESC NULLS LAST, p DESC LIMIT 3"
    )
    assert_same_ordered(got.select(["x", "p"]), duck.sql(sql))
