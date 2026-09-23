"""Multi-file schema modes and CSV reads: a distributed read must answer as a local one does.

Every case reads the same files twice, with ``collect(distributed=False)`` and with
``collect(distributed=True, num_workers=4)``, and demands the same row multiset *and* the
same column types, or the same exception. The defects behind it all passed a local test:

- a strict read (the default) of files that disagreed returned data distributed where it
  raised locally, because a split rebuilds a one-file reader whose contract is its own file:
  a column renamed from ``A`` to ``a`` came back as one row of two, an ``int64`` file as
  ``float64``;
- a CSV byte range that began inside a quoted field invented rows the file never held;
- ``schema_mode="union"`` over CSV failed distributed, each split asking its file for the
  union's columns;
- a UDF whose every partition came back empty returned a table with no columns.

`test_the_workers_really_split_the_work` is the positive control: without it a matrix that
agrees could be agreeing because one worker computed both sides.
"""

from __future__ import annotations

import dataclasses
import re

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher._internal.errors import SchemaError

pytestmark = pytest.mark.integration

pytest.importorskip("ray", reason="ray not installed")

from _ray_cluster import init_test_ray, shutdown_test_ray  # noqa: E402  (after importorskip)

_WORKERS = 4
_MODES = ("strict", "union", "latest")


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    started = init_test_ray(4)
    yield
    shutdown_test_ray(started)


def _fingerprint(table: pa.Table) -> tuple[str, list[str]]:
    """The schema and an order-independent row multiset, both exact."""
    ordered = table.select(sorted(table.column_names))
    return str(table.schema), sorted(repr(row) for row in ordered.to_pylist())


def _outcome(query: bt.Dataset, **kw):
    try:
        return _fingerprint(query.collect(**kw))
    except SchemaError as exc:
        return exc


def _assert_same_answer(make_query) -> object:
    """Same rows and types on both paths, or the same `SchemaError` about the same file."""
    local = _outcome(make_query(), distributed=False)
    spread = _outcome(make_query(), distributed=True, num_workers=_WORKERS)
    if isinstance(local, SchemaError):
        assert isinstance(spread, SchemaError), f"local raised {local}, distributed returned data"
        # A worker's error arrives wrapped in its traceback; `cause` is the error it raised.
        cause = getattr(spread, "cause", spread)
        assert _first_line(local) == _first_line(cause), (str(local), str(cause))
    else:
        assert not isinstance(spread, SchemaError), f"distributed raised {spread}"
        assert spread == local
    return local


def _first_line(exc: Exception) -> str:
    """The first sentence of a message, with the offending file's name made generic.

    When several files break the contract, which one is reported depends on which read
    fails first, and on the distributed path that is whichever worker gets there first.
    Every one of them is a correct answer, so the comparison is on what was said about it.
    """
    return re.sub(r"f\d+\.(parquet|csv)", "f<n>", str(exc).split(". ")[0])


# --- positive control --------------------------------------------------------------------


@pytest.fixture(scope="module")
def spread_source(cluster_scratch) -> str:
    directory = cluster_scratch("schema_csv_dist_control")
    for part in range(4):
        xs = list(range(part * 500, (part + 1) * 500))
        pq.write_table(
            pa.table({"x": xs, "g": [x % 17 for x in xs]}), directory / f"p{part}.parquet"
        )
    return str(directory)


def test_the_workers_really_split_the_work(spread_source):
    """A LIMIT over an unordered group_by keeps different groups once the work is split."""
    query = bt.read.parquet(spread_source).group_by("g").agg(n=bt.col("x").sum()).limit(3)
    single = query.collect(distributed=False)
    spread = query.collect(distributed=True, num_workers=_WORKERS)
    assert single.num_rows == spread.num_rows == 3
    assert set(single["g"].to_pylist()) != set(spread["g"].to_pylist())


# --- Parquet: every schema mode over drifting files --------------------------------------


def _four(first: pa.Table, rest: pa.Table) -> list[pa.Table]:
    return [first, first, rest, rest]


_PARQUET = {
    "homogeneous": _four(pa.table({"a": [1, 2]}), pa.table({"a": [3, 4]})),
    "added_column": _four(pa.table({"a": [1]}), pa.table({"a": [2], "b": ["x"]})),
    "narrower_later": _four(pa.table({"a": [1]}), pa.table({"a": pa.array([2], pa.int32())})),
    "renamed_column": [pa.table({"A": [1]})] + [pa.table({"a": [i]}) for i in range(3)],
    "int_then_float": _four(pa.table({"a": [1]}), pa.table({"a": [2.5]})),
    "missing_column": [pa.table({"a": [1], "b": ["x"]})] + [pa.table({"a": [i]}) for i in range(3)],
    "null_typed_then_int": _four(
        pa.table({"a": pa.array([None], pa.null())}), pa.table({"a": [5]})
    ),
}


@pytest.fixture(scope="module")
def parquet_dirs(cluster_scratch) -> dict[str, str]:
    out = {}
    for case, tables in _PARQUET.items():
        directory = cluster_scratch(f"schema_csv_dist_pq_{case}")
        for i, table in enumerate(tables):
            pq.write_table(table, directory / f"f{i}.parquet")
        out[case] = str(directory)
    return out


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("case", sorted(_PARQUET))
def test_parquet_schema_mode_answers_the_same_distributed(parquet_dirs, case, mode):
    _assert_same_answer(lambda: bt.read.parquet(parquet_dirs[case], schema_mode=mode))


def test_a_renamed_column_raises_rather_than_losing_a_row(parquet_dirs):
    """The audit's shape: four workers returned one row of the four, and said nothing."""
    outcome = _assert_same_answer(lambda: bt.read.parquet(parquet_dirs["renamed_column"]))
    assert isinstance(outcome, SchemaError)


def test_union_matches_duckdbs_union_by_name(parquet_dirs):
    directory = parquet_dirs["missing_column"]
    ours = bt.read.parquet(directory, schema_mode="union").collect(
        distributed=True, num_workers=_WORKERS
    )
    oracle = duckdb.sql(
        f"select * from read_parquet('{directory}/*.parquet', union_by_name=true)"
    ).to_arrow_table()
    assert _fingerprint(ours)[1] == _fingerprint(oracle)[1]


# --- CSV: every schema mode over drifting headers ----------------------------------------


_CSV = {
    "homogeneous": ["a,b\n1,x\n", "a,b\n2,y\n", "a,b\n3,z\n", "a,b\n4,w\n"],
    "added_column": ["a,b\n1,x\n", "a,b\n2,y\n", "a,b,c\n3,z,q\n", "a,b,c\n4,w,r\n"],
    "reordered_header": ["a,b\n1,x\n", "b,a\ny,2\n", "a,b\n3,z\n", "b,a\nw,4\n"],
    # Both later files hold the same bad value, so the error reads the same whichever of them
    # a worker happens to reach first.
    "int_then_float": ["a,b\n1,x\n", "a,b\n2,y\n", "a,b\n3.5,z\n", "a,b\n3.5,w\n"],
    "missing_column": ["a,b\n1,x\n", "a\n2\n", "a\n3\n", "a\n4\n"],
}


@pytest.fixture(scope="module")
def csv_dirs(cluster_scratch) -> dict[str, str]:
    out = {}
    for case, files in _CSV.items():
        directory = cluster_scratch(f"schema_csv_dist_csv_{case}")
        for i, content in enumerate(files):
            (directory / f"f{i}.csv").write_text(content)
        out[case] = str(directory)
    return out


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("case", sorted(_CSV))
def test_csv_schema_mode_answers_the_same_distributed(csv_dirs, case, mode):
    _assert_same_answer(lambda: bt.read.csv(csv_dirs[case], schema_mode=mode))


def test_csv_union_reads_distributed_as_duckdb_does(csv_dirs):
    directory = csv_dirs["added_column"]
    ours = bt.read.csv(directory, schema_mode="union").collect(
        distributed=True, num_workers=_WORKERS
    )
    oracle = duckdb.sql(
        f"select * from read_csv('{directory}/*.csv', union_by_name=true)"
    ).to_arrow_table()
    assert _fingerprint(ours)[1] == _fingerprint(oracle)[1]


# --- CSV byte ranges over a quoted newline -----------------------------------------------


@pytest.fixture
def small_ranges():
    from batcher.config import active_config, config_context

    config = active_config()
    small = config.replace(execution=dataclasses.replace(config.execution, split_bytes=300))
    with config_context(small):
        yield


@pytest.mark.usefixtures("small_ranges")
def test_a_quoted_newline_invents_no_rows_distributed(cluster_scratch):
    from batcher.io.formats.structured.csv import CSVRangeSplit, CSVSource

    directory = cluster_scratch("schema_csv_dist_quoted_newline")
    path = directory / "quoted.csv"
    path.write_text("a,b\n" + "".join(f'{i},"x\n{i + 1000},fake"\n' for i in range(200)))
    splits = CSVSource(str(path)).splits()
    assert sum(isinstance(s, CSVRangeSplit) for s in splits) > 1, "the file must be cut"
    local = _assert_same_answer(lambda: bt.read.csv(str(path)))
    oracle = duckdb.sql(f"select count(*) from read_csv('{path}')").fetchone()[0]
    assert len(local[1]) == oracle == 200
    assert all("1000" not in row.split(",")[0] for row in local[1])


# --- an empty UDF result keeps its columns -----------------------------------------------


_EMPTY_UDFS = {
    "map_batches_to_zero_rows": lambda ds: ds.map_batches(lambda b: b.slice(0, 0)),
    "map_batches_new_column": lambda ds: ds.map_batches(
        lambda b: pa.record_batch({"y": pa.array([], pa.string())})
    ),
    "flat_map_to_nothing": lambda ds: ds.flat_map(lambda row: []),
    "filter_fn_keeps_nothing": lambda ds: ds.filter(lambda b: pc.greater(b["x"], 10**9)),
}


@pytest.mark.parametrize("case", sorted(_EMPTY_UDFS))
def test_an_empty_udf_result_keeps_its_columns(spread_source, case):
    local = _assert_same_answer(lambda: _EMPTY_UDFS[case](bt.read.parquet(spread_source)))
    schema, rows = local
    assert rows == [] and schema != ""
