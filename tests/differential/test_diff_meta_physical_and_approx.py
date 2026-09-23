"""`ds.meta.storage`, `ds.meta.approx`, and the shapes the shortcut cross-check did not cover.

`test_diff_metadata_shortcuts` holds every *exact* shortcut to the executed answer over one
table. This file covers what it left out, which is where the audit found the bugs:

* **storage**, against what pyarrow reads from the same files: a single Parquet file, a
  directory of multi-row-group files, a hive-partitioned tree, and an in-memory relation.
  ``files()`` used to return ``[]`` for all of them, so a small-files check always passed.
* **approx**, against DuckDB and pyarrow with a stated tolerance rather than ``is not None``.
  An approximation with no bound checked is a number nobody verified.
* **"unknown" is None, not 0.0.** A plan the metadata layer cannot see used to report an
  estimated zero rows and zero bytes, which reads as "empty" and sizes a buffer at nothing.
* **type mismatches raise one `PlanError`** on both the metadata and the executed path,
  instead of a vacuous ``True`` from one and an engine error from the other.
* **fallbacks cost one query**, counted off the event bus rather than assumed.
* the fast-path-vs-execution comparison over a filtered, limited, projected, and joined
  Parquet scan, date and timestamp columns, and an empty Parquet file.

The forcing mechanism is the one `test_diff_metadata_shortcuts` explains: a ``map_batches``
identity is opaque to the IR, so every shortcut on it falls through to the engine.
"""

from __future__ import annotations

import datetime as dt
import glob
import math
import os
import random
from collections.abc import Callable
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytestmark = pytest.mark.differential

bt = pytest.importorskip("batcher")

from batcher._internal import events  # noqa: E402  (after the importorskip guard)
from batcher._internal.errors import ExecutionError, PlanError  # noqa: E402

_EMPTY_AGG_XFAIL = pytest.mark.xfail(
    raises=ExecutionError,
    strict=True,
    reason="engine: 'aggregation over empty input is not yet supported (no input schema)' "
    "when a map_batches stage over an empty Parquet file feeds an aggregate",
)


def _force(ds):
    """The same relation with the metadata layer switched off (an opaque identity stage)."""
    return ds.map_batches(lambda batch: batch)


def _same(a: Any, b: Any) -> bool:
    """Equality that treats NaN as NaN, ints as floats, and recurses into containers."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return bool(a == b)


def _executed_queries(call: Callable[[], Any]) -> tuple[Any, int]:
    """Run `call` and count the queries that *executed* — metadata shortcuts excluded.

    Every terminal closes with a ``QUERY_END`` event; one answered from metadata carries a
    ``shortcut`` field naming how. The rest ran the engine, and on a cluster each is a job.
    """
    executed: list[Any] = []

    def sink(event: Any) -> None:
        if event.kind == events.QUERY_END and not event.fields.get("shortcut"):
            executed.append(event)

    unsubscribe = events.subscribe(sink)
    try:
        result = call()
    finally:
        unsubscribe()
    return result, len(executed)


# --- fixtures ------------------------------------------------------------------------------

_N = 60
BASE = pa.table(
    {
        "k": pa.array(range(_N), pa.int64()),
        "v": pa.array([float(i % 17) * 1.5 for i in range(_N)], pa.float64()),
        "n": pa.array([None if i % 7 == 0 else i % 5 for i in range(_N)], pa.int64()),
        "s": pa.array([f"s{i % 9}" for i in range(_N)]),
        "g": pa.array(["x", "y", "z"] * (_N // 3)),
        "d": pa.array([dt.date(2024, 1, 1) + dt.timedelta(days=i % 40) for i in range(_N)]),
        "ts": pa.array(
            [dt.datetime(2024, 3, 1, 12) + dt.timedelta(hours=7 * i) for i in range(_N)],
            pa.timestamp("us"),
        ),
        "b": pa.array([i % 3 == 0 for i in range(_N)]),
    }
)


@pytest.fixture(scope="module")
def layouts(tmp_path_factory) -> dict[str, str]:
    """The same rows laid out four ways on disk, plus CSV and JSON copies."""
    root = tmp_path_factory.mktemp("meta_layouts")
    single = str(root / "single.parquet")
    pq.write_table(BASE, single, row_group_size=25)

    multi = root / "multi"
    multi.mkdir()
    for i in range(3):
        pq.write_table(BASE.slice(i * 20, 20), str(multi / f"part-{i}.parquet"), row_group_size=8)

    hive = str(root / "hive")
    pq.write_to_dataset(BASE.select(["k", "v", "g"]), hive, partition_cols=["g"])

    csv_dir = root / "csv"
    csv_dir.mkdir()
    for i in range(2):
        BASE.slice(i * 30, 30).select(["k", "s"]).to_pandas().to_csv(
            csv_dir / f"part-{i}.csv", index=False
        )
    json_file = str(root / "rows.jsonl")
    BASE.select(["k", "s"]).to_pandas().to_json(json_file, orient="records", lines=True)

    empty = str(root / "empty.parquet")
    pq.write_table(BASE.slice(0, 0), empty)

    other = str(root / "other.parquet")
    pq.write_table(
        pa.table({"k": pa.array(range(30, 90), pa.int64()), "w": list(range(60))}), other
    )
    return {
        "single": single,
        "multi": str(multi),
        "hive": hive,
        "csv_dir": str(csv_dir),
        "csv_file": str(csv_dir / "part-0.csv"),
        "json_file": json_file,
        "empty": empty,
        "other": other,
    }


def _parquet_files(path: str) -> list[str]:
    """Every Parquet file under `path` (or `path` itself), as pyarrow sees the layout."""
    if os.path.isfile(path):
        return [path]
    return sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))


def _norm(paths: list[str]) -> list[str]:
    return sorted(os.path.realpath(p.removeprefix("file://")) for p in paths)


# --- storage ------------------------------------------------------------------------------


@pytest.mark.parametrize("layout", ["single", "multi", "hive"])
def test_storage_answers_match_what_pyarrow_reads(layouts, layout):
    """Every `storage` method on a file-backed layout, against the files' own footers."""
    path = layouts[layout]
    files = _parquet_files(path)
    assert files, "the fixture must have written files for this layout"
    footers = [pq.ParquetFile(f).metadata for f in files]
    storage = bt.read.parquet(path).meta.storage

    assert _norm(storage.files()) == _norm(files)
    assert storage.num_files() == len(files)
    assert storage.num_sources() == 1
    assert storage.row_count() == sum(m.num_rows for m in footers) == _N
    assert storage.has_exact_row_count() is True
    assert storage.row_group_count() == sum(m.num_row_groups for m in footers)
    # The footer's per-row-group `total_byte_size`: column data before compression.
    recorded = sum(m.row_group(i).total_byte_size for m in footers for i in range(m.num_row_groups))
    assert storage.total_bytes() == recorded
    assert storage.bytes_per_row() == pytest.approx(recorded / _N)
    assert storage.partition_keys() == (("g",) if layout == "hive" else ())
    assert storage.is_partitioned() is (layout == "hive")
    assert storage.sorted_by() == ()


def test_the_multi_file_layout_really_has_several_row_groups_per_file(layouts):
    """The control for the test above: a one-row-group fixture would make that count vacuous."""
    assert all(pq.ParquetFile(f).num_row_groups > 1 for f in _parquet_files(layouts["multi"]))


def test_storage_on_an_in_memory_relation_has_no_files():
    """No file backing: the file questions do not apply, and the byte total is the buffers'."""
    table = BASE.select(["k", "v"])
    storage = bt.from_arrow(table).meta.storage
    assert storage.files() == []
    assert storage.num_files() == 0
    assert storage.num_sources() == 1
    assert storage.row_count() == _N
    assert storage.has_exact_row_count() is True
    assert storage.row_group_count() is None
    assert storage.total_bytes() is not None and storage.total_bytes() >= table.nbytes
    assert storage.bytes_per_row() == pytest.approx(storage.total_bytes() / _N)
    assert storage.partition_keys() == ()
    assert storage.is_partitioned() is False
    assert storage.sorted_by() == ()


@pytest.mark.parametrize(
    ("reader", "layout", "expected"),
    [
        ("csv", "csv_dir", ["part-0.csv", "part-1.csv"]),
        ("csv", "csv_file", ["part-0.csv"]),
        ("json", "json_file", ["rows.jsonl"]),
    ],
)
def test_storage_lists_the_files_of_text_formats_too(layouts, reader, layout, expected):
    """CSV and JSON are file sources as much as Parquet is, and list the same way."""
    storage = getattr(bt.read, reader)(layouts[layout]).meta.storage
    assert sorted(os.path.basename(p) for p in storage.files()) == expected
    assert storage.num_files() == len(expected)


def test_storage_files_follow_the_scan_through_a_plan(layouts):
    """`files` names what the sources hold, so a filter over the scan still lists them."""
    ds = bt.read.parquet(layouts["multi"]).filter(bt.col("k") > 10)
    assert ds.meta.storage.num_files() == 3


# --- relation-level: sorted_by and explain -------------------------------------------------


def test_sorted_by_reports_a_declared_order_that_really_holds(tmp_path):
    """A recorded order must be one the rows are actually in, direction included."""
    ds = bt.from_arrow(BASE.select(["k", "v"])).sort("v", descending=True)
    (order,) = ds.meta.sorted_by()
    assert (order.column, order.descending) == ("v", True)
    values = ds.to_pydict()["v"]
    assert values == sorted(values, reverse=True), "the recorded order must hold on the rows"
    assert bt.from_arrow(BASE).meta.sorted_by() == (), "nothing declared, nothing reported"

    path = str(tmp_path / "sorted.parquet")
    pq.write_table(BASE.select(["k"]), path, sorting_columns=[pq.SortingColumn(0)])
    (footer_order,) = bt.read.parquet(path).meta.sorted_by()
    assert (footer_order.column, footer_order.descending) == ("k", False)


def test_explain_reports_exactly_the_footer_facts(layouts, duck):
    """`explain` is what metadata knows: the exact count and bounds, and nothing after a filter."""
    duck.register("t", BASE)
    rows, lo, hi, nulls = duck.sql(
        "SELECT count(*), min(k), max(k), count(*) - count(n) FROM t"
    ).fetchone()
    report = bt.read.parquet(layouts["single"]).meta.explain()
    assert report["rows"] == rows
    assert report["estimated_rows"] == pytest.approx(rows)
    assert (report["columns"]["k"]["min"], report["columns"]["k"]["max"]) == (lo, hi)
    assert report["columns"]["n"]["null_count"] == nulls
    assert report["sorted_by"] == ()

    filtered = bt.read.parquet(layouts["single"]).filter(bt.col("k") > 10).meta.explain()
    assert filtered["rows"] is None, "a filter makes the row count a bound, not a fact"
    assert "min" not in filtered["columns"]["k"]

    opaque = _force(bt.read.parquet(layouts["single"])).meta.explain()
    assert opaque == {"rows": None, "estimated_rows": None, "sorted_by": (), "columns": {}}


# --- schema: temporal and select ------------------------------------------------------------


def test_schema_temporal_and_select_follow_the_arrow_types():
    """`temporal()` and `select(family)` against pyarrow's own type predicates."""
    ds = bt.from_arrow(BASE)
    schema = ds.meta.schema
    expected = [f.name for f in BASE.schema if pa.types.is_temporal(f.type)]
    assert schema.temporal() == expected == ["d", "ts"]
    assert ds.meta.schema.select("temporal").columns == expected
    assert schema.select("numeric").columns == ["k", "v", "n"]
    assert schema.select("boolean").to_pydict() == {"b": BASE.column("b").to_pylist()}


def test_schema_select_of_an_absent_family_names_the_ones_present():
    """No column of the family: a `PlanError` that says so, not "requires at least one column"."""
    ds = bt.from_pydict({"x": [1], "s": ["a"]})
    with pytest.raises(PlanError, match=r"no 'temporal' column.*numeric.*string"):
        ds.meta.schema.select("temporal")
    with pytest.raises(PlanError, match="unknown type family"):
        ds.meta.schema.select("colour")


# --- approx: every method, with a tolerance -------------------------------------------------

_SKEW_N = 5000


@pytest.fixture
def skewed(tmp_path):
    """A Parquet file with a skewed category, a uniform float, and a key; fresh per test."""
    rng = random.Random(7)
    table = pa.table(
        {
            "k": pa.array(range(_SKEW_N), pa.int64()),
            "g": [rng.choice("aaaaabbbcd") for _ in range(_SKEW_N)],
            "x": [rng.random() * 100 for _ in range(_SKEW_N)],
        }
    )
    path = str(tmp_path / "skewed.parquet")
    pq.write_table(table, path)
    return path, table


def test_approx_rows_and_widths_track_the_real_sizes(skewed):
    """`rows`, `column_bytes`, `row_bytes`, `memory_bytes` against pyarrow's buffers."""
    path, table = skewed
    approx = bt.read.parquet(path).meta.approx
    assert approx.rows() == _SKEW_N, "a footer row count is exact, so the estimate is too"
    assert approx.column_bytes("k") == table.column("k").nbytes, "int64 is 8 bytes a value"
    assert approx.column_bytes("x") == table.column("x").nbytes
    row_bytes = approx.row_bytes()
    assert row_bytes is not None and 16.0 < row_bytes <= 16.0 + 64.0
    memory = approx.memory_bytes()
    assert memory == pytest.approx(row_bytes * _SKEW_N)
    assert table.nbytes / 2 <= memory <= table.nbytes * 4, (memory, table.nbytes)


def test_approx_sketches_appear_after_a_run_and_match_the_oracle(skewed, duck):
    """`is_measured`, `n_unique`, `cardinality_ratio`, `top_k`, `frequency`, `histogram`."""
    path, table = skewed
    duck.register("t", table)
    before = bt.read.parquet(path).meta.approx
    assert before.is_measured("g") is False, "nothing has read `g` yet"
    assert before.top_k("g") is None and before.histogram("x", 4) is None

    ds = bt.read.parquet(path)
    ds.group_by("g").agg(n=bt.count()).collect()  # sketches `g`: distinct count, top values
    ds.filter(bt.col("x") > 50).collect()  # sketches `x`: a quantile grid
    approx = bt.read.parquet(path).meta.approx
    assert approx.is_measured("g") is True

    true_ndv = duck.sql("SELECT count(DISTINCT g) FROM t").fetchone()[0]
    assert approx.n_unique("g") == pytest.approx(true_ndv, rel=0.05)
    assert approx.cardinality_ratio("g") == pytest.approx(true_ndv / _SKEW_N, rel=0.05)

    shares = dict(duck.sql(f"SELECT g, count(*) / {_SKEW_N} FROM t GROUP BY g").fetchall())
    top = approx.top_k("g", 2)
    assert [value for value, _ in top] == sorted(shares, key=shares.get, reverse=True)[:2]
    for value, share in top:
        assert share == pytest.approx(shares[value], abs=0.02)
    assert approx.frequency("g", "a") == pytest.approx(shares["a"], abs=0.02)

    buckets = approx.histogram("x", 4)
    assert buckets is not None and len(buckets) == 4
    for low, high in buckets:
        inside = duck.sql(f"SELECT count(*) FROM t WHERE x >= {low} AND x <= {high}").fetchone()
        assert inside[0] / _SKEW_N == pytest.approx(0.25, abs=0.05), (low, high)


def test_approx_count_where_and_selectivity_bound_the_exact_count(skewed, duck):
    """Before any run the estimate is the planner's guess; once measured it is within 5%."""
    path, table = skewed
    duck.register("t", table)
    predicate = bt.col("x") > 50
    exact = duck.sql("SELECT count(*) FROM t WHERE x > 50").fetchone()[0]

    guess = bt.read.parquet(path).meta.approx
    assert 0.0 <= guess.count_where(predicate) <= _SKEW_N
    assert 0.0 <= guess.selectivity(predicate) <= 1.0

    bt.read.parquet(path).filter(predicate).collect()
    measured = bt.read.parquet(path).meta.approx
    assert measured.count_where(predicate) == pytest.approx(exact, rel=0.05)
    assert measured.selectivity(predicate) == pytest.approx(exact / _SKEW_N, rel=0.05)


@pytest.mark.xfail(
    strict=True,
    reason="kyber: a learned selectivity is keyed by a literal-blind filter signature, so after "
    "`k > 100` runs, `k > 4500` is estimated at the learned 0.98 instead of ~0.10",
)
def test_a_learned_selectivity_does_not_leak_onto_a_different_literal(skewed):
    path, _ = skewed
    bt.read.parquet(path).filter(bt.col("k") > 100).collect()
    approx = bt.read.parquet(path).meta.approx
    assert approx.selectivity(bt.col("k") > 4500) == pytest.approx(499 / _SKEW_N, abs=0.05)


@pytest.mark.xfail(
    strict=True,
    reason="core/kyber: on a hive-partitioned ParquetDatasetSource the filter is pushed into the "
    "pyarrow scan, so the quantile grid a filtered run records describes the surviving rows "
    "and is then attributed to the whole source",
)
def test_a_hive_sketch_describes_the_source_not_a_filtered_read(tmp_path):
    rng = random.Random(3)
    table = pa.table(
        {"day": [f"d{i % 4}" for i in range(4000)], "x": [rng.random() for _ in range(4000)]}
    )
    pq.write_to_dataset(table, str(tmp_path), partition_cols=["day"])
    bt.read.parquet(str(tmp_path)).filter(bt.col("x") > 0.8).collect()
    buckets = bt.read.parquet(str(tmp_path)).meta.approx.histogram("x", 2)
    assert buckets is not None
    assert buckets[0][0] == pytest.approx(0.0, abs=0.05), "the grid must span the whole column"


# --- unknown is None, never 0.0 ------------------------------------------------------------


def test_storage_totals_are_unknown_when_statistics_cannot_be_collected(layouts, monkeypatch):
    """A failed statistics pass is "could not tell" (None), not a confident 0 rows, 0 bytes."""
    import batcher.api.orchestration as orchestration

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("the driver cannot see this storage")

    monkeypatch.setattr(orchestration, "collect_source_stats", refuse)
    storage = bt.read.parquet(layouts["single"]).meta.storage
    assert storage.row_count() is None
    assert storage.total_bytes() is None
    assert storage.row_group_count() is None
    assert storage.has_exact_row_count() is False


def test_an_unknowable_estimate_is_none_not_zero(layouts):
    """An opaque stage hides the plan: every estimate is unknown, and must say so."""
    opaque = _force(bt.read.parquet(layouts["single"]))
    approx = opaque.meta.approx
    assert approx.rows() is None
    assert approx.row_bytes() is None
    assert approx.memory_bytes() is None
    assert approx.count_where(bt.col("k") > 3) is None
    assert approx.selectivity(bt.col("k") > 3) is None
    assert approx.is_measured("k") is False
    other = bt.from_pydict({"k": [1, 2]})
    assert opaque.meta.against(other).estimated_rows("k") is None
    assert other.meta.against(opaque).estimated_rows("k") is None


def test_a_known_empty_estimate_is_still_zero():
    """None means unknown, so a relation *known* to be empty keeps its zeros."""
    empty = bt.from_arrow(pa.table({"k": pa.array([], pa.int64())}))
    assert empty.meta.approx.rows() == 0.0
    assert empty.meta.approx.selectivity(bt.col("k") > 1) == 0.0
    left = bt.from_pydict({"k": [1, 2, 3]})
    assert left.meta.against(bt.from_pydict({"k": [900]})).estimated_rows("k") == 0.0


# --- non-numeric columns --------------------------------------------------------------------


def _meta_variants(layouts):
    return {
        "memory": bt.from_arrow(BASE),
        "parquet": bt.read.parquet(layouts["single"]),
        "forced": _force(bt.read.parquet(layouts["single"])),
    }


@pytest.mark.parametrize("variant", ["memory", "parquet", "forced"])
@pytest.mark.parametrize("column", ["s", "d", "ts", "b"])
@pytest.mark.parametrize("method", ["midpoint", "abs_max"])
def test_numeric_only_shortcuts_refuse_other_types(layouts, variant, column, method):
    """A string, date, timestamp, or boolean has no midpoint: one `PlanError`, on every path."""
    col = _meta_variants(layouts)[variant].meta.col(column)
    with pytest.raises(PlanError, match=rf"{method}\(\).*numeric"):
        getattr(col, method)()


@pytest.mark.parametrize("variant", ["memory", "parquet", "forced"])
@pytest.mark.parametrize("column", ["s", "b"])
def test_range_refuses_strings_and_booleans(layouts, variant, column):
    with pytest.raises(PlanError, match=r"range\(\)"):
        _meta_variants(layouts)[variant].meta.col(column).range()


@pytest.mark.parametrize("variant", ["memory", "parquet", "forced"])
@pytest.mark.parametrize("column", ["d", "ts"])
def test_range_of_a_temporal_column_is_a_timedelta_matching_duckdb(layouts, duck, variant, column):
    """`max - min` of a date or timestamp is a duration, as DuckDB's subtraction says."""
    duck.register("t", BASE)
    lo, hi = duck.sql(f"SELECT min({column}), max({column}) FROM t").fetchone()
    width = _meta_variants(layouts)[variant].meta.col(column).range()
    assert isinstance(width, dt.timedelta)
    assert width == hi - lo


# --- a type-mismatched check raises the same error on both paths ----------------------------


@pytest.mark.parametrize(
    ("column", "call"),
    [
        pytest.param("s", lambda c: c.all_greater_than(0), id="string-vs-int"),
        pytest.param("s", lambda c: c.all_positive(), id="string-all_positive"),
        pytest.param("s", lambda c: c.any_less_than(3), id="string-any_less_than"),
        pytest.param("s", lambda c: c.all_between(0, 5), id="string-all_between"),
        pytest.param("k", lambda c: c.contains("1"), id="int-contains-str"),
        pytest.param("k", lambda c: c.may_contain("1"), id="int-may_contain-str"),
        pytest.param("k", lambda c: c.never_equals("1"), id="int-never_equals-str"),
        pytest.param("k", lambda c: c.any_in([1, "2"]), id="int-any_in-mixed"),
        pytest.param("k", lambda c: c.none_in(["1"]), id="int-none_in-str"),
        pytest.param("d", lambda c: c.all_greater_than(5), id="date-vs-int"),
    ],
)
@pytest.mark.parametrize("shape", ["scan", "empty-filter"])
def test_a_type_mismatched_check_raises_on_the_metadata_path_and_executed_path(
    layouts, shape, column, call
):
    """Metadata used to answer these (vacuously, or "maybe") while the engine raised.

    `may_contain` never executes, so behind an opaque stage (whose column types static
    analysis cannot state) there is no executed path to disagree with: it stays "maybe".
    """
    ds = bt.read.parquet(layouts["single"])
    if shape == "empty-filter":
        ds = ds.filter(bt.col("k") > 10_000)
    with pytest.raises(PlanError, match="cannot compare"):
        call(ds.meta.col(column).check)
    if "may_contain" in call.__code__.co_names:
        assert call(_force(ds).meta.col(column).check) is True
        return
    with pytest.raises((PlanError, ExecutionError)):
        call(_force(ds).meta.col(column).check)


@pytest.mark.parametrize(
    ("column", "call", "expected"),
    [
        pytest.param("d", lambda c: c.contains(dt.date(2024, 1, 5)), True, id="date-contains"),
        pytest.param("d", lambda c: c.all_greater_equal(dt.date(2024, 1, 1)), True, id="date-all"),
        pytest.param("ts", lambda c: c.any_greater_than(dt.datetime(2030, 1, 1)), False, id="ts"),
        pytest.param("s", lambda c: c.any_in(["s3", "nope"]), True, id="string-in"),
        pytest.param("v", lambda c: c.all_less_than(100), True, id="float-vs-int"),
        pytest.param("k", lambda c: c.contains(None), False, id="null-literal"),
    ],
)
def test_a_well_typed_check_is_not_refused(layouts, column, call, expected):
    """The positive control: the gate must not refuse a comparison the engine accepts."""
    ds = bt.read.parquet(layouts["single"])
    assert call(ds.meta.col(column).check) is expected
    assert call(_force(ds).meta.col(column).check) is expected


# --- one query per fallback ----------------------------------------------------------------


def test_each_fallback_runs_one_query_and_keeps_its_answer(layouts, duck):
    """`summary`, a composite `is_key`, `all_match`, and `nulls.fractions`, forced to execute."""
    duck.register("t", BASE)
    opaque = _force(bt.read.parquet(layouts["single"]))

    summary, queries = _executed_queries(lambda: opaque.meta.col("n").summary())
    lo, hi, non_null, nulls, ndv = duck.sql(
        "SELECT min(n), max(n), count(n), count(*) - count(n), count(DISTINCT n) FROM t"
    ).fetchone()
    assert {k: v for k, v in summary.items() if k != "dtype"} == {
        "count": non_null,
        "null_count": nulls,
        "min": lo,
        "max": hi,
        "n_unique": ndv,
    }
    assert queries == 1

    for columns, expected in [(["g", "k"], True), (["g", "s"], False), (["n", "k"], False)]:
        is_key, queries = _executed_queries(lambda c=columns: opaque.meta.is_key(c))
        assert is_key is expected, columns
        assert queries == 1, columns

    for predicate, expected in [(bt.col("k") >= 0, True), (bt.col("n") >= 0, False)]:
        matched, queries = _executed_queries(lambda p=predicate: opaque.meta.all_match(p))
        assert matched is expected
        assert queries == 1

    fractions, queries = _executed_queries(lambda: opaque.meta.nulls.fractions())
    assert fractions["n"] == pytest.approx(nulls / _N)
    assert fractions["k"] == 0.0
    assert queries == 1


def test_a_footer_answer_still_costs_no_query(layouts):
    """The merged fallbacks must not have swallowed the free path."""
    ds = bt.read.parquet(layouts["single"])
    _, queries = _executed_queries(lambda: ds.meta.nulls.fractions())
    assert queries == 0
    _, queries = _executed_queries(lambda: ds.meta.is_key(["n", "k"]))
    assert queries == 0, "`n` has a recorded null, so no composite key can hold"


# --- fast path == execution over the shapes the other file does not build -------------------


def _summary_values(ds: Any, column: str) -> dict[str, Any]:
    """`summary()` minus `dtype`: the value facets, which are what the metadata path computes."""
    return {k: v for k, v in ds.meta.col(column).summary().items() if k != "dtype"}


_SHAPES: dict[str, Callable[[dict[str, str]], Any]] = {
    "filtered": lambda p: bt.read.parquet(p["single"]).filter(bt.col("k") > 12),
    "limit": lambda p: bt.read.parquet(p["single"]).limit(9),
    "computed": lambda p: bt.read.parquet(p["single"]).select(
        "k", "n", "s", "d", "ts", y=bt.col("k") * 2
    ),
    "join": lambda p: (
        bt.read.parquet(p["single"])
        .select("k", "n", "s", "d", "ts")
        .join(bt.read.parquet(p["other"]), on="k")
    ),
    "scan": lambda p: bt.read.parquet(p["single"]),
}

_CALLS: dict[str, Callable[[Any], Any]] = {
    "shape": lambda d: d.meta.shape(),
    "count_where": lambda d: d.meta.count_where(bt.col("n").is_null()),
    "none_match": lambda d: d.meta.none_match(bt.col("k") > 10_000),
    "all_match": lambda d: d.meta.all_match(bt.col("k") >= 0),
    "is_key": lambda d: d.meta.is_key("k"),
    "is_key-composite": lambda d: d.meta.is_key(["s", "k"]),
    "nulls": lambda d: d.meta.nulls.counts(),
    "fractions": lambda d: d.meta.nulls.fractions(),
    "k-bounds": lambda d: d.meta.col("k").bounds(),
    "k-summary": lambda d: _summary_values(d, "k"),
    "n-summary": lambda d: _summary_values(d, "n"),
    "s-summary": lambda d: d.meta.col("s").summary(),
    "s-n_unique": lambda d: d.meta.col("s").n_unique(),
    "n-null_fraction": lambda d: d.meta.col("n").null_fraction(),
    "k-midpoint": lambda d: d.meta.col("k").midpoint(),
    "d-bounds": lambda d: d.meta.col("d").bounds(),
    "d-range": lambda d: d.meta.col("d").range(),
    "d-n_unique": lambda d: d.meta.col("d").n_unique(),
    "d-is_constant": lambda d: d.meta.col("d").is_constant(),
    "d-contains": lambda d: d.meta.col("d").check.contains(dt.date(2024, 1, 20)),
    "d-all_between": lambda d: d.meta.col("d").check.all_between(
        dt.date(2024, 1, 1), dt.date(2024, 2, 9)
    ),
    "ts-bounds": lambda d: d.meta.col("ts").bounds(),
    "ts-range": lambda d: d.meta.col("ts").range(),
    "ts-any_greater_than": lambda d: d.meta.col("ts").check.any_greater_than(
        dt.datetime(2024, 3, 10)
    ),
}


@pytest.mark.parametrize("call", sorted(_CALLS))
@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_shortcut_equals_execution_over_more_shapes(layouts, shape, call):
    """Every shortcut over a filtered, limited, projected, and joined scan, and dates."""
    ds = _SHAPES[shape](layouts)
    shortcut = _CALLS[call](ds)
    executed = _CALLS[call](_force(ds))
    assert _same(shortcut, executed), f"metadata said {shortcut!r}, executing said {executed!r}"


_EMPTY_CALLS = {
    "is_empty": (lambda d: d.is_empty(), False),
    "none_match": (lambda d: d.meta.none_match(bt.col("k") > 1), False),
    "all_positive": (lambda d: d.meta.col("k").check.all_positive(), False),
    "contains": (lambda d: d.meta.col("k").check.contains(1), False),
    "shape": (lambda d: d.meta.shape(), True),
    "bounds": (lambda d: d.meta.col("k").bounds(), True),
    "n_unique": (lambda d: d.meta.col("k").n_unique(), True),
    "nulls": (lambda d: d.meta.nulls.counts(), True),
    "is_key": (lambda d: d.meta.is_key("k"), True),
    "summary": (lambda d: _summary_values(d, "k"), True),
}


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(name, marks=[_EMPTY_AGG_XFAIL] if aggregates else [])
        for name, (_, aggregates) in sorted(_EMPTY_CALLS.items())
    ],
)
def test_shortcut_equals_execution_on_an_empty_parquet_file(layouts, call):
    """An empty file: metadata answers everything; the forced path trips an engine gap."""
    fn, _ = _EMPTY_CALLS[call]
    ds = bt.read.parquet(layouts["empty"])
    shortcut = fn(ds)
    executed = fn(_force(ds))
    assert _same(shortcut, executed), f"metadata said {shortcut!r}, executing said {executed!r}"
