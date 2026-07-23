"""The top-level `bt.*` namespace behaves the way a Python data user expects.

The regression that motivated this file: ``bt.concat`` resolved to the *string*
concat expression, so ``bt.concat([ds1, ds2])`` — the universal pandas/Polars idiom —
returned a `Coalesce` expression and failed later with
``'Coalesce' object has no attribute 'collect'``. `concat` is now polymorphic, and
`concat_str` is the explicit name for the string form.
"""

from __future__ import annotations

import datetime

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.unit


# --- bt.concat: frames ------------------------------------------------------
def test_concat_of_datasets_stacks_rows():
    a = bt.from_pydict({"x": [1, 2]})
    b = bt.from_pydict({"x": [3, 4]})
    assert bt.concat([a, b]).to_pydict() == {"x": [1, 2, 3, 4]}


def test_concat_accepts_varargs_as_well_as_a_sequence():
    a, b = bt.from_pydict({"x": [1]}), bt.from_pydict({"x": [2]})
    assert bt.concat(a, b).to_pydict() == bt.concat([a, b]).to_pydict()


def test_concat_of_one_dataset_is_that_dataset():
    a = bt.from_pydict({"x": [1, 2]})
    assert bt.concat([a]).to_pydict() == {"x": [1, 2]}


def test_concat_vertical_relaxed_deduplicates():
    a = bt.from_pydict({"x": [1, 1]})
    b = bt.from_pydict({"x": [1]})
    assert bt.concat([a, b], how="vertical_relaxed").to_pydict() == {"x": [1]}


def test_concat_diagonal_fills_missing_columns_with_nulls():
    left = bt.from_pydict({"x": [1]})
    right = bt.from_pydict({"y": ["a"]})
    assert bt.concat([left, right], how="diagonal").to_pydict() == {
        "x": [1, None],
        "y": [None, "a"],
    }


def test_concat_horizontal_matches_rows_by_position():
    a = bt.from_pydict({"a": [1, 2]})
    b = bt.from_pydict({"b": ["x", "y"]})
    assert bt.concat([a, b], how="horizontal").to_pydict() == {"a": [1, 2], "b": ["x", "y"]}


def test_concat_horizontal_pads_the_shorter_side():
    a = bt.from_pydict({"a": [1, 2, 3]})
    b = bt.from_pydict({"b": ["x"]})
    assert bt.concat([a, b], how="horizontal").to_pydict() == {
        "a": [1, 2, 3],
        "b": ["x", None, None],
    }


def test_concat_horizontal_rejects_colliding_column_names():
    a = bt.from_pydict({"a": [1]})
    with pytest.raises(PlanError, match="horizontal"):
        bt.concat([a, a], how="horizontal")


def test_concat_rejects_an_unknown_how():
    a = bt.from_pydict({"x": [1]})
    with pytest.raises(PlanError, match="how must be one of"):
        bt.concat([a, a], how="sideways")


def test_concat_rejects_mixing_datasets_and_expressions():
    a = bt.from_pydict({"x": [1]})
    with pytest.raises(PlanError, match="mix of Datasets and expressions"):
        bt.concat([a, bt.col("x")])


def test_concat_rejects_no_arguments():
    with pytest.raises(PlanError, match="at least one"):
        bt.concat([])


# --- bt.concat: strings -----------------------------------------------------
def test_concat_of_expressions_still_builds_a_string():
    ds = bt.from_pydict({"a": ["x", "y"], "b": ["1", "2"]})
    assert ds.select(c=bt.concat(bt.col("a"), bt.col("b"))).to_pydict() == {"c": ["x1", "y2"]}


def test_concat_str_is_the_explicit_string_form():
    ds = bt.from_pydict({"a": ["x"], "b": ["y"]})
    assert ds.select(c=bt.concat_str(bt.col("a"), bt.col("b"))).to_pydict() == {"c": ["xy"]}


# --- constructors -----------------------------------------------------------
def test_from_dict_and_from_pydict_agree():
    assert bt.from_dict({"x": [1, 2]}).to_pydict() == bt.from_pydict({"x": [1, 2]}).to_pydict()


def test_from_dicts_and_from_pylist_agree():
    rows = [{"a": 1}, {"a": 2}]
    assert bt.from_dicts(rows).to_pydict() == bt.from_pylist(rows).to_pydict()


def test_from_records_names_tuple_rows():
    ds = bt.from_records([(1, "a"), (2, "b")], columns=["n", "s"])
    assert ds.to_pydict() == {"n": [1, 2], "s": ["a", "b"]}


def test_from_records_accepts_dict_rows_without_columns():
    assert bt.from_records([{"a": 1}]).to_pydict() == {"a": [1]}


def test_from_records_without_columns_is_an_actionable_error():
    with pytest.raises(PlanError, match="columns="):
        bt.from_records([(1, 2)])


def test_from_records_rejects_a_ragged_row():
    with pytest.raises(PlanError, match="column name"):
        bt.from_records([(1, 2), (3,)], columns=["a", "b"])


def test_from_iter_drains_a_generator():
    assert bt.from_iter(x * x for x in range(4)).to_pydict() == {"item": [0, 1, 4, 9]}


def test_from_iter_accepts_a_factory():
    assert bt.from_iter(lambda: iter([1, 2])).to_pydict() == {"item": [1, 2]}


def test_from_pydict_rejects_a_list_with_a_pointer_to_from_pylist():
    with pytest.raises(PlanError, match="from_pylist"):
        bt.from_pydict([{"a": 1}])


def test_from_pylist_rejects_a_mapping_with_a_pointer_to_from_pydict():
    with pytest.raises(PlanError, match="from_pydict"):
        bt.from_pylist({"a": [1]})


def test_from_arrow_accepts_an_arrow_c_stream_exporter():
    table = pa.table({"x": [1, 2]})
    assert bt.from_arrow(table.to_reader()).to_pydict() == {"x": [1, 2]}


def test_from_batches_infers_the_schema_when_it_is_omitted():
    ds = bt.from_batches(lambda: iter([pa.record_batch({"x": [1, 2]})]))
    assert ds.count() == 2


def test_from_numpy_accepts_a_mapping_of_arrays():
    np = pytest.importorskip("numpy")
    ds = bt.from_numpy({"a": np.array([1, 2]), "b": np.array([3, 4])})
    assert ds.to_pydict() == {"a": [1, 2], "b": [3, 4]}


# --- bt.from_any ------------------------------------------------------------
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ({"x": [1, 2]}, {"x": [1, 2]}),
        ([{"x": 1}, {"x": 2}], {"x": [1, 2]}),
        ([1, 2], {"item": [1, 2]}),
        (pa.table({"x": [1, 2]}), {"x": [1, 2]}),
    ],
)
def test_from_any_dispatches_on_type(value, expected):
    assert bt.from_any(value).to_pydict() == expected


def test_from_any_passes_a_dataset_through():
    ds = bt.from_pydict({"x": [1]})
    assert bt.from_any(ds) is ds


def test_from_any_reports_an_unknown_type():
    with pytest.raises(PlanError, match="no constructor"):
        bt.from_any(object())


# --- bt.range / bt.date_range ----------------------------------------------
def test_range_single_argument_matches_builtins_range():
    assert bt.range(5).to_pydict() == {"value": [0, 1, 2, 3, 4]}


def test_range_supports_a_negative_step():
    assert bt.range(3, 0, -1).to_pydict() == {"value": [3, 2, 1]}


def test_range_rejects_a_zero_step():
    with pytest.raises(PlanError, match="non-zero"):
        bt.range(0, 5, 0)


def test_date_range_periods_and_month_interval():
    got = bt.date_range("2024-01-01", periods=3, interval="1mo").to_pydict()["date"]
    assert got == [datetime.date(2024, 1, 1), datetime.date(2024, 2, 1), datetime.date(2024, 3, 1)]


def test_date_range_freq_is_the_pandas_spelling_of_interval():
    assert (
        bt.date_range("2024-01-01", "2024-01-10", freq="2d").to_pydict()
        == bt.date_range("2024-01-01", "2024-01-10", interval="2d").to_pydict()
    )


@pytest.mark.parametrize(
    ("closed", "expected"), [("both", 3), ("left", 2), ("right", 2), ("none", 1)]
)
def test_date_range_closed_drops_the_endpoints(closed, expected):
    assert bt.date_range("2024-01-01", "2024-01-03", closed=closed).count() == expected


def test_date_range_accepts_date_objects():
    got = bt.date_range(datetime.date(2024, 1, 1), datetime.date(2024, 1, 2))
    assert got.count() == 2


def test_date_range_sub_day_interval_yields_timestamps():
    ds = bt.date_range("2024-01-01", "2024-01-01T12:00:00", interval="6h")
    assert ds.count() == 3
    assert pa.types.is_timestamp(ds.schema.field("date").type)


def test_date_range_keeps_the_interval_days_spelling():
    assert bt.date_range("2024-01-01", "2024-01-07", interval_days=3).count() == 3


def test_date_range_rejects_two_stride_spellings():
    with pytest.raises(PlanError, match="only one of"):
        bt.date_range("2024-01-01", "2024-01-03", interval="1d", freq="1d")


def test_date_range_rejects_an_unreadable_interval():
    with pytest.raises(PlanError, match="could not read the interval"):
        bt.date_range("2024-01-01", "2024-01-03", interval="fortnightly")


def test_date_range_needs_an_end_or_a_period_count():
    with pytest.raises(PlanError, match="end= or periods="):
        bt.date_range("2024-01-01")


def test_date_range_rejects_a_bad_closed_value():
    with pytest.raises(PlanError, match="closed must be"):
        bt.date_range("2024-01-01", "2024-01-03", closed="inner")


# --- bt.sql -----------------------------------------------------------------
def test_sql_accepts_a_positional_mapping_of_tables():
    ds = bt.from_pydict({"x": [1, 2]})
    assert bt.sql("SELECT * FROM t", {"t": ds}).to_pydict() == {"x": [1, 2]}


def test_sql_binds_a_plain_dict_as_a_table():
    assert bt.sql("SELECT x * 2 AS y FROM t", t={"x": [1, 2]}).to_pydict() == {"y": [2, 4]}


def test_sql_binds_a_list_of_row_dicts_as_a_table():
    assert bt.sql("SELECT * FROM t", t=[{"x": 1}]).to_pydict() == {"x": [1]}


def test_sql_rejects_a_non_string_query():
    with pytest.raises(PlanError, match="SQL string"):
        bt.sql(bt.from_pydict({"x": [1]}))


def test_sql_rejects_a_non_mapping_positional_argument():
    with pytest.raises(PlanError, match="mapping"):
        bt.sql("SELECT 1", ["t"])


# --- versions and the top-level namespace -----------------------------------
def test_versions_reports_batcher_and_the_engine():
    info = bt.versions()
    assert info["batcher"] == bt.__version__
    assert info["engine"] == bt.engine_version()
    assert "pyarrow" in info


def test_show_versions_prints_a_report(capsys):
    bt.show_versions()
    assert "batcher" in capsys.readouterr().out


@pytest.mark.parametrize(
    "name",
    [
        "read_csv",
        "read_parquet",
        "read_json",
        "read_ndjson",
        "read_ipc",
        "read_orc",
        "read_avro",
        "read_excel",
        "read_delta",
        "read_iceberg",
        "read_database",
        "read_table",
    ],
)
def test_toplevel_reader_shorthands_exist(name):
    assert callable(getattr(bt, name))


@pytest.mark.parametrize(
    "name",
    ["Config", "ExecutionConfig", "MemoryConfig", "DistributedConfig", "active_config"],
)
def test_config_names_are_reachable_from_the_top_level(name):
    assert hasattr(bt, name)


def test_every_exported_name_resolves():
    missing = [n for n in bt.__all__ if not hasattr(bt, n)]
    assert not missing
