"""Plan-time contracts of the nested-data API: JSONPath subset, typed errors, IR shapes."""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, element
from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.namespaces._json_path import check_json_path, split_wildcard_tail
from batcher.plan.types.registry import dtype_from_wire, dtype_to_wire

pytestmark = pytest.mark.unit

# --- the supported JSONPath subset (RFC 9535 singular queries) -----------------
#
# One row per form. `None` means accepted; a string is a fragment the refusal must name.
# The Rust parser (`eval/str/json/path.rs`) carries the same table; the per-row test
# below proves the engine refuses what the plan-time check refuses.
CONFORMANCE = [
    ("$", None),
    ("$.a.b", None),
    ("a.b", None),
    ("$.tags[0]", None),
    ("$.a[-1]", None),
    ("$[0][1]", None),
    ("$.a[ 1 ]", None),
    ('$."x.y"', None),
    ('$["x.y"].z', None),
    ("$['x.y']", None),
    ("$['it\\'s']", None),
    ("$.my-key", None),
    ("$.a[*]", "wildcard"),
    ("$.a.*", "wildcard"),
    ("$..b", "recursive descent"),
    ("$.a[0:1]", "slice"),
    ("$.a[0,1]", "union"),
    ("$.a[?(@>1)]", "filter"),
    ("$.a[#-1]", "[-n]"),
    ("$.a[x]", "integer or a quoted"),
    ("$.a[0", "closing"),
    ('$."x', "unterminated"),
    ("$.a.", "end with"),
    ("$.a[0]x", "expected `.` or `[`"),
]


@pytest.mark.parametrize(("path", "reason"), CONFORMANCE)
def test_the_conformance_table(path, reason):
    if reason is None:
        assert check_json_path(path) == path
    else:
        with pytest.raises(PlanError, match="unsupported JSONPath") as info:
            check_json_path(path)
        assert reason in str(info.value)


@pytest.mark.parametrize(("path", "reason"), [r for r in CONFORMANCE if r[1] is not None])
def test_every_accessor_refuses_at_plan_build(path, reason):
    with pytest.raises(PlanError, match="unsupported JSONPath"):
        col("j").json.extract_string(path)
    with pytest.raises(PlanError, match="unsupported JSONPath"):
        col("j").json.exists(path)


@pytest.mark.parametrize("path", ["$.a[0:1]", "$.a[*]", "$..b", "$.a[#-1]"])
def test_the_engine_refuses_a_per_row_path_too(path):
    ds = bt.from_pydict({"j": ['{"a": [1, 2]}'], "p": [path]})
    with pytest.raises(Exception, match="unsupported JSONPath"):
        bt.sql("SELECT j ->> p AS r FROM t", t=ds).collect()


def test_a_slice_no_longer_reads_the_whole_array():
    ds = bt.from_pydict({"j": ['{"a": [1, 2, 3]}']})
    with pytest.raises(PlanError, match="slice"):
        ds.select(r=col("j").json.extract_string("$.a[0:1]")).collect()


def test_sql_refuses_what_the_accessor_refuses():
    ds = bt.from_pydict({"j": ['{"a": {"b": 1}}']})
    for path in ["$..b", "$.a[0:1]", "$.a[0,1]"]:
        with pytest.raises(PlanError, match="unsupported JSONPath"):
            bt.sql(f"SELECT json_extract(j, '{path}') AS r FROM t", t=ds).collect()


def test_split_wildcard_tail():
    assert split_wildcard_tail("$.a[*]") == "$.a"
    assert split_wildcard_tail("$[ * ]") == "$"
    assert split_wildcard_tail("$.a") is None
    with pytest.raises(PlanError, match="wildcard"):
        split_wildcard_tail("$.a[*].b[*]")


# --- typed errors that used to surface only at execution -----------------------


def test_a_nested_union_conflict_names_the_column_and_path():
    a = bt.from_pydict({"s": [{"x": {"y": 1}}]})
    b = bt.from_pydict({"s": [{"x": {"y": "z"}}]})
    with pytest.raises(Exception, match=r"column `s`.*at `s\.x\.y`, Int64"):
        a.union(b).collect()


def test_fixed_size_lists_of_different_dimensions_fail_at_plan_time():
    t = pa.table(
        {
            "a": pa.array([[1.0, 2.0]], pa.list_(pa.float64(), 2)),
            "b": pa.array([[1.0, 2.0, 3.0]], pa.list_(pa.float64(), 3)),
        }
    )
    ds = bt.from_arrow(t)
    for build in (lambda a, b: a.list.dot(b), lambda a, b: a.list.add(b)):
        with pytest.raises(PlanError, match=r"2 elements .* 3"):
            _ = ds.select(r=build(col("a"), col("b"))).schema


def test_equal_fixed_dimensions_still_plan():
    t = pa.table({"a": pa.array([[1.0, 2.0]], pa.list_(pa.float64(), 2))})
    ds = bt.from_arrow(t)
    assert ds.select(r=col("a").list.dot(col("a"))).to_pydict() == {"r": [5.0]}


def test_sequence_refuses_temporal_and_text_operands():
    ds = bt.from_pydict({"a": [dt.datetime(2024, 1, 1)], "b": [dt.datetime(2024, 1, 3)]})
    with pytest.raises(PlanError, match=r"integer series.*timestamp"):
        _ = ds.select(s=bt.sequence(col("a"), col("b"))).schema
    ints = bt.from_pydict({"a": [1]})
    with pytest.raises(PlanError, match="step is string"):
        _ = ints.select(s=bt.sequence(col("a"), 5, "1 day")).schema
    assert ints.select(s=bt.sequence(col("a"), 3)).to_pydict() == {"s": [[1, 2, 3]]}


def test_list_moments_take_only_the_two_ddofs_the_engine_computes():
    with pytest.raises(PlanError, match="ddof=2"):
        col("a").list.std(ddof=2)
    assert col("a").list.var(ddof=0).to_ir()["fn"] == "var_pop"
    assert col("a").list.std().to_ir()["fn"] == "std"


def test_jaccard_names_its_three_modes():
    with pytest.raises(PlanError, match="'set'"):
        col("a").list.jaccard(col("b"), mode="sets")


def test_struct_edits_naming_a_missing_field_fail_at_plan_time():
    ds = bt.from_pydict({"s": [{"x": 1}]})
    with pytest.raises(PlanError, match="no field 'zz'"):
        _ = ds.select(r=col("s").struct.drop_fields("zz")).schema
    with pytest.raises(PlanError, match="at least one field"):
        _ = ds.select(r=col("s").struct.drop_fields("x")).schema
    with pytest.raises(PlanError, match="no field 'q'"):
        _ = ds.select(r=col("s").struct.rename_fields({"q": "r"})).schema


def test_struct_update_keeps_field_metadata_and_null_rows():
    inner = pa.field("x", pa.int64(), nullable=False, metadata={"unit": "m"})
    arr = pa.StructArray.from_arrays(
        [pa.array([1, 2]), pa.array(["a", "b"])],
        fields=[inner, pa.field("y", pa.string())],
        mask=pa.array([False, True]),
    )
    ds = bt.from_arrow(pa.table({"s": arr}))
    out = ds.select(r=col("s").struct.with_fields(z=1)).collect()
    field = out.schema.field("r").type.field("x")
    assert field.metadata == {b"unit": b"m"}
    assert not field.nullable
    assert out.to_pydict()["r"] == [{"x": 1, "y": "a", "z": 1}, None]


def test_with_fields_types_its_result_without_running():
    ds = bt.from_pydict({"s": [{"x": 1, "y": "a"}]})
    t = ds.select(r=col("s").struct.with_fields(x="t", z=1.5).struct.rename_fields({"y": "w"}))
    assert t.schema.field("r").type == pa.struct(
        [("x", pa.string()), ("w", pa.string()), ("z", pa.float64())]
    )


# --- lambda scope ---------------------------------------------------------------


def test_a_lambda_captures_the_outer_columns_it_reads():
    e = col("a").list.filter(element() > col("th"))
    ir = e.to_ir()
    assert ir["capture_names"] == ["th"]
    assert ir["captures"] == [{"e": "col", "name": "th"}]
    # A body reading only its own bindings serializes exactly as before.
    assert "captures" not in col("a").list.transform(element() * 2).to_ir()


def test_an_unknown_outer_column_fails_at_plan_build():
    ds = bt.from_pydict({"a": [[1]]})
    with pytest.raises(Exception, match="nope"):
        ds.select(r=col("a").list.filter(element() > col("nope"))).collect()


def test_a_transform_body_is_typed_against_its_own_scope():
    ds = bt.from_pydict({"a": [[1, 2]], "element": ["shadowed"], "k": [1.5]})
    body = element() * col("k") + bt.element_index()
    assert ds.select(r=col("a").list.transform(body)).schema.field("r").type == pa.list_(
        pa.float64()
    )


# --- the nested-dtype wire encoding ---------------------------------------------


@pytest.mark.parametrize(
    "dtype",
    [
        pa.int64(),
        pa.list_(pa.string()),
        pa.struct([("a", pa.int32()), ("b", pa.list_(pa.struct([("c", pa.bool_())])))]),
        pa.map_(pa.string(), pa.float64()),
        pa.timestamp("us"),
    ],
)
def test_the_wire_encoding_round_trips(dtype):
    assert dtype_from_wire(dtype_to_wire(dtype)) == dtype


def test_the_engine_reads_the_wire_encoding_back():
    t = pa.struct([("a", pa.list_(pa.int16())), ("m", pa.map_(pa.string(), pa.int64()))])
    ds = bt.from_pydict({"j": ['{"a": [1, 2], "m": {"k": 3}}']})
    out = ds.select(r=col("j").json.decode(t)).collect()
    assert out.schema.field("r").type == t
    assert out.to_pydict()["r"] == [{"a": [1, 2], "m": [("k", 3)]}]


def test_decode_refuses_a_non_text_map_key():
    with pytest.raises(PlanError, match="map's keys"):
        col("j").json.decode(pa.map_(pa.int64(), pa.int64()))
