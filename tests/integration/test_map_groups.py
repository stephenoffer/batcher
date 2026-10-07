"""`GroupBy.map_groups`: one Python call per group, whole.

The property that matters is the one `map_batches` after a `group_by` does not have — the
callback sees every row of a group and no row of another. It is also the property that
fails silently when it fails, so these tests count the calls and the rows rather than
trusting the aggregate to look right.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError


def _summary(group: pa.RecordBatch) -> dict:
    values = group.column("v").to_pylist()
    return {
        "k": [group.column("k")[0].as_py()],
        "n": [group.num_rows],
        "total": [sum(values)],
    }


@pytest.fixture
def ds() -> bt.Dataset:
    return bt.from_pydict({"k": ["a", "b", "a", "b", "a"], "v": [1, 10, 3, 20, 5]})


def test_each_group_arrives_whole(ds: bt.Dataset) -> None:
    out = ds.group_by("k").map_groups(_summary, output_columns=["k", "n", "total"])
    assert out.sort("k").to_pydict() == {"k": ["a", "b"], "n": [3, 2], "total": [9, 30]}


def test_every_row_reaches_exactly_one_call() -> None:
    """The scale the ledger measured: `map_batches` split all 20 keys, `map_groups` splits none."""
    rows = 50_000
    keys = 20
    wide = bt.from_pydict({"k": [f"k{i % keys}" for i in range(rows)], "x": list(range(rows))})
    out = (
        wide.group_by("k")
        .map_groups(
            lambda g: {"k": [g.column("k")[0].as_py()], "n": [g.num_rows]},
            output_columns=["k", "n"],
        )
        .to_pydict()
    )
    assert len(out["k"]) == keys
    assert sum(out["n"]) == rows
    assert set(out["n"]) == {rows // keys}


def test_the_callback_may_change_the_row_count(ds: bt.Dataset) -> None:
    """A group can expand or shrink; results concatenate."""
    out = ds.group_by("k").map_groups(
        lambda g: {"k": [g.column("k")[0].as_py()] * 2, "half": [g.num_rows, -g.num_rows]},
        output_columns=["k", "half"],
    )
    assert sorted(out.to_pydict()["half"]) == [-3, -2, 2, 3]


def test_group_sees_every_source_column(ds: bt.Dataset) -> None:
    named = bt.from_pydict({"k": ["a", "a"], "v": [1, 2], "t": ["x", "y"]})
    seen: list[list[str]] = []

    def capture(group: pa.RecordBatch) -> pa.RecordBatch:
        seen.append(sorted(group.schema.names))
        return group.select(["k"])

    named.group_by("k").map_groups(capture, output_columns=["k"]).to_pydict()
    assert seen == [["k", "t", "v"]]


def test_pandas_batch_format_is_the_applyinpandas_shape(ds: bt.Dataset) -> None:
    """The frame must be the *group's rows*, not the aggregated row of list columns."""
    pytest.importorskip("pandas")

    def ranked(df):
        return df.assign(rank=df["v"].rank().astype("int64"))

    out = ds.group_by("k").map_groups(
        ranked, batch_format="pandas", output_columns=["v", "k", "rank"]
    )
    assert out.sort("k", "v").to_pydict()["rank"] == [1, 2, 3, 1, 2]


def test_multiple_keys(ds: bt.Dataset) -> None:
    two = bt.from_pydict({"a": [1, 1, 2], "b": ["x", "x", "y"], "v": [1, 2, 3]})
    out = two.group_by("a", "b").map_groups(
        lambda g: {"a": [g.column("a")[0].as_py()], "n": [g.num_rows]},
        output_columns=["a", "n"],
    )
    assert sorted(out.to_pydict()["n"]) == [1, 2]


def test_columns_stay_row_aligned_across_nulls() -> None:
    """Each column is aggregated separately, so alignment is the property to pin.

    If `array_agg` dropped nulls, one column's list would be shorter than its siblings' and
    the rebuilt group would pair a feature with the wrong label — a wrong answer that no
    row count would show.
    """
    ds = bt.from_pydict({"k": ["a", "a", "a"], "x": [1, None, 3], "y": ["p", "q", None]})

    def pairs(group: pa.RecordBatch) -> dict:
        xs = group.column("x").to_pylist()
        ys = group.column("y").to_pylist()
        assert len(xs) == len(ys) == group.num_rows
        return {"pair": [f"{a}:{b}" for a, b in zip(xs, ys, strict=True)]}

    out = ds.group_by("k").map_groups(pairs, output_columns=["pair"]).to_pydict()
    assert sorted(out["pair"]) == ["1:p", "3:None", "None:q"]


def test_empty_input_produces_no_groups() -> None:
    empty = bt.from_pydict({"k": ["a"], "v": [1]}).filter(bt.col("v") > 99)
    assert empty.group_by("k").map_groups(_summary).count() == 0


def test_all_columns_are_keys_is_rejected(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="at least one non-key column"):
        ds.group_by("k", "v").map_groups(_summary).to_pydict()


def test_input_columns_is_rejected(ds: bt.Dataset) -> None:
    """It would describe the aggregated relation, not the one the caller sees."""
    with pytest.raises(PlanError, match="does not take input_columns"):
        ds.group_by("k").map_groups(_summary, input_columns=["v"]).to_pydict()


def test_derived_keys_are_rejected(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="derived key"):
        ds.group_by(doubled=bt.col("v") * 2).map_groups(_summary).to_pydict()


def test_a_group_of_one_row(ds: bt.Dataset) -> None:
    single = bt.from_pydict({"k": ["only"], "v": [7]})
    assert single.group_by("k").map_groups(
        _summary, output_columns=["k", "n", "total"]
    ).to_pydict()["total"] == [7]


def test_nulls_in_the_key_form_their_own_group() -> None:
    nulled = bt.from_pydict({"k": ["a", None, None], "v": [1, 2, 3]})
    out = (
        nulled.group_by("k")
        .map_groups(lambda g: {"n": [g.num_rows]}, output_columns=["n"])
        .to_pydict()
    )
    assert sorted(out["n"]) == [1, 2]


def test_a_tensor_column_survives_the_regroup() -> None:
    """An embedding per row is the ML shape, and the rebuild slices a list's child buffer."""
    import numpy as np

    tensored = bt.from_pydict({"k": ["a", "a", "b"]}).map_batches(
        lambda b: {"k": b.column("k").to_pylist(), "e": np.eye(3, 4, dtype="float32")},
        output_columns=["k", "e"],
    )
    out = tensored.group_by("k").map_groups(
        lambda g: {
            "k": [g.column("k")[0].as_py()],
            "rows": [np.asarray(g.column("e").to_pylist()).shape[0]],
            "dim": [np.asarray(g.column("e").to_pylist()).shape[1]],
        },
        output_columns=["k", "rows", "dim"],
    )
    assert sorted(zip(out.to_pydict()["k"], out.to_pydict()["rows"], strict=True)) == [
        ("a", 2),
        ("b", 1),
    ]
    assert set(out.to_pydict()["dim"]) == {4}


def test_a_list_column_survives_the_regroup() -> None:
    lists = bt.from_pydict({"k": ["a", "a", "b"], "v": [[1, 2], [3], [4, 5, 6]]})
    out = lists.group_by("k").map_groups(
        lambda g: {
            "k": [g.column("k")[0].as_py()],
            "flat": [sum(len(v) for v in g.column("v").to_pylist())],
        },
        output_columns=["k", "flat"],
    )
    assert sorted(zip(*(out.to_pydict()[c] for c in ("k", "flat")), strict=True)) == [
        ("a", 3),
        ("b", 3),
    ]


def test_many_tiny_groups() -> None:
    """`O(groups)` control-plane work is the cost of the operation; it must still be correct."""
    wide = bt.from_pydict({"k": [f"k{i}" for i in range(5000)], "x": list(range(5000))})
    out = (
        wide.group_by("k")
        .map_groups(lambda g: {"n": [g.num_rows]}, output_columns=["n"])
        .to_pydict()
    )
    assert len(out["n"]) == 5000
    assert sum(out["n"]) == 5000


def test_one_large_group() -> None:
    """The documented memory trade: a whole group is materialized for the callback."""
    one = bt.from_pydict({"k": ["same"] * 200_000, "x": list(range(200_000))})
    out = one.group_by("k").map_groups(lambda g: {"n": [g.num_rows]}, output_columns=["n"])
    assert out.to_pydict()["n"] == [200_000]


def test_a_raising_callback_propagates_its_own_error(ds: bt.Dataset) -> None:
    def boom(group: pa.RecordBatch) -> dict:
        raise RuntimeError("model died")

    with pytest.raises(RuntimeError, match="model died"):
        ds.group_by("k").map_groups(boom).to_pydict()


def test_iteration_error_points_at_map_groups(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="map_groups"):
        iter(ds.group_by("k"))


def test_apply_guidance_points_at_map_groups(ds: bt.Dataset) -> None:
    """A pandas/Spark migrant must be sent to the correct spelling, not to map_batches."""
    for name in ("apply", "applyInPandas"):
        with pytest.raises(AttributeError, match="map_groups"):
            getattr(ds.group_by("k"), name)


_SUMMARY_SCHEMA = pa.schema([("k", pa.string()), ("n", pa.int64()), ("total", pa.int64())])


def test_an_input_with_no_groups_keeps_the_declared_schema(ds: bt.Dataset) -> None:
    """With `output_schema`, no groups is an empty table of it, so the union is legal.

    Without it the empty result had no columns while `.columns` reported three, and the
    union below raised "union inputs must have identical columns".
    """
    empty = (
        ds.filter(bt.col("v") > 99)
        .group_by("k")
        .map_groups(_summary, output_schema=_SUMMARY_SCHEMA)
    )
    got = empty.collect()
    assert got.schema == _SUMMARY_SCHEMA
    assert got.num_rows == 0
    assert empty.columns == ["k", "n", "total"]
    full = ds.group_by("k").map_groups(_summary, output_schema=_SUMMARY_SCHEMA)
    assert full.union(empty).sort("k").to_pydict() == {
        "k": ["a", "b"],
        "n": [3, 2],
        "total": [9, 30],
    }


def test_results_are_cast_to_the_declared_schema(ds: bt.Dataset) -> None:
    """A result column is cast to `output_schema` and reordered to it."""
    schema = pa.schema([("total", pa.float64()), ("k", pa.string())])

    def fn(group: pa.RecordBatch) -> dict:
        return {"k": [group.column("k")[0].as_py()], "total": [sum(group.column("v").to_pylist())]}

    out = ds.group_by("k").map_groups(fn, output_schema=schema).collect()
    assert out.schema == schema
    assert sorted(out.to_pydict()["total"]) == [9.0, 30.0]


def test_a_result_missing_a_declared_column_raises(ds: bt.Dataset) -> None:
    def fn(group: pa.RecordBatch) -> dict:
        return {"k": [group.column("k")[0].as_py()]}

    with pytest.raises(bt.ExecutionError, match="output_schema declares"):
        ds.group_by("k").map_groups(fn, output_schema=_SUMMARY_SCHEMA).to_pydict()


def test_output_schema_disagreeing_with_output_columns_is_refused(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="disagree"):
        ds.group_by("k").map_groups(_summary, output_schema=_SUMMARY_SCHEMA, output_columns=["k"])
    with pytest.raises(PlanError, match=r"pyarrow\.Schema"):
        ds.group_by("k").map_groups(_summary, output_schema=["k"])


@pytest.mark.parametrize(
    ("limits", "unit"),
    [({"max_group_rows": 2}, "rows"), ({"max_group_bytes": 20}, "bytes")],
)
def test_an_oversized_group_is_refused_before_the_call(ds: bt.Dataset, limits, unit) -> None:
    """Group `a` (3 rows, 24 bytes of `v` plus its key) trips the limit; `fn` never sees it."""
    seen: list[str] = []

    def fn(group: pa.RecordBatch) -> dict:
        seen.append(group.column("k")[0].as_py())
        return _summary(group)

    with pytest.raises(bt.ExecutionError, match=rf"group \{{'k': 'a'\}} holds .* {unit}"):
        ds.group_by("k").map_groups(fn, **limits).to_pydict()
    assert "a" not in seen


def test_limits_that_every_group_fits_change_nothing(ds: bt.Dataset) -> None:
    out = ds.group_by("k").map_groups(
        _summary, output_columns=["k", "n", "total"], max_group_rows=3, max_group_bytes=10**6
    )
    assert out.sort("k").to_pydict() == {"k": ["a", "b"], "n": [3, 2], "total": [9, 30]}


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_a_malformed_limit_is_refused(ds: bt.Dataset, bad) -> None:
    with pytest.raises(PlanError, match="positive integer"):
        ds.group_by("k").map_groups(_summary, max_group_rows=bad)
