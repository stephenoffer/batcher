"""Dataset column and terminal methods: option/column clashes, validation messages, and opt-ins.

Covers the filter mapping form and its clash message, the same-call reference hint on
`with_columns`/`select`, `drop(strict=)`, rename collision checks, the selector reuse hint,
slice messages, `pipe`'s `ParamSpec`, the `collect`/`iter_batches` placement defaults,
`collect(max_rows=)`, `explain(backend="gpu")`, keyed `with_random`, `use_learned=` on the
approximate terminals, and field metadata on an empty result.
"""

from __future__ import annotations

import inspect
import json

import pyarrow as pa
import pytest

import batcher as bt
from batcher import Dataset
from batcher._internal.errors import ColumnNotFoundError, PlanError, ResourceError

pytestmark = pytest.mark.unit


# --- filter -----------------------------------------------------------------------------


def test_filter_option_that_is_also_a_column_suggests_the_column_spellings() -> None:
    ds = bt.from_pydict({"num_workers": [1, 2, 1], "x": [1, 2, 3]})
    with pytest.raises(PlanError) as err:
        ds.filter(num_workers=1)
    message = str(err.value)
    assert "bt.col('num_workers') == 1" in message
    assert "filter({'num_workers': 1})" in message


def test_filter_option_that_is_not_a_column_keeps_the_callable_message() -> None:
    ds = bt.from_pydict({"x": [1, 2, 3]})
    with pytest.raises(PlanError, match="apply only to a callable predicate; drop them"):
        ds.filter(bt.col("x") > 1, num_gpus=1)


def test_filter_mapping_compares_a_column_named_like_a_parameter() -> None:
    ds = bt.from_pydict({"batch_size": [5, 5, 6], "num_workers": [2, 3, 2], "x": [1, 2, 3]})
    out = ds.filter({"batch_size": 5, "num_workers": 2}).to_pydict()
    assert out == {"batch_size": [5], "num_workers": [2], "x": [1]}


def test_filter_mapping_composes_with_expressions_and_keywords() -> None:
    ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 3], "y": [0, 1, 1]})
    out = ds.filter({"g": "a"}, bt.col("x") > 0, y=1).to_pydict()
    assert out == {"g": ["a"], "x": [2], "y": [1]}


def test_filter_mapping_with_an_unknown_column_raises() -> None:
    ds = bt.from_pydict({"x": [1]})
    with pytest.raises(PlanError, match="unknown column"):
        ds.filter({"nope": 1})


# --- with_columns / select --------------------------------------------------------------


@pytest.mark.parametrize("method", ["with_columns", "select"])
def test_reference_to_a_sibling_output_names_the_chained_fix(method: str) -> None:
    ds = bt.from_pydict({"a": [1, 2]})
    with pytest.raises(ColumnNotFoundError) as err:
        getattr(ds, method)(c=bt.col("a") + 1, d=bt.col("c") * 2)
    message = str(err.value)
    assert f"which this same {method}() call defines" in message
    assert "chain a second .with_columns(...)" in message


@pytest.mark.parametrize("method", ["with_columns", "select"])
def test_an_aggregate_over_an_unknown_column_reports_the_column(method: str) -> None:
    """The sibling check runs on the failure path, and must read an aggregate too.

    It called `referenced_columns` on the raw aggregate, which raises for any `AggExpr`, so
    ``select(col("missing").sum())`` failed with "an aggregate expression can only be used
    inside group_by().agg()" -- the opposite of true -- instead of naming the column.
    """
    ds = bt.from_pydict({"a": [1, 2]})
    with pytest.raises(ColumnNotFoundError) as err:
        getattr(ds, method)(s=bt.col("missing").sum())
    assert err.value.column == "missing"
    assert getattr(ds, method)(s=bt.col("a").sum()).to_pydict()["s"][-1] == 3


def test_an_ordinary_unknown_column_keeps_the_plain_message() -> None:
    ds = bt.from_pydict({"a": [1, 2]})
    with pytest.raises(ColumnNotFoundError) as err:
        ds.with_columns(d=bt.col("zz"))
    assert "same with_columns() call" not in str(err.value)
    assert "unknown column" in str(err.value)


def test_chained_with_columns_is_the_working_spelling() -> None:
    ds = bt.from_pydict({"a": [1, 2]})
    out = ds.with_columns(c=bt.col("a") + 1).with_columns(d=bt.col("c") * 2).to_pydict()
    assert out == {"a": [1, 2], "c": [2, 3], "d": [4, 6]}


def test_a_keyword_string_value_is_a_literal() -> None:
    ds = bt.from_pydict({"a": [1, 2]})
    assert ds.with_columns(tag="a").to_pydict()["tag"] == ["a", "a"]
    assert ds.select("a", tag="a").to_pydict()["tag"] == ["a", "a"]


# --- drop / rename ----------------------------------------------------------------------


def test_drop_is_strict_by_default() -> None:
    ds = bt.from_pydict({"a": [1], "b": [2]})
    assert inspect.signature(Dataset.drop).parameters["strict"].default is True
    with pytest.raises(PlanError, match="unknown column"):
        ds.drop("zzz")


def test_drop_strict_false_ignores_a_missing_name() -> None:
    ds = bt.from_pydict({"a": [1], "b": [2]})
    assert ds.drop("b", "zzz", strict=False).to_pydict() == {"a": [1]}
    assert ds.drop("zzz", strict=False).to_pydict() == {"a": [1], "b": [2]}
    with pytest.raises(PlanError, match="remove all columns"):
        ds.drop("a", "b", "zzz", strict=False)


def test_rename_swap_is_allowed() -> None:
    ds = bt.from_pydict({"a": [1], "b": [2], "c": [3]})
    assert ds.rename({"a": "b", "b": "a"}).to_pydict() == {"b": [1], "a": [2], "c": [3]}


def test_rename_onto_an_untouched_column_names_rename() -> None:
    ds = bt.from_pydict({"a": [1], "b": [2]})
    with pytest.raises(PlanError) as err:
        ds.rename(a="b")
    message = str(err.value)
    assert message.startswith("rename(): target(s) ['b'] already name a column")
    assert "swaps such as" in message


def test_rename_two_columns_onto_one_name_raises() -> None:
    ds = bt.from_pydict({"a": [1], "b": [2]})
    with pytest.raises(PlanError, match=r"several columns would be renamed onto \['z'\]"):
        ds.rename({"a": "z", "b": "z"})


def test_rename_absent_source_still_raises() -> None:
    with pytest.raises(PlanError, match="unknown column"):
        bt.from_pydict({"a": [1]}).rename(zzz="y")


# --- selectors, slices, pipe ------------------------------------------------------------


def test_one_selector_spelled_three_times_suggests_binding_it() -> None:
    ds = bt.from_pydict({"a": [1.0, 2.0, 3.0], "s": ["x", "y", "z"]})
    spelled = (bt.numeric() - bt.numeric().mean()) / bt.numeric().std()
    with pytest.raises(PlanError, match=r"bind it once: n = bt\.numeric\(\)"):
        ds.with_columns(spelled)
    n = bt.numeric()
    assert ds.with_columns((n - n.mean()) / n.std()).to_pydict()["a"] == [-1.0, 0.0, 1.0]


def test_distinct_selectors_get_no_reuse_hint() -> None:
    ds = bt.from_pydict({"a": [1.0], "b": [2]})
    with pytest.raises(PlanError) as err:
        ds.with_columns(bt.numeric() + bt.floating())
    assert "bind it once" not in str(err.value)


def test_slice_step_message_names_gather_every_and_reverse() -> None:
    ds = bt.from_pydict({"x": [1, 2, 3]})
    with pytest.raises(PlanError, match="gather_every") as err:
        ds[::2]
    assert "reverse()" in str(err.value)


def test_negative_slice_message_names_tail() -> None:
    ds = bt.from_pydict({"x": [1, 2, 3]})
    with pytest.raises(PlanError, match=r"use tail\(n\)"):
        ds[-1:]


def test_pipe_is_typed_with_a_paramspec() -> None:
    params = inspect.signature(Dataset.pipe).parameters
    assert params["fn"].annotation == "Callable[Concatenate[Dataset, _P], _T]"
    assert params["args"].annotation == "_P.args"
    assert params["kwargs"].annotation == "_P.kwargs"


# --- collect / iter_batches -------------------------------------------------------------


def test_collect_and_iter_batches_placement_defaults_are_pinned() -> None:
    assert inspect.signature(Dataset.collect).parameters["distributed"].default == "auto"
    assert inspect.signature(Dataset.iter_batches).parameters["distributed"].default is False


def test_collect_max_rows_boundary() -> None:
    ds = bt.from_pydict({"x": [3, 1, 2]})
    assert ds.collect(max_rows=3).num_rows == 3
    with pytest.raises(ResourceError, match="more than max_rows=2"):
        ds.collect(max_rows=2)
    assert ds.filter(bt.col("x") > 5).collect(max_rows=0).num_rows == 0
    with pytest.raises(ResourceError):
        ds.collect(max_rows=0)


def test_collect_max_rows_keeps_the_sorted_order() -> None:
    out = bt.from_pydict({"x": [3, 1, 2]}).sort("x").collect(max_rows=3)
    assert out.column("x").to_pylist() == [1, 2, 3]


def test_collect_max_rows_rejects_a_negative_budget() -> None:
    with pytest.raises(PlanError):
        bt.from_pydict({"x": [1]}).collect(max_rows=-1)


# --- explain(backend="gpu") -------------------------------------------------------------


def _device_line(ds: Dataset) -> str:
    return ds.explain(backend="gpu").splitlines()[-1]


def test_explain_gpu_reports_an_eligible_chain() -> None:
    ds = bt.from_pydict({"g": ["a", "b"], "x": [1, 2]})
    plan = ds.filter(bt.col("x") > 1).group_by("g").agg(s=bt.col("x").sum())
    assert _device_line(plan) == "device tier (backend='gpu'): eligible"


def test_explain_gpu_names_a_declined_expression_tag() -> None:
    ds = bt.from_pydict({"x": [1, 2]}).with_columns(h=bt.col("x").hash())
    line = _device_line(ds)
    assert line.startswith("device tier (backend='gpu'): declined:")
    assert "expr hash" in line


def test_explain_gpu_names_a_declined_operator_and_a_python_stage() -> None:
    ds = bt.from_pydict({"x": [1, 2, 3]})
    assert "operator 'sample'" in _device_line(ds.sample(fraction=0.5, seed=1))
    assert "Python-only stage" in _device_line(ds.map_batches(lambda batch: batch))


def test_explain_gpu_names_a_declined_aggregate_option() -> None:
    ds = bt.from_pydict({"g": ["a"], "x": [1]})
    plan = ds.group_by("g").agg(q=bt.col("x").quantile(0.5, interpolation="lower"))
    assert "'quantile' (interpolation='lower')" in _device_line(plan)


def test_explain_gpu_json_carries_a_device_object() -> None:
    ds = bt.from_pydict({"x": [1, 2]})
    doc = json.loads(ds.explain(format="json", backend="gpu"))
    assert "device" in doc  # positive control for the absence check below
    assert doc["device"] == {"eligible": True, "reason": None}
    assert "device" not in json.loads(ds.explain(format="json"))


def test_explain_cpu_default_has_no_device_line() -> None:
    ds = bt.from_pydict({"x": [1]})
    assert "device tier" in ds.explain(backend="gpu")  # positive control
    assert "device tier" not in ds.explain()


def test_explain_rejects_an_unknown_backend() -> None:
    with pytest.raises(PlanError, match=r"explain\(backend=\.\.\.\)"):
        bt.from_pydict({"x": [1]}).explain(backend="tpu")


# --- with_random(key=) ------------------------------------------------------------------


def _draws(ds: Dataset, **kw: object) -> dict[int, float]:
    out = ds.with_random("r", seed=11, **kw).to_pydict()
    return dict(zip(out["id"], out["r"], strict=True))


@pytest.mark.parametrize("normal", [False, True])
def test_keyed_random_ignores_order_filtering_and_partitioning(normal: bool) -> None:
    ds = bt.from_pydict({"id": list(range(200)), "g": [i % 7 for i in range(200)]})
    base = _draws(ds, key="id", normal=normal)
    assert len(set(base.values())) == 200
    reordered = _draws(ds.sort("id", descending=True), key="id", normal=normal)
    filtered = _draws(ds.filter(bt.col("id") % 3 == 0), key="id", normal=normal)
    partitioned = _draws(ds.repartition(4), key="id", normal=normal)
    assert reordered == base
    assert filtered == {k: v for k, v in base.items() if k % 3 == 0}
    assert partitioned == base


def test_unkeyed_random_depends_on_order_which_is_why_key_exists() -> None:
    ds = bt.from_pydict({"id": list(range(50))})
    assert _draws(ds.sort("id", descending=True)) != _draws(ds)


def test_keyed_random_gives_one_draw_per_key_and_uniform_range() -> None:
    ds = bt.from_pydict({"g": [1, 2, 1, 2, 3]})
    out = ds.with_random(seed=0, key=["g"]).to_pydict()
    by_group: dict[int, set[float]] = {}
    for g, r in zip(out["g"], out["random"], strict=True):
        by_group.setdefault(g, set()).add(r)
    assert all(len(v) == 1 for v in by_group.values())
    assert all(0.0 <= r < 1.0 for r in out["random"])


def test_keyed_random_rejects_an_unknown_key() -> None:
    with pytest.raises(ColumnNotFoundError):
        bt.from_pydict({"x": [1]}).with_random(key="nope")


# --- approx_*(use_learned=) -------------------------------------------------------------


def test_use_learned_false_skips_the_learned_distinct_count(monkeypatch) -> None:
    import batcher.api.terminal.metadata_answer as answers

    monkeypatch.setattr(answers, "metadata_approx_n_unique", lambda *a, **k: 999)
    ds = bt.from_pydict({"x": [1, 2, 3, 3]})
    assert ds.approx_count_distinct("x") == 999  # the learned answer is used by default
    assert ds.approx_count_distinct("x", use_learned=False) == 3


def test_use_learned_false_skips_the_learned_quantile(monkeypatch) -> None:
    import batcher.api.terminal.metadata_answer as answers

    monkeypatch.setattr(answers, "metadata_learned_quantile", lambda *a, **k: -1.0)
    ds = bt.from_pydict({"x": [float(i) for i in range(1, 102)]})
    assert ds.approx_median("x") == -1.0
    assert ds.approx_quantile("x", 0.5) == -1.0
    assert ds.approx_median("x", use_learned=False) == pytest.approx(51.0, abs=2.0)
    assert ds.approx_percentile("x", 50, use_learned=False) == pytest.approx(51.0, abs=2.0)


# --- empty results keep field metadata --------------------------------------------------


def _with_metadata(dtype: pa.DataType) -> Dataset:
    schema = pa.schema(
        [pa.field("a", dtype, metadata={"k": "v"}), pa.field("b", pa.int64())],
        metadata={"tk": "tv"},
    )
    return bt.from_arrow(pa.table({"a": pa.array([1, 2, 3], dtype), "b": [1, 2, 3]}, schema))


@pytest.mark.parametrize("dtype", [pa.int64(), pa.int32()])
@pytest.mark.parametrize(
    "shape",
    [
        lambda ds: ds,
        lambda ds: ds.filter(bt.col("a") > 0),
        lambda ds: ds.sort("a"),
        lambda ds: ds.select("a"),
        lambda ds: ds.rename(a="x"),
        lambda ds: ds.with_columns(a=bt.col("a") * 2),
        lambda ds: ds.filter(bt.col("a") > 0).select("b", "a").sort("a"),
    ],
    ids=["scan", "filter", "sort", "select", "rename", "derived", "filter-select-sort"],
)
def test_limit_zero_keeps_the_schema_one_row_returns(dtype: pa.DataType, shape) -> None:
    ds = shape(_with_metadata(dtype))
    empty, one = ds.limit(0).collect().schema, ds.limit(1).collect().schema
    assert empty.equals(one, check_metadata=True), (empty, one)


def test_an_emptying_filter_keeps_field_metadata() -> None:
    out = _with_metadata(pa.int64()).filter(bt.col("a") > 10).collect()
    assert out.num_rows == 0
    assert out.schema.field("a").metadata == {b"k": b"v"}
    assert out.schema.metadata == {b"tk": b"tv"}
