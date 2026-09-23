"""Edges of the UDF verbs' plumbing: argument validation, row-key drift, and error hints.

`map_batches`, `map`, `flat_map` and the callable `filter` share one validation and binding
path (`api/dataset/_udf`). Each case here was a real defect: an option that raised Python's
own `TypeError` (or nothing at all) instead of a `PlanError` naming it, a `ds.map` that kept
only the first row's keys, and failures whose message did not say what to change.
"""

from __future__ import annotations

import logging
import threading

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError


@pytest.fixture
def ds() -> bt.Dataset:
    return bt.from_pydict({"x": list(range(10)), "g": [1, 2] * 5})


def _keep(batch):
    return batch


class _Model:
    def __init__(self, n):
        self.n = n

    def __call__(self, batch):
        return batch


# --- ds.map keeps every key any row produced --------------------------------------------


def test_map_keeps_a_key_the_first_row_lacked(ds):
    """Alternating keys used to lose `b`: the table took its columns from row one only."""
    out = ds.map(lambda r: {"a": 1} if r["x"] % 2 == 0 else {"b": 2}).to_pydict()
    assert out == {"a": [1, None] * 5, "b": [None, 2] * 5}


def test_map_with_uniform_keys_is_unchanged(ds):
    """The positive control: the common case builds the same table it always did."""
    out = ds.map(lambda r: {"x": r["x"], "y": r["x"] * 2}).to_pydict()
    assert out == {"x": list(range(10)), "y": [2 * i for i in range(10)]}


def test_flat_map_keeps_every_key_too(ds):
    out = ds.flat_map(lambda r: [{"a": r["x"]}] if r["x"] < 5 else [{"b": r["x"]}]).to_pydict()
    assert out == {"a": [0, 1, 2, 3, 4, *[None] * 5], "b": [*[None] * 5, 5, 6, 7, 8, 9]}


def test_a_dropped_key_is_warned_about_and_an_added_one_is_not(ds, caplog):
    """The `map_batches` drift rule, per row: a key that disappears is warned, a new one is not."""
    with caplog.at_level(logging.WARNING, logger="batcher"):
        ds.map(lambda r: {"a": 1, **({"b": 2} if r["x"] == 5 else {})}).to_pydict()
    assert "dropped a column" not in caplog.text
    with caplog.at_level(logging.WARNING, logger="batcher"):
        ds.map(lambda r: {"a": 1} if r["x"] % 2 == 0 else {"b": 2}).to_pydict()
    assert "dropped a column between rows" in caplog.text


# --- options are refused at definition, by name ------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_concurrency": "x"}, "max_concurrency must be a number"),
        ({"max_concurrency": -1}, "max_concurrency must be >= 0"),
        ({"fn_args": 5}, "fn_args must be a tuple"),
        ({"fn_args": "abc"}, "fn_args must be a tuple"),
        ({"fn_kwargs": [1]}, "fn_kwargs must be a dict"),
        ({"fn_kwargs": {1: 2}}, "fn_kwargs must be a dict"),
        ({"input_columns": "x"}, r"input_columns=\.\.\.\) must be a list"),
        ({"preserves_columns": "x"}, r"preserves_columns=\.\.\.\) must be a list"),
        ({"num_workers": 0}, "num_workers must be >= 1"),
        ({"num_workers": -1}, "num_workers must be >= 1"),
    ],
)
def test_map_batches_options_fail_at_definition(ds, kwargs, match):
    with pytest.raises(PlanError, match=match):
        ds.map_batches(_keep, **kwargs)


@pytest.mark.parametrize("param", ["input_columns", "preserves_columns"])
def test_an_unknown_declared_column_is_named(ds, param):
    with pytest.raises(ColumnNotFoundError, match="nope"):
        ds.map_batches(_keep, **{param: ["nope"]})


def test_constructor_bindings_are_checked_too(ds):
    with pytest.raises(PlanError, match="fn_constructor_args must be a tuple"):
        ds.map_batches(_Model, fn_constructor_args=5)
    with pytest.raises(PlanError, match="fn_constructor_kwargs must be a dict"):
        ds.map_batches(_Model, fn_constructor_kwargs=[1])


@pytest.mark.parametrize("verb", ["map", "flat_map", "filter"])
def test_every_verb_validates_max_concurrency(ds, verb):
    with pytest.raises(PlanError, match="max_concurrency must be a number"):
        getattr(ds, verb)(lambda r: r, max_concurrency="x")


def test_valid_options_still_run(ds):
    """The control for the refusals above: the same options, well-formed, execute."""
    out = ds.map_batches(
        lambda b, a, k=0: {"x": pc.add(b["x"], a + k)},
        fn_args=(1,),
        fn_kwargs={"k": 2},
        input_columns=["x"],
        num_workers=2,
        max_concurrency=4,
    ).to_pydict()
    assert out == {"x": [i + 3 for i in range(10)]}


# --- failures say what to change ---------------------------------------------------------


def test_a_spent_error_budget_says_so(ds):
    def boom(b):
        if b["x"][0].as_py() >= 5:
            raise ValueError("bad row")
        return b

    with pytest.raises(ValueError, match="bad row") as err:
        ds.map_batches(boom, batch_size=5, max_errored_rows=3).to_pydict()
    assert any("max_errored_rows exceeded" in note for note in err.value.__notes__)
    # Within budget the same stage drops the rows and says nothing.
    assert ds.map_batches(boom, batch_size=5, max_errored_rows=5).count() == 5


def test_a_strict_failure_carries_no_budget_note(ds):
    """Budget 0 is strict mode: the note would claim a budget the caller never set."""

    def boom(b):
        raise ValueError("strict")

    with pytest.raises(ValueError, match="strict") as err:
        ds.map_batches(boom).to_pydict()
    assert not any("max_errored_rows" in n for n in getattr(err.value, "__notes__", []))


def test_writing_into_a_zero_copy_batch_names_the_fix(ds):
    def write(batch):
        batch["x"][0] = 99
        return batch

    with pytest.raises(ValueError, match="read-only") as err:
        ds.map_batches(write, batch_format="numpy").to_pydict()
    assert any("zero_copy_batch=False" in note for note in err.value.__notes__)
    out = ds.map_batches(write, batch_format="numpy", zero_copy_batch=False).to_pydict()
    assert out["x"][0] == 99


def test_a_column_a_udf_added_points_at_output_columns(ds):
    added = ds.map_batches(lambda b: b.append_column("z", pc.multiply(b["x"], 2)))
    with pytest.raises(ColumnNotFoundError, match="output_columns"):
        added.group_by("g").agg(s=bt.col("z").sum())
    declared = ds.map_batches(
        lambda b: b.append_column("z", pc.multiply(b["x"], 2)), output_columns=["x", "g", "z"]
    )
    assert declared.group_by("g").agg(s=bt.col("z").sum()).sort("g").to_pydict() == {
        "g": [1, 2],
        "s": [40, 50],
    }


def test_an_ordinary_typo_carries_no_udf_hint(ds):
    """The hint is for an undeclared UDF upstream, not for every unknown column."""
    with pytest.raises(ColumnNotFoundError) as err:
        ds.select("nope")
    assert "output_columns" not in str(err.value)


# --- the distributed submission helpers, without a cluster --------------------------------


def test_an_unpicklable_closure_is_named_briefly(ds):
    pytest.importorskip("ray", reason="ray not installed")
    from batcher.api.dataset._udf.cluster import unpicklable_udf_error

    lock = threading.Lock()
    plan = ds.map_batches(lambda b: (lock, b)[1])._plan
    err = unpicklable_udf_error(plan, TypeError("Could not serialize the argument ..."))
    assert isinstance(err, PlanError)
    assert "'lock'" in str(err) and "distributed=False" in str(err)
    assert len(str(err)) < 600
    # Something else failing to serialize is not attributed to the fn.
    assert unpicklable_udf_error(ds.map_batches(_keep)._plan, TypeError("serialize")) is None
    assert unpicklable_udf_error(plan, TypeError("unrelated")) is None


def test_a_gpu_request_on_a_local_process_with_no_ray_is_not_checked(ds):
    """No cluster is up, so nothing can be read about it and nothing is refused."""
    ray = pytest.importorskip("ray", reason="ray not installed")
    if ray.is_initialized():
        pytest.skip("a Ray cluster is attached to this process")
    from batcher.api.dataset._udf.cluster import require_placeable_accelerators

    require_placeable_accelerators(ds.map_batches(_Model, num_gpus=1)._plan)


def test_null_filling_a_missing_key_keeps_each_column_type():
    """Null-filling a missing key keeps the other column's type."""
    ds = bt.from_arrow(pa.table({"x": pa.array([1, 2], pa.int64())}))
    out = ds.map(lambda r: {"f": 1.5} if r["x"] == 1 else {"s": "t"}).collect()
    assert out.schema.field("f").type == pa.float64()
    assert out.schema.field("s").type == pa.string()
