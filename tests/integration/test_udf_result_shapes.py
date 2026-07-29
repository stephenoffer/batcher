"""What a `map_batches` function is allowed to return, and what it is told when it isn't.

The Python-workload surface meets users at this boundary more than anywhere else: a model
wrapper returns whatever its framework hands back, and before these tests a pandas frame, a
polars frame, a list of batches, or a generator all died with the same
``must return a pyarrow RecordBatch, Table, or dict; got <type>`` — a message that names the
type it rejected and nothing about the fix. Each case here pins one accepted shape, or one
rejection whose message says what to do instead.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt


@pytest.fixture
def ds() -> bt.Dataset:
    return bt.from_pydict({"x": [1, 2, 3], "name": ["a", "b", "c"]})


def test_returns_pandas_frame_under_pyarrow_format(ds: bt.Dataset) -> None:
    """A `fn` written for pandas still works when the format was left at the default."""
    out = ds.map_batches(lambda b: b.to_pandas().assign(y=lambda d: d["x"] * 2)).to_pydict()
    assert out["y"] == [2, 4, 6]
    assert out["name"] == ["a", "b", "c"]


def test_returns_polars_frame_under_pyarrow_format(ds: bt.Dataset) -> None:
    pl = pytest.importorskip("polars")
    out = ds.map_batches(lambda b: pl.from_arrow(pa.Table.from_batches([b]))).to_pydict()
    assert out["x"] == [1, 2, 3]


def test_returns_list_of_batches(ds: bt.Dataset) -> None:
    """A list is the shape a fan-out `fn` (one batch per decoded item) produces naturally."""
    out = ds.map_batches(lambda b: [b.slice(0, 1), b.slice(1)]).to_pydict()
    assert out["x"] == [1, 2, 3]


def test_returns_generator_of_batches(ds: bt.Dataset) -> None:
    """A `yield`-per-item `fn` is the natural spelling of a row-expanding ML stage."""

    def gen(batch: pa.RecordBatch):
        for i in range(batch.num_rows):
            yield batch.slice(i, 1)

    assert ds.map_batches(gen).to_pydict()["x"] == [1, 2, 3]


def test_returns_generator_of_column_dicts(ds: bt.Dataset) -> None:
    def gen(batch: pa.RecordBatch):
        yield {"x": batch.column("x").to_pylist()}
        yield {"x": [99]}

    assert ds.map_batches(gen).to_pydict()["x"] == [1, 2, 3, 99]


def test_row_dict_list_is_rejected_with_the_fix(ds: bt.Dataset) -> None:
    """Row-oriented output is a different operator, so say which one rather than accepting it."""
    with pytest.raises(TypeError, match="flat_map"):
        ds.map_batches(lambda b: [{"x": 1}, {"x": 2}]).to_pydict()


def test_unsupported_return_names_every_accepted_shape(ds: bt.Dataset) -> None:
    with pytest.raises(TypeError, match="pandas/polars DataFrame"):
        ds.map_batches(lambda b: np.arange(3)).to_pydict()


def test_pandas_format_may_return_a_named_series(ds: bt.Dataset) -> None:
    """``df["x"] * 2`` is the obvious one-column transform; it used to raise from inside pyarrow."""
    out = ds.map_batches(lambda df: df["x"] * 2, batch_format="pandas").to_pydict()
    assert out["x"] == [2, 4, 6]


def test_pandas_format_unnamed_series_says_how_to_name_it(ds: bt.Dataset) -> None:
    with pytest.raises(ValueError, match="rename"):
        ds.map_batches(lambda df: df["x"].rename(None) * 2, batch_format="pandas").to_pydict()


def test_polars_format_may_return_a_series(ds: bt.Dataset) -> None:
    pytest.importorskip("polars")
    out = ds.map_batches(lambda df: df["x"] * 2, batch_format="polars").to_pydict()
    assert out["x"] == [2, 4, 6]


def test_torch_format_warns_about_the_columns_it_cannot_pass(ds: bt.Dataset) -> None:
    """A dropped string ``name``/``id`` is invisible otherwise, and it is usually the join key."""
    pytest.importorskip("torch")
    from batcher.ml import batch_format as bf

    bf._WARNED_DROPS.clear()
    with pytest.warns(UserWarning, match=r"non-numeric column\(s\) \['name'\]"):
        ds.map_batches(lambda b: {"x": b["x"]}, batch_format="torch").to_pydict()


@pytest.mark.parametrize("terminal", ["collect", "iter_batches"])
def test_the_drop_warning_reaches_the_streaming_path_too(ds: bt.Dataset, terminal: str) -> None:
    """`iter_batches` runs a different executor, and a guard that misses it is the usual gap."""
    pytest.importorskip("torch")
    from batcher.ml import batch_format as bf

    bf._WARNED_DROPS.clear()
    plan = ds.map_batches(lambda b: {"x": b["x"]}, batch_format="torch")
    with pytest.warns(UserWarning, match="non-numeric"):
        plan.to_pydict() if terminal == "collect" else list(plan.iter_batches())


@pytest.mark.parametrize("terminal", ["collect", "iter_batches"])
def test_a_frame_return_works_on_the_streaming_path_too(ds: bt.Dataset, terminal: str) -> None:
    plan = ds.map_batches(lambda b: b.to_pandas())
    out = plan.to_pydict() if terminal == "collect" else list(plan.iter_batches())
    assert out if terminal == "collect" else sum(b.num_rows for b in out) == 3


def test_torch_format_warns_only_once_per_column_set(ds: bt.Dataset) -> None:
    """A per-batch warning would emit thousands of identical lines on a real scan."""
    pytest.importorskip("torch")
    from batcher.ml import batch_format as bf

    bf._WARNED_DROPS.clear()
    with pytest.warns(UserWarning):
        ds.map_batches(lambda b: {"x": b["x"]}, batch_format="torch").to_pydict()
    import warnings

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        ds.map_batches(lambda b: {"x": b["x"]}, batch_format="torch").to_pydict()
    assert not [w for w in seen if "non-numeric" in str(w.message)]
