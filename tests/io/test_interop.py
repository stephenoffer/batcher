"""Framework-interop ingestion coverage.

The CORE adapters (``from_arrow`` / ``from_pydict`` / ``from_numpy``) are tested
unconditionally — they only need pyarrow/numpy. The optional-framework adapters
(pandas, Polars, HuggingFace) are gated with ``pytest.importorskip`` so the suite
runs without those extras installed. Each adapter must return a `Source` whose
schema and values round-trip the input, batch-granularly.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from batcher.io.interop import (
    from_arrow,
    from_huggingface,
    from_items,
    from_numpy,
    from_pandas,
    from_polars,
    from_pydict,
    from_pylist,
    from_ray_dataset,
)
from batcher.io.source import Source


def _table(source: Source) -> pa.Table:
    return pa.Table.from_batches(source.read(), schema=source.schema())


def test_from_pylist_row_oriented():
    src = from_pylist([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, {"a": 3}])
    assert isinstance(src, Source)
    table = _table(src)
    assert table.column("a").to_pylist() == [1, 2, 3]
    assert table.column("b").to_pylist() == ["x", "y", None]  # missing key -> null


def test_from_items_scalars_and_dicts():
    scalars = _table(from_items([1, 2, 3]))
    assert scalars.column_names == ["item"]
    assert scalars.column("item").to_pylist() == [1, 2, 3]

    dicts = _table(from_items([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]))
    assert dicts.column("a").to_pylist() == [1, 2]
    assert dicts.column("b").to_pylist() == ["x", "y"]

    named = _table(from_items(["x", "y"], column="word"))
    assert named.column("word").to_pylist() == ["x", "y"]


def test_from_ray_dataset_streams_blocks():
    pytest.importorskip("ray", reason="ray not installed")
    import ray
    import ray.data

    ray.init(
        include_dashboard=False,
        ignore_reinit_error=True,
        configure_logging=False,
        log_to_driver=False,
    )
    rds = ray.data.from_arrow(pa.table({"k": [1, 2, 3, 4], "v": [10, 20, 30, 40]}))
    src = from_ray_dataset(rds)
    assert isinstance(src, Source)
    table = _table(src)
    assert sorted(table.column("v").to_pylist()) == [10, 20, 30, 40]
    assert src.schema().names == ["k", "v"]


def test_from_arrow_table_roundtrip():
    table = pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    src = from_arrow(table)
    assert isinstance(src, Source)
    assert _table(src).to_pydict() == table.to_pydict()
    assert src.schema().names == ["id", "name"]


def test_from_arrow_batch_and_list():
    batch = pa.record_batch({"x": [1, 2]})
    assert _table(from_arrow(batch)).to_pydict() == {"x": [1, 2]}
    assert _table(from_arrow([batch, batch])).num_rows == 4


def test_from_arrow_empty_list_raises():
    with pytest.raises(ValueError, match="empty batch list"):
        from_arrow([])


def test_from_pydict_roundtrip():
    data = {"a": [1, 2, 3], "b": [1.5, 2.5, 3.5]}
    src = from_pydict(data)
    got = _table(src)
    assert got.column("a").to_pylist() == [1, 2, 3]
    assert got.column("b").to_pylist() == [1.5, 2.5, 3.5]


def test_from_numpy_1d_default_column():
    src = from_numpy(np.array([10, 20, 30]))
    got = _table(src)
    assert got.column_names == ["data"]
    assert got.column("data").to_pylist() == [10, 20, 30]


def test_from_numpy_named_column():
    src = from_numpy(np.array([1.0, 2.0]), column="score")
    assert _table(src).column_names == ["score"]


def test_from_numpy_2d_fixed_size_list():
    arr = np.arange(6, dtype=np.float32).reshape(3, 2)
    src = from_numpy(arr, column="emb")
    got = _table(src)
    assert pa.types.is_fixed_size_list(got.schema.field("emb").type)
    assert got.column("emb").to_pylist() == [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]


def test_from_numpy_3d_is_tensor_column():
    # A rank->=2-per-row array becomes a fixed-shape-tensor column (the ML tensor path).
    from batcher.io.formats.ml.tensor import is_tensor_column

    src = from_numpy(np.zeros((4, 2, 2, 3), dtype=np.uint8))
    table = _table(src)
    field = table.schema.field("data")
    assert is_tensor_column(field.type)
    assert field.type.shape == [2, 2, 3]


def test_from_pandas_roundtrip():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"k": [1, 2], "v": ["x", "y"]})
    got = _table(from_pandas(df))
    assert got.column("k").to_pylist() == [1, 2]
    assert got.column("v").to_pylist() == ["x", "y"]


def test_from_polars_roundtrip():
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"k": [1, 2, 3], "v": [4, 5, 6]})
    got = _table(from_polars(df))
    assert got.column("k").to_pylist() == [1, 2, 3]
    assert got.column("v").to_pylist() == [4, 5, 6]


def test_from_huggingface_roundtrip():
    datasets = pytest.importorskip("datasets")
    ds = datasets.Dataset.from_dict({"text": ["a", "b"], "label": [0, 1]})
    got = _table(from_huggingface(ds))
    assert got.column("text").to_pylist() == ["a", "b"]
    assert got.column("label").to_pylist() == [0, 1]


def test_optional_adapter_missing_dep_raises_backenderror(monkeypatch):
    """A missing optional framework surfaces a typed BackendError with a hint."""
    import builtins

    from batcher._internal.errors import BackendError

    real_import = builtins.__import__

    def _no_polars(name, *args, **kwargs):
        if name == "polars":
            raise ImportError("no polars")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_polars)
    with pytest.raises(BackendError, match=r"\[polars\]"):
        from_polars(object())
