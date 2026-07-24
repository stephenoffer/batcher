"""`batch_format` conversion around a `map_batches` fn — new formats and result robustness.

Covers the polars/jax parity formats and the torch-result path that used to crash when a
UDF returned a dict mixing tensors with a plain ndarray or scalar. Polars/jax tests skip
when the library is absent; the torch-result test needs no torch.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from batcher.ml.batch_format import FORMATS, result_to_arrowable, to_format


def test_polars_and_jax_are_accepted_formats():
    assert "polars" in FORMATS
    assert "jax" in FORMATS


def test_torch_result_tolerates_non_tensor_values():
    # A dict mixing a tensor-like ndarray with a Python scalar used to crash on `.detach()`.
    out = result_to_arrowable({"a": np.array([1, 2]), "b": 5}, "torch")
    assert out["a"].tolist() == [1, 2]
    assert out["b"] == 5


def test_torch_result_accepts_a_bare_array():
    out = result_to_arrowable(np.array([1.0, 2.0]), "torch")
    assert out.tolist() == [1.0, 2.0]


def test_polars_roundtrip():
    pl = pytest.importorskip("polars")
    batch = pa.record_batch({"x": [1, 2], "y": [3.0, 4.0]})
    df = to_format(batch, "polars")
    assert isinstance(df, pl.DataFrame)
    assert isinstance(result_to_arrowable(df, "polars"), pa.Table)


def test_jax_roundtrip():
    jnp = pytest.importorskip("jax.numpy")
    batch = pa.record_batch({"x": [1, 2], "y": [3.0, 4.0]})
    arrays = to_format(batch, "jax")
    assert arrays["x"].tolist() == [1, 2]
    back = result_to_arrowable({"x": jnp.asarray([1, 2])}, "jax")
    assert back["x"].tolist() == [1, 2]
