"""Casts on the way into a device or a graph must round, never rewrite, a value.

Two narrowing points: the Apple MPS loader (no 64-bit dtypes) and an ONNX graph's declared
input dtypes. Each used to cast unconditionally, so an int64 id past the int32 range
wrapped, a float64 past float32's range became infinity, and a float column fed to an
``int64`` token-id input was truncated. They now refuse by name. The MPS path is driven
directly, since no Apple device is available here; the ONNX path runs a real graph.
"""

from __future__ import annotations

import numpy as np
import pytest

from batcher._internal.errors import PlanError

torch = pytest.importorskip("torch")


def _mps_safe(tensor):
    from batcher.ml.loader.tensors import _mps_safe_dtype

    return _mps_safe_dtype(tensor, "x")


def test_an_int64_that_fits_is_narrowed_to_int32():
    out = _mps_safe(torch.tensor([1, -5, 2**31 - 1], dtype=torch.int64))
    assert out.dtype == torch.int32
    assert out.tolist() == [1, -5, 2**31 - 1]


@pytest.mark.parametrize("value", [2**31, -(2**31) - 1, 1_700_000_000_000_000_000])
def test_an_int64_outside_int32_is_refused_rather_than_wrapped(value):
    with pytest.raises(PlanError, match=r"'x'.*int32 range"):
        _mps_safe(torch.tensor([0, value], dtype=torch.int64))


def test_a_float64_past_float32_range_is_refused_rather_than_made_infinite():
    with pytest.raises(PlanError, match="float32 range"):
        _mps_safe(torch.tensor([1.0, 1e300], dtype=torch.float64))


def test_a_float64_that_fits_is_narrowed_with_a_warning():
    with pytest.warns(UserWarning, match="loses precision"):
        out = _mps_safe(torch.tensor([0.1, float("inf"), float("nan")], dtype=torch.float64))
    assert out.dtype == torch.float32


# --- ONNX ------------------------------------------------------------------------------------

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")


def _identity_graph(tmp_path, elem_type, name):
    """A graph casting one declared input to float, so any input dtype can be read back."""
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", elem_type, [None, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 2])
    node = helper.make_node("Cast", ["x"], ["y"], to=TensorProto.FLOAT)
    model = helper.make_model(
        helper.make_graph([node], "g", [x], [y]),
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=9,
    )
    path = tmp_path / f"{name}.onnx"
    onnx.save(model, str(path))
    from batcher.ml.runtimes.onnx import OnnxSession

    return OnnxSession(str(path), providers=["cpu"])


def test_an_integral_float_feed_into_an_int64_input_is_cast(tmp_path):
    from onnx import TensorProto

    session = _identity_graph(tmp_path, TensorProto.INT64, "ids")
    got = session.predict({"x": np.array([[1.0, 2.0], [3.0, 4.0]])})["y"]
    assert got.tolist() == [[1.0, 2.0], [3.0, 4.0]]


@pytest.mark.parametrize("bad", [0.5, float("nan"), float("inf")])
def test_a_fractional_or_non_finite_float_into_an_int64_input_is_refused(tmp_path, bad):
    from onnx import TensorProto

    session = _identity_graph(tmp_path, TensorProto.INT64, "ids")
    with pytest.raises(PlanError, match=r"'x'.*int64"):
        session.predict({"x": np.array([[1.0, bad]])})


def test_an_integer_outside_a_narrower_input_is_refused(tmp_path):
    from onnx import TensorProto

    session = _identity_graph(tmp_path, TensorProto.INT8, "small")
    with pytest.raises(PlanError, match="range"):
        session.predict({"x": np.array([[1, 300]], dtype=np.int64)})


def test_a_float_precision_narrowing_is_still_applied(tmp_path):
    from onnx import TensorProto

    session = _identity_graph(tmp_path, TensorProto.FLOAT16, "half")
    got = session.predict({"x": np.array([[0.1, 2.0]])})["y"]
    assert got[0, 1] == 2.0
    assert got[0, 0] == pytest.approx(0.1, abs=1e-3)


def test_a_bfloat16_input_is_fed_natively(tmp_path):
    """ONNX Runtime rejects a float32 feed for a bfloat16 input; this used to raise."""
    from onnx import TensorProto

    session = _identity_graph(tmp_path, TensorProto.BFLOAT16, "bf16")
    feed = np.array([[1.5, -7.0], [3.14159, float("nan")]], dtype=np.float32)
    got = session.predict({"x": feed})["y"]
    # bfloat16 keeps 8 significant bits: exact for 1.5 and -7, 3.140625 for pi-ish.
    assert got[0].tolist() == [1.5, -7.0]
    assert got[1, 0] == 3.140625
    assert np.isnan(got[1, 1])
