"""Local model runtimes — provider resolution, port renaming, and dtype/shape coercion.

Every case here runs without an accelerator and without a model, because the parts that go
wrong are the parts that decide *where* and *how* a model runs, not the forward pass itself.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from batcher._internal.errors import PerformanceWarning
from batcher.ml.runtimes.onnx import ONNX_TO_NUMPY, _static_batch_size
from batcher.ml.runtimes.providers import (
    PROVIDER_ALIASES,
    RenamedPorts,
    onnx_providers,
    port_mapping,
    resolve_device_id,
    runtime_thread_target,
)

pytestmark = pytest.mark.unit

_ALL = [
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "CPUExecutionProvider",
]


def _names(resolved: list) -> list[str]:
    return [entry if isinstance(entry, str) else entry[0] for entry in resolved]


class _Spec:
    """Stands in for an ``onnxruntime`` node arg (name, declared type, declared shape)."""

    def __init__(self, name: str, type_: str = "tensor(float)", shape=("batch", 3)) -> None:
        self.name = name
        self.type = type_
        self.shape = list(shape)


def test_alias_table_maps_the_friendly_names_onto_onnx_spelling():
    assert PROVIDER_ALIASES["cuda"] == "CUDAExecutionProvider"
    assert PROVIDER_ALIASES["trt"] == PROVIDER_ALIASES["tensorrt"]
    assert PROVIDER_ALIASES["dml"] == PROVIDER_ALIASES["directml"]


def test_explicit_provider_is_honored_and_cpu_is_appended_as_the_fallback():
    resolved = onnx_providers("cuda", _ALL)
    assert _names(resolved) == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert resolved[0][1]["device_id"] == 0


def test_gpu_providers_are_pinned_to_the_requested_ordinal():
    resolved = onnx_providers(["cuda"], _ALL, device_id=3)
    assert resolved[0][1]["device_id"] == 3


def test_tensorrt_enables_the_engine_cache_so_a_worker_compiles_once():
    resolved = onnx_providers(["tensorrt"], _ALL)
    assert resolved[0][1]["trt_engine_cache_enable"] is True


def test_caller_provider_options_win_over_the_defaults():
    resolved = onnx_providers(
        ["tensorrt"], _ALL, provider_options={"tensorrt": {"trt_fp16_enable": True, "device_id": 7}}
    )
    assert resolved[0][1] == {
        "device_id": 7,
        "trt_engine_cache_enable": True,
        "trt_fp16_enable": True,
    }


def test_an_unavailable_provider_warns_rather_than_silently_running_on_cpu():
    # The whole point: dropping it quietly is a correct answer at CPU speed, which is
    # invisible in the results and expensive in the bill.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = onnx_providers("cuda", ["CPUExecutionProvider"])
    assert _names(resolved) == ["CPUExecutionProvider"]
    assert any(issubclass(w.category, PerformanceWarning) for w in caught)


def test_auto_selection_names_nothing_accelerated_without_a_visible_device(monkeypatch):
    monkeypatch.setattr("batcher.ml.gpu.detect_backend", lambda: "cpu")
    assert _names(onnx_providers(None, _ALL)) == ["CPUExecutionProvider"]


def test_auto_selection_prefers_cuda_and_never_defaults_to_tensorrt(monkeypatch):
    # TensorRT compiles on the first batch (minutes for a transformer), so it must be a
    # deliberate choice rather than something an auto-selection lands on.
    monkeypatch.setattr("batcher.ml.gpu.detect_backend", lambda: "cuda")
    names = _names(onnx_providers(None, _ALL))
    assert names[0] == "CUDAExecutionProvider"
    assert "TensorrtExecutionProvider" not in names


def test_duplicate_requests_are_collapsed_in_order():
    assert _names(onnx_providers(["cuda", "cuda", "cpu"], _ALL)) == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_device_id_defaults_to_the_actors_own_visible_card():
    assert resolve_device_id(None) == 0
    assert resolve_device_id(2) == 2
    assert resolve_device_id(-1) == 0


def test_thread_target_honors_an_explicit_request_over_the_host(monkeypatch):
    assert runtime_thread_target(4) == 4
    monkeypatch.setenv("OMP_NUM_THREADS", "6")
    assert runtime_thread_target(None) == 6


def test_port_mapping_pairs_positionally_and_skips_identical_names():
    assert port_mapping(["a", "b"], ["x", "y"]) == {"a": "x", "b": "y"}
    assert port_mapping(["a", "b"], None) == {}
    assert port_mapping(["a", "b"], ["a", "y"]) == {"b": "y"}


def test_renamed_ports_translates_both_directions():
    class _Client:
        def predict(self, feed):
            assert set(feed) == {"input_ids"}
            return {"logits": feed["input_ids"]}

    wrapped = RenamedPorts(_Client(), {"tokens": "input_ids"}, {"logits": "score"})
    assert wrapped.predict({"tokens": 1}) == {"score": 1}


def test_renamed_ports_declines_the_optional_halves_a_client_does_not_define():
    class _Bare:
        def predict(self, feed):
            return feed

    wrapped = RenamedPorts(_Bare(), {}, {})
    assert wrapped.batch_window() is None
    wrapped.close()  # must not raise


def test_renamed_ports_forwards_the_declared_batch_window():
    class _Windowed:
        def predict(self, feed):
            return feed

        def batch_window(self):
            return 32

    assert RenamedPorts(_Windowed(), {}, {}).batch_window() == 32


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        ("tensor(float)", "float32"),
        ("tensor(float16)", "float16"),
        ("tensor(double)", "float64"),
        ("tensor(int64)", "int64"),
        ("tensor(bool)", "bool"),
    ],
)
def test_every_declared_element_type_maps_to_the_dtype_the_runtime_demands(declared, expected):
    # A graph does not coerce: feeding float32 to a float16 input is rejected, and reported
    # as a message about the tensor's *shape*.
    assert ONNX_TO_NUMPY[declared] == expected


def test_a_fixed_batch_dimension_is_reported_as_the_models_window():
    assert _static_batch_size([_Spec("a", shape=(32, 3)), _Spec("b", shape=(32, 5))]) == 32


def test_a_dynamic_batch_dimension_declares_no_window():
    assert _static_batch_size([_Spec("a", shape=("batch", 3))]) is None
    assert _static_batch_size([_Spec("a", shape=(None, 3))]) is None


def test_a_leading_one_is_a_single_example_export_not_a_one_row_window():
    # Splitting a batch into 1-row requests would be correct and catastrophically slow.
    assert _static_batch_size([_Spec("a", shape=(1, 3))]) is None


def test_inputs_disagreeing_on_the_batch_dimension_declare_no_window():
    assert _static_batch_size([_Spec("a", shape=(32, 3)), _Spec("b", shape=(16, 3))]) is None


def test_tabular_onnx_feed_is_cast_to_the_graphs_own_element_type():
    # The substring test this replaced read "tensor(float16)" as float32 — so every
    # half-precision export was fed the wrong width — and "tensor(double)" as not-float.
    from batcher.ml.tabular.estimators import _as_graph_dtype

    matrix = np.zeros((2, 3), dtype="float64")
    assert _as_graph_dtype(matrix, "tensor(float)").dtype == np.dtype("float32")
    assert _as_graph_dtype(matrix, "tensor(float16)").dtype == np.dtype("float16")
    assert _as_graph_dtype(matrix, "tensor(double)").dtype == np.dtype("float64")
    assert _as_graph_dtype(matrix, "tensor(string)") is matrix


class TestTorchInputPrecision:
    """A module is fed the precision its own weights are in.

    Not an optimization: the engine normalizes Float32 to Float64 at the FFI boundary, so an
    ordinary numeric column arrives as float64 and every float32 checkpoint — nearly all of
    them — refused it with "expected scalar type Float but found Double" from inside the
    forward.
    """

    @staticmethod
    def _dtype_of(module):
        from batcher.ml.runtimes.torch_module import _parameter_dtype

        return _parameter_dtype(module)

    def test_a_float32_module_reports_float32(self):
        torch = pytest.importorskip("torch", reason="torch not installed")

        assert self._dtype_of(torch.nn.Linear(2, 2)) == torch.float32

    def test_a_half_module_reports_half(self):
        torch = pytest.importorskip("torch", reason="torch not installed")

        assert self._dtype_of(torch.nn.Linear(2, 2).to(torch.float16)) == torch.float16

    def test_an_integer_buffer_does_not_decide_the_precision(self):
        # A module's integer buffers say nothing about the precision its matmuls run in.
        torch = pytest.importorskip("torch", reason="torch not installed")

        module = torch.nn.Linear(2, 2)
        module.register_buffer("position_ids", torch.zeros(4, dtype=torch.int64))
        assert self._dtype_of(module) == torch.float32

    def test_a_module_with_no_parameters_leaves_the_input_alone(self):
        torch = pytest.importorskip("torch", reason="torch not installed")

        assert self._dtype_of(torch.nn.ReLU()) is None

    def test_a_float64_column_reaches_a_float32_module(self):
        torch = pytest.importorskip("torch", reason="torch not installed")

        from batcher.ml.runtimes import TorchModule

        module = TorchModule(lambda: torch.nn.Linear(3, 2), device="cpu")
        out = module.predict({"x": np.zeros((2, 3), dtype="float64")})
        assert out["output"].shape == (2, 2)

    def test_an_integer_input_keeps_its_dtype(self):
        # An integer input is an index — a token id, a class — and casting one to a float is
        # not a precision change, it is destroying what the embedding table is looked up by.
        torch = pytest.importorskip("torch", reason="torch not installed")

        from batcher.ml.runtimes import TorchModule

        module = TorchModule(lambda: torch.nn.Embedding(10, 4), device="cpu")
        out = module.predict({"ids": np.array([[1, 2, 3]], dtype="int64")})
        assert out["output"].shape == (1, 3, 4)
