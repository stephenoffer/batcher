"""Local model runtimes — run an exported model without a serving process.

`ml.serving` reaches a model that lives somewhere else (Triton, TorchServe, an HTTP
endpoint). This package runs the model *in the worker*, which is what batch inference
usually wants: there is no network between the data and the model, no second fleet to
size, and the batch the engine already assembled is the batch the model sees.

Three runtimes, one shape. Each is a `ServingClient` — ``{name: ndarray}`` in,
``{name: ndarray}`` out — wrapped by `ml.serving.serving_udf`, so all three inherit batch
splitting against the model's own window, in-flight pipelining, output alignment and
replacement, and worker teardown, rather than each restating them:

* `onnx_predictor` — ONNX Runtime, and through its execution providers also TensorRT,
  OpenVINO, ROCm, CoreML and DirectML. The portable choice: a model exported once runs
  everywhere, and the provider list is the only thing that changes per host.
* `torch_predictor` — a TorchScript archive, a pickled module, or a factory, for a model
  that was never exported. Applies `eval()` and `inference_mode` for you.
* `openvino_predictor` — OpenVINO directly, for CPU fleets, where owning the scheduling
  (streams, throughput hint) is the difference between a CPU's latency floor and its
  throughput ceiling.

`providers` holds the execution-provider and thread-count resolution the three share.
"""

from __future__ import annotations

from batcher.ml.runtimes.onnx import OnnxSession, onnx_predictor
from batcher.ml.runtimes.openvino import OpenVinoModel, openvino_predictor
from batcher.ml.runtimes.providers import (
    PROVIDER_ALIASES,
    RenamedPorts,
    onnx_providers,
    port_mapping,
    resolve_device_id,
    runtime_thread_target,
)
from batcher.ml.runtimes.torch_module import TorchModule, torch_predictor

__all__ = [
    "PROVIDER_ALIASES",
    "OnnxSession",
    "OpenVinoModel",
    "RenamedPorts",
    "TorchModule",
    "onnx_predictor",
    "onnx_providers",
    "openvino_predictor",
    "port_mapping",
    "resolve_device_id",
    "runtime_thread_target",
    "torch_predictor",
]
