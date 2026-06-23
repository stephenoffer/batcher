"""Model-serving adapters — call an external inference server from `map_batches`.

Each adapter returns a load-once class UDF (instantiate-once-per-worker) for
``ds.ml.map_batches`` / ``ds.ml.infer``, so a Batcher pipeline can run batch inference
against Triton, TorchServe, or any columnar-JSON HTTP endpoint while keeping the
preprocessing on CPU and the model call on the server.

    from batcher.ml.serving import triton_client

    udf = triton_client("triton:8000", "resnet50", input_columns=["image"],
                        output_columns=["logits"])
    scored = ds.ml.map_batches(udf, concurrency=4)
"""

from __future__ import annotations

from batcher.ml.serving.base import ServingClient, serving_udf
from batcher.ml.serving.http import http_client
from batcher.ml.serving.online import serve_deployment
from batcher.ml.serving.torchserve import torchserve_client
from batcher.ml.serving.triton import triton_client

__all__ = [
    "ServingClient",
    "http_client",
    "serve_deployment",
    "serving_udf",
    "torchserve_client",
    "triton_client",
]
