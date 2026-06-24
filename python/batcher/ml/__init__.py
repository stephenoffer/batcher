"""ML data plane — actor-pool batch inference orchestration.

The native-pipeline foundation lives in the Rust `bc-udf` crate; this package is the
control-plane orchestration (model-once-per-worker pools, dynamic batching) over
whole Arrow batches.
"""

from __future__ import annotations

from batcher.ml.converters import to_numpy_batches, to_tf_dataset, to_torch_iterable
from batcher.ml.embed import EncoderFactory, build_vector_index, embed, vector_search
from batcher.ml.inference import InferencePool, Worker, WorkerFactory
from batcher.ml.llm import Engine, EngineFactory, http_engine, llm_generate, vllm_engine
from batcher.ml.loader import (
    iter_torch_batches,
    shard_stream_loader,
    stream_loader,
    streaming_split,
)
from batcher.ml.pipeline import Stage, run_pipeline
from batcher.ml.serving import (
    ServingClient,
    http_client,
    serve_deployment,
    serving_udf,
    torchserve_client,
    triton_client,
)
from batcher.ml.streaming_sampler import elastic_shard, epoch_order, rank_shard, usable_length

__all__ = [
    "EncoderFactory",
    "Engine",
    "EngineFactory",
    "InferencePool",
    "ServingClient",
    "Stage",
    "Worker",
    "WorkerFactory",
    "build_vector_index",
    "elastic_shard",
    "embed",
    "epoch_order",
    "http_client",
    "http_engine",
    "iter_torch_batches",
    "llm_generate",
    "rank_shard",
    "run_pipeline",
    "serve_deployment",
    "serving_udf",
    "shard_stream_loader",
    "stream_loader",
    "streaming_split",
    "to_numpy_batches",
    "to_tf_dataset",
    "to_torch_iterable",
    "torchserve_client",
    "triton_client",
    "usable_length",
    "vector_search",
    "vllm_engine",
]
