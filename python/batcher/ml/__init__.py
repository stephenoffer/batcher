"""ML data plane — actor-pool batch inference, training ingest, and preprocessing.

The native-pipeline foundation lives in the Rust `bc-udf` crate; this package is the
control-plane orchestration (model-once-per-worker pools, dynamic batching) over
whole Arrow batches.

It also re-exports the sklearn-style `Preprocessor` family (scalers, encoders,
imputers, binning, tokenization): `fit` learns state with one mergeable aggregate
over the engine and `transform` is a lazy `Expr` projection, so feature engineering
never touches a row in Python. Fit on the train split, transform both splits —
`ds.ml.train_test_split` produces them.
"""

from __future__ import annotations

from batcher.ml.converters import to_numpy_batches, to_tf_dataset, to_torch_iterable
from batcher.ml.embed import EncoderFactory, build_vector_index, embed, vector_search
from batcher.ml.inference import InferencePool, Worker, WorkerFactory
from batcher.ml.llm import (
    Engine,
    EngineFactory,
    http_engine,
    llm_generate,
    llm_udf,
    pack_sequences,
    vllm_engine,
)
from batcher.ml.loader import (
    iter_torch_batches,
    shard_stream_loader,
    stream_loader,
    streaming_split,
)
from batcher.ml.pipeline import Stage, run_pipeline
from batcher.ml.preprocessors import (
    Chain,
    Concatenator,
    KBinsDiscretizer,
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    MultiHotEncoder,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    Preprocessor,
    RobustScaler,
    SimpleImputer,
    StandardScaler,
    Tokenizer,
)
from batcher.ml.serving import (
    ServingClient,
    http_client,
    serve_deployment,
    serving_udf,
    torchserve_client,
    triton_client,
)
from batcher.ml.streaming_sampler import (
    ResumableSampler,
    epoch_order,
    epoch_permutation,
    rank_index_batches,
    usable_length,
)

__all__ = [
    "Chain",
    "Concatenator",
    "EncoderFactory",
    "Engine",
    "EngineFactory",
    "InferencePool",
    "KBinsDiscretizer",
    "LabelEncoder",
    "MaxAbsScaler",
    "MinMaxScaler",
    "MultiHotEncoder",
    "Normalizer",
    "OneHotEncoder",
    "OrdinalEncoder",
    "Preprocessor",
    "ResumableSampler",
    "RobustScaler",
    "ServingClient",
    "SimpleImputer",
    "Stage",
    "StandardScaler",
    "Tokenizer",
    "Worker",
    "WorkerFactory",
    "build_vector_index",
    "embed",
    "epoch_order",
    "epoch_permutation",
    "http_client",
    "http_engine",
    "iter_torch_batches",
    "llm_generate",
    "llm_udf",
    "pack_sequences",
    "rank_index_batches",
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
