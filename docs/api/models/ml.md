# The ML accessor

This page covers the `.ml` accessor on a {py:class}`Dataset <batcher.Dataset>` and the `batcher.ml` package behind it. For the relational surface these compose with, see {doc}`Dataset </api/relational/dataset>`.

ML work attaches to a `Dataset` through the `.ml` accessor:

| Method | Use |
| --- | --- |
| {py:meth}`ds.ml.map_batches(fn, ...) <batcher.api.dataset.ml.DatasetML.map_batches>` | Apply an arbitrary function to each Arrow batch. |
| {py:meth}`ds.ml.infer(model, ...) <batcher.api.dataset.ml.DatasetML.infer>` | Run batch inference from a model id plus `column`, or from a model callable. |
| {py:meth}`ds.ml.embed(model, ...) <batcher.api.dataset.ml.DatasetML.embed>` | Generate embeddings from a model id plus `column`, or from a model callable. |
| {py:meth}`ds.ml.generate(engine, ...) <batcher.api.dataset.ml.DatasetML.generate>` | Offline LLM text generation, appending the response column. |
| {py:meth}`ds.ml.download(url_col, ...) <batcher.api.dataset.ml.DatasetML.download>` | Fetch bytes at each URL/path into a column. |
| {py:meth}`ds.ml.upload(data_col, dir, ...) <batcher.api.dataset.ml.DatasetML.upload>` | Write a bytes column out to object storage. |
| {py:meth}`ds.ml.iter_torch_batches(...) <batcher.api.dataset.ml.DatasetML.iter_torch_batches>` | Stream the dataset to PyTorch as tensor batches. |
| {py:meth}`ds.ml.stream_loader(...) <batcher.api.dataset.ml.DatasetML.stream_loader>` | A distributed-training `IterableDataset` for one rank. |
| {py:meth}`ds.ml.train_test_split(test_size, seed=0) <batcher.api.dataset.ml.DatasetML.train_test_split>` | Disjoint, reproducible train/test `Dataset`s. |
| {py:meth}`ds.ml.random_split(fractions, seed=0) <batcher.api.dataset.ml.DatasetML.random_split>` | The n-way generalization (train/val/test). |
| {py:meth}`ds.ml.near_duplicates(column, threshold=0.8) <batcher.api.dataset.ml.DatasetML.near_duplicates>` | MinHash + LSH near-duplicate pairs. |
| {py:meth}`ds.ml.drop_near_duplicates(column, threshold=0.8) <batcher.api.dataset.ml.DatasetML.drop_near_duplicates>` | Fuzzy dedup, keeping one per cluster. |
| {py:meth}`ds.ml.nearest_neighbors(query, column="embedding", k=10, metric="cosine") <batcher.api.dataset.ml.DatasetML.nearest_neighbors>` | Exact brute-force top-`k` retrieval against a query vector. |
| {py:meth}`ds.ml.similarity_to(query, column="embedding", metric="cosine") <batcher.api.dataset.ml.DatasetML.similarity_to>` | Score every row against a query vector (no top-`k` cut). |
| {py:meth}`ds.ml.normalize_embeddings(column, output_column=None) <batcher.api.dataset.ml.DatasetML.normalize_embeddings>` | Unit-normalize an embedding column (L2 = 1). |

These operate on whole `pyarrow.RecordBatch` objects, never on individual rows. They're lazy, as every other transformation is, and return a new `Dataset`. The loaders are the exception, returning a torch iterator.

## Whole-batch semantics

A function passed to `map_batches` takes one `pyarrow.RecordBatch` and returns one
`pyarrow.RecordBatch`. Because it sees the whole batch, it can use vectorized
Arrow compute rather than per-row Python loops.

```python
import batcher as bt
import pyarrow.compute as pc

ds = bt.from_pydict({"x": [1, 2, 3, 4], "y": [10, 20, 30, 40]})


def add_sum(batch):
    total = pc.add(batch.column("x"), batch.column("y"))
    return batch.append_column("sum", total)


print(ds.ml.map_batches(add_sum).to_pydict())
# {'x': [1, 2, 3, 4], 'y': [10, 20, 30, 40], 'sum': [11, 22, 33, 44]}
```

## Class-based functions load once per worker

A plain function is re-imported on each worker. A class is instantiated once per
worker and then called per batch, so any expensive setup (loading a model,
opening a tokenizer) happens once and is reused across all batches that worker
processes. The class implements `__call__(self, batch) -> batch`.

```python
import pyarrow as pa


class Scale:
    def __init__(self, factor):
        self.factor = pa.scalar(factor)

    def __call__(self, batch):
        scaled = pc.multiply(batch.column("x"), self.factor)
        return batch.set_column(0, "x", scaled)


print(ds.ml.map_batches(Scale(10)).to_pydict())
# {'x': [10, 20, 30, 40], 'y': [10, 20, 30, 40]}
```

For a real model, the constructor loads the weights and `__call__` runs the
forward pass. That needs a GPU and a model, so it's shown but not run here.

```python
# docs: skip
class Classifier:
    def __init__(self):
        import torch

        self.model = torch.load("model.pt").cuda().eval()

    def __call__(self, batch):
        import torch

        x = torch.tensor(batch.column("features").to_pylist()).cuda()
        with torch.no_grad():
            preds = self.model(x).argmax(dim=1).cpu().tolist()
        return batch.append_column("prediction", pa.array(preds))


labelled = ds.ml.map_batches(Classifier, num_gpus=1, concurrency=4)
```

## Common arguments

`map_batches`, `infer`, and `embed` share these keywords:

| Argument | Meaning |
| --- | --- |
| `batch_size` | Rows per batch handed to `fn`. Defaults to the engine morsel size. |
| `output_columns` | Names of the columns the function produces, when they differ from the input. |
| `batch_format` | What `fn` receives/returns: `"pyarrow"` (default), `"numpy"`, `"pandas"`, or `"torch"`. |
| `num_gpus` | GPUs to reserve per worker (a fraction packs several workers onto one GPU). |
| `concurrency` | Actor-pool size: an `int` for a fixed pool, or a `(min, max)` tuple to autoscale to the workload. |
| `accelerator_type` | Pin GPU actors to a device model (a `ray.util.accelerators` name such as `"NVIDIA_A100"`). |
| `model_memory_gb` | The model's GB footprint. Budgets host memory per worker to protect against OOM, and VRAM-packs small models onto a shared GPU. |
| `num_workers` | Number of workers (`map_batches`). |

`num_gpus` and `concurrency` together describe a GPU actor pool: each actor holds
`num_gpus` of a device, and `concurrency` actors run in parallel. `batch_format` converts only around the call, and the engine boundary stays Arrow. See {doc}`GPU scheduling </ml/inference/gpu>`.

## infer and embed

`ds.ml.infer(model, ...)` and `ds.ml.embed(model, ...)` are the inference-shaped calls. The quickest form is a **model identifier** plus the `column` to run on. The model loads once per worker and the result is appended, as a prediction for `infer` and a vector for `embed`. `infer` resolves a HuggingFace `transformers` pipeline, and `embed` resolves a `sentence-transformers` model.

```python
# docs: skip
scored = ds.ml.infer("distilbert-base-uncased-finetuned-sst-2-english", column="text")
vectors = ds.ml.embed("sentence-transformers/all-MiniLM-L6-v2", column="text")
```

For full control over a custom model, a non-text modality, or your own batching, pass a callable or a class that loads weights once per worker, and declare the result schema with `output_columns`. Both forms take `batch_size`, `num_gpus`, and `concurrency`. Real models need GPUs, so these aren't run here.

```python
# docs: skip
scored = ds.ml.infer(Classifier, output_columns=[...], batch_size=512, num_gpus=1, concurrency=4)
vectors = ds.ml.embed(Embedder, output_columns=[...], batch_size=256, num_gpus=1, concurrency=2)
```

See {doc}`Inference </ml/inference/inference>` for the inference workflow and
{doc}`Streaming </ml/inference/streaming>` for feeding training loops.

## What lives outside the accessor

Operators that aren't `Dataset` methods live in `batcher.ml`: the standalone `embed` and `llm_generate` functions, the {doc}`preprocessors </ml/preparing/preprocessors/index>`, the {doc}`serving adapters </ml/training/serving>`, {doc}`vector search </ml/preparing/multimodal/index>`, the {py:class}`Chain <batcher.ml.preprocessors.Chain>` preprocessor pipeline, the {py:class}`ResumableSampler <batcher.ml.ResumableSampler>` checkpointable per-rank index stream, and the {doc}`LLM engines </ml/retrieval/llm/index>`.

A *callable* model passed to `map_batches` or `infer` receives the whole batch and picks its own columns, so there's no `input_columns=` keyword. The model-identifier form of `infer` and `embed` takes the `column` to run on instead.

## `batcher.ml` reference

The `.ml` accessor above covers the common path. Underneath it, `batcher.ml` exports the same machinery as plain functions over Arrow batch iterators. Reach for those when you're driving the pipeline yourself, such as from a custom training loop or a serving process, rather than executing a `Dataset`.

```python
import batcher.ml as ml
```

### LLM inference

An *engine* is any callable from a list of prompts to a list of completions. That is the
whole contract, which is why a local vLLM engine, a local SGLang engine, a remote
OpenAI-compatible endpoint, and a hosted Claude model are interchangeable: swap
{py:obj}`vllm_engine <batcher.ml.vllm_engine>` for
{py:obj}`sglang_engine <batcher.ml.sglang_engine>`,
{py:obj}`http_engine <batcher.ml.http_engine>` or
{py:obj}`anthropic_engine <batcher.ml.anthropic_engine>`,
{py:obj}`bedrock_engine <batcher.ml.bedrock_engine>` or
{py:obj}`gemini_engine <batcher.ml.gemini_engine>` and nothing else changes.

```{eval-rst}
.. currentmodule:: batcher.ml

.. autosummary::
   :toctree: generated
   :nosignatures:

   vllm_engine
   sglang_engine
   http_engine
   anthropic_engine
   bedrock_engine
   gemini_engine
   llm_generate
   llm_udf
   json_schema

.. autodata:: Engine

.. autodata:: EngineFactory
```

### Training-corpus preparation

Mix several sources at declared weights, drop the documents that are not prose, and remove the
evaluation data that leaked into a web-scale corpus. See {doc}`/ml/training/training-corpus`.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   mix_corpora
   MixtureReport
   quality_filter
   quality_flags
   quality_report
   QualityThresholds
   decontaminate
   contamination_rate
   length_grouped_order
   padding_waste
```

### Retrieval reranking

Rerank a retrieved candidate set before it reaches a model.
{py:obj}`cross_encoder_rerank_udf <batcher.ml.cross_encoder_rerank_udf>` rescores candidates
with a model that reads the query and each passage together, which a vector search cannot do;
{py:obj}`mmr_rerank_udf <batcher.ml.mmr_rerank_udf>` then drops the near-duplicates, so a
context window is not spent on the same passage several times. See {doc}`/ml/retrieval/rag`.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   cross_encoder_rerank_udf
   mmr_rerank_udf

.. autodata:: CrossEncoderScorer
```

### Model-graded evaluation

Score generations with a judge model. Each is a load-once class UDF over the same `Engine`
contract, appending one parsed column rather than a string you still have to interpret.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   llm_score_udf
   llm_pairwise_udf
   llm_verify_udf
```

### Local model runtimes

Run an exported model inside the worker, with no serving process between the data and the
model. Each is a load-once class UDF built on `serving_udf`, so all three inherit its batch
splitting, in-flight pipelining, and output alignment. ONNX Runtime's execution providers are
also how TensorRT is reached. See {doc}`/ml/inference/runtimes`.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   onnx_predictor
   torch_predictor
   openvino_predictor
```

### Model serving

Call a model that lives in another process or on another host. Each client turns a
served endpoint into a UDF you can drop into a pipeline.

```{eval-rst}
.. autoclass:: ServingClient
   :members:

.. autosummary::
   :toctree: generated
   :nosignatures:

   serving_udf
   serve_deployment
   triton_client
   torchserve_client
   http_client
```

### Inference pools and pipelines

{py:obj}`InferencePool <batcher.ml.InferencePool>` keeps model-loading off the hot path:
workers load once and are reused across batches. {py:obj}`run_pipeline <batcher.ml.run_pipeline>`
chains {py:obj}`Stage <batcher.ml.Stage>`s with credit-based backpressure, which is what
overlaps a CPU decode with the GPU forward of the previous batch instead of running them
in lockstep.

When the embedding model runs behind a service rather than in the worker,
{py:obj}`openai_embedding_encoder <batcher.ml.openai_embedding_encoder>` (any
OpenAI-compatible `/embeddings` endpoint) and {py:obj}`tei_encoder <batcher.ml.tei_encoder>`
(a HuggingFace Text-Embeddings-Inference server) are load-once encoders that drop into
`ds.ml.embed(...)` in place of a local model id.

```{eval-rst}
.. autoclass:: InferencePool
   :members:

.. autoclass:: Stage
   :members:

.. autosummary::
   :toctree: generated
   :nosignatures:

   run_pipeline
   embed
   openai_embedding_encoder
   tei_encoder

.. autodata:: Worker

.. autodata:: WorkerFactory
```

### Training loaders

Stream a dataset into a training loop as tensors, without materializing it.
{py:obj}`streaming_split <batcher.ml.streaming_split>` gives each DDP rank a disjoint
shard.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   iter_torch_batches
   to_torch_iterable
   to_tf_dataset
   to_numpy_batches
   stream_loader
   shard_stream_loader
   streaming_split
```

### Feature contract

A trained model is only valid against the exact columns, order, and dtypes it saw during
training. {py:class}`FeatureSpec <batcher.ml.FeatureSpec>` pins that contract so scoring can be checked against it rather than
failing silently on a reordered or retyped frame.

```{eval-rst}
.. autoclass:: FeatureSpec
   :members:
```

### Sampling and resumption

These give deterministic, resumable epoch ordering. A training run that dies at step 40,000 restarts at step 40,000 seeing the same samples in the same order, rather than silently re-showing data it already trained on.

```{eval-rst}
.. autoclass:: ResumableSampler
   :members:
   :special-members: __len__, __iter__

.. autosummary::
   :toctree: generated
   :nosignatures:

   epoch_order
   epoch_permutation
   rank_index_batches
   usable_length
   pack_sequences
```

### Vector search

Build an index over an embedding column and query it:

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   build_vector_index
   vector_search

.. autodata:: EncoderFactory
```

## The rest of the ML reference

The `batcher.ml` surface is large enough to be split by what you are doing:

- {doc}`/api/models/preprocessors`: the fit/transform estimators, `Chain`, and persistence.
- {doc}`/api/models/ml-models`: tabular scoring, in-engine estimators, and evaluation metrics.
- {doc}`/api/models/ml-statistics`: drift, fairness, resampling, cross-validation, interpretation.

## See also

- {doc}`Inference </ml/inference/inference>`: batch prediction and embeddings.
- {doc}`Preprocessors </ml/preparing/preprocessors/index>`: fit/transform feature engineering.
- {doc}`Multimodal </ml/preparing/multimodal/index>`: download, decode, tensors, vector search.
- {doc}`Serving </ml/training/serving>` and {doc}`LLM inference </ml/retrieval/llm/index>`.
- {doc}`PyTorch </ml/inference/pytorch>` and {doc}`streaming </ml/inference/streaming>` training loaders.
- {doc}`GPU scheduling </ml/inference/gpu>`: how `num_gpus` and `concurrency` map to actors.
- {doc}`Tabular models </ml/inference/tabular-models>`: scoring XGBoost, LightGBM, and scikit-learn.
- {doc}`Evaluation </ml/evaluation/evaluation>`: metrics, per-segment scoring, diagnostic tables.
- {doc}`Statistics and drift </ml/evaluation/statistics-and-drift>`: feature screening and monitoring.- {doc}`/cookbook/ml/index`: 16 runnable recipes across the `batcher.ml` surface.
