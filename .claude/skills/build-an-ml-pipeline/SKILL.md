---
name: build-an-ml-pipeline
description: Build batch inference, embeddings/vector search, multimodal (image/audio/video) decode, and training-data loading on Batcher — the ds.ml entry points, class-based model reuse, GPU sizing and autobatching, batch-first UDFs, streaming so nothing materializes, error handling, and writing results back out. Invoke when writing or debugging an inference, embedding, multimodal, or data-loader pipeline.
---

# Build an ML pipeline

ML on Batcher is not a separate system: it is the same lazy `Dataset` with an `.ml` accessor.
Relational ops (`filter`, `join`, `group_by`) compose with an inference stage in one plan, so
the optimizer prunes columns and pushes filters *through* your model stage. Two rules dominate
everything below:

1. **UDFs take whole Arrow batches, never rows.** Python must not touch a tuple in the hot
   path. `ds.ml.map_batches(fn)` hands `fn` a `pyarrow.RecordBatch` and expects one back.
   `ds.map` exists and is per-row Python — it is the slow path.
2. **Pass a class, not a function**, whenever a model is involved. The engine instantiates the
   class **once per worker**; `__call__` runs per batch. A plain function with `num_gpus > 0`
   reloads the model every batch, and Batcher emits a `PerformanceWarning` saying exactly that.

## The `ds.ml` surface

`ds.ml` is `DatasetML` (`python/batcher/api/dataset/ml.py`). The workhorses:

```
map_batches(fn: Callable | type, *, batch_size=None, input_columns=None, output_columns=None,
            num_workers="auto", num_gpus=0.0, concurrency=None, batch_format="pyarrow",
            accelerator_type=None, model_memory_gb=0.0, multiprocessing=False,
            max_errored_rows=0) -> Dataset
infer(model: str | Callable | type, *, column=None, output_column="prediction", task=None,
      ...same GPU/batch kwargs as map_batches...) -> Dataset
embed(model: str | Callable | type, *, column=None, output_column="embedding", ...) -> Dataset
generate(engine, *, prompt_column, output_column="response", parse_json=False, ...)
classify(engine, *, labels: list[str], ...)        extract(engine, *, schema: dict[str, str], ...)
download(url_column, *, output_column="bytes", max_concurrency=16, on_error="raise")
upload(data_column, directory, *, output_column="path", name_column=None, extension="", ...)
train_test_split(test_size=0.25, *, seed=0, key=None)   random_split(fractions, *, seed=0, ...)
iter_torch_batches(...)  stream_loader(...)  similarity_join(...)  drop_near_duplicates(...)
```

`infer` and `embed` are `map_batches` with a model-id shortcut in front; there is no separate
inference engine to learn.

## Shape 1 — batch inference over a model

```python
# Needs a GPU + a real checkpoint.
import batcher as bt
import pyarrow as pa
import torch


class Classifier:
    def __init__(self) -> None:
        self.model = torch.load("model.pt").eval()   # once per worker

    def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        features = torch.tensor(batch.column("feature").to_numpy())
        with torch.no_grad():
            preds = self.model(features).argmax(dim=1)
        return batch.append_column("label", pa.array(preds.tolist()))


labeled = (
    bt.read.parquet("s3://bucket/features.parquet")
    .filter(bt.col("active"))                       # pushed below the model stage
    .ml.map_batches(Classifier, batch_size=1024, num_gpus=1.0, concurrency=4,
                    input_columns=["feature", "active"])
)
labeled.write.parquet("s3://bucket/labeled.parquet")
```

For a HuggingFace `transformers` pipeline, skip the wrapper entirely (needs the
`transformers` extra):

```python
scored = reviews.ml.infer("distilbert-base-uncased-finetuned-sst-2-english",
                          column="text", output_column="sentiment", task="sentiment-analysis",
                          batch_size=64, num_gpus=1, concurrency=(1, 4))
```

`input_columns=` must name **every** column the callable reads — projection pushdown prunes the
scan to that list, so an omission is a correctness bug, not a perf nit; leave it `None`
(the default) when unsure, which keeps every column alive. `output_columns=`
declares the schema the stage produces so later ops can name those columns.
`batch_format="numpy"` gives `{column: ndarray}` instead of a `RecordBatch`; conversion happens
only around the call, the engine boundary stays Arrow. Fully runnable with no GPU or model:
`examples/ml_inference.py`.

## Shape 2 — embeddings and vector output

```python
# Needs the sentence-transformers extra + a GPU.
import batcher as bt
from batcher.ml import build_vector_index, vector_search

vectors = bt.read.parquet("s3://bucket/chunks.parquet").ml.embed(
    "sentence-transformers/all-MiniLM-L6-v2", column="text",
    output_column="embedding", batch_size=256, num_gpus=1, concurrency=2,
)
vectors.write.lance("s3://bucket/chunks.lance")
build_vector_index("s3://bucket/chunks.lance", "embedding")

hits = vector_search("s3://bucket/chunks.lance", query_vector, column="embedding", k=10)
```

`vector_search(uri, query, *, column="embedding", k=10, columns=None, filter=None,
nprobes=None, refine_factor=None) -> Dataset` returns a `Dataset`, so hits join, filter and
aggregate like anything else. For dedup without an index, `ds.ml.near_duplicates` /
`drop_near_duplicates` / `similarity_join` are MinHash/LSH based and need no model.

## Shape 3 — multimodal decode from cloud storage

Decode runs in the Rust data plane (SIMD JPEG + SIMD resize, fanned out per row), not a Python
loop. Two ways in — the reader decodes on the way, or you fetch bytes and decode as an
expression:

```python
# Needs cloud credentials.
import batcher as bt
from batcher import col

imgs = bt.read.images("s3://bucket/photos/", decode=True, size=(224, 224))
tensors = (
    bt.read.parquet("s3://bucket/catalog.parquet")
    .ml.download("url", output_column="bytes", on_error="null")
    .with_columns(image=col("bytes").image.to_tensor(224, 224))
)
```

Accessor methods (all take **positional ints**, unlike `read.images(size=(h, w))`):

| Accessor | Methods | Output |
|---|---|---|
| `.image` | `decode()`, `resize(width, height)`, `to_tensor(width, height)` | `resize` re-encodes to PNG bytes; `to_tensor` gives a flat uint8 tensor |
| `.audio` | `decode()`, `resample(rate)`, `to_waveform()` | mono PCM `list<float>` |
| `.video` | `decode()` | frames |

Readers: `read.images(path, *, decode=False, size=None)`, `read.audio(path, *, decode=False,
sample_rate=None)`, `read.video(path, *, decode=False, size=None, num_frames=8)`. Use
`to_tensor` when a model is next; `resize` when you want a small blob before a shuffle, a
spill, or a write. Keep `batch_size` small for multi-GB video — the payload is the memory.

## Shape 4 — feeding a training loop

**`ds.ml.iter_torch_batches` streams; `ds.to_torch` materializes.** That distinction is
load-bearing — use the streaming one for anything that does not comfortably fit.

```python
# Needs torch.
from batcher.ml import StandardScaler

train, test = featured.ml.train_test_split(test_size=0.25, seed=7, key="user_id")

scaler = StandardScaler(["clicks", "spend"])
scaler.fit(train)                       # fit on train only
train_x = scaler.transform(train)

for batch in train_x.select("clicks", "spend", "label").ml.iter_torch_batches(
    batch_size=16, device="cpu", prefetch_batches=2, pin_memory=True,
):
    ...  # batch is a {column: tensor} dict
```

- Full signature: `iter_torch_batches(*, batch_size=None, columns=None, device="auto",
  collate_fn=None, prefetch_batches=1, pin_memory=False, zero_copy=False,
  local_shuffle_buffer_size=None, seed=0)`.
- Multi-rank DDP: `ds.ml.stream_loader(*, batch_size, world_size=1, rank=0, epoch=0,
  seed=0, shuffle=True, drop_last=True, columns=None, global_consumed=0)` — resumable
  (`global_consumed` restarts mid-epoch); `batcher.ml.streaming_split(dataset, world_size,
  *, rank=None, queue_depth=2)` splits one dataset across ranks.
- `Dataset.to_torch / to_tf / to_torch_dataloader(*, columns=None, batch_size=None)` are the
  materializing conveniences for data that fits.
- Preprocessors are the next section — `StandardScaler` above is one of fifteen.

## Preprocessors

`batcher.ml.preprocessors` (re-exported from `batcher.ml`) is a stateful, sklearn-shaped
feature layer over lazy `Dataset`s. Fifteen public names:

| Group | Classes |
|---|---|
| Scaling | `StandardScaler(cols, *, with_mean=True, with_std=True)`, `MinMaxScaler(cols, *, feature_range=(0.0, 1.0))`, `MaxAbsScaler(cols)`, `RobustScaler(cols, *, quantile_range=(25.0, 75.0))`, `Normalizer(cols, *, norm="l2")` |
| Missing / binning | `SimpleImputer(cols, *, strategy="mean", fill_value=None)`, `KBinsDiscretizer(cols, *, n_bins=5, strategy="quantile")` |
| Categorical | `OneHotEncoder(cols, *, drop_first=False)`, `OrdinalEncoder(cols, *, unknown_value=-1)`, `LabelEncoder(col, *, unknown_value=-1)`, `MultiHotEncoder(col, *, categories=None)` |
| Assembly / text | `Concatenator(cols, *, output_column="features", drop=False)`, `Tokenizer(col, tokenizer, *, output_column=None)` |
| Base / composition | `Preprocessor` (the `fit`/`transform`/`fit_transform` Protocol), `Chain(*steps)` |

**The state model.** `fit(ds) -> Preprocessor` **executes** — it collects the statistics
(means, quantiles, category vocabularies) and stores them on the instance. `transform(ds)
-> Dataset` is lazy and returns a new plan; nothing runs until a terminal op.
`fit_transform(ds)` is the two in sequence. So a `fit` is a materialization point in an
otherwise lazy pipeline — fit once, reuse the fitted object.

**`Chain` fits sequentially.** Each step is fit on the output of the previous step's
transform, so ordering matters: impute before you scale, scale before you concatenate.

**The leakage hazard — fit on train only.** `fit` reads the whole dataset it is given. Fit
on the full dataset and test-set statistics leak into the training features; your offline
metric is optimistic and production is worse. Split *first*, `fit` on train, then
`transform` both:

```python
import batcher as bt
from batcher.ml import Chain, OneHotEncoder, SimpleImputer, StandardScaler

ds = bt.from_pydict({"clicks": [1.0, 2.0, 3.0, 4.0],
                     "spend": [10.0, 20.0, 30.0, 40.0],
                     "city": ["a", "b", "a", "b"]})
train, test = ds.ml.train_test_split(test_size=0.5, seed=0)

chain = Chain(
    SimpleImputer(["spend"], strategy="mean"),
    StandardScaler(["clicks", "spend"]),
    OneHotEncoder(["city"]),
)
chain.fit(train)                    # statistics come from train only
train_x = chain.transform(train)
test_x = chain.transform(test)      # same fitted statistics — no leakage
```

The encoders hold the same line: the vocabulary is frozen at `fit`, so a category first
seen in test maps to `unknown_value` (`-1`) under `OrdinalEncoder`/`LabelEncoder` and to an
all-zero row under `OneHotEncoder` — the columns never change shape between train and
serve.
`examples/preprocessors.py` and `docs/ml/preprocessors/index.md` are the fuller treatment.

## GPU and autobatching

Five keywords, on every `.ml` stage: `num_gpus`, `concurrency`, `batch_size`,
`accelerator_type`, `model_memory_gb`.

- `num_gpus` is what **each actor reserves**; `concurrency` is **how many actors**, or a
  `(min, max)` tuple for an autoscaling pool. `num_gpus=1, concurrency=4` → four actors each
  owning a GPU; `num_gpus=0.5, concurrency=4` → four actors packed two per GPU; `num_gpus=0.0`
  (default) is CPU-only. `accelerator_type="NVIDIA_A100"` pins actors to a device.
- **`model_memory_gb` is the one knob to set first.** Declare the footprint and the resource
  layer packs actors per device and seeds `batch_size` from the leftover VRAM.
- **Leave `batch_size` unset by default.** The inference pool autobatches — a PID controller
  toward a latency target, or a hill-climb on throughput — and it warm-starts from what
  previous runs measured. A CUDA OOM is survived by recursively halving the batch, re-raising
  only if a single row still OOMs. Set `batch_size` only when a measurement says to. So:
  `ds.ml.infer(Model, model_memory_gb=1.5)` (let it size itself) before
  `ds.ml.infer(Model, num_gpus=0.25, concurrency=8, batch_size=256)` (explicit override).

## LLM stages

`ds.ml.generate` / `classify` / `extract` take an *engine factory*, not a model:
`batcher.ml.vllm_engine(model, *, chat=False, system=None, sampling=None, guided_json=None,
lora_path=None, quantization="auto", **engine_kwargs)` for local vLLM, or `http_engine(base_url,
model, *, api_key=None, chat=True, max_tokens=512, temperature=0.0, concurrency=8)` for an
OpenAI-compatible endpoint. `extract(schema={...})` and `parse_json=True` give structured
output; `image_column=` handles VLM prompts.

## Errors, laziness, and output

- **Errors** — exactly two real options, both narrow.
  `ds.ml.map_batches(..., max_errored_rows=N)` bisects a batch whose `fn` raises, drops the
  offending rows, and gives up past the budget (default `0` = strict); it exists **only** on
  `map_batches`, not on `infer`/`embed`/`generate`. `ds.ml.download(..., on_error="null")`
  turns a failed fetch into a null (`"raise"` or `"null"` only). There is no per-row error
  column and no dead-letter sink — anything richer, encode as a column your `fn` sets.
- **Stay lazy, stay streaming.** Every `.ml` call returns a new `Dataset`; nothing runs until a
  terminal op. Never `collect()` a multimodal dataset to iterate it — decoded images and audio
  are orders of magnitude larger than their encoded bytes. Use `iter_batches()`,
  `iter_torch_batches()`, or write straight to a sink so memory stays bounded.
- **Writing out.** `ds.write.parquet(path, compression="zstd")`, `.lance`, `.delta(uri,
  mode="append", merge_on=[...])`, `.iceberg`, `.json`; `ds.ml.upload(data_column, directory)`
  for bytes back to object storage. A retried or preempted worker recomputes its partition, so
  a `fn` with an external side effect (vector-DB insert, REST POST) can apply it **twice** —
  make sinks upserts on a stable key. A pure transform is already safe.

## Self-review checklist

1. Model arrives as a **class**, not a closure over a loaded model.
2. `input_columns=` names every column the callable reads; `output_columns=` declares what it adds.
3. No per-row Python — no `ds.map`, no `.to_pylist()` inside a UDF loop.
4. `batch_size` left unset unless measured; `model_memory_gb` set if there is a GPU.
5. Nothing large is `collect()`ed; the pipeline ends in a write or a streaming iterator.
6. Pure column arithmetic is an `Expr`, not a UDF — expressions run in Rust and JIT.
7. `ds.explain()` shows the filter/projection landing below the model stage; `ds.stats()`
   after a sample run says where the time went.
8. Preprocessors are `fit` on the **train split only**, then `transform` applied to train and
   test — never `fit_transform` on the full dataset.

## See also

- `docs/ml/{index,inference,batch-scoring,embeddings,multimodal,gpu,llm,vector-search,rag,
  data-loaders,distributed-training,preprocessors,streaming,tokenization,serving}.md`;
  `docs/user-guide/{udfs,cloud-storage,writing-data,explain-plans}.md`.
- `docs/tutorials/{batch-inference,distributed-training-pipeline,feature-engineering}.md`;
  `examples/ml_inference.py`, `examples/preprocessors.py`.
- Skills: `run-a-distributed-job` (taking this to a cluster — GPU stages force distribution),
  `write-a-batcher-pipeline`, `debug-a-batcher-query`, `migrate-from-daft`.
