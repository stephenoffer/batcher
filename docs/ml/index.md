# Machine learning

This section covers running models where the data already lives, from batch inference and LLM generation to media decoding, evaluation, and the loop that feeds training.

The model reads the same {py:class}`Dataset <batcher.Dataset>` you filtered and joined. Its `.ml` accessor hands your model whole Arrow batches, and the engine places that work on GPUs and worker actors. Inference is an operator in the plan, so it streams. Scoring more data than fits in memory is the ordinary case.

## One pipeline, from raw files to predictions

A model is a class. The constructor loads the weights once per worker, and `__call__` runs on each batch. The stand-in below multiplies a column so it runs anywhere. Swap in a real model and add `num_gpus=1` to put it on a device:

```python
import batcher as bt
import pyarrow as pa
import pyarrow.compute as pc


class Scorer:
    def __init__(self):
        self.weight = 2.0  # load real weights here, once per worker

    def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        return batch.append_column("score", pc.multiply(batch.column("x"), self.weight))


ds = bt.from_pydict({"id": [1, 2, 3], "x": [0.5, 1.0, 1.5]})
scored = ds.ml.infer(Scorer, output_columns=["id", "x", "score"])
print(scored.sort("id").to_pydict())
# {'id': [1, 2, 3], 'x': [0.5, 1.0, 1.5], 'score': [1.0, 2.0, 3.0]}
```

Everything around that call is ordinary dataset work. The read, the label join and the write sit in the same lazy plan as the model. The following diagram shows that plan:

![Raw files to predictions as one lazy plan, where nothing runs until a write or a collect. Read takes Parquet or images and hands bytes to decode, which runs as an .image expression in Rust. Decode hands tensors to a filter and a join that attaches labels. Decode, filter and join run on CPU cores. The rows go to infer, ds.ml.infer with your model class, which runs in a GPU actor pool, and the rows come out with a score and go to a write or an aggregate. Running it as one plan buys three things. The model loads once per worker, in its constructor. The model scores partition k while the CPU stages prepare partition k+1. Batches stream, so scoring more data than fits in memory is the ordinary case.](/_static/diagrams/ml_one_plan.svg)

## The toolbox in a few lines

Each step of an ML workflow is a short call on one lazy dataset. The cells below share a small sales table:

```python
sales = bt.from_pydict(
    {
        "region": ["eu", "eu", "us", "us"],
        "ads": [1.0, 2.0, 3.0, 4.0],
        "revenue": [3.1, 4.9, 7.0, 9.1],
    }
)
train, test = sales.ml.train_test_split(test_size=0.25, seed=7)
print(train.count(), test.count())
# 3 1
```

Scale features with a fitted preprocessor, which keeps its statistics for the held-out split:

```python
from batcher.ml.preprocessors import StandardScaler

scaler = StandardScaler(["ads"]).fit(sales)
print([round(v, 2) for v in scaler.transform(sales).to_pydict()["ads"]])
# [-1.34, -0.45, 0.45, 1.34]
```

Fit a baseline in the engine, in one scan, and score with it:

```python
from batcher.ml.linear import LinearRegression

model = LinearRegression(["ads"], "revenue").fit(sales)
print(round(model.coef_[0], 2), round(model.intercept_, 2))
# 2.01 1.0
scored = model.predict(sales)
```

Evaluate per segment in the same pass. Metrics are aggregates, so `by=` is a group-by:

```python
report = scored.ml.evaluate("revenue", y_pred="prediction", by="region", metrics=["rmse"])
print({r: round(v, 3) for r, v in zip(*report.sort("region").to_pydict().values())})
# {'eu': 0.106, 'us': 0.047}
```

Search vectors with an ordinary column of embeddings:

```python
docs = bt.from_pydict({"id": [1, 2, 3], "vec": [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]]})
print(docs.ml.nearest_neighbors([1.0, 0.0], column="vec", k=2).select("id").to_pydict())
# {'id': [1, 3]}
```

## Why it is fast

Media decoding runs in Rust. Image decode, resize, perceptual hashes and curation measures are expressions on the {py:class}`.image <batcher.plan.expr_ir.image._ImageNamespace>`, {py:class}`.audio <batcher.plan.expr_ir.audio._AudioNamespace>` and {py:class}`.video <batcher.plan.expr_ir.video._VideoNamespace>` namespaces, parallel across cores, and so are audio mel spectrograms and MFCCs. Those two match `torchaudio` to 1e-6.

Models load once per session. Inference actor pools stay warm across `collect()` calls, so re-running a scoring cell doesn't reload the checkpoint. When a chain crosses from CPU work to GPU work, it splits into per-stage actor pools, and the model scores partition *k* while the stage below prepares *k+1*. The GPU rarely waits.

Training ingest is deterministic too. {py:meth}`ds.ml.stream_loader <batcher.api.dataset.ml.DatasetML.stream_loader>` gives every rank the same number of batches in a seed-reproducible order, and a job restarts mid-epoch with no repeated or skipped samples.

On an 8xT4 Ray cluster, sentence-transformers MiniLM embeds 33,611 texts per second. A two-stage JPEG-to-ResNet-50 pipeline holds 81% GPU utilization.

::::{dropdown} Headline measurements
The following table lists single-node media results on 96 cores with a release build, 2,000 JPEG frames or audio clips per run:

| Workload | Batcher | Comparison |
|---|---:|---|
| JPEG decode and resize to 224x224 | 4,649-4,788 img/s | 1.87-1.96x Daft 0.7.23, 6.35-6.61x Ray Data 2.56 |
| Entropy, pHash, and horizontal flip per image | 1,514-1,541 img/s | 5.67-5.76x a per-row Pillow loop |
| Audio decode, 2,000 clips | 20.4 ms | 19.8x a per-clip `soundfile` loop |

The following table lists GPU workload families on an 8xT4 Ray cluster with real models and 100% output agreement:

| Workload | Model | Batcher |
|---|---|---:|
| Text embeddings | sentence-transformers MiniLM | 33,611 text/s |
| Audio feature extraction | torchaudio mel and ResNet-18 | 38,546 clip/s |
| JPEG decode into a model, two stages | ResNet-50 | 2,504 img/s at 81% GPU |
| Fractional-GPU packing | EfficientNet-B0, 2 per GPU | 6,764 img/s at 89% GPU |
| LLM batch inference | HF gpt2 | 814.8 prompt/s |
| Training-data ingest | `iter_torch_batches`, zero-copy DLPack, no shuffle | 1.06 M rows/s |

{doc}`/benchmarks/results/ai-and-gpu` and {doc}`/benchmarks/results/multimodal-ingest` have the full results.
::::

## Find your workload

The following table maps common ML tasks to the entry point and the page that covers it:

| You want to | Start with | Read |
|---|---|---|
| Score a PyTorch, Hugging Face, or custom model | {py:meth}`ds.ml.infer <batcher.api.dataset.ml.DatasetML.infer>` | {doc}`/ml/inference/inference` |
| Score a fitted XGBoost, LightGBM, CatBoost, scikit-learn, or ONNX model | {py:meth}`ds.ml.predict <batcher.api.dataset.ml.DatasetML.predict>` | {doc}`/ml/inference/tabular-models` |
| Put the model on GPUs and keep them busy | `num_gpus`, `concurrency` | {doc}`/ml/inference/gpu` |
| Turn text or images into vectors | {py:meth}`ds.ml.embed <batcher.api.dataset.ml.DatasetML.embed>` | {doc}`/ml/retrieval/embeddings` |
| Search vectors or build RAG | {py:meth}`ds.ml.nearest_neighbors <batcher.api.dataset.ml.DatasetML.nearest_neighbors>` | {doc}`/ml/retrieval/vector-search` |
| Run an LLM over millions of rows | {py:meth}`ds.ml.generate <batcher.api.dataset.ml.DatasetML.generate>` | {doc}`/ml/retrieval/llm/index` |
| Fetch and decode images, audio, or video | {py:meth}`ds.ml.download <batcher.api.dataset.ml.DatasetML.download>`, `.image.to_tensor` | {doc}`/ml/preparing/multimodal/index` |
| Scale, encode, and impute features | `batcher.ml.preprocessors` | {doc}`/ml/preparing/preprocessors/index` |
| Measure a model per segment, or detect drift | {py:meth}`ds.ml.evaluate <batcher.api.dataset.ml.DatasetML.evaluate>`, {py:meth}`ds.ml.drift <batcher.api.dataset.ml.DatasetML.drift>` | {doc}`/ml/evaluation/index` |
| Feed PyTorch training ranks | {py:meth}`ds.ml.stream_loader <batcher.api.dataset.ml.DatasetML.stream_loader>` | {doc}`/ml/training/data-loaders` |

## In this section

Start at inference if you already have a model, and at preparing if the data isn't yet in shape for one.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`cpu;1.1em` Run a model
:link: /ml/inference/index
:link-type: doc
Batch inference over Arrow on CPU or GPU, including tabular models and exported runtimes.
:::

:::{grid-item-card} {octicon}`filter;1.1em` Prepare the data
:link: /ml/preparing/index
:link-type: doc
Feature preprocessors, media decode for every modality, and tokenization.
:::

:::{grid-item-card} {octicon}`graph;1.1em` Measure the model
:link: /ml/evaluation/index
:link-type: doc
Metrics per segment in one pass, model selection over leak-free splits, and drift monitoring for a deployed model.
:::

:::{grid-item-card} {octicon}`search;1.1em` Embeddings, retrieval, and LLMs
:link: /ml/retrieval/index
:link-type: doc
Encode and search vectors, then generate text and grade what came back.
:::

:::{grid-item-card} {octicon}`workflow;1.1em` Serve and train
:link: /ml/training/index
:link-type: doc
Call served models, and feed training ranks a balanced, resumable stream.
:::
::::

## Requirements and limitations

Three practical constraints apply:

- The engine installs with `pip install batcher-engine`. Model frameworks are extras: `torch`, `transformers`, `st` for sentence-transformers, `vllm`, `tabular` for the gradient-boosting and scikit-learn stack, and `multimodal` for the image, audio, video and PDF readers.
- GPU reservation with `num_gpus` and multi-worker actor pools run on a Ray cluster. Without one, the same pipeline runs on a single machine.
- Models are fitted and engines are registered from Python. SQL calls them as table functions: `ML_PREDICT` scores a registered model, `AI_GENERATE`, `AI_CLASSIFY` and `AI_EXTRACT` call a registered language-model engine, and `AI_EMBED` runs a sentence-transformers encoder. {doc}`/user-guide/analyze/sql-model-functions` lists the syntax and the calls that stay DataFrame-only.

## See also

Read next:

- {doc}`/getting-started/tutorials/ml/index`: four end-to-end tutorials that run on a laptop.
- {doc}`/cookbook/ml/index`: shorter recipes for the workloads above.
- {doc}`/user-guide/transform/columns/udfs`: batch UDFs generally, of which model inference is one case.
- {doc}`/api/models/ml`: the reference for the `.ml` accessor and the `batcher.ml` package.
- {doc}`/architecture/deep-dives/distribution/gpu-execution`: how device work is scheduled.
- {doc}`/benchmarks/results/ai-and-gpu`: the full GPU measurements and how they were taken.
- {doc}`/examples/machine-learning`: 43 model and metric scripts, each run on every commit.
- {doc}`/getting-started/concepts/glossary`: morsel, breaker, mergeable, spill, and the rest, defined in one line each.

```{toctree}
:hidden:

inference/index
preparing/index
evaluation/index
retrieval/index
training/index
```
