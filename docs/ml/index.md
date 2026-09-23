# Machine learning

This section covers running models where the data already lives: batch inference, embeddings and retrieval, LLM generation, media decoding, feature preparation, evaluation, and feeding a training loop.

Most ML pipelines pay a tax at the seam. A query engine produces rows, something converts them, and a separate system runs the model. Batcher removes the seam. The {py:class}`Dataset <batcher.Dataset>` you filtered and joined is the one the model reads, and the `.ml` accessor hands your model whole Arrow batches rather than one row at a time. The engine places that work on GPUs and across worker actors, and because inference is an operator rather than a separate job, it streams. Scoring more data than fits in memory is the ordinary case.

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

Everything around that call stays ordinary dataset work. Read Parquet or a directory of images, filter, join labels, score, then write the result or aggregate it, all in one lazy plan.

The following diagram shows that plan and what running it as one plan buys:

![Raw files to predictions as one lazy plan, where nothing runs until a write or a collect. Read takes Parquet or images and hands bytes to decode, which runs as an .image expression in Rust. Decode hands tensors to a filter and a join that attaches labels. Decode, filter and join run on CPU cores. The rows go to infer, ds.ml.infer with your model class, which runs in a GPU actor pool, and the rows come out with a score and go to a write or an aggregate. Running it as one plan buys three things. The model loads once per worker, in its constructor. The model scores partition k while the CPU stages prepare partition k+1. Batches stream, so scoring more data than fits in memory is the ordinary case.](/_static/diagrams/ml_one_plan.svg)

## What Batcher brings to an ML workload

Media decoding runs in the engine. Image decode, resize, crop and normalize, perceptual hashes, curation measures, and audio mel spectrograms and MFCCs are expressions on the {py:class}`.image <batcher.plan.expr_ir.image._ImageNamespace>`, {py:class}`.audio <batcher.plan.expr_ir.audio._AudioNamespace>` and {py:class}`.video <batcher.plan.expr_ir.video._VideoNamespace>` namespaces, implemented in Rust and parallel across cores. Daft decodes and resizes natively too, but its users write the hashes, curation measures and audio features as per-row Pillow or torch UDFs, and Ray Data runs all of it as UDFs. The mel spectrogram and MFCC match `torchaudio` to 1e-6.

Models load once per session. Inference actor pools stay warm across `collect()` calls, so re-running a scoring cell in a notebook doesn't reload the checkpoint.

The CPU stage and the GPU stage overlap. A `map_batches` chain that crosses from CPU work to GPU work splits into per-stage actor pools, so the model runs partition *k* while the stage below prepares *k+1*. That is the default, `distributed.stream_inference=True`. It holds a two-stage JPEG-to-ResNet-50 pipeline at 81% GPU utilization, as the measurements below show.

Training ingest is deterministic and resumable. {py:meth}`ds.ml.stream_loader <batcher.api.dataset.ml.DatasetML.stream_loader>` gives every rank the same number of batches in a seed-reproducible global order that doesn't depend on `world_size`. The shuffle is a computed permutation rather than a materialized index, so the order itself costs constant memory at any corpus size. A job restarts mid-epoch, even on a differently sized cluster, with no repeated or skipped samples.

## Measured

Every figure below passed a correctness gate before it was timed: the same predictions, or the same frame count at the same shape, as the comparison. The source is [`benchmarks/BENCHMARK_RESULTS.md`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/BENCHMARK_RESULTS.md), and {doc}`/benchmarks/results/ai-and-gpu` and {doc}`/benchmarks/results/multimodal-ingest` walk through them.

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

Against Ray Data on a compute-bound inference pipeline, 17 nodes and 8 T4s, warm pools decide the small and medium jobs. A 125,000-row job finishes in 1.2 s against Ray Data's 10.5 s, 8.6x, and a 500,000-row job in 4.0 s against 10.2 s, 2.5x. The two engines meet at about 2M rows, and at 4M Ray Data is ahead, 22.6 s against 29.8 s. Quote a factor on this shape only with its row count.

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

The guide has five groups. Start at inference if you already have a model, and at preparing if the data isn't yet in shape for one.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`cpu;1.1em` Run a model
:link: /ml/inference/index
:link-type: doc
Batch inference over Arrow on CPU or GPU, tabular models, exported runtimes, and streaming sources.
:::

:::{grid-item-card} {octicon}`filter;1.1em` Prepare the data
:link: /ml/preparing/index
:link-type: doc
Feature preprocessors, image, audio and video decode, and tokenization.
:::

:::{grid-item-card} {octicon}`graph;1.1em` Measure the model
:link: /ml/evaluation/index
:link-type: doc
Metrics per segment in one pass, model selection, honest splits, and drift.
:::

:::{grid-item-card} {octicon}`search;1.1em` Embeddings, retrieval, and LLMs
:link: /ml/retrieval/index
:link-type: doc
Encode, search, retrieve, generate, and evaluate what came back.
:::

:::{grid-item-card} {octicon}`workflow;1.1em` Serve and train
:link: /ml/training/index
:link-type: doc
Call served models, and feed training ranks a balanced, resumable stream.
:::
::::

## Requirements and limitations

- The engine installs with `pip install batcher-engine`. Model frameworks are extras: `torch`, `transformers`, `st` for sentence-transformers, `vllm`, `tabular` for the gradient-boosting and scikit-learn stack, and `multimodal` for the image, audio, video and PDF readers.
- GPU reservation with `num_gpus` and multi-worker actor pools run on a Ray cluster. Without one, the same pipeline runs on a single machine.
- Models run from Python. SQL queries can compute vector similarity but can't call a model.

## See also

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
