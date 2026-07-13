# ML recipes

Turning a table, or a folder of media, into something a model can consume. Then running the
model over it without wasting the GPU you are paying for.

The data half of an ML pipeline is where the time actually goes, and it is where most of the
throughput gets lost. A model that reloads on every batch. A GPU that sits idle through a CPU
decode. One corrupt JPEG that kills a six-hour job at hour five. These recipes are mostly
about not doing that.

:::{tip}
The single idiom that carries most of these pages: pass a **class** to `map_batches`, `infer`,
`embed`, or `generate`, never an instance and never a plain function. The engine constructs it
once per worker, so the weights load in the constructor and stay loaded. A function is rebuilt
on every batch, and on a GPU stage that is usually the whole performance story.
:::

## Inference

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`zap;1.1em` Batch scoring an LLM
:link: llm-batch-scoring
:link-type: doc
Structured output, and why the engine you pick barely matters.
:::

:::{grid-item-card} {octicon}`image;1.1em` Image classification
:link: image-classification
:link-type: doc
Decode on the CPU, forward pass on the GPU, and never in lockstep.
:::

:::{grid-item-card} {octicon}`pencil;1.1em` Image captioning
:link: image-captioning
:link-type: doc
A vision-language model over a URL column, with the dead links survived.
:::

:::{grid-item-card} {octicon}`broadcast;1.1em` Audio transcription
:link: audio-transcription
:link-type: doc
Decode and resample in the engine, not in a `librosa` loop.
:::
::::

## Embeddings and retrieval

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`search;1.1em` Text embeddings
:link: text-embeddings
:link-type: doc
Clean the text, embed once, normalize at write time.
:::

:::{grid-item-card} {octicon}`stack;1.1em` Building a RAG index
:link: rag-index
:link-type: doc
Chunk, embed, index, retrieve.
:::
::::

## Training data

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`beaker;1.1em` Feature pipeline
:link: feature-pipeline
:link-type: doc
Fit on train, transform everywhere.
:::

:::{grid-item-card} {octicon}`git-branch;1.1em` Train/test split
:link: train-test-split
:link-type: doc
The leak you get for free from a naive random split.
:::

:::{grid-item-card} {octicon}`filter;1.1em` Deduplicating training data
:link: training-data-dedup
:link-type: doc
Including the test-set contamination nobody checks for.
:::

:::{grid-item-card} {octicon}`graph;1.1em` Recommender features
:link: recommender-features
:link-type: doc
Aggregates and windows over an event log, without the future in them.
:::
::::

## What the engine buys you here

Measured on 8×T4 against Ray Data, with a prediction-agreement gate on every run. These are
the workloads the recipes above are built on; the [full table and the honest exceptions live
in the benchmarks](../../benchmarks/ai-and-gpu.md).

| Workload | Model | vs Ray Data |
| --- | --- | ---: |
| Text embeddings | sentence-transformers MiniLM | **47×** |
| Audio feature extraction | torchaudio mel + ResNet-18 | **12.5×** |
| LLM batch inference | HF gpt2 | **11.1×** |
| Batch inference | ResNet-50 | **2.05×** |
| Fractional-GPU packing | EfficientNet-B0, 2 per GPU | **1.96×** |

None of that comes from per-workload tuning. It comes from the model loading once, the CPU
stage overlapping the GPU stage, and the decode running in the data plane.

:::{seealso}
- [ML guide](../../ml/index.md): the reference for every surface these recipes call.
- [Inference](../../ml/inference.md) and [GPU scheduling](../../ml/gpu.md): pools, stage
  overlap, adaptive batch sizing, fractional packing.
- [ML API reference](../../api/ml.md): the `ds.ml` namespace and the `batcher.ml` functions.
- [GPU execution](../../deep-dives/gpu-execution.md) and
  [tensor columns](../../deep-dives/tensor-columns.md): the mechanisms underneath.
- [Streaming recipes](../streaming/index.md): the same model stages, over a source that
  never ends.
:::

```{toctree}
:hidden:

text-embeddings
image-classification
llm-batch-scoring
feature-pipeline
train-test-split
rag-index
training-data-dedup
image-captioning
audio-transcription
recommender-features
```
