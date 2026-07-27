# Machine learning

Run your models where the data already is.

Most ML pipelines pay a tax at the seam: a query engine produces rows, something
converts them, and a separate system runs the model. Batcher removes the seam. The `ml`
accessor hands your Python functions and models whole Arrow batches rather than one row
at a time, and it places that work on GPUs and across worker actors for you. The same
`Dataset` you filtered and joined is the one the model reads.

That has a practical consequence worth stating up front. Because inference is an
operator rather than a separate job, it streams. A batch-scoring run over more data than
memory is the ordinary case, not something you build around.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`cpu;1.1em` Run a model
:link: inference
:link-type: doc
Batched inference over Arrow, on CPU or GPU.
:::

:::{grid-item-card} {octicon}`filter;1.1em` Prepare the data
:link: preprocessors/index
:link-type: doc
Feature transforms, media decode, tokenization.
:::

:::{grid-item-card} {octicon}`graph;1.1em` Measure the model
:link: evaluation
:link-type: doc
Metrics, per-segment scoring, drift.
:::

:::{grid-item-card} {octicon}`search;1.1em` Embeddings and retrieval
:link: embeddings
:link-type: doc
Encode, index, search, and generate.
:::

:::{grid-item-card} {octicon}`workflow;1.1em` Feed a training loop
:link: distributed-training
:link-type: doc
Sharding that is balanced, resumable, and elastic.
:::
::::

## Run a model over data

Start here. `ds.ml.predict` and its siblings take a callable or a model object and run it
over batches, reusing one loaded model across the whole scan rather than reloading it per
call. Everything else in this section is about where that work runs and how it is fed.

- {doc}`inference`: the core loop, batch-first UDFs, and model reuse.
- {doc}`tabular-models`: scoring a fitted XGBoost, LightGBM, CatBoost, scikit-learn, or ONNX model.
- {doc}`gpu`: placing work on devices, sizing batches, and keeping the GPU busy.
- {doc}`batch-scoring`: the offline scoring job end to end.
- {doc}`pytorch`: handing batches straight to Torch with zero copies.
- {doc}`streaming`: the same operators against an unbounded source.

## Prepare the data

Models rarely read raw columns. These pages cover the transforms that sit between a
source and a model, all of which run as ordinary operators, so they stream and they
distribute like everything else.

- {doc}`preprocessors/index`: scalers, encoders, imputers, binning, and composition.
- {doc}`multimodal`: decoding images, audio, and video into tensor columns.
- {doc}`tokenization`: tokenizing as a pipeline stage, and packing sequences.

## Measure what the model does

A model is only as trustworthy as the numbers around it, and those numbers are queries.
Every metric here is an expression the engine evaluates, so a report over a billion scored
rows is one pass and the same report *per segment* costs the same.

- {doc}`evaluation`: metrics, per-segment scoring, and the diagnostic tables.
- {doc}`statistics-and-drift`: statistical expressions, input drift, and honest splits.

## Embeddings, retrieval, and generation

Vectors are first-class columns rather than a bolted-on index, which is what lets one
pipeline chunk a corpus, embed it, retrieve against it, and call a model on the result
without leaving the engine.

- {doc}`embeddings`: encoding a text or image column into vectors at scale.
- {doc}`vector-search`: brute-force search in-engine, or an approximate index.
- {doc}`rag`: chunk, embed, retrieve, generate, as one pipeline.
- {doc}`llm`: batched text generation, engines, prompts, and throughput.
- {doc}`llm-outputs`: parsing generated strings into typed columns, and guided decoding.
- {doc}`llm-evaluation`: scoring generations, and the reference-free output monitors.

## Serve and train

The two ends of the lifecycle. Serving covers reaching a model that lives elsewhere;
training covers feeding a loop that lives elsewhere. Both keep the data plane in Batcher.

- {doc}`serving`: standing models up behind the engine.
- {doc}`model-serving-patterns`: running in-process against calling a served model.
- {doc}`distributed-training`: balanced, resumable, elastic sharding across ranks.
- {doc}`data-loaders`: which loader to use, and what each one guarantees.

## See also

:::{seealso}
- {doc}`../user-guide/index`: the relational half of the pipeline that feeds all of this.
- {doc}`../user-guide/udfs`: batch UDFs generally, of which model inference is one case.
- {doc}`../api/ml`: the reference for the `.ml` accessor and the `batcher.ml` package.
- {doc}`../examples/ml/index`: runnable recipes for the workloads above.
- {doc}`../deep-dives/gpu-execution`: how device work is actually scheduled.
- {doc}`../benchmarks/index`: the measured throughput behind the claims here.
:::

```{toctree}
:hidden:
:caption: Run a model

inference
tabular-models
gpu
batch-scoring
pytorch
streaming
```

```{toctree}
:hidden:
:caption: Measure what the model does

evaluation
statistics-and-drift
```

```{toctree}
:hidden:
:caption: Prepare the data

preprocessors/index
multimodal
tokenization
```

```{toctree}
:hidden:
:caption: Embeddings and retrieval

embeddings
vector-search
rag
llm
llm-outputs
llm-evaluation
```

```{toctree}
:hidden:
:caption: Serve and train

serving
model-serving-patterns
distributed-training
data-loaders
```
