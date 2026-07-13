# Machine learning

Run your models where the data already is. The `ml` accessor hands your Python
functions and models whole Arrow batches instead of one row at a time, and it places
that work on GPUs and across worker actors for you.

::::{grid} 1 2 3 4
:gutter: 3

:::{grid-item-card} {octicon}`cpu;1.1em` Inference
:link: inference
:link-type: doc
Run a model over Arrow batches.
:::

:::{grid-item-card} {octicon}`filter;1.1em` Preprocessors
:link: preprocessors
:link-type: doc
Feature transforms over batches.
:::

:::{grid-item-card} {octicon}`image;1.1em` Multimodal
:link: multimodal
:link-type: doc
Decode images and audio into tensors; sample video frames.
:::

:::{grid-item-card} {octicon}`server;1.1em` Serving
:link: serving
:link-type: doc
Stand models up behind the engine.
:::

:::{grid-item-card} {octicon}`comment-discussion;1.1em` LLM generation
:link: llm
:link-type: doc
Batched text generation.
:::

:::{grid-item-card} {octicon}`package;1.1em` PyTorch
:link: pytorch
:link-type: doc
Hand batches straight to Torch.
:::

:::{grid-item-card} {octicon}`broadcast;1.1em` Streaming
:link: streaming
:link-type: doc
Inference over live streams.
:::

:::{grid-item-card} {octicon}`zap;1.1em` GPU execution
:link: gpu
:link-type: doc
Place work on GPUs and actors.
:::

:::{grid-item-card} {octicon}`git-branch;1.1em` Embeddings
:link: embeddings
:link-type: doc
Encode a column into vectors, at scale.
:::

:::{grid-item-card} {octicon}`search;1.1em` Vector search
:link: vector-search
:link-type: doc
Brute force in-engine, or an ANN index.
:::

:::{grid-item-card} {octicon}`book;1.1em` RAG
:link: rag
:link-type: doc
Chunk, embed, retrieve, generate.
:::

:::{grid-item-card} {octicon}`typography;1.1em` Tokenization
:link: tokenization
:link-type: doc
Tokenize as a stage; pack sequences.
:::

:::{grid-item-card} {octicon}`workflow;1.1em` Distributed training
:link: distributed-training
:link-type: doc
Balanced, resumable, elastic sharding.
:::

:::{grid-item-card} {octicon}`stack;1.1em` Data loaders
:link: data-loaders
:link-type: doc
Which loader, and what it guarantees.
:::

:::{grid-item-card} {octicon}`meter;1.1em` Batch scoring
:link: batch-scoring
:link-type: doc
The offline scoring job, end to end.
:::

:::{grid-item-card} {octicon}`plug;1.1em` Serving patterns
:link: model-serving-patterns
:link-type: doc
In-process, or call a served model.
:::
::::

```{toctree}
:hidden:

inference
preprocessors
multimodal
serving
llm
pytorch
streaming
gpu
embeddings
vector-search
rag
tokenization
distributed-training
data-loaders
batch-scoring
model-serving-patterns
```
