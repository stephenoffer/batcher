# Machine learning

Run your models where your data already lives. The `ml` accessor hands your Python
functions and models whole Arrow batches instead of one row at a time, and the
scheduler places that work on GPUs and across worker actors for you. From there the
pages cover the common jobs: batch inference and embeddings, feature preprocessing,
decoding images and audio, serving models, LLM generation, and loading training data.

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
Decode images, audio, and video.
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
::::

```{toctree}
:maxdepth: 1

inference
preprocessors
multimodal
serving
llm
pytorch
streaming
gpu
```
