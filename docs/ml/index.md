# Machine learning

The `ml` accessor applies Python functions and models to Arrow batches, on whole
batches rather than per row, and the scheduler can place that work on GPUs and across
actors. The pages here cover inference and embeddings, feature preprocessing,
multimodal decode, model serving, LLM generation, and training-data loaders.

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
