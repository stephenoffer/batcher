# Machine learning

Run your models where your data already lives. The `ml` accessor hands your Python
functions and models whole Arrow batches instead of one row at a time, and the
scheduler places that work on GPUs and across worker actors for you. From there the
pages cover the common jobs: batch inference and embeddings, feature preprocessing,
decoding images and audio, serving models, LLM generation, and loading training data.

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
