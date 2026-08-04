# Serve and train

The two ends of the lifecycle. Serving covers reaching a model that lives elsewhere;
training covers feeding a loop that lives elsewhere. Both keep the data plane in Batcher.

- {doc}`/ml/training/serving`: standing models up behind the engine.
- {doc}`/ml/training/model-serving-patterns`: running in-process against calling a served model.
- {doc}`/ml/training/distributed-training`: sharding across ranks that stays balanced and resumes cleanly.
- {doc}`/ml/training/data-loaders`: which loader to use, and what each one guarantees.

```{toctree}
:hidden:

serving
model-serving-patterns
ensembling
distributed-training
data-loaders
training-corpus
```
