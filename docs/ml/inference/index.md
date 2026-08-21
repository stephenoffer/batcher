# Run a model over data

Start here. {py:meth}`ds.ml.predict <batcher.api.dataset.ml.DatasetML.predict>` and its siblings take a callable or a model object and run it
over batches, reusing one loaded model across the whole scan rather than reloading it per
call. Everything else in this section is about where that work runs and how it is fed.

- {doc}`/ml/inference/inference`: the core loop, batch-first UDFs, and model reuse.
- {doc}`/ml/inference/tabular-models`: scoring a fitted XGBoost, LightGBM, CatBoost, scikit-learn, or ONNX model.
- {doc}`/ml/inference/runtimes`: running an exported model with ONNX Runtime, TensorRT, TorchScript, or OpenVINO.
- {doc}`/ml/inference/calibration`: turning a classifier's scores into probabilities.
- {doc}`/ml/inference/gpu`: placing work on devices, sizing batches, and keeping the GPU busy.
- {doc}`/ml/inference/batch-scoring`: the offline scoring job end to end.
- {doc}`/ml/inference/pytorch`: handing batches straight to Torch with zero copies.
- {doc}`/ml/inference/streaming`: the same operators against an unbounded source.

```{toctree}
:hidden:

batch-scoring
calibration
gpu
inference
pytorch
runtimes
streaming
tabular-models
```
