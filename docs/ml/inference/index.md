# Run a model over data

This section covers batch inference: applying a model to every row of a dataset, on CPUs or GPUs, from a laptop to a Ray cluster.

In Batcher a model is an operator in the plan. It sits between a scan and a sink like a filter would, receives whole Arrow batches, and runs as many copies as the engine decides. The filter in front of it is pushed down to the scan. The result streams. The model loads once per worker.

## Pick an entry point

Four calls cover almost every inference job. Each returns a new lazy {py:class}`Dataset <batcher.Dataset>`. Nothing runs until you collect or write.

{py:meth}`ds.ml.infer <batcher.api.dataset.ml.DatasetML.infer>` is the general one. Pass a class whose constructor loads the model and whose `__call__` scores a batch:

```python
import batcher as bt
import pyarrow as pa
import pyarrow.compute as pc


class Upper:
    def __init__(self):
        pass  # load weights here, once per worker

    def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        return batch.append_column("loud", pc.utf8_upper(batch.column("text")))


reviews = bt.from_pydict({"text": ["hi", "yo"]})
print(reviews.ml.infer(Upper, output_columns=["text", "loud"]).to_pydict())
# {'text': ['hi', 'yo'], 'loud': ['HI', 'YO']}
```

`infer` also takes a Hugging Face `transformers` model id with the `column` to score, and {py:meth}`ds.ml.embed <batcher.api.dataset.ml.DatasetML.embed>` is the same call shaped for sentence-transformers encoders.

{py:meth}`ds.ml.predict <batcher.api.dataset.ml.DatasetML.predict>` scores a fitted tabular model from XGBoost, LightGBM, CatBoost, scikit-learn, or ONNX, and checks the feature order against the names the model recorded. Batcher's own estimators skip that step and predict directly:

```python
from batcher.ml.linear import LogisticRegression

train = bt.from_pydict({"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "y": [0, 0, 1, 0, 1, 1]})
model = LogisticRegression(["x"], "y").fit(train)
print(model.predict(train).to_pydict()["prediction"])
# [0, 0, 0, 1, 1, 1]
```

{py:meth}`ds.map_batches <batcher.Dataset.map_batches>` is the primitive underneath. Reach for it when the step isn't a model, or when you want `batch_format="numpy"`, `"pandas"` or `"torch"`:

```python
def double(df):
    df["x2"] = df["x"] * 2
    return df


ds = bt.from_pydict({"id": [1, 2], "x": [0.5, 1.0]})
print(ds.map_batches(double, batch_format="pandas").to_pydict())
# {'id': [1, 2], 'x': [0.5, 1.0], 'x2': [1.0, 2.0]}
```

For a model that left its training framework, {py:func}`bt.ml.onnx_predictor <batcher.ml.onnx_predictor>`, {py:func}`bt.ml.torch_predictor <batcher.ml.torch_predictor>` and {py:func}`bt.ml.openvino_predictor <batcher.ml.openvino_predictor>` build the load-once class for you.

## The shape of a scoring job

A production scoring job reads, cuts the input down before the model sees it, scores on a GPU pool, and writes. The block below needs a GPU and real weights, so it is shown but not executed:

```python
# docs: skip
import batcher as bt
import pyarrow as pa


class Classifier:
    def __init__(self):
        import torch

        self.model = torch.jit.load("model.pt").cuda().eval()

    def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        import torch

        x = torch.tensor(batch.column("features").to_pylist()).cuda()
        with torch.no_grad():
            preds = self.model(x).argmax(dim=1).cpu().tolist()
        return batch.append_column("prediction", pa.array(preds))


(
    bt.read.parquet("s3://<your-bucket>/features/")
    .filter(bt.col("country") == "US")
    .ml.infer(
        Classifier,
        batch_size=512,
        num_gpus=1,
        concurrency=(1, 8),
        max_errored_rows=100,
        output_columns=["id", "features", "country", "prediction"],
    )
    .write.parquet("s3://<your-bucket>/scored/")
)
```

Replace `<your-bucket>` with a bucket you can read and write. Four arguments do most of the work:

- `batch_size` is the model's batch size, not the file's.
- `num_gpus` reserves devices per actor. A fraction such as `0.5` packs two actors onto one GPU.
- `concurrency=(1, 8)` scales the pool between one and eight actors with the backlog.
- `max_errored_rows` bisects a batch that raises and drops the bad rows, so one corrupt input costs a row rather than the run.

## Why it stays fast

A pipeline that decodes on CPUs and scores on GPUs splits into separate actor pools that overlap, so the next batch is prepared while the current one is on the device. Pools also stay warm across `collect()` calls. That matters when a checkpoint loads slower than the scoring runs. On an 8xT4 Ray cluster, EfficientNet-B0 packed two to a GPU sustained 6,764 img/s at 89% utilization. {doc}`/benchmarks/results/ai-and-gpu` has the full results.

## In this section

The following table lists the pages in this section, in the order a first scoring job usually needs them:

| Page | Covers |
|---|---|
| {doc}`/ml/inference/inference` | The core call, model-id shortcuts, batch formats, and driving the actor pool yourself. |
| {doc}`/ml/inference/batch-scoring` | The offline scoring job end to end: pool sizing, dirty data, retries, idempotent writes, and checkpoints. |
| {doc}`/ml/inference/gpu` | Placing actors on devices, autoscaling, fractional GPUs, memory-based packing, and non-GPU accelerators. |
| {doc}`/ml/inference/tabular-models` | Scoring gradient-boosted and scikit-learn models, SHAP contributions, and Batcher's own linear models. |
| {doc}`/ml/inference/runtimes` | Running an exported model with ONNX Runtime, TensorRT, TorchScript, or OpenVINO. |
| {doc}`/ml/inference/calibration` | Turning a classifier's scores into probabilities you can make decisions against. |
| {doc}`/ml/inference/pytorch` | Handing batches to PyTorch as tensors, with zero-copy views and DataLoader integration. |
| {doc}`/ml/inference/streaming` | Which plans stream batch by batch into a training loop in bounded memory. The ordering and resume contract lives in {doc}`/ml/training/distributed-training`. |

## See also

These pages pick up where this one stops:

- {doc}`/getting-started/tutorials/ml/batch-inference`: build a scoring pipeline step by step on a laptop.
- {doc}`/ml/retrieval/llm/index`: running language models, which have their own engines and batching.
- {doc}`/ml/training/serving`: calling a model that is served elsewhere instead of loading it in the worker.
- {doc}`/ml/evaluation/evaluation`: scoring the predictions you just produced.
- {doc}`/user-guide/transform/columns/udfs`: batch UDFs in general.

```{toctree}
:hidden:

inference
tabular-models
runtimes
calibration
gpu
batch-scoring
pytorch
streaming
```
