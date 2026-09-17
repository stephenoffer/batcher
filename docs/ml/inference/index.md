# Run a model over data

This section covers batch inference: applying a model to every row of a dataset, on CPUs or GPUs, from a laptop to a Ray cluster.

In Batcher a model is an operator in the plan. It sits between a scan and a sink like a filter would, it receives whole Arrow batches, and the engine decides how many copies of it run and where. You don't write a job that loads data, loops over it, and writes results. You write a pipeline, and scoring is one step of it. That changes what you get for free: the filter before the model is pushed down to the scan, the result streams instead of materializing, and the model loads once per worker and stays loaded for the rest of the session.

## Pick an entry point

Four calls on the `.ml` accessor cover almost every inference job. Each returns a new lazy {py:class}`Dataset <batcher.Dataset>`, so nothing runs until you collect or write.

{py:meth}`ds.ml.infer <batcher.api.dataset.ml.DatasetML.infer>` is the general one. Pass a class whose constructor loads the model and whose `__call__` scores a batch, or pass a Hugging Face `transformers` model id with the `column` to score. {py:meth}`ds.ml.embed <batcher.api.dataset.ml.DatasetML.embed>` is the same call shaped for encoders, and takes a sentence-transformers model id.

{py:meth}`ds.ml.predict <batcher.api.dataset.ml.DatasetML.predict>` scores a fitted tabular model: XGBoost, LightGBM, CatBoost, scikit-learn, or ONNX. It detects the framework, assembles each batch's features into one dense matrix, and checks the feature order against the names the model recorded, because a reordered feature list changes every prediction without raising anywhere.

{py:meth}`ds.ml.map_batches <batcher.api.dataset.ml.DatasetML.map_batches>` is the primitive underneath both. Reach for it when the step isn't a model at all, or when you need `batch_format="numpy"`, `"pandas"` or `"torch"`.

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

Replace `<your-bucket>` with a bucket you can read and write. Four arguments do most of the work. `batch_size` is the model's batch size, not the file's. `num_gpus` reserves devices per actor, and a fraction such as `0.5` packs two actors onto one GPU. `concurrency=(1, 8)` lets the pool scale between one and eight actors with the backlog. `max_errored_rows` bisects a batch that raises and drops the bad rows, so one corrupt input in ten million costs a row rather than the run.

## Why it stays fast

Batcher keeps the device busy with scheduling rather than per-workload tuning. A pipeline that decodes on CPUs and scores on GPUs splits into separate actor pools that overlap, so the next batch is being prepared while the current one is on the device. Actor pools stay warm across `collect()` calls in a session, which matters when a checkpoint takes longer to load than the scoring takes to run.

On an 8xT4 Ray cluster, a two-stage JPEG-decode-into-ResNet-50 pipeline sustained 2,504 img/s at 81% GPU utilization, and EfficientNet-B0 packed two to a GPU sustained 6,764 img/s at 89%. Against Ray Data on a compute-bound pipeline, warm pools finished a 125,000-row job in 1.2 s against 10.5 s. The gap narrows as jobs grow, and at 4M rows Ray Data is ahead. {doc}`/benchmarks/results/ai-and-gpu` has the full measurements.

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
