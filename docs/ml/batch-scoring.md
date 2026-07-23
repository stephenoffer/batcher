# Batch scoring

An offline scoring job is a scan with a model in the middle. The model is the expensive
part, so everything else in the pipeline exists to keep it busy. Filter before the model
so you do not score rows you throw away, load the weights once per worker rather than
once per batch, and size the batch to the device rather than to the file.

## The shape of the job

The job below reads reviews, cuts them down before the GPU sees them, scores what is left
on an actor pool, and writes the result partitioned by label.

```python
# docs: skip
import batcher as bt
import pyarrow as pa


class Classifier:
    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased"
        ).cuda().eval()
        self._torch = torch

    def __call__(self, batch):
        enc = self.tok(
            batch.column("text").to_pylist(),
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to("cuda")
        with self._torch.no_grad():
            preds = self.model(**enc).logits.argmax(dim=1).cpu().tolist()
        return batch.append_column("label", pa.array(preds))


scored = (
    bt.read.parquet("s3://bucket/reviews/*.parquet")
    .filter(bt.col("lang") == "en")                    # cut rows before the GPU
    .select("id", "text")                              # cut columns before the GPU
    .ml.infer(
        Classifier,                                     # the CLASS: loads once per worker
        output_columns=["id", "text", "label"],
        batch_size=256,
        num_gpus=1,
        concurrency=4,
        model_memory_gb=0.3,
    )
)
scored.write.parquet("s3://bucket/scored/", partition_by=["label"])
```

Everything above the `infer` is an ordinary lazy pipeline. The filter and the projection
get pushed into the scan, so the Parquet reader skips row groups and never decodes the
columns the model does not read. That is not a micro-optimization. On a wide table it is
most of the I/O.

## Pass the class, not an instance, not a function

:::{tip}
This is the one mistake that costs an order of magnitude. A plain function is rebuilt per
batch, which reloads the model per batch. A class is constructed **once per worker** and
then called per batch. The engine raises a `PerformanceWarning` if a GPU stage gets a
bare function, because it is the most common inference mistake there is.
:::

The mechanics are visible without a GPU. The "model" here is arithmetic, but the contract
is exactly the real one: the constructor runs once, and `__call__` runs per batch.

```python
import pyarrow as pa
import pyarrow.compute as pc

import batcher as bt


class ToyScorer:
    def __init__(self, threshold):
        self.threshold = threshold      # in a real model: load the weights here
        self.calls = 0

    def __call__(self, batch):
        self.calls += 1
        score = pc.divide(pc.cast(batch.column("clicks"), "float64"), 100.0)
        label = pc.greater(score, self.threshold)
        return batch.append_column("score", score).append_column("label", label)


ds = bt.from_pydict({"id": [1, 2, 3, 4], "clicks": [10, 90, 55, 30]})
scored = ds.ml.infer(ToyScorer(0.5), output_columns=["id", "clicks", "score", "label"])
print(scored.to_pydict())
# {'id': [1, 2, 3, 4], 'clicks': [10, 90, 55, 30], 'score': [0.1, 0.9, 0.55, 0.3],
#  'label': [False, True, True, False]}
```

`infer` is `map_batches` with inference defaults, so use whichever name reads better. Both
take `batch_size`, `num_gpus`, `concurrency`, and `output_columns`.

## Sizing the pool

`num_gpus` is what each actor holds, and `concurrency` is how many actors run.

| The situation | The knobs | What happens |
| --- | --- | --- |
| A model that fills a device | `num_gpus=1, concurrency=4` | four actors, each holding a whole GPU |
| A model small enough to share one | `num_gpus=0.25, concurrency=8` | four actors per GPU, which usually beats one per GPU because a single actor rarely saturates the device |
| You would rather not work it out | `model_memory_gb=...` | the engine budgets host RAM per worker and packs small models onto shared GPUs |
| A backlog that comes and goes | `concurrency=(2, 8)` | the pool autoscales to it |

`model_memory_gb` is the better lever most of the time. State the model's footprint and
eight workers will not each load a 20 GB model into a 64 GB box. See
[GPU scheduling](gpu.md).

`batch_size` should be the model's batch size, not the file's. Too small a `batch_size`
leaves the GPU launching kernels on tiny inputs. Too large and the activations do not
fit. It is a property of the model and the device, so pin it explicitly rather than
inheriting the morsel size.

## Dirty data should not kill a six-hour job

One corrupt image in ten million rows should cost you one row, not the run.
`max_errored_rows` bisects a batch whose `fn` raises, isolates the offending rows, and
drops them, up to the budget. Past the budget the error propagates, so a genuine bug on
clean data still fails fast.

```python
def parse_score(batch):
    return pa.RecordBatch.from_pydict(
        {"score": [float(v) for v in batch.column("raw").to_pylist()]}
    )


dirty = bt.from_pydict({"raw": ["0.1", "0.9", "corrupt", "0.3"]})
print(dirty.map_batches(parse_score, output_columns=["score"], max_errored_rows=10).to_pydict())
# {'score': [0.1, 0.9, 0.3]}
```

:::{important}
Set it deliberately and keep it small. A budget of a million silently deleted rows is not
resilience. It is a data-loss bug with a config flag.
:::

## Idempotency, because workers get preempted

:::{warning}
Under `distributed=True`, a worker whose node is reclaimed mid-batch is reassigned and
its partition **recomputed** from the durable input. So the scoring function must be
idempotent. A pure transform is. A function that POSTs a prediction to an API, upserts
into a vector store, or increments an external counter is not. On a retry it applies the
effect twice.
:::

The fix is to keep side effects out of the model stage. Return the prediction as a
column, and let a `write` land it. If you must call an external sink from inside the UDF,
make it idempotent by upserting on a key. The retry is not optional. It is how a
spot-instance job survives at all.

## Checkpoint by partition

A 10-hour scoring job that dies at hour 9 should not restart at hour 0. Partition the
input and write each partition's output as it completes, so a rerun skips the partitions
that already landed.

```python
# docs: skip
import batcher as bt

for day in days:
    out = f"s3://bucket/scored/dt={day}/"
    if already_written(out):        # your check: a manifest, a marker, a listing
        continue
    (
        bt.read.parquet(f"s3://bucket/events/dt={day}/")
        .ml.infer(Classifier, output_columns=[...], num_gpus=1, concurrency=4)
        .write.parquet(out)
    )
```

It is crude, and it works. The alternative, one enormous job with an internal checkpoint,
is a lot of machinery to rebuild what a partitioned write gives you for free.

## Verify before you scale

:::{tip}
Run the pipeline over `head(1000)` first and look at the output. A model that returns the
wrong label for every row costs you the same GPU-hours as one that works, and the
distribution of the predictions is the cheapest possible check.
:::

```python
print(scored.group_by("label").agg(n=bt.count()).sort("label").to_pydict())
# {'label': [False, True], 'n': [2, 2]}
```

If every row came back with the same class, the pipeline is wrong somewhere. Look for a
column mix-up, a truncation, or a preprocessing step that did not run. Find that on a
thousand rows, not on a billion.

## See also

- [Inference](inference.md): the model-as-callable contract and the batch formats.
- [GPU scheduling](gpu.md): `num_gpus`, `concurrency`, and actor autoscaling.
- [Model serving patterns](model-serving-patterns.md): calling a model that lives in
  another process.
- [Multimodal](multimodal.md): scoring images, audio, and video.
- [GPU execution](../deep-dives/gpu-execution.md): what an actor pool actually is, and
  what keeps the device fed.
- [Batch inference tutorial](../tutorials/batch-inference.md): this job, built up from
  nothing.
- [Image classification](../examples/ml/image-classification.md) and
  [LLM batch scoring](../examples/ml/llm-batch-scoring.md): the same shape, two models.
- [AI and GPU benchmarks](../benchmarks/ai-and-gpu.md): the throughput this path reaches.
- [UDFs](../user-guide/udfs.md): `max_errored_rows`, `output_columns`, and the rest of
  the batch-function contract.
