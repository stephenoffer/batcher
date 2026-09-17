# Batch scoring

This page walks through an offline scoring job: read a table, run a model over it, and write the predictions, with the settings that decide whether it finishes in an hour or a day. The job is a scan with a model in the middle. The model is the expensive part, so the rest of the pipeline exists to keep it busy: filter before the model, load the weights once per worker, and size the batch to the device rather than to the file.

## The shape of the job

The job below reads reviews, cuts them down before the GPU sees them, scores the rest on an actor pool, and writes the result partitioned by label.

```python
# docs: skip
import batcher as bt
import pyarrow as pa


class Classifier:
    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = (
            AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
            .cuda()
            .eval()
        )
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
    .filter(bt.col("lang") == "en")  # cut rows before the GPU
    .select("id", "text")  # cut columns before the GPU
    .ml.infer(
        Classifier,  # the CLASS: loads once per worker
        output_columns=["id", "text", "label"],
        batch_size=256,
        num_gpus=1,
        concurrency=4,
        model_memory_gb=0.3,
    )
)
scored.write.parquet("s3://bucket/scored/", partition_by=["label"])
```

Everything above the `infer` is an ordinary lazy pipeline. The filter and the projection are pushed into the scan, so the Parquet reader skips row groups and never decodes the columns the model doesn't read. On a wide table that is most of the I/O.

## Pass a class, not a function

:::{tip}
A plain function is rebuilt per batch, so a model loaded inside it reloads per batch. A class is constructed once per worker and then called per batch. Batcher emits a `PerformanceWarning` when a stage with `num_gpus > 0` gets a plain function, because it is the most common inference mistake.
:::

The contract is visible without a GPU. The "model" below is arithmetic, but the constructor runs once and `__call__` runs per batch, as with a real one. The example passes an instance because its constructor takes an argument and loads nothing. An instance is built on the driver and shipped to the workers, so when the constructor loads weights, pass the class.

```python
import pyarrow as pa
import pyarrow.compute as pc

import batcher as bt


class ToyScorer:
    def __init__(self, threshold):
        self.threshold = threshold  # in a real model: load the weights here
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

`infer` is `map_batches` with inference defaults. Both take `batch_size`, `num_gpus`, `concurrency`, and `output_columns`, so use whichever name reads better.

## Size the pool

Each actor holds `num_gpus` of a device, and `concurrency` sets how many actors run. Behind that sit two nested pools that are easy to conflate: Ray actors across GPUs, and threads inside each actor sharing one model.

![Two nested pools sit behind one ds.ml.infer or ds.map_batches call, and they are not the same thing. The outer pool is one Ray actor per GPU and exists only on the distributed path: each partition goes to the emptiest actor, each actor builds your class once in __init__, num_gpus is a Ray reservation, and concurrency=(min, max) grows the pool while work waits and reaps an idle actor. Inside a single actor, an InferencePool of threads shares one model object and one CUDA context, so the threads buy overlap with host work rather than extra model replicas. They call your __call__ with a whole Arrow RecordBatch, in input order, and an autobatch controller hill-climbs the batch size under a VRAM cap from the measured rows per second; an out-of-memory error bisects the batch and records a ceiling for the run, and the size that worked is written back for the next one. batch_format reframes only the call, never the data plane, which stays Arrow, and ds.map is the row-at-a-time escape hatch, marked as one so a profile can price what it costs.](/_static/diagrams/inference_actor_pool.svg)

The table maps common situations to the knobs that fit them:

| The situation | The knobs | What happens |
| --- | --- | --- |
| A model that fills a device | `num_gpus=1, concurrency=4` | four actors, each holding a whole GPU |
| A model small enough to share one | `num_gpus=0.25, concurrency=8` | four actors per GPU, which usually beats one per GPU because a single actor rarely saturates the device |
| You would rather not work it out | `model_memory_gb=...` | the engine budgets host RAM per worker and packs small models onto shared GPUs |
| A backlog that comes and goes | `concurrency=(2, 8)` | the pool autoscales to it |

`model_memory_gb` is usually the better lever. State the model's footprint and eight workers won't each load a 20 GB model into a 64 GB box. See {doc}`GPU scheduling </ml/inference/gpu>`.

`batch_size` is the model's batch size, not the file's. Too small and the GPU launches kernels on tiny inputs. Too large and the activations don't fit. It belongs to the model and the device, so pin it rather than inheriting the morsel size.

## Survive dirty data

One corrupt image in ten million rows should cost one row, not the run. `max_errored_rows` bisects a batch whose `fn` raises, isolates the offending rows, and drops them, up to the budget. Past the budget the error propagates, so a genuine bug on clean data still fails fast.

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
Set it deliberately and keep it small. A budget of a million silently deleted rows is a data-loss bug with a config flag.
:::

## Retry the endpoint, not the job

A hosted model answers with a 429 or a 503 routinely, and one throttled request shouldn't rerun a six-hour job. `max_retries` retries a batch whose model call raised, with jittered exponential backoff starting from `retry_backoff` seconds. `timeout` bounds one call so a hung request can't stall the run, and `retry_on` narrows which exceptions count so a genuine bug still fails fast.

`generate`, `extract`, `classify`, `infer`, `predict`, and `embed` take these options exactly as `map_batches` does:

```python
import batcher as bt

attempts = {"n": 0}


def flaky_engine():
    def engine(prompts):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("503 from the provider")
        return [p.upper() for p in prompts]

    return engine


reviews = bt.from_pydict({"text": ["good", "bad"]})
print(
    reviews.ml.generate(flaky_engine, prompt_column="text", max_retries=3, timeout=30.0).to_pydict()
)
# {'text': ['good', 'bad'], 'response': ['GOOD', 'BAD']}
```

Retries run first, and only a failure that survives every attempt is charged against `max_errored_rows`. A transient outage costs latency, and a row the model will never accept costs one row.

## Keep the model stage idempotent

:::{warning}
Under `distributed=True`, a worker whose node is reclaimed mid-batch is reassigned and its partition recomputed from the durable input, so the scoring function must be idempotent. A pure transform is. A function that POSTs a prediction to an API, upserts into a vector store, or increments an external counter applies its effect twice on a retry.
:::

Keep side effects out of the model stage. Return the prediction as a column and let a `write` land it. If you must call an external sink from inside the UDF, upsert on a key. You can't turn the recompute off, because it is how a spot-instance job survives at all.

## Checkpoint by partition

A 10-hour scoring job that dies at hour 9 shouldn't restart at hour 0. Partition the input and write each partition's output as it completes, so a rerun skips what already landed.

```python
# docs: skip
import batcher as bt

for day in days:
    out = f"s3://bucket/scored/dt={day}/"
    if already_written(out):  # your check: a manifest, a marker, a listing
        continue
    (
        bt.read.parquet(f"s3://bucket/events/dt={day}/")
        .ml.infer(Classifier, output_columns=[...], num_gpus=1, concurrency=4)
        .write.parquet(out)
    )
```

It is crude, and it works. One enormous job with an internal checkpoint rebuilds, with a lot of machinery, what a partitioned write already gives you.

## Verify before you scale

:::{tip}
Run the pipeline over `limit(1000)` first and look at the output. A model that returns the wrong label for every row costs the same GPU-hours as one that works, and the distribution of predictions is the cheapest check there is.
:::

```python
print(scored.group_by("label").agg(n=bt.count()).sort("label").to_pydict())
# {'label': [False, True], 'n': [2, 2]}
```

If every row comes back with the same class, look for a column mix-up, a truncation, or a preprocessing step that didn't run. Find it on a thousand rows, not a billion.

## See also

- {doc}`Inference </ml/inference/inference>`: the model-as-callable contract and the batch formats.
- {doc}`GPU scheduling </ml/inference/gpu>`: `num_gpus`, `concurrency`, and actor autoscaling.
- {doc}`/ml/inference/tabular-models`: the same job for XGBoost, LightGBM, and scikit-learn models.
- {doc}`Model serving patterns </ml/training/model-serving-patterns>`: calling a model that lives in
  another process.
- {doc}`Multimodal </ml/preparing/multimodal/index>`: scoring images, audio, and video.
- {doc}`GPU execution </architecture/deep-dives/distribution/gpu-execution>`: what an actor pool actually is, and
  what keeps the device fed.
- {doc}`Batch inference tutorial </getting-started/tutorials/ml/batch-inference>`: this job, built up from
  nothing.
- {doc}`Image classification </cookbook/ml/pipelines/multimodal/image-classification>` and
  {doc}`LLM batch scoring </cookbook/ml/pipelines/text/llm-batch-scoring>`: the same shape, two models.
- {doc}`AI and GPU benchmarks </benchmarks/results/ai-and-gpu>`: the throughput this path reaches.
- {doc}`UDFs </user-guide/transform/columns/udfs>`: `output_columns` and the rest of the batch-function
  contract, with {doc}`Running a UDF at scale </user-guide/transform/columns/udfs-at-scale>` for
  `max_errored_rows`.
