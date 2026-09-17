# Model serving patterns

This page covers the ways a model's predictions reach a Batcher pipeline, and how to pick
between loading the model in the worker and calling a service that already has it loaded.

Load the model in the worker when you can. There's no network, no serialization, and no shared
queue, and it's what a batch job should do. Call a service when the model doesn't belong to you:
another team owns it, it runs on hardware you can't schedule, or the same endpoint serves an
online path that your backfill must not starve.

:::{warning}
Don't pick the service because it's architecturally tidier. A 10-million-row backfill through an
HTTP endpoint pays a network round trip and a serialization per request, on top of the forward
pass the in-process path runs anyway.
:::

## The two shapes

::::{tab-set}
:::{tab-item} In-process (the default)

Load the model in the worker. A class is constructed once per worker and called per batch, so
the weights land once and the forward pass is a local call.

```python
# docs: skip
scored = ds.ml.infer(Classifier, output_columns=[...], num_gpus=1, concurrency=4)
```

That's {doc}`batch scoring </ml/inference/batch-scoring>`, and it's the right answer for anything
you can schedule yourself.

:::

:::{tab-item} A served model

An adapter turns the endpoint into a UDF, a class you drop into `map_batches`, which connects
once per worker rather than once per row.

```python
# docs: skip
import batcher as bt
from batcher.ml import http_client

Score = http_client(
    "http://model-service/predict",
    input_columns=["features"],
    output_columns=["prediction"],
    timeout=30.0,
    retries=3,
)

scored = bt.read.parquet("s3://bucket/rows.parquet").map_batches(Score, batch_size=64)
```

:::
::::

## Calling a served model

When the model lives elsewhere, pick the adapter for the backend that holds it:

| Adapter | Backend |
| --- | --- |
| `http_client(url, input_columns=, output_columns=)` | Any JSON HTTP endpoint |
| `triton_client(...)` | NVIDIA Triton |
| `torchserve_client(...)` | TorchServe |
| `serving_udf(connect, ...)` | Your own {py:class}`ServingClient <batcher.ml.ServingClient>` |

The adapter sends a *batch* per request, not a row, and that's what makes the pattern viable.
Ten million rows at `batch_size=64` is 156,250 requests instead of ten million.

:::{important}
Most serving stacks cap the rows or bytes in one request, and a request over the cap doesn't
degrade: it fails. Set `max_batch_size` on the adapter to the window the endpoint was built for,
and each engine batch is split into requests it can hold. {doc}`Serving </ml/training/serving>`
covers how each adapter finds that number.
:::

`retries` handles the transient failure. A request that keeps failing raises, so if the endpoint
is flaky enough that you'd rather lose rows than the job, pair `retries` with `max_errored_rows`
on `map_batches`.

To write your own adapter, implement `ServingClient` and hand `serving_udf` a `connect`
callable. The connection is still made once per worker.

## Overlapping stages with run_pipeline

A single map stage doing decode-then-forward makes both halves wait on each other: the CPU
decodes batch *n+1* only after the GPU finishes batch *n*. `run_pipeline` chains
{py:class}`Stage <batcher.ml.Stage>`s with credit-based backpressure instead, so each stage runs
while the next one is still working, and the credits bound the queue between them so no stage
runs far enough ahead to blow up memory.

```python
import pyarrow as pa
import pyarrow.compute as pc

import batcher as bt
from batcher.ml import Stage, run_pipeline


class Decode:  # stands in for a CPU stage (image decode, tokenize)
    def __call__(self, batch):
        scaled = pc.multiply(pc.cast(batch.column("x"), "float64"), 2.0)
        return batch.set_column(0, "x", scaled)


class Forward:  # stands in for the GPU forward pass
    def __call__(self, batch):
        label = pc.greater(batch.column("x"), 4.0)
        return batch.append_column("label", label)


ds = bt.from_pydict({"x": [1, 2, 3, 4]})
out = list(
    run_pipeline(
        ds.iter_batches(),
        [
            Stage(Decode, credits=2, name="decode"),
            Stage(Forward, credits=2, num_gpus=0, name="gpu"),
        ],
    )
)
print(out[0].to_pydict())
# {'x': [2.0, 4.0, 6.0, 8.0], 'label': [False, False, True, True]}
```

`credits` is how many finished batches may sit queued between a stage and the next. One credit
is one batch slot, and the producer blocks at zero. The default of two overlaps the stages
without buffering a pipeline's worth of decoded images in RAM. The engine's shuffle uses the
same credit-based flow control.

`num_gpus` on a `Stage` is recorded but not yet used for placement. Single-node execution ignores
it, so don't rely on it to put a stage on a device.

## Adaptive batching with InferencePool

{py:class}`InferencePool <batcher.ml.InferencePool>` sits underneath the `infer` path. Use it
directly when you drive the stream yourself, in a serving process or a custom loop. It keeps
workers alive, so the factory runs once per worker and the model loads once. It also *rebatches*
the incoming stream to a target size rather than feeding the GPU whatever size the reader
produced.

```python
import pyarrow.compute as pc

import batcher as bt
from batcher.ml import InferencePool


class Model:
    def __call__(self, batch):
        return batch.set_column(0, "x", pc.multiply(batch.column("x"), 2))


ds = bt.from_pydict({"x": [1, 2, 3, 4, 5, 6]})
pool = InferencePool(Model, num_workers=2, target_batch_rows=2)
print([b.to_pydict() for b in pool.run(ds.iter_batches())])
# [{'x': [2, 4]}, {'x': [6, 8]}, {'x': [10, 12]}]
```

Results come back in input order, whichever worker produced them, so a downstream join on row
position stays valid. Setting `target_latency_ms` retunes the batch size online toward that
per-batch latency instead of maximizing throughput, which is the online-serving trade.
`min_batch_rows` and `max_batch_rows` bound the adaptation.

## Batch and online, one model

:::{tip}
Run the same worker class in the offline pipeline and inside the serving process. Offline it's
handed to `map_batches`, and online to `InferencePool`, fed by request handlers. With one
implementation, a preprocessing step can't drift between training-time and serving-time scoring.
That drift is the most common source of training/serving skew and the hardest to find, because
both halves look correct in isolation.
:::

```python
# docs: skip
# offline: score the backfill
scored = ds.ml.infer(Model, output_columns=[...], num_gpus=1, concurrency=4)

# online: the same class, in the serving process
pool = InferencePool(Model, num_workers=4, target_latency_ms=50, objective="latency")
for result in pool.run(request_batches()):
    respond(result)
```

To put a model behind an endpoint of your own instead, `serve_deployment` wraps a load-once
factory as a Ray Serve deployment. {doc}`Serving </ml/training/serving>` covers it.

## Choosing

The pattern follows from who owns the model and where it runs, not from how large it is. Find
your situation in the table:

| Situation | Reach for |
| --- | --- |
| A batch job, model you can schedule | {py:meth}`ds.ml.infer(ModelClass, num_gpus=...) <batcher.api.dataset.ml.DatasetML.infer>` |
| Model owned by another team or another cluster | `http_client`, `triton_client`, or `torchserve_client` |
| CPU preprocessing starving a GPU stage | `run_pipeline` with `Stage` credits |
| You're driving the stream, and want adaptive batching | `InferencePool` |
| A model you validated in batch that must also answer online requests | `serve_deployment` |
| An LLM behind an OpenAI-compatible endpoint | `http_engine`, covered in {doc}`LLM inference </ml/retrieval/llm/index>` |

## See also

- {doc}`Serving </ml/training/serving>`: the adapters, request sizing, retries, and `serve_deployment` in full.
- {doc}`Batch scoring </ml/inference/batch-scoring>`: the offline job end to end.
- {doc}`Inference </ml/inference/inference>`: the load-once-per-worker contract.
- {doc}`GPU scheduling </ml/inference/gpu>`: sizing actors and packing models onto devices.
- {doc}`Credit flow control </architecture/deep-dives/distribution/credit-flow-control>`: the mechanism behind `Stage` credits and the engine's shuffle.
- {doc}`GPU execution </architecture/deep-dives/distribution/gpu-execution>`: why a CPU stage starves a GPU stage.
- {doc}`Streaming inference </cookbook/streaming/streaming-inference>`: the online half, as a runnable recipe.
- {doc}`AI and GPU benchmarks </benchmarks/results/ai-and-gpu>`: measured in-process model throughput.
- {doc}`ML API </api/models/ml>`: the `InferencePool`, `Stage`, and `run_pipeline` reference.
