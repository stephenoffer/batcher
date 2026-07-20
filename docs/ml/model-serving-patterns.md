# Model serving patterns

There are two ways to get a model's predictions into a pipeline. You either load the
weights into the worker, or call a service that already has them loaded. The first is
faster, with no network, no serialization, and no shared queue, and it is what a batch job
should do. The second is what you do when the model does not belong to you. Another team
owns it, it runs on hardware you cannot schedule, or the same endpoint has to serve an
online path that must not be starved by your backfill.

:::{warning}
The mistake is picking the second because it is architecturally tidier. A 10-million-row
backfill through an HTTP endpoint is 10 million round trips, and it will be slower than
the model by an order of magnitude.
:::

## The two shapes

::::{tab-set}
:::{tab-item} In-process (the default)

Load the model in the worker. A class is constructed once per worker and called per
batch, so the weights land once and the forward pass is a local call.

```python
# docs: skip
scored = ds.ml.infer(Classifier, output_columns=[...], num_gpus=1, concurrency=4)
```

That is [batch scoring](batch-scoring.md), and it is the right answer for anything you
can schedule yourself.

:::

:::{tab-item} A served model

An adapter turns the endpoint into a UDF, a class you drop into `map_batches`, which
connects once per worker rather than once per row.

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

scored = bt.read.parquet("s3://bucket/rows.parquet").ml.map_batches(Score, batch_size=64)
```

:::
::::

## Calling a served model

When the model lives elsewhere, pick the adapter for the backend that holds it.

| Adapter | Backend |
| --- | --- |
| `http_client(url, input_columns=, output_columns=)` | Any JSON HTTP endpoint |
| `triton_client(...)` | NVIDIA Triton |
| `torchserve_client(...)` | TorchServe |
| `serve_deployment(...)` | A Ray Serve deployment |
| `serving_udf(connect, ...)` | Your own `ServingClient` |

The adapter sends a **batch** per request, not a row. That is the whole reason the
pattern is viable. Ten million rows at `batch_size=64` is 156,000 requests instead of ten
million.

:::{important}
Tune `batch_size` against what the endpoint will accept. Most serving stacks have a max
payload, and a batch that exceeds it does not degrade. It fails every request.
:::

`retries` handles the transient failure, and a request that keeps failing raises. If the
endpoint is flaky enough that you would rather lose rows than the job, pair it with
`max_errored_rows`.

Writing your own adapter means implementing `ServingClient` and handing `serving_udf` a
`connect` callable. Same shape, so the connection is made once per worker.

## Overlapping stages with run_pipeline

The problem with a single map stage doing decode-then-forward is that both wait on each
other. The CPU decodes batch *n+1* only after the GPU finishes batch *n*.
`run_pipeline` chains `Stage`s with credit-based backpressure, so each stage runs while
the next one is still working. No stage can run ahead far enough to blow up memory,
because credits bound the queue between them.

```python
import pyarrow as pa
import pyarrow.compute as pc

import batcher as bt
from batcher.ml import Stage, run_pipeline


class Decode:              # stands in for a CPU stage (image decode, tokenize)
    def __call__(self, batch):
        scaled = pc.multiply(pc.cast(batch.column("x"), "float64"), 2.0)
        return batch.set_column(0, "x", scaled)


class Forward:             # stands in for the GPU forward pass
    def __call__(self, batch):
        label = pc.greater(batch.column("x"), 4.0)
        return batch.append_column("label", label)


ds = bt.from_pydict({"x": [1, 2, 3, 4]})
out = list(
    run_pipeline(
        ds.iter_batches(),
        [Stage(Decode, credits=2, name="decode"), Stage(Forward, credits=2, num_gpus=0, name="gpu")],
    )
)
print(out[0].to_pydict())
# {'x': [2.0, 4.0, 6.0, 8.0], 'label': [False, False, True, True]}
```

`credits` is how many batches may sit queued ahead of a stage. One credit is one batch
slot, and the producer blocks at zero. Two is a sane default, enough to overlap without
buffering a pipeline's worth of decoded images in RAM. This is the same credit-based flow
control the engine's shuffle uses.

## Adaptive batching with InferencePool

`InferencePool` sits underneath the `infer` path and is worth reaching for directly when
you are driving the stream yourself, in a serving process or a custom loop. It keeps
workers alive, so the factory runs once per worker and the model loads once, and it
*rebatches* the incoming stream to a target size. That is the difference between feeding
a GPU 64-row batches and feeding it whatever size the reader happened to produce.

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

Results come back **in input order**, whichever worker produced them, so a downstream
join on row position stays valid. `target_latency_ms` with `objective="latency"` shrinks
the batch to hold a latency target instead of maximizing throughput, which is the
online-serving trade. `min_batch_rows` and `max_batch_rows` bound the adaptation.

## Batch and online, one model

:::{tip}
The pattern that keeps a team honest is to run the same worker class in the offline
pipeline and inside the serving process. Offline it is handed to `map_batches`. Online it
is handed to `InferencePool` and fed by request handlers. One implementation, so a preprocessing
step cannot drift between training-time scoring and serving-time scoring. That drift is
the most common source of training/serving skew and the hardest to find, because both
halves look correct in isolation.
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

## Choosing

| Situation | Reach for |
| --- | --- |
| A batch job, model you can schedule | `ds.ml.infer(ModelClass, num_gpus=…)` |
| Model owned by another team or another cluster | `http_client`, `triton_client`, or `serve_deployment` |
| CPU preprocessing starving a GPU stage | `run_pipeline` with `Stage` credits |
| You are driving the stream, and want adaptive batching | `InferencePool` |
| An LLM behind an OpenAI-compatible endpoint | `http_engine`, covered in [LLM inference](llm.md) |

## See also

- [Serving](serving.md): the adapters and the `ServingClient` contract in full.
- [Batch scoring](batch-scoring.md): the offline job end to end.
- [Inference](inference.md): the load-once-per-worker contract.
- [GPU scheduling](gpu.md): sizing actors and packing models onto devices.
- [Credit flow control](../deep-dives/credit-flow-control.md): the credits `Stage` hands
  out, and the same mechanism in the engine's shuffle.
- [GPU execution](../deep-dives/gpu-execution.md): why a CPU stage starves a GPU stage,
  and what overlapping them buys.
- [Streaming inference](../examples/streaming/streaming-inference.md): the online half,
  as a runnable recipe.
- [AI and GPU benchmarks](../benchmarks/ai-and-gpu.md): in-process against the served
  path, measured.
- [ML API](../api/ml.md): the `InferencePool`, `Stage`, and `run_pipeline` reference.
