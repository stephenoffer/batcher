# Streaming inference

:::{warning}
The classic way to make a streaming inference job useless is to pass a plain function to
`map_batches` that loads the model inside it. The model then loads once per batch. On a
five-second trigger with a model that takes four seconds to initialize, the job spends its
life loading weights, and the throughput number you report is a measurement of `from_pretrained`.
:::

The fix is one character of API: pass the **class** itself, rather than an instance or a
closure. The engine constructs it once and reuses that instance across every micro-batch
of the query.

| What you pass to `map_batches` | What the engine does with it | The cost |
| --- | --- | --- |
| a class, `Scorer` | constructs it once, reuses the instance for every micro-batch | the weights load once |
| a plain function or a closure | calls it per batch, so the model is rebuilt inside it | the load cost, on every micro-batch, forever |

## Resident by construction

::::{tab-set}
:::{tab-item} A stub scorer, no model

```python
import pyarrow as pa
import pyarrow.compute as pc

import batcher as bt
from batcher import col

LOADS = []


class Scorer:
    def __init__(self):
        LOADS.append(1)  # stands in for loading real weights

    def __call__(self, batch):
        length = pc.cast(pc.utf8_length(batch.column("text")), "float64")
        return batch.append_column("score", pc.divide(length, 10.0))


schema = pa.schema([("id", pa.int64()), ("text", pa.string())])


def feed():
    yield pa.record_batch({"id": [1, 2], "text": ["hello", "hi"]}, schema=schema)
    yield pa.record_batch({"id": [3], "text": ["streaming"]}, schema=schema)


stream = bt.from_batches(feed, schema, bounded=False)
scored = stream.map_batches(Scorer, output_columns=["id", "text", "score"], num_workers=1)

query = scored.filter(col("score") > 0.2).write.memory(
    "scored_stream", trigger=bt.Trigger.available_now(), query_name="scorer"
)
query.await_termination()

print(bt.read_memory("scored_stream").to_pydict())
# {'id': [1, 3], 'text': ['hello', 'streaming'], 'score': [0.5, 0.9]}
print(len(LOADS))
# 1
```

One construction, two micro-batches. The `filter` after `map_batches` runs in Rust on the
model's output, so rows you are going to throw away never reach Python. Put the cheap
predicates before the model when you can, and the expensive ones after it.
:::

:::{tab-item} A real model, on a topic

Same shape, different `__call__`. `ds.ml.infer` takes a HuggingFace model id and does the
class-per-worker construction for you; `num_gpus` and `concurrency` place it:

```python
# docs: skip
sentiment = (
    bt.read.kafka("reviews", bootstrap_servers="broker-1:9092")
    .with_columns(text=col("value").cast("string"))
    .ml.infer(
        "distilbert-base-uncased-finetuned-sst-2-english",
        column="text",
        output_column="sentiment",
        batch_size=64,
        num_gpus=1,
        model_memory_gb=1.5,
    )
)

query = sentiment.write.delta(
    "s3://lake/gold/review_sentiment",
    trigger=bt.Trigger.processing_time("10 seconds"),
    checkpoint="s3://lake/gold/review_sentiment/_ckpt",
    query_name="review-sentiment",
)
query.await_termination()
```

`batch_size` here is the model's batch size, not the source's. A Kafka poll assembles up to
`poll_size` messages into one Arrow batch; `batch_size=64` rebatches to what the GPU
actually wants. Those are independent numbers and conflating them is how you end up with a
GPU that is 90% idle.
:::
::::

The load is once **per query**, not once per process. Start the query again and the class
is constructed again. That matters if your model takes minutes to load and you were
planning to restart the job on a schedule.

## The poison record

A 24/7 job will eventually meet a row that breaks the model: a truncated UTF-8 sequence, a
zero-byte image, a text field that is 400KB of base64. If the exception propagates, the
query stops. At 3am.

`max_errored_rows` caps how many failures the pipeline absorbs before it gives up:

```python
class Flaky:
    def __call__(self, batch):
        if 2 in batch.column("id").to_pylist():
            raise ValueError("poison record")
        return batch.append_column("score", pc.cast(batch.column("id"), "float64"))


def three_batches():
    yield pa.record_batch({"id": [1], "text": ["a"]}, schema=schema)
    yield pa.record_batch({"id": [2], "text": ["b"]}, schema=schema)
    yield pa.record_batch({"id": [3], "text": ["c"]}, schema=schema)


tolerant = bt.from_batches(three_batches, schema, bounded=False).map_batches(
    Flaky, output_columns=["id", "text", "score"], num_workers=1, max_errored_rows=5
)
q = tolerant.write.memory("tolerant", trigger=bt.Trigger.available_now())
q.await_termination()

print(bt.read_memory("tolerant").to_pydict()["id"])
# [1, 3]
print(q.exception())
# None
```

Row 2 is gone and the query survived.

:::{important}
Be clear-eyed about what that means: the failing *batch* is dropped, not only the offending
row, and the dropped rows go nowhere. There is no dead-letter output. `max_errored_rows` buys
you uptime at the price of silent data loss. If the loss matters, catch the exception inside
`__call__`, emit a null prediction plus an error string as columns, and route those rows
yourself downstream.
:::

## Watching it run

The query handle is how you know whether the model is keeping up:

:::{dropdown} Per-batch progress, straight off the query handle

```python
progress = q.recent_progress()
print([(p.batch_id, p.num_input_rows, p.num_output_rows) for p in progress])
# [(0, 1, 1), (1, 1, 0), (2, 1, 1)]
print(q.is_active)
# False
```
:::

`num_input_rows` versus `num_output_rows` on batch 1 is the dropped poison batch, visible
in the metrics even though nothing raised. `p.input_rows_per_second` is the throughput
number to alert on; if it is falling while the source's lag is rising, the model is your
bottleneck and no amount of trigger tuning changes that.

:::{important}
Two limits worth knowing before you scale this out. The resident-model path shown here is
single-node: the model lives in the driver process for the life of the query. And
`distributed=True` on a streaming write only covers an `available_now`/`once` drain of a
stateless pipeline, with no checkpoint, so it is a backfill tool. A GPU inference stream
that must span nodes is not something Batcher does today; run the inference as a
distributed *batch* job over the landed bronze table instead.
:::

## See also

- [Batch inference](../../ml/inference.md): the same `map_batches`/`ml.infer` surface,
  bounded.
- [ML for streaming](../../ml/streaming.md): the model-over-a-stream surface in full.
- [GPU pipelines](../../ml/gpu.md): actor pools, `concurrency`, and accelerator placement.
- [Kafka to the lake](kafka-etl.md): landing the raw events this job scores.
- [Exactly-once sinks](exactly-once-sink.md): what the Delta write above actually guarantees.
- [LLM batch scoring](../ml/llm-batch-scoring.md): the same stage, run over the landed table
  as a distributed batch.
- [Streaming](../../user-guide/streaming.md): triggers, checkpoints, and `recent_progress()`.
- [ML API reference](../../api/ml.md): `ds.ml.infer`, `map_batches`, `max_errored_rows`.
- [AI and GPU benchmarks](../../benchmarks/ai-and-gpu.md): what a warm, resident model is worth.
- [GPU execution](../../deep-dives/gpu-execution.md): the pool the model lives in.
