# Pub/Sub

This page covers reading Google Cloud Pub/Sub. {py:meth}`bt.read.pubsub(subscription) <batcher.api.io_namespace.reader.Reader.pubsub>` consumes a subscription as an unbounded {py:class}`Dataset <batcher.Dataset>`, acknowledging messages only after the micro-batch that carries them is published, so a crash costs a redelivery rather than lost data.

The following table summarizes the connector:

| | |
| --- | --- |
| Read | `bt.read.pubsub("projects/<project>/subscriptions/<name>")` |
| Write | No sink. Write the stream to Delta or another streaming sink. |
| Extra | `pip install 'batcher-engine[pubsub]'` |
| Parallelism | One split per subscription, since Pub/Sub exposes no partitions |
| Credentials | The ambient `google.auth` environment; `roles/pubsub.subscriber` |
| Restart | The subscription's own unacked backlog |

```bash
pip install 'batcher-engine[pubsub]'
```

## Read a subscription

:::{important}
The argument is the fully-qualified subscription path, `projects/<project>/subscriptions/<name>`, not a topic or a bare name. It goes straight to `SubscriberClient.pull` as the `subscription` field, and a short name produces an `InvalidArgument` from the API.
:::

```python
# docs: skip
import batcher as bt

events = bt.read.pubsub(
    "projects/acme-prod/subscriptions/events-batcher",
    poll_size=1_000,
    pull_timeout=5.0,
)
```

`pull_timeout` bounds one pull, 10 seconds by default. An idle subscription that reaches the deadline returns an empty poll rather than blocking the trigger.

Credentials come from the ambient `google.auth` environment:
`GOOGLE_APPLICATION_CREDENTIALS`, an application-default login, or the service account
attached to the node. The identity needs `roles/pubsub.subscriber` on the subscription. There's no credential keyword.

Rows arrive in the fixed broker schema (`key`, `value`, `partition`, `offset`, `timestamp`,
`topic`). The following table shows what Pub/Sub puts in each column:

| Column | What Pub/Sub puts there |
| --- | --- |
| `key` | The message's *ordering key*, UTF-8 encoded, or null if it has none |
| `value` | The message data, as raw bytes |
| `partition` | Always `0`, since Pub/Sub has no user-visible partitions |
| `offset` | A SHA-256 hash of the message id, folded into int64 |
| `timestamp` | The publish time, in milliseconds |
| `topic` | The subscription path you passed, not the topic name |

That `offset` is stable per message across runs and workers, which makes it a de-duplication key, but it isn't ordered and it isn't a position you can seek to.

Message *attributes* are Pub/Sub's spelling of Kafka's headers, and `include_headers=True`
adds them as a `headers` column of `array<struct<key:string,value:binary>>`, the same type
and the same option every broker here uses:

```python
# docs: skip
events = bt.read.pubsub("projects/p/subscriptions/s", include_headers=True)
routed = events.with_columns(tenant=bt.col("headers").list.get(0).struct.field("value"))
```

It's off by default because the nested column costs on every message of every poll, and a
message that carried no attributes reads as `null` rather than as an empty list.

## Decode the payload

Pass `value_format=` to decode the payload in the source, as described in {doc}`/integrations/streams/payload-formats`. Otherwise decoding is your first transformation. Write it as
expressions and it runs in Rust rather than in a Python loop.

The block below stands a local batch in for the subscription, using the same six columns
Pub/Sub delivers, so the decode pipeline runs here exactly as it would against the real
source:

```python
import batcher as bt
import pyarrow as pa
from batcher import col

schema = pa.schema(
    [
        ("key", pa.binary()),
        ("value", pa.binary()),
        ("partition", pa.int64()),
        ("offset", pa.int64()),
        ("timestamp", pa.int64()),
        ("topic", pa.string()),
    ]
)
batch = pa.record_batch(
    {
        "key": [b"order-1", b"order-2", b"order-1"],  # the ordering key
        "value": [b'{"sku":"a","qty":2}', b'{"sku":"b","qty":1}', b'{"sku":"a","qty":3}'],
        "partition": [0, 0, 0],  # always 0 on Pub/Sub
        "offset": [7314159265358979, 2718281828459045, 1414213562373095],
        "timestamp": [1700000000000, 1700000001000, 1700000002000],
        "topic": ["projects/acme-prod/subscriptions/events-batcher"] * 3,
    },
    schema=schema,
)

# Stand in for the subscription; the pipeline below is what you run against the real one.
events = bt.from_batches(lambda: iter([batch]), schema)

decoded = events.select(
    col("key").cast("string").alias("ordering_key"),
    col("value").cast("string").json.extract_string("$.sku").alias("sku"),
    col("value").cast("string").json.extract_int("$.qty").alias("qty"),
)
print(decoded.group_by("sku").agg(units=col("qty").sum()).sort("sku").to_pydict())
# {'sku': ['a', 'b'], 'units': [5, 1]}
```

Against the live subscription, the only line that changes is the source:
`events = bt.read.pubsub("projects/acme-prod/subscriptions/events-batcher")`.

## How it parallelizes

A {py:class}`Source <batcher.io.Source>` divides into {py:class}`Split <batcher.io.Split>` objects, and each split is a unit of read parallelism. Pub/Sub exposes no partitions, so the source models a subscription as one logical partition and `splits()` returns one split, read on one worker.

Pub/Sub scales ingest with many concurrent subscriber clients on one subscription, and a split names a piece of the source, so there's nothing for a second split to name. If ingest rate is your bottleneck, run several queries against several subscriptions on the same topic
and union the results downstream, or write a custom source (see
{doc}`custom connectors </user-guide/moving-data/custom-connectors>`).

`poll_size` maps to `max_messages` on the pull request, and Pub/Sub rejects a request above 1,000. Batcher clamps to that ceiling before sending, so the default of 16,384 never reaches the API. A smaller value passes through untouched. `max_bytes_per_trigger` trims a pull to its payload budget, and the messages it trims are never acknowledged, so Pub/Sub redelivers them.

## Delivery and the ack deadline

Messages are acknowledged once the micro-batch carrying them has been published, not when the poll assembles them. A crash in between means Pub/Sub redelivers, so nothing is lost.

:::{warning}
The other half of that trade is duplicates, and the ack deadline is where they come from. The
subscription's deadline (10 seconds by default) starts when the message is pulled, not when
your micro-batch finishes. Pull a thousand messages, spend fifteen seconds in a model call or
a slow sink, and Pub/Sub has already redelivered them to somebody. There is no ack-extension
loop in the source.
:::

There are two ways to live with it.

::::{tab-set}

:::{tab-item} Raise the deadline
Raise the subscription's `ackDeadlineSeconds` above your worst-case micro-batch time. This is
a subscription-side setting, so it is a `gcloud` change, not a Batcher one.
:::

:::{tab-item} Deduplicate downstream
Deduplicate on the message id, which the `offset` column carries. {py:meth}`drop_duplicates_within_watermark <batcher.Dataset.drop_duplicates_within_watermark>` does it with bounded state:

```python
# docs: skip
clean = (
    bt.read.pubsub("projects/acme-prod/subscriptions/events-batcher", poll_size=1_000)
    .with_columns(published=bt.col("timestamp").cast("timestamp(ms)"))
    .drop_duplicates_within_watermark(["offset"], event_time="published", lateness="10 minutes")
)
```
:::

::::

Pub/Sub isn't replayable by position, so no seek is applied on restart. The resume point is the subscription's own unacknowledged backlog, which means the subscription, not the checkpoint, is what you must keep between runs.

## Writing

:::{dropdown} A checkpointed write into a bronze Delta table
```python
# docs: skip
q = bt.read.pubsub("projects/acme-prod/subscriptions/events-batcher", poll_size=1_000).write.delta(
    "lake/bronze/events",
    trigger=bt.Trigger.processing_time("30 seconds"),
    checkpoint="/var/lib/batcher/ckpt/bronze-events",
    query_name="bronze-events",
)
q.await_termination()
```
:::

The checkpoint can be a local path or a `gs://` URI. `query_name` must be stable across restarts: it is the Delta transaction id a replayed
micro-batch is recognized by.

{py:meth}`collect() <batcher.Dataset.collect>` raises {py:exc}`PlanError <batcher.PlanError>` on an unbounded source. Use {py:meth}`iter_batches() <batcher.Dataset.iter_batches>`, a triggered write,
or {py:meth}`bt.Trigger.available_now() <batcher.Trigger.available_now>` to drain the current backlog and stop.

## Requirements and limitations

A slow sink means duplicates, per the ack deadline above. It's the failure you're most likely to hit.

Ordering keys are not honored on read. The key is exposed as a column, but the pull path does
nothing to preserve per-key order across a batch.

Dead-letter topics keep working, since they are a subscription-side setting. A message Batcher
never acks will eventually be routed there if you configured one, which is a good backstop for
a poison payload that keeps crashing your decode.

Pub/Sub's own topic schemas aren't read. Name the payload format with `value_format=` and a `value_schema=` instead.

## See also

- {doc}`Streaming </user-guide/moving-data/streaming>`: triggers, watermarks, dedup, checkpointing.
- {doc}`Late data and watermarks </cookbook/streaming/late-data-watermarks>`: the bounded-state
  dedup above, in a full pipeline.
- {doc}`Custom connectors </user-guide/moving-data/custom-connectors>`: the `Source`/`Split` protocol.
- {doc}`Reading and writing </api/relational/io>`: the full reader/writer surface.
- {doc}`Kafka </integrations/streams/kafka>`: the shared broker schema and the JSON-payload decode example.
- {doc}`Kinesis </integrations/streams/kinesis>`: the AWS broker, one reader per shard.
- {doc}`Payload formats </integrations/streams/payload-formats>`: decoding Avro, JSON, and Protobuf in the source.
