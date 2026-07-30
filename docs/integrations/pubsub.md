# Pub/Sub

`bt.read.pubsub(subscription)` consumes a Google Cloud Pub/Sub subscription as an unbounded
`Dataset`. Read only. There is no Pub/Sub sink in Batcher.

| | |
| --- | --- |
| **Read** | `bt.read.pubsub("projects/<project>/subscriptions/<name>")` |
| **Write** | Not supported |
| **Extra** | `pip install 'batcher-engine[pubsub]'` |
| **Parallelism** | One split, always. Pub/Sub exposes no partitions. |
| **Credentials** | The ambient `google.auth` environment; `roles/pubsub.subscriber` |
| **Restart** | The subscription's own unacked backlog |

```bash
pip install 'batcher-engine[pubsub]'
```

## You pass a subscription, not a topic

:::{warning}
The argument is the fully-qualified subscription path,
`projects/<project>/subscriptions/<name>`. Not the topic. Not a bare name. It goes straight to
`SubscriberClient.pull` as the `subscription` field, and a short name produces an
`InvalidArgument` from the API.
:::

```python
# docs: skip
import batcher as bt

events = bt.read.pubsub(
    "projects/acme-prod/subscriptions/events-batcher",
    poll_size=1_000,
)
```

Credentials come from the ambient `google.auth` environment:
`GOOGLE_APPLICATION_CREDENTIALS`, an application-default login, or the service account
attached to the node. The identity needs `roles/pubsub.subscriber` on the subscription. There
is no credential keyword.

Rows arrive in the fixed broker schema (`key`, `value`, `partition`, `offset`, `timestamp`,
`topic`), and Pub/Sub bends most of it:

| Column | What Pub/Sub puts there |
| --- | --- |
| `key` | The message's *ordering key*, UTF-8 encoded, or null if it has none |
| `value` | The message data, as raw bytes |
| `partition` | Always `0`, since Pub/Sub has no user-visible partitions |
| `offset` | A hash of the opaque message id, folded into int64 |
| `timestamp` | The publish time, in milliseconds |
| `topic` | The subscription path you passed, not the topic name |

That `offset` is stable per message and useful for de-duplication, but it is not ordered and
it is not a position you can seek to.

Message *attributes* are not surfaced. If your producers put routing metadata in attributes
rather than the body, Batcher will not show it to you.

## Decoding the payload

The payload is opaque bytes, so decoding it is your first transformation. Write it as
expressions and it runs in Rust rather than in a Python loop.

The block below stands a local batch in for the subscription, using the same six columns
Pub/Sub delivers, so the decode pipeline runs here exactly as it would against the real
source:

```python
import batcher as bt
import pyarrow as pa
from batcher import col

schema = pa.schema([
    ("key", pa.binary()), ("value", pa.binary()), ("partition", pa.int64()),
    ("offset", pa.int64()), ("timestamp", pa.int64()), ("topic", pa.string()),
])
batch = pa.record_batch({
    "key": [b"order-1", b"order-2", b"order-1"],          # the ordering key
    "value": [b'{"sku":"a","qty":2}', b'{"sku":"b","qty":1}', b'{"sku":"a","qty":3}'],
    "partition": [0, 0, 0],                                # always 0 on Pub/Sub
    "offset": [7314159265358979, 2718281828459045, 1414213562373095],
    "timestamp": [1700000000000, 1700000001000, 1700000002000],
    "topic": ["projects/acme-prod/subscriptions/events-batcher"] * 3,
}, schema=schema)

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

## It does not parallelize

:::{important}
A `Source` divides into `Split`s and each split is a unit of read parallelism. Pub/Sub exposes
no partitions, so the source models the stream as one logical partition and `splits()` returns
exactly one split. One split means one reader, on one worker, however large your cluster.
:::

This is the honest limit of the connector. Pub/Sub's own scaling story is many concurrent
subscriber clients on one subscription, and Batcher's split model has no way to express that
today: a split is a *named piece of the source*, and there is nothing here to name. If ingest
rate is your bottleneck, run several queries against several subscriptions on the same topic
and union the results downstream, or write a custom source (see
{doc}`custom connectors <../user-guide/custom-connectors>`).

`poll_size` maps to `max_messages` on the pull request. The service treats it as an upper
bound and routinely returns far fewer, so the default 16,384 is optimistic rather than wrong.
A value around 1,000 matches what the API will actually hand back.

## Delivery: at-least-once, and the ack deadline

Messages are acked after a batch has been assembled, in one `acknowledge` call for the whole
poll. A crash before that ack means Pub/Sub redelivers, so nothing is lost.

:::{warning}
The other half of that trade is duplicates, and the ack deadline is where they come from. The
subscription's deadline (10 seconds by default) starts when the message is pulled, not when
your micro-batch finishes. Pull a thousand messages, spend fifteen seconds in a model call or
a slow sink, and Pub/Sub has already redelivered them to somebody. There is no ack-extension
loop in the source.
:::

Two ways to live with it.

::::{tab-set}

:::{tab-item} Raise the deadline
Raise the subscription's `ackDeadlineSeconds` above your worst-case micro-batch time. This is
a subscription-side setting, so it is a `gcloud` change, not a Batcher one.
:::

:::{tab-item} Deduplicate downstream
Deduplicate on the message id, which is what the `offset` column is for.
`drop_duplicates_within_watermark` does it with bounded state:

```python
# docs: skip
clean = (
    bt.read.pubsub("projects/acme-prod/subscriptions/events-batcher", poll_size=1_000)
    .drop_duplicates_within_watermark(["offset"], event_time="ts", lateness="10 minutes")
)
```
:::

::::

Batcher's checkpoint records a position for Pub/Sub, but no seek is applied to a live
subscriber. The resume point on restart is the subscription's own unacked backlog. That is the
right behavior for Pub/Sub, and it means the subscription, not the checkpoint, is what you
must not delete between runs.

## Writing the stream out

:::{dropdown} A checkpointed write into a bronze Delta table
```python
# docs: skip
q = (
    bt.read.pubsub("projects/acme-prod/subscriptions/events-batcher", poll_size=1_000)
    .write.delta(
        "lake/bronze/events",
        trigger=bt.Trigger.processing_time("30 seconds"),
        checkpoint="/var/lib/batcher/ckpt/bronze-events",
        query_name="bronze-events",
    )
)
q.await_termination()
```
:::

The checkpoint directory is SQLite plus Arrow IPC on a real filesystem, not a `gs://` URI.
`query_name` must be stable across restarts: it is the Delta transaction id a replayed
micro-batch is recognized by.

`collect()` raises `PlanError` on an unbounded source. Use `iter_batches()`, a triggered write,
or `bt.Trigger.available_now()` to drain the current backlog and stop.

## Failure modes worth knowing

A slow sink means duplicates, per the ack deadline above. This is the failure you will
actually hit.

Ordering keys are not honored on read. The key is exposed as a column, but the pull path does
nothing to preserve per-key order across a batch.

Dead-letter topics keep working, since they are a subscription-side setting. A message Batcher
never acks will eventually be routed there if you configured one, which is a good backstop for
a poison payload that keeps crashing your decode.

`value` is bytes. Pub/Sub's own schema support (Avro and Protobuf schemas registered on the
topic) is not applied. Decode in `map_batches`, per batch, never per row.

## See also

- {doc}`Streaming <../user-guide/streaming>`: triggers, watermarks, dedup, checkpointing.
- {doc}`Late data and watermarks <../examples/streaming/late-data-watermarks>`: the bounded-state
  dedup above, in a full pipeline.
- {doc}`Custom connectors <../user-guide/custom-connectors>`: the `Source`/`Split` protocol.
- {doc}`Reading and writing <../api/io>`: the full reader/writer surface.
- {doc}`Kafka <kafka>`: the shared broker schema and the JSON-payload decode example.
- {doc}`Kinesis <kinesis>`: the cloud broker that does split, one worker per shard.
