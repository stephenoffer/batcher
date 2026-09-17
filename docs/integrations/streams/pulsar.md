# Pulsar

This page covers reading Apache Pulsar topics. {py:meth}`bt.read.pulsar(topic) <batcher.api.io_namespace.reader.Reader.pulsar>` consumes a topic as an unbounded {py:class}`Dataset <batcher.Dataset>` through `pulsar-client`, with one reader per partition, acknowledgement only after a micro-batch is published, and checkpointed recovery applied as a consumer seek.

The following table summarizes the connector:

| | |
| --- | --- |
| Read | `bt.read.pulsar(topic)` |
| Write | No sink. Write the stream to Delta or another streaming sink. |
| Extra | `pip install 'batcher-engine[pulsar]'` |
| Parallelism | One split per partition, from the `num_partitions` you declare |
| Subscription | `ConsumerType.Shared`, so no per-key ordering |
| Auth | None. The client is built from `service_url` alone. |
| Restart | A checkpointed `MessageId`, applied as a consumer seek; else the broker cursor |

```bash
pip install 'batcher-engine[pulsar]'
```

## Read a topic

```python
# docs: skip
import batcher as bt

events = bt.read.pulsar(
    "persistent://tenant/ns/events",
    service_url="pulsar://broker:6650",
    subscription="batcher-etl",
    num_partitions=8,
    starting_position="earliest",
)
```

`topic` can be the short name or the fully-qualified `persistent://tenant/namespace/topic`
form. It is passed to the client verbatim, so use whatever your cluster expects.

`starting_position` is `"earliest"` or `"latest"` and applies only when the subscription is new, because Pulsar remembers a subscription's cursor on the broker.

Rows come back in the fixed broker schema every streaming source shares (`key`, `value`, `partition`, `offset`, `timestamp`, `topic`), with `value` the raw payload bytes and `key` the partition key. Pass `value_format=` to decode the payload in the source, as described in {doc}`/integrations/streams/payload-formats`, or decode it with expressions as shown below.

Two of those columns mean something slightly different on Pulsar.

:::{dropdown} What `offset` and `timestamp` actually hold here
`offset` is not a Pulsar concept at all. The `MessageId` is a `(ledger_id, entry_id)` pair,
and Batcher folds it into one int64 (`ledger << 32 | entry`) so it fits the shared schema. It
is monotonic within a ledger. It is not something you can hand back to a Pulsar client, and it
is not comparable across ledgers.

`timestamp` is the publish time in milliseconds, not the event time. If your payload carries
an event time, extract it and use *that* for {py:meth}`with_watermark <batcher.Dataset.with_watermark>` and windowing.
:::

## Declare the partition count

:::{warning}
Batcher doesn't ask the broker how many partitions the topic has. `num_partitions` defaults to 1, and `splits()` trusts it. Point the reader at a
12-partition topic without saying so and you get one split, one consumer, and one worker doing
all the ingest, on a cluster that could have used twelve.
:::

Set `num_partitions` to the topic's real partition count. Get it wrong upward and `splits()`
will address partition topics (`<topic>-partition-N`) that do not exist; the client errors out
on subscribe.

## How it parallelizes

A {py:class}`Source <batcher.io.Source>` divides into {py:class}`Split <batcher.io.Split>` objects, and each split is a unit of read parallelism. Pulsar's split
is the partition: split *n* subscribes to the physical topic `<topic>-partition-<n>`, so a
worker reads exactly one partition. A non-partitioned topic is one split, and reads on one
worker. That is the same rule Kafka has, expressed through Pulsar's partitioned-topic naming.

Consumed with no split (the single-node path), the client subscribes to the base topic and
fans out across its partitions itself.

## Subscriptions, acks, and ordering

Batcher subscribes with `ConsumerType.Shared`, under the subscription name you pass (default
`"batcher"`).

:::{important}
Shared means round-robin dispatch across consumers on that subscription, with no per-key ordering guarantee. If your pipeline needs messages for a key in publish order, as a CDC
stream or a state machine or an SCD feed does, a shared subscription is the wrong shape, and
Batcher offers no `Key_Shared` or `Failover` subscription type. Reorder downstream, or accept
the ordering you get.
:::

Messages are acknowledged once the micro-batch that carries them has been published, not when the poll assembles them, so a crash in between leaves them unacked and Pulsar redelivers them. That's at-least-once. With a `checkpoint=` set, the recorded `MessageId` is applied as a real
per-consumer `seek` on restart, so recovery resumes from the checkpoint rather than from
wherever the subscription cursor happened to sit. Without one, the cursor is the resume point.
Either way, give every distinct pipeline its own `subscription` name, keep it stable across
restarts, and make your sink idempotent.

The `subscription` name doubles as the isolation boundary. Two pipelines sharing a name share
the message stream, and each sees roughly half the messages.

## Decoding the payload

The payload arrives as opaque bytes, so decoding is your first transformation. Expressed as
expressions rather than a Python loop, it runs in the engine. The block below stands a local
batch in for the topic, using the same six columns the reader delivers:

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
        "key": [b"acct-1", b"acct-2", b"acct-1"],
        "value": [
            b'{"account":"acct-1","delta":50}',
            b'{"account":"acct-2","delta":-20}',
            b'{"account":"acct-1","delta":15}',
        ],
        "partition": [0, 1, 0],
        "offset": [881, 12, 882],
        "timestamp": [1700000000000, 1700000001000, 1700000002000],
        "topic": ["persistent://public/default/ledger"] * 3,
    },
    schema=schema,
)

# Stand in for the topic; the pipeline below is what you run against the real one.
ledger = bt.from_batches(lambda: iter([batch]), schema)

moves = ledger.select(
    col("value").cast("string").json.extract_string("$.account").alias("account"),
    col("value").cast("string").json.extract_int("$.delta").alias("delta"),
)
print(moves.group_by("account").agg(balance=col("delta").sum()).sort("account").to_pydict())
# {'account': ['acct-1', 'acct-2'], 'balance': [65, -20]}
```

Because a shared subscription gives no per-key ordering, an aggregate like the one above is
safe (addition commutes) while a last-write-wins or state-machine transition is not. That is
the practical shape of the warning above.

## Writing

```python
# docs: skip
q = bt.read.pulsar(
    "events", service_url="pulsar://broker:6650", subscription="bronze-events", num_partitions=8
).write.delta(
    "lake/bronze/events",
    trigger=bt.Trigger.processing_time("30 seconds"),
    checkpoint="/var/lib/batcher/ckpt/bronze-events",
    query_name="bronze-events",
)
q.await_termination()
```

The checkpoint can be a local path or an object-store URI such as `s3://`. {py:meth}`collect() <batcher.Dataset.collect>` on an unbounded source raises {py:exc}`PlanError <batcher.PlanError>`; use {py:meth}`iter_batches() <batcher.Dataset.iter_batches>`, a triggered
write, or {py:meth}`bt.Trigger.available_now() <batcher.Trigger.available_now>` to drain the backlog that has already arrived and stop.

## Message metadata

Pulsar calls them *properties*, and Kafka calls them headers. They are the same idea, so they
arrive under the same option and in the same column type,
`array<struct<key:string,value:binary>>`:

```python
# docs: skip
events = bt.read.pulsar("persistent://public/default/events", include_headers=True)
traced = events.with_columns(trace=bt.col("headers").list.get(0).struct.field("value"))
```

It is off by default because the nested column costs on every message of every poll. Values
are carried as bytes whatever the client hands back, and a message that carried none reads
as `null` rather than as an empty list.

## Requirements and limitations

Authentication isn't wired. The source builds `pulsar.Client(service_url)` with no other parameters, so token auth, TLS, and OAuth2 have no way in, and a cluster that requires authentication can't be read without extending the source. See {doc}`custom connectors </user-guide/moving-data/custom-connectors>`.

A poll waits for its first message for up to `receive_timeout_millis`, 1,000 by default, then drains whatever the client has already buffered in one batch call. On a quiet topic a poll often returns nothing, and the loop goes round again.

A single consumer that spans several partitions can't resume from a checkpoint holding positions for more than one of them, because a Pulsar seek repositions the whole consumer. Batcher raises `PlanError` rather than resuming at the wrong position. Read the topic distributed, one split per partition, so each partition seeks its own consumer.

Ack timeouts produce duplicates. If a micro-batch takes longer than the subscription's ack
timeout, the broker redelivers messages Batcher is still working on. Keep micro-batches short,
or raise the ack timeout on the subscription.

Backlog quota bites eventually. A subscription that Batcher stops consuming, because a crashed
query was never restarted, accrues backlog against the namespace quota, and the broker will
start rejecting producers. Delete the subscription when you retire a pipeline.

## See also

- {doc}`Streaming </user-guide/moving-data/streaming>`: triggers, watermarks, output modes, checkpoints.
- {doc}`Windowed aggregation </cookbook/streaming/windowed-aggregation>`: the shape most
  Pulsar pipelines end up in.
- {doc}`Custom connectors </user-guide/moving-data/custom-connectors>`: the `Source`/`Split` protocol, if
  you need auth or a `Key_Shared` subscription.
- {doc}`Reading and writing </api/relational/io>`: the full reader/writer surface.
- {doc}`Kafka </integrations/streams/kafka>`: the same broker schema, and the payload-decoding example.
- {doc}`Payload formats </integrations/streams/payload-formats>`: decoding Avro, JSON, and Protobuf in the source.
