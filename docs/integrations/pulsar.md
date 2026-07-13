# Pulsar

`bt.read.pulsar(topic)` consumes an Apache Pulsar topic as an unbounded `Dataset`, via
`pulsar-client`. Read only. Batcher has no Pulsar sink.

| | |
| --- | --- |
| **Read** | `bt.read.pulsar(topic)` |
| **Write** | Not supported |
| **Extra** | `pip install 'batcher-engine[pulsar]'` |
| **Parallelism** | One split per partition, from the `num_partitions` you declare |
| **Subscription** | `ConsumerType.Shared`, so no per-key ordering |
| **Auth** | Not wired. `pulsar.Client(service_url)` and nothing else. |
| **Restart** | The subscription cursor on the broker; no seek is applied |

```
pip install 'batcher-engine[pulsar]'
```

## The read

```python
# docs: skip
import batcher as bt

events = bt.read.pulsar(
    "persistent://tenant/ns/events",
    service_url="pulsar://broker:6650",
    subscription="batcher-etl",
    num_partitions=8,
    poll_size=16_384,
)
```

`topic` can be the short name or the fully-qualified `persistent://tenant/namespace/topic`
form. It is passed to the client verbatim, so use whatever your cluster expects.

Rows come back in the fixed broker schema every streaming source shares (`key`, `value`,
`partition`, `offset`, `timestamp`, `topic`), with `value` the raw payload bytes and `key` the
partition key. Decoding is a transformation, not a reader option; the [Kafka page](kafka.md)
shows the JSON-payload pattern, and it is identical here.

Two of those columns mean something slightly different on Pulsar.

:::{dropdown} What `offset` and `timestamp` actually hold here
`offset` is not a Pulsar concept at all. The `MessageId` is a `(ledger_id, entry_id)` pair,
and Batcher folds it into one int64 (`ledger << 20 | entry`) so it fits the shared schema. It
is monotonic within a ledger. It is not something you can hand back to a Pulsar client, and it
is not comparable across ledgers.

`timestamp` is the publish time in milliseconds, not the event time. If your payload carries
an event time, extract it and use *that* for `with_watermark` and windowing.
:::

## `num_partitions` is a claim, not a lookup

:::{warning}
This is the thing that will bite you. Batcher does not ask the broker how many partitions the
topic has. `num_partitions` defaults to **1**, and `splits()` trusts it. Point the reader at a
12-partition topic without saying so and you get one split, one consumer, and one worker doing
all the ingest, on a cluster that could have used twelve.
:::

Set `num_partitions` to the topic's real partition count. Get it wrong upward and `splits()`
will address partition topics (`<topic>-partition-N`) that do not exist; the client errors out
on subscribe.

## How it parallelizes

A `Source` divides into `Split`s, and each split is a unit of read parallelism. Pulsar's split
is the partition: split *n* subscribes to the physical topic `<topic>-partition-<n>`, so a
worker reads exactly one partition. A non-partitioned topic is one split, and reads on one
worker. That is the same rule Kafka has, expressed through Pulsar's partitioned-topic naming.

Consumed with no split (the single-node path), the client subscribes to the base topic and
fans out across its partitions itself.

## Subscriptions, acks, and ordering

Batcher subscribes with `ConsumerType.Shared`, under the subscription name you pass (default
`"batcher"`).

:::{important}
Shared means round-robin dispatch across consumers on that subscription, with **no per-key
ordering guarantee**. If your pipeline needs messages for a key in publish order, as a CDC
stream or a state machine or an SCD feed does, a shared subscription is the wrong shape, and
Batcher does not currently offer `Key_Shared` or `Failover`. Reorder downstream, or accept the
ordering you get.
:::

Messages are acknowledged after a batch has been assembled, so a crash before the ack leaves
them unacked and Pulsar redelivers them. That is at-least-once. Your resume point is the
subscription cursor on the broker, not Batcher's checkpoint: a checkpointed position is
recorded, but no seek is applied to a live Pulsar consumer, so a restarted query picks up
wherever the cursor left off. Give every distinct pipeline its own `subscription` name, keep
it stable across restarts, and make your sink idempotent.

The `subscription` name doubles as the isolation boundary. Two pipelines sharing a name share
the message stream, and each sees roughly half the messages.

## Writing the stream out

```python
# docs: skip
q = (
    bt.read.pulsar("events", service_url="pulsar://broker:6650",
                   subscription="bronze-events", num_partitions=8)
    .write.delta(
        "lake/bronze/events",
        trigger=bt.Trigger.processing_time("30 seconds"),
        checkpoint="/var/lib/batcher/ckpt/bronze-events",
        query_name="bronze-events",
    )
)
q.await_termination()
```

The checkpoint directory is SQLite plus Arrow IPC on a real filesystem, not an object-store
path. `collect()` on an unbounded source raises `PlanError`; use `iter_batches()`, a triggered
write, or `bt.Trigger.available_now()` to drain what is currently there and stop.

## Failure modes worth knowing

:::{warning}
Auth is not wired. `PulsarSource` builds `pulsar.Client(service_url)` and nothing else. Token
auth, TLS, and OAuth2 parameters have no way in today, so a cluster that requires
authentication cannot be read without extending the source. See
[custom connectors](../user-guide/custom-connectors.md).
:::

A poll waits. Each `receive` blocks for up to one second before giving up, and the poll stops
at the first timeout. On a quiet topic a poll returns whatever arrived in that first second,
which is often nothing, and the loop goes round again.

Ack timeouts produce duplicates. If a micro-batch takes longer than the subscription's ack
timeout, the broker redelivers messages Batcher is still working on. Keep micro-batches short,
or raise the ack timeout on the subscription.

Backlog quota bites eventually. A subscription that Batcher stops consuming, because a crashed
query was never restarted, accrues backlog against the namespace quota, and the broker will
start rejecting producers. Delete the subscription when you retire a pipeline.

## See also

- [Streaming](../user-guide/streaming.md): triggers, watermarks, output modes, checkpoints.
- [Windowed aggregation](../examples/streaming/windowed-aggregation.md): the shape most
  Pulsar pipelines end up in.
- [Custom connectors](../user-guide/custom-connectors.md): the `Source`/`Split` protocol, if
  you need auth or a `Key_Shared` subscription.
- [Reading and writing](../api/io.md): the full reader/writer surface.
- [Kafka](kafka.md): the same broker schema, and the payload-decoding example.
- [Event Hubs](eventhubs.md): the other broker whose resume path needs care.
