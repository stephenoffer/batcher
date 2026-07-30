# Kafka

`bt.read.kafka(topic)` consumes a Kafka topic as an unbounded `Dataset`. It is a read path
only. Batcher has no Kafka sink, so producing back to Kafka goes through
`ds.write.for_each_batch` with your own producer, covered at the end of this page.

| | |
| --- | --- |
| **Read** | `bt.read.kafka(topic)` |
| **Write** | Not supported. `ds.write.for_each_batch` with your own producer. |
| **Extra** | `pip install 'batcher-engine[kafka]'` |
| **Parallelism** | One split per topic partition |
| **Pushdown** | None. The payload arrives as opaque bytes. |
| **Restart** | Batcher's checkpoint, applied on partition assignment; the group offset is the fallback |

```bash
pip install 'batcher-engine[kafka]'
```

That pulls `confluent-kafka`. Without it, construction raises a `BackendError` saying so.

## The read

```python
# docs: skip
import batcher as bt

clicks = bt.read.kafka(
    "clicks",
    bootstrap_servers="broker-1:9092,broker-2:9092",
    group="clickstream-etl",
    poll_size=16_384,
)
```

Every broker source in Batcher (Kafka, Kinesis, Pulsar, Pub/Sub, Event Hubs) hands you the
same six columns and nothing else:

| Column | Type | Meaning |
| --- | --- | --- |
| `key` | binary | The message key, or null if unkeyed |
| `value` | binary | The raw payload, undecoded |
| `partition` | int64 | The topic partition |
| `offset` | int64 | The Kafka offset |
| `timestamp` | int64 | Milliseconds since the Unix epoch |
| `topic` | string | The topic name |

The payload stays opaque bytes. Decoding is your first transformation, expressed as ordinary
expressions so it runs in Rust rather than in a Python loop:

```python
import batcher as bt
import pyarrow as pa
from batcher import col

schema = pa.schema([
    ("key", pa.binary()), ("value", pa.binary()), ("partition", pa.int64()),
    ("offset", pa.int64()), ("timestamp", pa.int64()), ("topic", pa.string()),
])
batch = pa.record_batch({
    "key": [b"u1", b"u2", b"u1"],
    "value": [b'{"user":"u1","amount":10}', b'{"user":"u2","amount":5}',
              b'{"user":"u1","amount":7}'],
    "partition": [0, 0, 1],
    "offset": [11, 12, 4],
    "timestamp": [1700000000000, 1700000001000, 1700000002000],
    "topic": ["orders"] * 3,
}, schema=schema)

# Stand in for the Kafka source; the pipeline below is what you run against the real one.
orders = bt.from_batches(lambda: iter([batch]), schema)

decoded = orders.select(
    col("value").cast("string").json.extract_string("$.user").alias("user"),
    col("value").cast("string").json.extract_int("$.amount").alias("amount"),
)
totals = decoded.group_by("user").agg(total=col("amount").sum())
print(sorted(zip(*[totals.to_pydict()[c] for c in ("user", "total")], strict=True)))
```

```text
[('u1', 17), ('u2', 5)]
```

Avro and Protobuf payloads have no built-in decoder, and there is no Schema Registry client.
Reach for `map_batches` and decode the whole Arrow `value` column at once, per batch, never
per row.

## Consumer groups, and what happens on restart

Read this part before you deploy.

:::{warning}
`group=` defaults to `"batcher"`. Two unrelated pipelines against the same topic with that
default land in the same consumer group, and Kafka will happily split the partitions between
them. Each pipeline then sees half the data and neither complains.
:::

Give every query its own group id, and keep it stable across restarts, because that id *is*
the offset bookmark.

Batcher disables `enable.auto.commit` and advances the group only after a micro-batch is
*published*, never when it is merely polled. A crash in between re-delivers the batch, which
an idempotent sink absorbs. The ordering is chosen so the failure mode is always a duplicate
and never a gap.

The restart path follows from that. Batcher's own checkpoint, not the group offset, is the
source of truth. Under explicit `partitions=`, which is what the distributed split path uses,
the consumer already owns its partitions and repositions immediately to `offset + 1`. Under a
plain group subscription the partitions aren't known until the group assigns them, so the
resume happens in the assignment callback: each partition Kafka hands over is rewound to the
checkpointed position before reading starts. A partition with no checkpointed position keeps
the offset the broker assigned, which is what `auto_offset_reset` controls.

:::{note}
`auto_offset_reset` defaults to `"earliest"`: a brand-new group id starts at the head of the
retained log, not at the tip. Pointing a fresh group at a topic with a week of retention will
replay the week. Pass `auto_offset_reset="latest"` if that is not what you meant.
:::

## How it parallelizes

A `Source` divides into `Split`s, and a split is the unit of read parallelism. Kafka's split
is the topic-partition. `splits()` asks the broker for the topic's partition list and returns
one split per partition, each of which rebuilds a consumer on its worker with an explicit
assignment for that one partition.

:::{important}
The consequence is blunt. **A single-partition topic reads on one worker.** No amount of
cluster gets you a second reader, because there is no second split. If ingest throughput is
your bottleneck, the fix is upstream, in the topic's partition count.
:::

You can also pin a reader to a subset yourself:

```python
# docs: skip
shard = bt.read.kafka("clicks", bootstrap_servers="broker-1:9092",
                      group="etl", partitions=[0, 1, 2])
```

## Security and client config

Anything else you pass through goes to the `confluent-kafka` consumer config with underscores
rewritten as dots, so `security_protocol` becomes `security.protocol`. Batcher owns three of
those keys: `enable.auto.commit` (false, for the commit-after-batch behavior above),
`bootstrap.servers`, and `group.id`, which come from the named arguments. Everything else in
the librdkafka configuration surface is available this way.

:::{dropdown} A SASL_SSL read against Confluent Cloud
```python
# docs: skip
secure = bt.read.kafka(
    "clicks",
    bootstrap_servers="pkc-xxxxx.us-east-1.aws.confluent.cloud:9092",
    group="clickstream-etl",
    security_protocol="SASL_SSL",
    sasl_mechanisms="PLAIN",
    sasl_username="<key>",
    sasl_password="<secret>",
)
```
:::

## Writing

The stream is a `Dataset`, so it goes to any sink. There is no Kafka sink, so publishing back
to a topic is a `for_each_batch` over a producer you own.

::::{tab-set}

:::{tab-item} To a sink

A `trigger` sets the cadence:

```python
# docs: skip
q = (
    bt.read.kafka("clicks", bootstrap_servers="broker-1:9092", group="bronze")
    .with_watermark("ts", "10 minutes")
    .write.delta(
        "lake/bronze/clicks",
        trigger=bt.Trigger.processing_time("30 seconds"),
        checkpoint="/var/lib/batcher/ckpt/bronze-clicks",
        query_name="bronze-clicks",
    )
)
q.await_termination()
```

The checkpoint directory is SQLite plus Arrow IPC on a real filesystem, a local disk or a
mounted volume. It is not an object-store URI. Give the query a stable `query_name`: the Delta
sink writes one transaction id per micro-batch under that name, and that is what makes a
replayed batch a no-op instead of a duplicate.
:::

:::{tab-item} Back to Kafka

If a pipeline has to publish results, hand each micro-batch to a producer you own:

```python
# docs: skip
from confluent_kafka import Producer

producer = Producer({"bootstrap.servers": "broker-1:9092"})

def publish(table, batch_id):
    for payload in table.column("value").to_pylist():
        producer.produce("enriched", value=payload)
    producer.flush()

enriched.write.for_each_batch(publish, trigger=bt.Trigger.processing_time("5 seconds"))
```

`for_each_batch` gets the whole Arrow table, so the per-message loop is at the edge, in your
producer call, not in the engine.
:::

::::

:::{note}
`collect()` on a Kafka dataset raises `PlanError`. The source is unbounded; there is nothing
to materialize. Use `iter_batches()`, a write with a trigger, or bound it with
`Trigger.available_now()`.
:::

## See also

- {doc}`Streaming <../user-guide/streaming>`: triggers, watermarks, output modes, checkpoints.
- {doc}`Kafka ETL <../examples/streaming/kafka-etl>`: this connector end to end, decode to sink.
- {doc}`Exactly-once sink <../examples/streaming/exactly-once-sink>`: what the stable
  `query_name` above is buying you.
- {doc}`Reading and writing <../api/io>`: the full reader/writer surface.
- {doc}`Event Hubs <eventhubs>`: its Kafka protocol endpoint works here, without the Azure SDK.
- {doc}`Kinesis <kinesis>`: the same broker schema, a different shard model.
