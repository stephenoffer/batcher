# Kafka

{py:meth}`bt.read.kafka(topic) <batcher.api.io_namespace.reader.Reader.kafka>` consumes a Kafka topic as an unbounded {py:class}`Dataset <batcher.Dataset>`. It is a read path
only. Batcher has no Kafka sink, so producing back to Kafka goes through
{py:meth}`ds.write.for_each_batch <batcher.api.io_namespace.writer.Writer.for_each_batch>` with your own producer, covered at the end of this page.

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

That pulls `confluent-kafka`. Without it, construction raises a {py:exc}`BackendError <batcher.BackendError>` saying so.

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

That is the right shape for an ad-hoc look at a JSON topic. For anything long-running,
name the wire format instead and the source decodes it for you, so `value` arrives as a
typed column and the stream's schema is known before a message is polled:

```python
# docs: skip
orders = bt.read.kafka(
    "orders",
    bootstrap_servers="broker-1:9092",
    value_format="avro",
    schema_registry="http://schema-registry:8081",
)
totals = orders.group_by(bt.col("value").struct.field("user")).agg(
    total=bt.col("value").struct.field("amount").sum()
)
```

Avro, JSON, Protobuf and text are supported, with Confluent Schema Registry framing and a
policy for malformed records. See {doc}`Payload formats </integrations/streams/payload-formats>`.

## Where a query starts, and what happens when offsets age out

`starting_offsets` is Spark's `startingOffsets`: `"earliest"` (the default), `"latest"`, or
an explicit `{partition: offset}` map. Spark's nested `{"topic": {"0": 123}}` form and its
`-2`/`-1` sentinels are accepted too, so a map copied out of a Spark job keeps meaning what
it meant.

```python
# docs: skip
bt.read.kafka("clicks", bootstrap_servers="broker-1:9092", starting_offsets="latest")
bt.read.kafka("clicks", bootstrap_servers="broker-1:9092", starting_offsets={0: 4096})
```

It applies only to a **first** run. Once a query has a checkpoint, the recorded position
always wins, or every restart would rewind to the configured start and reprocess.

`fail_on_data_loss` is Spark's `failOnDataLoss` and defaults to `True`: if the offsets the
query wants have aged out of the log, the read fails rather than skipping to whatever is
left. Set it `False` to keep running through the gap. The skip is logged at warning level,
because a stream that says nothing here cannot be told apart from one that lost nothing.

:::{warning}
`fail_on_data_loss=False` is a decision to accept missing rows. Reach for it when the
alternative is a dead pipeline, not to quiet a recurring alert: a query that keeps hitting
it is falling behind retention, and the fix is more throughput or a longer retention.
:::

## Reading a bounded offset range

A topic is unbounded, so `collect()` on one refuses: it could never terminate. Declaring
`ending_offsets` makes the read a finite range instead, which is how you express a backfill,
a reprocess, or a one-off query over a window of history:

```python
# docs: skip
window = bt.read.kafka(
    "orders",
    bootstrap_servers="broker-1:9092",
    starting_offsets={0: 1_000, 1: 1_000},
    ending_offsets={0: 2_000, 1: 2_000},
)
window.count()
```

The end is exclusive, as in Spark: an end of 2000 reads up to and including offset 1999.
`ending_offsets="latest"` reads to the head of each partition as of query start, so a
partition that keeps growing during the read does not extend it and the same command run
twice covers the same rows.

A range read assigns every partition of the topic rather than joining the consumer group.
It has to: a group hands this consumer whichever partitions a rebalance decides, so a
subscribed read would stop at the end of *its* partitions and silently omit the rest of the
range.

## Rate-limiting a micro-batch

A backlogged topic hands over as much as one poll allows, and the two bounds that matter
ask different questions. `max_offsets_per_trigger` caps the message count;
`max_bytes_per_trigger` caps the payload size, which is the one that decides whether a
micro-batch fits in memory when a message can be a megabyte. Both are Spark's spellings and
both compose:

```python
# docs: skip
bt.read.kafka(
    "clicks",
    bootstrap_servers="broker-1:9092",
    max_offsets_per_trigger=50_000,
    max_bytes_per_trigger=64 << 20,
)
```

Both bounds hold on every broker connector, not only this one. A byte bound is what keeps a
poll inside Arrow's own limit as well as the machine's: a `binary` column carries 32-bit
offsets, so a batch past 2 GiB fails inside the array builder rather than at any boundary
you named.

On a connector that sweeps several partitions or shards in one poll, the sweep stops when
the budget runs out and starts from a different partition next time. Rotating matters more
than it looks: a partition that is never reached never advances its event time, and the
stream's watermark is the minimum across partitions, so a starved partition stalls the whole
query exactly as a silent one would.

## Message headers

Headers carry the metadata that is not the payload: a trace id, a schema-registry id, a
routing hint. `include_headers=True` adds a `headers` column typed exactly as Spark's
Kafka source types it, `array<struct<key:string,value:binary>>`:

```python
# docs: skip
events = bt.read.kafka("clicks", bootstrap_servers="broker-1:9092", include_headers=True)
traced = events.with_columns(trace=col("headers").list.get(0).struct.field("value"))
```

It is off by default for the same reason it is in Spark: most pipelines never read headers,
and the nested column costs on every message of every poll. A message that carried none
reads as `null` rather than as an empty list, so "this broker has no headers" and "this
message had none" stay distinguishable.

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

A {py:class}`Source <batcher.io.Source>` divides into {py:class}`Split <batcher.io.Split>`s, and a split is the unit of read parallelism. Kafka's split
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

The stream is a `Dataset`, so it goes to any sink, including back to Kafka.

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

`write.kafka` publishes one message per row. The column contract is Spark's: `value` is
required, `key`, `topic`, `partition`, and `headers` are optional, and both binary and
string are accepted for the payload columns.

```python
# docs: skip
q = enriched.select(
    key=col("user_id").cast("string"),
    value=col("payload"),
).write.kafka(
    "enriched",
    bootstrap_servers="broker-1:9092",
    trigger=bt.Trigger.processing_time("5 seconds"),
    compression_type="zstd",
)
```

Any further option is passed to `confluent-kafka` with underscores turned into dots, so
`compression_type="zstd"` sets `compression.type`.

Each micro-batch is flushed and acknowledged before the sink reports it written, so a
broker rejection fails the query instead of silently dropping records. Delivery is
**at-least-once**: a replayed micro-batch republishes its rows, so the consumer must be
idempotent or dedup on the key. Kafka's transactional produce is the only way to do better,
and it requires the consumer to read committed-only; Spark's Kafka sink makes the same
tradeoff.

A pipeline that needs to publish through a producer it configures itself can still use
{py:meth}`for_each_batch <batcher.api.io_namespace.writer.Writer.for_each_batch>`, which gets the whole Arrow table so the per-message loop stays at the edge.
:::

::::

:::{note}
{py:meth}`collect() <batcher.Dataset.collect>` on a Kafka dataset raises {py:exc}`PlanError <batcher.PlanError>`. The source is unbounded; there is nothing
to materialize. Use {py:meth}`iter_batches() <batcher.Dataset.iter_batches>`, a write with a trigger, or bound it with
{py:meth}`Trigger.available_now() <batcher.Trigger.available_now>`.
:::

## See also

- {doc}`Streaming </user-guide/moving-data/streaming>`: triggers, watermarks, output modes, checkpoints.
- {doc}`Payload formats </integrations/streams/payload-formats>`: decoding Avro, JSON, and Protobuf payloads.
- {doc}`Kafka ETL </cookbook/streaming/kafka-etl>`: this connector end to end, decode to sink.
- {doc}`Exactly-once sink </cookbook/streaming/exactly-once-sink>`: what the stable
  `query_name` above is buying you.
- {doc}`Reading and writing </api/relational/io>`: the full reader/writer surface.
- {doc}`Event Hubs </integrations/streams/eventhubs>`: its Kafka protocol endpoint works here, without the Azure SDK.
- {doc}`Kinesis </integrations/streams/kinesis>`: the same broker schema, a different shard model.
