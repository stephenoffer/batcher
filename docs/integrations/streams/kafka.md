# Kafka

This page covers reading from and writing to Apache Kafka. {py:meth}`bt.read.kafka(topic) <batcher.api.io_namespace.reader.Reader.kafka>` consumes a topic as an unbounded {py:class}`Dataset <batcher.Dataset>`, with one reader per partition across the cluster, and {py:meth}`ds.write.kafka(topic) <batcher.api.io_namespace.writer.Writer.kafka>` publishes each micro-batch back to a topic. Both sides use Spark's option names and column contract, so a ported job keeps its offsets, rate limits, and schema.

The following table summarizes the connector:

| | |
| --- | --- |
| Read | `bt.read.kafka(topic)` |
| Write | `ds.write.kafka(topic)`, one message per row. At-least-once. |
| Extra | `pip install 'batcher-engine[kafka]'` |
| Parallelism | One split per topic partition |
| Decoding | Raw bytes, or typed columns with `value_format=` (Avro, JSON, Protobuf, text) |
| Restart | Batcher's checkpoint, applied on partition assignment; the group offset is the fallback |

```bash
pip install 'batcher-engine[kafka]'
```

That pulls `confluent-kafka`. Without it, construction raises a {py:exc}`BackendError <batcher.BackendError>` saying so.

## Read a topic

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

Every broker source in Batcher (Kafka, Kinesis, Pulsar, Pub/Sub, Event Hubs) delivers the same six columns, plus an optional `headers` column:

| Column | Type | Meaning |
| --- | --- | --- |
| `key` | binary | The message key, or null if unkeyed |
| `value` | binary | The raw payload, undecoded |
| `partition` | int64 | The topic partition |
| `offset` | int64 | The Kafka offset |
| `timestamp` | int64 | Milliseconds since the Unix epoch |
| `topic` | string | The topic name |

Without a declared format the payload stays opaque bytes, and decoding is your first transformation. Write it as ordinary expressions and it runs in Rust rather than in a Python loop:

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
        "key": [b"u1", b"u2", b"u1"],
        "value": [
            b'{"user":"u1","amount":10}',
            b'{"user":"u2","amount":5}',
            b'{"user":"u1","amount":7}',
        ],
        "partition": [0, 0, 1],
        "offset": [11, 12, 4],
        "timestamp": [1700000000000, 1700000001000, 1700000002000],
        "topic": ["orders"] * 3,
    },
    schema=schema,
)

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

That's the right shape for an ad-hoc look at a JSON topic. For anything long-running,
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

## Choose where a query starts

`starting_offsets` is Spark's `startingOffsets`: `"earliest"` (the default), `"latest"`, or
an explicit `{partition: offset}` map. Spark's nested `{"topic": {"0": 123}}` form and its
`-2`/`-1` sentinels are accepted too, so a map copied out of a Spark job keeps meaning what
it meant.

```python
# docs: skip
bt.read.kafka("clicks", bootstrap_servers="broker-1:9092", starting_offsets="latest")
bt.read.kafka("clicks", bootstrap_servers="broker-1:9092", starting_offsets={0: 4096})
```

It applies only to a first run. Once a query has a checkpoint, the recorded position
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

## Read a bounded offset range

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
`ending_offsets="latest"` reads to the head of each partition as of the first poll, so a
partition that keeps growing during the read does not extend it and the same command run
twice covers the same rows.

A range read assigns every partition of the topic rather than joining the consumer group. A group hands a consumer whichever partitions a rebalance decides, so a subscribed read would stop at the end of its own partitions and omit the rest of the range.

## Rate-limit a micro-batch

A backlogged topic hands over as much as one poll allows, and two bounds ask different questions. `max_offsets_per_trigger` caps the message count, 16,384 by default. `max_bytes_per_trigger` caps the payload size, 128 MiB by default, which is the one that decides whether a
micro-batch fits in memory when a message can be a megabyte. Both are Spark's spellings, `poll_size` and `poll_bytes` are the native names for the same bounds, and the two compose:

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
the budget runs out and starts from a different partition next time. Rotating matters because a partition that is never reached never advances its event time, and the
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
and the nested column costs on every message of every poll. A message that carried none reads as `null` rather than as an empty list.

## Consumer groups and restarts

:::{warning}
`group=` defaults to `"batcher"`. Two unrelated pipelines against the same topic with that
default land in the same consumer group, and Kafka will happily split the partitions between
them. Each pipeline then sees half the data and neither complains.
:::

Give every query its own group id, and keep it stable across restarts, because that id *is*
the offset bookmark.

Batcher disables `enable.auto.commit` and advances the group only after a micro-batch is published, never when it's merely polled. A crash in between re-delivers the batch, which
an idempotent sink absorbs. The failure mode is always a duplicate, never a gap.

The figure puts that ordering beside the split model. Before a micro-batch is published, Batcher writes the consumed position ahead to its own checkpoint, and the group commit comes last.

![Left, a topic's partitions P0, P1, and P2 are each assigned to their own reader with its own consumer, one split per partition, so read parallelism equals the partition count. The readers' messages feed four steps per micro-batch, in order: poll, with no commit at poll time; write ahead the consumed position as offsets to the checkpoint, the source of truth; publish the micro-batch's rows to the sink, which should be idempotent; and commit the group offset synchronously, after publish, to the consumer group, which is only the fallback. On restart, each partition resumes at its checkpointed offset plus 1. A partition with no checkpointed position keeps the group's committed offset, or starting_offsets for a new group. A crash between publish and commit replays a batch, a duplicate and never a gap.](/_static/diagrams/kafka_offsets.svg)

The restart path follows from that. Batcher's own checkpoint, not the group offset, is the
source of truth. Under explicit `partitions=`, which is what the distributed split path uses,
the consumer already owns its partitions and repositions immediately to `offset + 1`. Under a
plain group subscription the partitions aren't known until the group assigns them, so the
resume happens in the assignment callback: each partition Kafka hands over is rewound to the
checkpointed position before reading starts. A partition with no checkpointed position keeps the group's committed offset, or the `starting_offsets` position for a new group.

:::{note}
`starting_offsets` defaults to `"earliest"`, so a brand-new group id starts at the beginning of the retained log, not at the tip. Pointing a fresh group at a topic with a week of retention replays the week. Pass `starting_offsets="latest"` if that isn't what you meant.
:::

## How it parallelizes

A {py:class}`Source <batcher.io.Source>` divides into {py:class}`Split <batcher.io.Split>` objects, and a split is the unit of read parallelism. Kafka's split is the topic-partition. `splits()` asks the broker for the topic's partition list and returns
one split per partition, each of which rebuilds a consumer on its worker with an explicit
assignment for that one partition.

Read parallelism therefore scales with the partition count. A 48-partition topic can read on 48 workers at once, and a single-partition topic reads on one, so if ingest throughput is the bottleneck, raise the topic's partition count.

You can also pin a reader to a subset yourself:

```python
# docs: skip
shard = bt.read.kafka(
    "clicks", bootstrap_servers="broker-1:9092", group="etl", partitions=[0, 1, 2]
)
```

## Security and client configuration

Anything else you pass through goes to the `confluent-kafka` consumer config with underscores
rewritten as dots, so `security_protocol` becomes `security.protocol`. Batcher owns four of those keys. `enable.auto.commit` is false, for the commit-after-batch behavior above. `bootstrap.servers` and `group.id` come from the named arguments. `auto.offset.reset` comes from `starting_offsets`, unless you pass `auto_offset_reset=` explicitly, which overrides it. Everything else in the librdkafka configuration surface is available this way.

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

## Write a stream

The stream is a `Dataset`, so it goes to any streaming sink, including back to Kafka.

::::{tab-set}

:::{tab-item} To a sink

A `trigger` sets the cadence:

```python
# docs: skip
q = (
    bt.read.kafka("clicks", bootstrap_servers="broker-1:9092", group="bronze")
    .write.delta(
        "lake/bronze/clicks",
        trigger=bt.Trigger.processing_time("30 seconds"),
        checkpoint="/var/lib/batcher/ckpt/bronze-clicks",
        query_name="bronze-clicks",
    )
)
q.await_termination()
```

The checkpoint can be a local path or an object-store URI such as `s3://` or `gs://`. A node-local directory is lost with the node, so use durable storage for a production query. Give the query a stable `query_name`: the Delta
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
broker rejection fails the query instead of silently dropping records. Delivery is at-least-once: a replayed micro-batch republishes its rows, so the consumer must be
idempotent or dedup on the key. Kafka's transactional produce is the only way to do better, and it requires the consumer to read committed-only. Spark's Kafka sink makes the same tradeoff.

`value_format=`, `schema_registry=`, and the other payload options work on the write too, encoding a struct `value` column before it's produced. See {doc}`/integrations/streams/payload-formats`.

A pipeline that needs to publish through a producer it configures itself can still use
{py:meth}`for_each_batch <batcher.api.io_namespace.writer.Writer.for_each_batch>`, which gets the whole Arrow table so the per-message loop stays at the edge.
:::

::::

:::{note}
{py:meth}`collect() <batcher.Dataset.collect>` on a live Kafka dataset raises {py:exc}`PlanError <batcher.PlanError>`, because an unbounded source has nothing to materialize. A read with `ending_offsets=` is bounded and collects normally. Use {py:meth}`iter_batches() <batcher.Dataset.iter_batches>`, a write with a trigger, or bound it with
{py:meth}`Trigger.available_now() <batcher.Trigger.available_now>`.
:::

## Requirements and limitations

Every query left on the default `group="batcher"` joins one consumer group, so give each query its own. A topic's partition count caps its read parallelism.

The Kafka sink delivers at-least-once, so consumers of its output must be idempotent or dedup on the key. `fail_on_data_loss=False` skips rows that aged out of retention, with a warning in the log.

Client options such as `sasl_password` go to librdkafka as given. They aren't resolved as secret references, so read them from your own secret store before building the source.

## See also

- {doc}`Streaming </user-guide/moving-data/streaming>`: triggers, watermarks, output modes, checkpoints.
- {doc}`Payload formats </integrations/streams/payload-formats>`: decoding Avro, JSON, and Protobuf payloads.
- {doc}`Kafka ETL </cookbook/streaming/kafka-etl>`: this connector end to end, decode to sink.
- {doc}`Exactly-once sink </cookbook/streaming/exactly-once-sink>`: what the stable
  `query_name` above is buying you.
- {doc}`Reading and writing </api/relational/io>`: the full reader/writer surface.
- {doc}`Event Hubs </integrations/streams/eventhubs>`: its Kafka protocol endpoint works here, without the Azure SDK.
- {doc}`Kinesis </integrations/streams/kinesis>`: the same broker schema, a different shard model.
