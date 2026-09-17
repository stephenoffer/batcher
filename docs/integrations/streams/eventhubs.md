# Event Hubs

This page covers reading Azure Event Hubs. Batcher reaches a hub two ways: {py:meth}`bt.read.eventhubs(hub) <batcher.api.io_namespace.reader.Reader.eventhubs>` over the native AMQP client, or {py:meth}`bt.read.kafka <batcher.api.io_namespace.reader.Reader.kafka>` against the hub's Kafka endpoint. Both read one partition per worker and resume from a checkpoint.

The following table summarizes the connector:

| | |
| --- | --- |
| Read | {py:meth}`bt.read.eventhubs(hub) <batcher.api.io_namespace.reader.Reader.eventhubs>`, or {py:meth}`bt.read.kafka <batcher.api.io_namespace.reader.Reader.kafka>` against port 9093 |
| Write | No sink. Write the stream to Delta or another streaming sink. |
| Extra | `pip install 'batcher-engine[eventhubs]'` |
| Parallelism | One split per partition, fixed at hub creation |
| Auth | Connection string only. No `DefaultAzureCredential`, no managed identity. |
| Restart | The checkpointed per-partition offset; `starting_position` when there is none |

```bash
pip install 'batcher-engine[eventhubs]'
```

## Choose a reader

Every Event Hubs namespace at Standard tier or above speaks the Kafka protocol on port 9093, and `bt.read.kafka` works against it with no Azure SDK at all. Prefer it. Both readers split per
partition and both honour a checkpoint, so what the Kafka path adds is a consumer-group offset
to fall back on when there is no checkpoint, a bounded offset-range read for a backfill, and a
client Batcher is not reaching into the private internals of. Take the native reader when you
are already on the Azure SDK and would rather not add `confluent-kafka`, or when the protocol
endpoint is unavailable (Basic tier).

::::{tab-set}

:::{tab-item} The Kafka endpoint (preferred)

```python
# docs: skip
import batcher as bt

events = bt.read.kafka(
    "telemetry",
    bootstrap_servers="acme-ns.servicebus.windows.net:9093",
    group="telemetry-etl",
    security_protocol="SASL_SSL",
    sasl_mechanisms="PLAIN",
    sasl_username="$ConnectionString",
    sasl_password="Endpoint=sb://acme-ns.servicebus.windows.net/;SharedAccessKeyName=...",
)
```

The username is the literal string `$ConnectionString`; the password is the whole connection
string. That's Azure's convention, not a typo.
:::

:::{tab-item} The native AMQP reader

```python
# docs: skip
import batcher as bt

events = bt.read.eventhubs(
    "telemetry",
    connection_str="Endpoint=sb://acme-ns.servicebus.windows.net/;SharedAccessKeyName=reader;SharedAccessKey=...",
    consumer_group="$Default",
    starting_position="earliest",
    poll_size=1_000,
)
```

`connection_str` is the namespace-level connection string, or an entity-scoped one for the
hub. The hub name goes in the first argument, not in the connection string's `EntityPath`.
Only connection-string auth is wired. There's no hook for `DefaultAzureCredential` or a managed identity.
:::

::::

## The rows you get

`starting_position` defaults to `"earliest"`, the beginning of the retained stream, so a new query replays whatever retention holds. `"latest"` starts at
the tip. Those are the words every broker in this section shares. Event Hubs' own sentinels `"-1"` and `"@latest"` are accepted too, as is an explicit offset string you recorded earlier.

Rows arrive in the fixed broker schema (`key`, `value`, `partition`, `offset`, `timestamp`,
`topic`), with `partition` the Event Hubs partition id, `offset` an int64 projection of the native offset, `timestamp` the enqueued time in milliseconds, and `topic` the hub name. The native offset string itself is kept as the resume token.

## What lands in `value`

An ordinary producer sends an AMQP `DATA` body, and that reaches `value` as the raw bytes it
was, undecoded, exactly as every other broker source here delivers a payload. A multi-section
event has its sections joined. Protobuf, an Avro frame and a compressed blob all pass through
intact; nothing is decoded or re-encoded on the way.

The exception is a structured AMQP body, `SEQUENCE` or `VALUE`, which has no byte encoding of
its own. Those are rendered as JSON, so the `.json` accessor can parse them downstream.

## Decoding the payload

Pass `value_format=` to decode the payload in the source, as described in {doc}`/integrations/streams/payload-formats`. Otherwise decoding is your first transformation. Write it as expressions and it runs in Rust rather than in a Python loop. The block below stands a local batch in for the hub,
using the same six columns the reader delivers, so the pipeline runs here as written:

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
        "key": [b"dev-1", b"dev-2", b"dev-1"],
        "value": [
            b'{"device":"dev-1","temp_c":21}',
            b'{"device":"dev-2","temp_c":25}',
            b'{"device":"dev-1","temp_c":23}',
        ],
        "partition": [0, 1, 0],  # the Event Hubs partition id
        "offset": [4021, 991, 4022],  # projected from the native offset
        "timestamp": [1700000000000, 1700000001000, 1700000002000],
        "topic": ["telemetry"] * 3,  # the hub name
    },
    schema=schema,
)

# Stand in for the hub; the pipeline below is what you run against the real one.
events = bt.from_batches(lambda: iter([batch]), schema)

readings = events.select(
    col("value").cast("string").json.extract_string("$.device").alias("device"),
    col("value").cast("string").json.extract_int("$.temp_c").alias("temp_c"),
)
print(readings.group_by("device").agg(peak=col("temp_c").max()).sort("device").to_pydict())
# {'device': ['dev-1', 'dev-2'], 'peak': [23, 25]}
```

Against the live hub the only line that changes is the source, and the streaming controls
(triggers, watermarks, checkpoints) attach to that source rather than to the transformation.
See {doc}`/user-guide/moving-data/streaming`.

## How it parallelizes

A {py:class}`Source <batcher.io.Source>` divides into {py:class}`Split <batcher.io.Split>` objects, and each split is a unit of read parallelism. The split here
is the partition: `splits()` calls `get_partition_ids()` and returns one split per partition,
each rebuilt on its worker scoped to that one id. A partition's consumer is opened once and reused across polls, so its prefetch survives and a short trigger interval doesn't pay for an AMQP link negotiation per partition per poll.

On Standard tier the partition count is fixed when the hub is created, so choose it up front with your read parallelism in mind. Four partitions means four workers can read.

## Restarts

With a `checkpoint=` set, the native reader resumes where it stopped. Each partition's
checkpointed offset is what its consumer is opened at, and a recovery drops the cached consumer
so the next poll reopens it there. A live offset is exclusive, so the resume is strictly after
the last delivered event. Without a checkpoint, every restart begins at `starting_position`,
which with the default `"earliest"` replays the whole retained stream.

The delivery guarantee is at-least-once: a micro-batch read but not yet committed when the query stopped is read again on restart. Make the sink idempotent: a Delta sink
with a stable `query_name` commits one transaction per micro-batch and recognizes a replayed
one, and {py:meth}`drop_duplicates_within_watermark <batcher.Dataset.drop_duplicates_within_watermark>`
handles the rest.

Azure's Blob checkpoint store isn't used. Batcher's own checkpoint is the resume point.

## Writing

```python
# docs: skip
q = bt.read.eventhubs("telemetry", connection_str="Endpoint=sb://...", poll_size=1_000).write.delta(
    "lake/bronze/telemetry",
    trigger=bt.Trigger.processing_time("30 seconds"),
    checkpoint="/var/lib/batcher/ckpt/bronze-telemetry",
    query_name="bronze-telemetry",
)
q.await_termination()
```

The checkpoint holds the per-partition offsets the reader resumes from, and it can be a local path or an object-store URI. The stable `query_name` makes the Delta sink recognize a replayed micro-batch.

{py:meth}`collect() <batcher.Dataset.collect>` raises {py:exc}`PlanError <batcher.PlanError>` on an unbounded source. Use {py:meth}`iter_batches() <batcher.Dataset.iter_batches>`, a triggered write,
or {py:meth}`bt.Trigger.available_now() <batcher.Trigger.available_now>`.

## Message metadata

Event Hubs calls them *application properties*, and Kafka calls them headers. They're the same idea, so they arrive under the same option and in the same column type,
`array<struct<key:string,value:binary>>`:

```python
# docs: skip
events = bt.read.eventhubs("hub", connection_str="<your-connection-string>", include_headers=True)
traced = events.with_columns(trace=bt.col("headers").list.get(0).struct.field("value"))
```

It is off by default because the nested column costs on every message of every poll. Values
are carried as bytes whatever the client hands back, and a message that carried none reads
as `null` rather than as an empty list.

## Requirements and limitations

Azure allows five readers per consumer group per partition. Batcher's split assignment is one reader per partition, so two queries on the same `consumer_group` start crowding that limit. Give each pipeline its own consumer group.

Throughput units cap ingress and egress for the namespace, and exceeding egress returns `ServerBusyError` rather than a slower read.

The native reader authenticates with a connection string only, and `connection_str` isn't resolved as a secret reference. The reader also calls a private method on `EventHubConsumerClient` to create per-partition consumers, so pin `azure-eventhub` and test before upgrading it.

## See also

- {doc}`Kafka </integrations/streams/kafka>`: the protocol-compatible path, and the payload-decoding example.
- {doc}`Streaming </user-guide/moving-data/streaming>`: triggers, watermarks, dedup, checkpointing.
- {doc}`Exactly-once sink </cookbook/streaming/exactly-once-sink>`: the idempotent sink a replayed micro-batch depends on.
- {doc}`Custom connectors </user-guide/moving-data/custom-connectors>`: the `Source`/`Split` protocol.
- {doc}`Reading and writing </api/relational/io>`: the full reader/writer surface.
- {doc}`Payload formats </integrations/streams/payload-formats>`: decoding Avro, JSON, and Protobuf in the source.
