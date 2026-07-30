# Event Hubs

`bt.read.eventhubs(hub)` consumes an Azure Event Hub as an unbounded `Dataset` over the AMQP
client. Read only. Batcher has no Event Hubs sink.

| | |
| --- | --- |
| **Read** | `bt.read.eventhubs(hub)`, or `bt.read.kafka` against port 9093 |
| **Write** | Not supported |
| **Extra** | `pip install 'batcher-engine[eventhubs]'` |
| **Parallelism** | One split per partition, fixed at hub creation |
| **Auth** | Connection string only. No `DefaultAzureCredential`, no managed identity. |
| **Restart** | None. The native reader re-applies `starting_position` on every restart. |

```bash
pip install 'batcher-engine[eventhubs]'
```

## Two ways in, and they are not equivalent

Every Event Hubs namespace at Standard tier or above speaks the Kafka protocol on port 9093.
`bt.read.kafka` works against it, needs no Azure SDK, and (this is the part that matters) gets
you a working seek-on-resume and a per-partition split assignment. Use the native reader when
you are already on the Azure SDK and would rather not add `confluent-kafka`, or when the
protocol endpoint is unavailable (Basic tier).

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
string. That is Azure's convention, not a typo.
:::

:::{tab-item} The native AMQP reader

```python
# docs: skip
import batcher as bt

events = bt.read.eventhubs(
    "telemetry",
    connection_str="Endpoint=sb://acme-ns.servicebus.windows.net/;SharedAccessKeyName=reader;SharedAccessKey=...",
    consumer_group="$Default",
    starting_position="-1",
    poll_size=1_000,
)
```

`connection_str` is the namespace-level connection string, or an entity-scoped one for the
hub. The hub name goes in the first argument, not in the connection string's `EntityPath`.
Only connection-string auth is wired. There is no hook for `DefaultAzureCredential` or a
managed identity today.
:::

::::

## The rows you get

`starting_position` defaults to `"-1"`, which is the beginning of the retained stream, so a
new query replays whatever retention holds (one to seven days on Standard). `"@latest"` starts
at the tip.

Rows arrive in the fixed broker schema (`key`, `value`, `partition`, `offset`, `timestamp`,
`topic`), with `partition` the Event Hubs partition id, `offset` the native offset, `timestamp`
the enqueued time in milliseconds, and `topic` the hub name.

## `value` must be UTF-8 text

:::{warning}
The reader converts each event with `body_as_str()` and re-encodes to bytes. A payload that is
not valid UTF-8, meaning Avro, Protobuf, or anything else binary, raises on decode rather than
arriving as opaque bytes. Every other broker source in Batcher hands you the raw payload. This
one does not.
:::

If your hub carries binary bodies, read it through the Kafka endpoint above, where the payload
passes through untouched.

## Decoding the payload

Because the body arrives as UTF-8 text in `value`, decoding is an ordinary expression, and it
runs in Rust rather than in a Python loop. The block below stands a local batch in for the hub,
using the same six columns the reader delivers, so the pipeline runs here as written:

```python
import batcher as bt
import pyarrow as pa
from batcher import col

schema = pa.schema([
    ("key", pa.binary()), ("value", pa.binary()), ("partition", pa.int64()),
    ("offset", pa.int64()), ("timestamp", pa.int64()), ("topic", pa.string()),
])
batch = pa.record_batch({
    "key": [b"dev-1", b"dev-2", b"dev-1"],
    "value": [b'{"device":"dev-1","temp_c":21}', b'{"device":"dev-2","temp_c":25}',
              b'{"device":"dev-1","temp_c":23}'],
    "partition": [0, 1, 0],                                # the Event Hubs partition id
    "offset": [4021, 991, 4022],                            # the native offset
    "timestamp": [1700000000000, 1700000001000, 1700000002000],
    "topic": ["telemetry"] * 3,                             # the hub name
}, schema=schema)

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
See {doc}`../user-guide/streaming`.

## How it parallelizes

A `Source` divides into `Split`s, and each split is a unit of read parallelism. The split here
is the partition: `splits()` calls `get_partition_ids()` and returns one split per partition,
each rebuilt on its worker scoped to that one id.

Partition count is fixed at hub creation and cannot be raised on an existing hub (Standard
tier), so it is your read-parallelism ceiling and you have to pick it up front. Four partitions
means four workers can read; the fifth has nothing to do.

## Restart semantics, before you rely on a checkpoint

:::{important}
The native reader has no working resume. It records a checkpointed position as every other
broker source does, but nothing consults it: each poll re-derives its consumer from
`starting_position`. A restarted query starts from `starting_position` again, not from where it
stopped. Concretely, with the default `"-1"`, a restart replays the whole retained stream.
:::

Live with it one of three ways:

1. Use the Kafka endpoint at the top of this page, where consumer-group offsets and the
   checkpointed seek both work.
2. Make the sink idempotent and let the replay wash out. A Delta sink with a stable
   `query_name` commits one transaction per micro-batch and recognizes a replayed batch, and
   `drop_duplicates_within_watermark` handles the rest.
3. Set `starting_position` on restart from a position you tracked yourself. Workable, but you
   are now doing the checkpoint's job by hand.

:::{dropdown} What the reader is coupled to underneath
Azure's own Blob checkpoint store is not used. Neither is the SDK's public receive loop: the
source reaches into a private method on `EventHubConsumerClient` to create a per-partition
consumer, which is a real coupling to the SDK's internals. Pin `azure-eventhub`, and test
before you upgrade it.
:::

## Writing the stream out

```python
# docs: skip
q = (
    bt.read.eventhubs("telemetry", connection_str="Endpoint=sb://...", poll_size=1_000)
    .write.delta(
        "lake/bronze/telemetry",
        trigger=bt.Trigger.processing_time("30 seconds"),
        checkpoint="/var/lib/batcher/ckpt/bronze-telemetry",
        query_name="bronze-telemetry",
    )
)
q.await_termination()
```

The checkpoint directory is SQLite plus Arrow IPC on a real filesystem, not an `abfss://` URI.
It buys you sink-side idempotency, not source-side resume, for the reason above.

`collect()` raises `PlanError` on an unbounded source. Use `iter_batches()`, a triggered write,
or `bt.Trigger.available_now()`.

## Failure modes worth knowing

A new consumer is constructed on every poll rather than held open. On a busy hub that is AMQP
link setup in the hot loop, and it costs you throughput.

Azure allows five readers per consumer group per partition. Batcher's per-partition split
assignment is one reader each, which is fine, until you run two queries on the same
`consumer_group` and start crowding it. Give each pipeline its own consumer group.

Throughput units throttle you. Ingress and egress are capped by the namespace's TUs (1 MB/s in
and 2 MB/s out, per TU). Exceeding egress gets you `ServerBusyError`, not a slower read.

## See also

- {doc}`Kafka <kafka>`: the protocol-compatible path, and the payload-decoding example.
- {doc}`Streaming <../user-guide/streaming>`: triggers, watermarks, dedup, checkpointing.
- {doc}`Exactly-once sink <../examples/streaming/exactly-once-sink>`: the idempotent sink that
  option 2 above leans on.
- {doc}`Custom connectors <../user-guide/custom-connectors>`: the `Source`/`Split` protocol.
- {doc}`Reading and writing <../api/io>`: the full reader/writer surface.
- {doc}`Pulsar <pulsar>`: the other broker whose checkpoint does not drive a seek.
