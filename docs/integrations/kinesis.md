# Kinesis

`bt.read.kinesis(stream_name)` consumes an AWS Kinesis Data Stream as an unbounded `Dataset`,
over `boto3` and the classic `GetRecords` API. Read only. There is no Kinesis sink.

| | |
| --- | --- |
| **Read** | `bt.read.kinesis(stream_name)` |
| **Write** | Not supported |
| **Extra** | `pip install 'batcher-engine[kinesis]'` |
| **Parallelism** | One split per shard |
| **Credentials** | The ambient `boto3` chain. There is no credential keyword. |
| **Restart** | The raw sequence number per shard, re-obtained with `AFTER_SEQUENCE_NUMBER` |

```
pip install 'batcher-engine[kinesis]'
```

## Set `poll_size` before anything else

:::{warning}
The default `poll_size` is 16,384, inherited from the broker base. Kinesis caps the
`GetRecords` `Limit` at 10,000 records, so that default is over the line and the API rejects
it.
:::

Pass a legal one:

```python
# docs: skip
import batcher as bt

payments = bt.read.kinesis(
    "payments",
    region="us-east-1",
    poll_size=5_000,
    iterator_type="TRIM_HORIZON",
)
```

5,000 is a reasonable starting point. `GetRecords` also caps a single response at 10 MB, so
with fat records you will get fewer rows than you asked for regardless.

Credentials come from the ambient `boto3` chain: environment, profile, instance role, IRSA.
There is no credential keyword. If `boto3` can find them, so can Batcher.

## The rows you get

Every broker source shares one fixed schema: `key`, `value`, `partition`, `offset`,
`timestamp`, `topic`. For Kinesis they map like this.

| Column | What Kinesis puts there |
| --- | --- |
| `key` | The record's partition key, UTF-8 encoded |
| `value` | The record `Data` blob, raw bytes, undecoded |
| `partition` | The index of the shard in the discovered shard list |
| `offset` | The sequence number, reduced modulo 2⁶³ |
| `timestamp` | `ApproximateArrivalTimestamp`, in milliseconds |
| `topic` | The stream name |

The two that surprise people are `partition` and `offset`. `partition` is the *index* of the
shard in the discovered shard list, not the shard id. `offset` is the sequence number, which
is a large decimal string, reduced modulo 2⁶³ so it fits an int64 column. It stays monotonic
within a shard, but it is not the sequence number and you cannot hand it back to AWS. The
real sequence number is kept out of band as the resume token, which is what recovery uses; it
simply does not appear as a column.

```python
import batcher as bt
import pyarrow as pa
from batcher import col

schema = pa.schema([
    ("key", pa.binary()), ("value", pa.binary()), ("partition", pa.int64()),
    ("offset", pa.int64()), ("timestamp", pa.int64()), ("topic", pa.string()),
])
batch = pa.record_batch({
    "key": [b"cust-1", b"cust-2", b"cust-1"],
    "value": [b'{"amount": 10}', b'{"amount": 5}', b'{"amount": 7}'],
    "partition": [0, 1, 0],
    "offset": [4956, 4957, 4958],
    "timestamp": [1700000000000, 1700000001000, 1700000002000],
    "topic": ["payments"] * 3,
}, schema=schema)

records = bt.from_batches(lambda: iter([batch]), schema)
per_shard = records.group_by("partition").agg(records_read=col("offset").count())
d = per_shard.to_pydict()
print(sorted(zip(d["partition"], d["records_read"], strict=True)))
```

```text
[(0, 2), (1, 1)]
```

Grouping by `partition` like that is the cheapest way to see whether your producer's partition
key is spreading traffic or piling one shard high.

## How it parallelizes

A shard is a split. `splits()` calls `ListShards` and returns one `BrokerSplit` per shard;
each worker rebuilds a source scoped to its shard index and drives a shard iterator of its
own. Parallelism is therefore exactly your shard count, and the per-shard read ceiling (2 MB
per second, five `GetRecords` calls per second, shared across every consumer of that shard) is
the ceiling for a Batcher worker too.

:::{important}
That last point deserves emphasis. This is the *shared-throughput* API, not enhanced fan-out.
If a Lambda and a Firehose delivery stream are already reading the shard, you are splitting
2 MB/s with them, and `ProvisionedThroughputExceededException` is what you will see when the
budget runs out. Add shards, or move the other consumers to enhanced fan-out.
:::

## Resharding

:::{warning}
The shard list is fetched once and cached for the life of the source, and `ListShards` is read
without pagination, so a stream with more shards than one page returns will read only the
first page.
:::

Splits are addressed by *index into that list*. Split a shard, merge two, or scale a stream
while a query is running, and the index-to-shard mapping shifts under you. A resumed query can
end up reading a different shard than the one its checkpointed sequence number belongs to.

Restart the query after a resharding event. Do not reshard under a long-lived reader and
assume it followed.

## Restart semantics

`iterator_type` defaults to `"TRIM_HORIZON"`: a fresh query replays the entire retention
window (24 hours by default, up to 365 days if you have paid for it). `"LATEST"` starts at the
tip and drops the backlog. Neither is right for every job; pick deliberately.

::::{tab-set}

:::{tab-item} Replay the retention window
```python
# docs: skip
backfill = bt.read.kinesis(
    "payments", region="us-east-1", poll_size=5_000, iterator_type="TRIM_HORIZON"
)
```
:::

:::{tab-item} Start at the tip
```python
# docs: skip
live = bt.read.kinesis(
    "payments", region="us-east-1", poll_size=5_000, iterator_type="LATEST"
)
```
:::

::::

With `checkpoint=` set, recovery works properly here. The raw sequence number is stored per
shard and the shard iterator is re-obtained with `AFTER_SEQUENCE_NUMBER`, so no record is
replayed and none is skipped. Of the five broker sources, Kinesis has the sturdiest resume
path.

:::{dropdown} A checkpointed write into a bronze Delta table
```python
# docs: skip
q = (
    bt.read.kinesis("payments", region="us-east-1", poll_size=5_000)
    .write.delta(
        "lake/bronze/payments",
        trigger=bt.Trigger.processing_time("30 seconds"),
        checkpoint="/var/lib/batcher/ckpt/bronze-payments",
        query_name="bronze-payments",
    )
)
```
:::

The checkpoint directory is SQLite plus Arrow IPC on a local filesystem, not an S3 URI. The
`query_name` is the Delta transaction id prefix, so keep it stable across restarts or a
replayed micro-batch will write twice.

## Failure modes worth knowing

:::{warning}
A shard iterator is valid for five minutes. A worker that stalls longer than that (a slow
sink, a long GC pause, backpressure from a downstream credit stall) gets
`ExpiredIteratorException` on its next call. Keep micro-batches short.
:::

Empty polls are normal. `GetRecords` returns an empty record list constantly on a quiet shard;
the poll loop skips those and keeps going. An idle stream produces no batches. It is not an
end-of-stream.

`collect()` raises, because the source is unbounded. Use `iter_batches()` or a triggered
write, or bound the read with `bt.Trigger.available_now()`.

Records are bytes, and that includes KPL aggregation. Producers using the Kinesis Producer
Library pack several user records into one Kinesis record inside a protobuf envelope. Batcher
hands you that envelope undecoded; de-aggregate it in `map_batches` if your producers use it.

## See also

- [Streaming](../user-guide/streaming.md): triggers, watermarks, and checkpointing.
- [Windowed aggregation](../examples/streaming/windowed-aggregation.md): what to do with the
  records once they land.
- [Exactly-once sink](../examples/streaming/exactly-once-sink.md): the `query_name` contract
  the write above depends on.
- [Reading and writing](../api/io.md): the full reader/writer surface.
- [Kafka](kafka.md): the same broker schema and the same decode pattern.
- [Pub/Sub](pubsub.md): the other cloud broker, and the one that does not split.
