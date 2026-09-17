# Kinesis

This page covers reading AWS Kinesis Data Streams. {py:meth}`bt.read.kinesis(stream_name) <batcher.api.io_namespace.reader.Reader.kinesis>` consumes a stream as an unbounded {py:class}`Dataset <batcher.Dataset>` over `boto3` and the `GetRecords` API, with one reader per shard, exact resume from the stored sequence number, and resharding followed without a restart.

The following table summarizes the connector:

| | |
| --- | --- |
| Read | `bt.read.kinesis(stream_name)` |
| Write | No sink. Write the stream to Delta or another streaming sink. |
| Extra | `pip install 'batcher-engine[kinesis]'` |
| Parallelism | One split per shard |
| Credentials | The ambient `boto3` chain. There is no credential keyword. |
| Restart | The raw sequence number per shard, re-obtained with `AFTER_SEQUENCE_NUMBER` |

```bash
pip install 'batcher-engine[kinesis]'
```

## Read a stream

Kinesis caps the `GetRecords` `Limit` at 10,000 records, so `poll_size` defaults to 10,000 and a larger value is clamped rather than sent. A smaller value passes through untouched, and is worth setting when you want tighter micro-batches:

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

`GetRecords` also caps a single response at 10 MB, so large records return fewer rows than you asked for.

Credentials come from the ambient `boto3` chain: environment, profile, instance role, IRSA.
There's no credential keyword. If `boto3` can find them, so can Batcher.

Pass `value_format=` to decode the `Data` blob in the source, as described in {doc}`/integrations/streams/payload-formats`.

## The rows you get

Every broker source shares one fixed schema: `key`, `value`, `partition`, `offset`, `timestamp`, `topic`. The following table shows what each column holds for Kinesis:

| Column | What Kinesis puts there |
| --- | --- |
| `key` | The record's partition key, UTF-8 encoded |
| `value` | The record `Data` blob, raw bytes, undecoded |
| `partition` | The shard's own number, parsed out of its `ShardId` |
| `offset` | An int64 projection of the sequence number |
| `timestamp` | `ApproximateArrivalTimestamp`, in milliseconds |
| `topic` | The stream name |

The two that surprise people are `partition` and `offset`. `partition` is the number embedded
in the shard's `ShardId`, so `shardId-000000000005` reads as `5`. Kinesis assigns that number
once and never reuses it, which is what keeps a checkpointed position pointing at the same shard
across a reshard. A `ShardId` that does not fit the format falls back to a `sha256` digest,
stable across processes and workers. `offset` is the sequence number, a large decimal string, reduced modulo 2^63 so it fits an int64 column. It's stable across runs and workers, but it isn't the sequence number, you can't hand it back to AWS, and it isn't safe to sort by. The real sequence number is kept out of band as the resume token that recovery uses, and doesn't appear as a column.

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
        "key": [b"cust-1", b"cust-2", b"cust-1"],
        "value": [b'{"amount": 10}', b'{"amount": 5}', b'{"amount": 7}'],
        "partition": [0, 1, 0],
        "offset": [4956, 4957, 4958],
        "timestamp": [1700000000000, 1700000001000, 1700000002000],
        "topic": ["payments"] * 3,
    },
    schema=schema,
)

records = bt.from_batches(lambda: iter([batch]), schema)
per_shard = records.group_by("partition").agg(records_read=col("offset").count())
d = per_shard.to_pydict()
print(sorted(zip(d["partition"], d["records_read"], strict=True)))
```

```text
[(0, 2), (1, 1)]
```

Grouping by `partition` is the cheapest way to see whether your producer's partition
key is spreading traffic or piling one shard high.

## How it parallelizes

A shard is a split. `splits()` calls `ListShards` and returns one split per shard, and each worker drives a shard iterator of its own. Parallelism is your shard count. A single reader that owns several shards fetches them concurrently, up to 16 threads.

The per-shard read ceiling is AWS's: 2 MB per second and five `GetRecords` calls per second, shared across every consumer of that shard. Batcher reads through the shared-throughput API, not enhanced fan-out, so a Lambda or Firehose on the same shard splits that budget with it. When the budget runs out, `ProvisionedThroughputExceededException` is treated as back-pressure and the shard is retried on the next poll rather than failing the query. Add shards, or move the other consumers to enhanced fan-out, if the throttling persists.

## Resharding

`ListShards` is paginated, so every shard is discovered however many there are, and the listing is cached. A reshard invalidates it rather than outliving it: a shard closes when its children
replace it, `GetRecords` says so by returning no next iterator, and the reader retires the
parent, drops the cached listing, and reads the fresh one's lineage to adopt the children. So a
running query follows a split or a merge without a restart, and it costs one `ListShards` per
reshard.

That works because `partition` is the shard's own number rather than its position in a list. A
checkpointed sequence number therefore still names the shard it was taken from, whatever the
listing looks like afterwards.

## Choose a starting position and restart

`iterator_type` defaults to `"TRIM_HORIZON"`, so a fresh query replays the stream's whole retention window. `"LATEST"` starts at the tip and drops the backlog. Pick deliberately. `starting_position`
is accepted as well, with `"earliest"` and `"latest"` mapping onto those two, so a pipeline that
reads several brokers can spell the choice one way throughout.

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
live = bt.read.kinesis("payments", region="us-east-1", poll_size=5_000, iterator_type="LATEST")
```
:::

::::

With `checkpoint=` set, recovery is exact. The raw sequence number is stored per shard and
the shard iterator is re-obtained with `AFTER_SEQUENCE_NUMBER`, so no record is replayed and
none is skipped.

:::{dropdown} A checkpointed write into a bronze Delta table
```python
# docs: skip
q = bt.read.kinesis("payments", region="us-east-1", poll_size=5_000).write.delta(
    "lake/bronze/payments",
    trigger=bt.Trigger.processing_time("30 seconds"),
    checkpoint="/var/lib/batcher/ckpt/bronze-payments",
    query_name="bronze-payments",
)
```
:::

The checkpoint can be a local path or an `s3://` URI. The `query_name` is the Delta transaction id prefix, so keep it stable across restarts or a replayed micro-batch writes twice.

A shard iterator expires after five minutes. A trigger interval longer than that, or a shard that goes quiet while its siblings are polled, gets `ExpiredIteratorException`, and the reader re-obtains the iterator from the last delivered sequence number instead of failing the query.

## Requirements and limitations

Empty polls are normal. `GetRecords` returns an empty record list constantly on a quiet shard, and the poll loop skips those and keeps going. An idle stream produces no batches, which isn't an end-of-stream.

{py:meth}`collect() <batcher.Dataset.collect>` raises, because the source is unbounded. Use {py:meth}`iter_batches() <batcher.Dataset.iter_batches>` or a triggered
write, or bound the read with {py:meth}`bt.Trigger.available_now() <batcher.Trigger.available_now>`.

KPL aggregation isn't unpacked. Producers using the Kinesis Producer Library pack several user records into one Kinesis record inside a protobuf envelope, and Batcher hands you that envelope as the `value`. De-aggregate it in `map_batches` if your producers use it.

Only the ambient `boto3` credential chain and the `region` option reach the client. There's no endpoint override or credential keyword.

## See also

- {doc}`Streaming </user-guide/moving-data/streaming>`: triggers, watermarks, and checkpointing.
- {doc}`Windowed aggregation </cookbook/streaming/windowed-aggregation>`: what to do with the
  records once they land.
- {doc}`Exactly-once sink </cookbook/streaming/exactly-once-sink>`: the `query_name` contract
  the write above depends on.
- {doc}`Reading and writing </api/relational/io>`: the full reader/writer surface.
- {doc}`Kafka </integrations/streams/kafka>`: the same broker schema and the same decode pattern.
- {doc}`Pub/Sub </integrations/streams/pubsub>`: the Google Cloud broker.
- {doc}`Payload formats </integrations/streams/payload-formats>`: decoding Avro, JSON, and Protobuf in the source.
