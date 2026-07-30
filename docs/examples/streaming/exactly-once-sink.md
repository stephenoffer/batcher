# Exactly-once sinks

Here is the sequence that duplicates your data. The engine records micro-batch 7's source
offsets in the offset log, processes the batch, writes it to the sink. Then the process is
killed before the commit log records it. On restart, recovery finds batch 7 in the offset
log but not in the commit log, so it replays it. The rows go out a second time.

:::{important}
That is not a bug. It is the only safe design: the alternative (commit the offset first)
loses the batch instead. The engine is deliberately at-least-once, and end-to-end
exactly-once is bought by the **sink** being idempotent about a replayed batch. If you
understand which of Batcher's sinks are, and how, you can reason about your delivery
semantics. If you don't, you will assume a guarantee you do not have.
:::

## The checkpoint is what makes a restart mean anything

`checkpoint=` is a directory holding three things: an offset log (per batch, per source,
the position read, written *ahead* of processing), a commit log (which batches reached
the sink), and state snapshots for stateful operators. Recovery is a pure function over
those logs: resume at the first uncommitted batch, seek each source to the position
recorded at the last committed batch, restore that batch's state.

Run a query twice against the same checkpoint. The second run has nothing to do:

::::{tab-set}
:::{tab-item} The first run

```python
import os
import tempfile

import batcher as bt

root = tempfile.mkdtemp()
out = os.path.join(root, "bronze")
ckpt = os.path.join(root, "_ckpt")


def run():
    source = bt.read.rate(rows_per_second=4, num_rows=8, pace=False)
    query = source.write(
        out,
        format="parquet",
        trigger=bt.Trigger.available_now(),
        checkpoint=ckpt,
        query_name="bronze-rate",
    )
    query.await_termination()
    return [(p.batch_id, p.num_input_rows) for p in query.recent_progress()]


print(run())
# [(0, 4), (1, 4)]
print(bt.read.parquet(out).count())
# 8
```

Two micro-batches, eight rows.
:::

:::{tab-item} The restart

```python
print(run())
# []
print(bt.read.parquet(out).count())
# 8
print(sorted(os.listdir(ckpt)))
# ['commits.sqlite', 'offsets.sqlite', 'state']
```

Zero micro-batches processed, eight rows on disk. The `rate` source is *replayable*: it
can snapshot a position and seek back to it, so recovery could tell the source it had
already consumed everything.
:::
::::

## Which sinks actually dedup a replayed batch

Sinks differ in what they deduplicate on, and that decides whether a replay is safe. The
key column below is the identity the sink compares:

| Sink | Dedup | Key |
| --- | --- | --- |
| `ds.write(path, format=...)` | by **position** | the file name, `part-batch00007.parquet` |
| `ds.write.delta(uri)` | by **transaction id** | a `txn` action of `(app_id, batch_id)` in the log |
| `for_each_batch` | none, but you are handed the `batch_id` | whatever idempotency key you build from it |
| `for_each`, `console`, `memory` | none | none |

**File sinks** (`ds.write(path, format="parquet")`) write one file per micro-batch, named
`part-batch00000.parquet`, `part-batch00001.parquet`, and so on. On a replay of batch 7,
the writer finds `part-batch00007.parquet` already there and skips it. Idempotence **by
position**.

**Delta** (`ds.write.delta(uri)`) commits each micro-batch as one transaction carrying a
`txn` action of `(app_id, batch_id)`, and checks the log for that pair before it writes
anything. A replayed batch finds its own transaction already recorded, writes no file, and
commits nothing. Idempotence **by transaction id**, which is strictly stronger: the log
ends up with exactly one commit per micro-batch no matter how often it was retried.

`app_id` is the `query_name` you passed, or a value derived from the destination table.
It must be stable across restarts, or the check will never find the previous run's
commits. Name your queries.

**`for_each_batch`, `for_each`, `console`, `memory`**: no dedup at all. `for_each_batch`
does receive the `batch_id`, which is the hook. It is the same id on a replay, so you can
use it as the idempotency key of your own upsert (a `MERGE` keyed on batch id, a Redis
`SETNX`, whatever your target supports). The engine hands you the identifier and steps
back; the guarantee is then yours to implement.

:::{dropdown} Rolling your own idempotent upsert on top of `batch_id`

```python
# docs: skip
def upsert(table, batch_id):
    with warehouse.transaction() as tx:
        if tx.already_applied("clicks", batch_id):
            return
        tx.merge("clicks", table, keys=["id"])
        tx.record_applied("clicks", batch_id)


bt.read.kafka("clicks").write.for_each_batch(
    upsert,
    trigger=bt.Trigger.processing_time("30 seconds"),
    checkpoint="s3://lake/clicks/_ckpt",
    query_name="clicks-upsert",
)
```
:::

## The file sink's sharp edge

:::{important}
Position-based idempotence means the file sink assumes batch *N* always contains the same
rows. That holds when the checkpoint replays the same source positions. It does not hold
when the numbering restarts against a source that has moved on, and the sink does not
notice. It skips, silently, and the rows are gone.
:::

```python
import pyarrow as pa

schema = pa.schema([("v", pa.int64())])
shared = os.path.join(root, "shared")


def write_once(values):
    def feed():
        yield pa.record_batch({"v": values}, schema=schema)

    query = bt.from_batches(feed, schema, bounded=False).write(
        shared, format="parquet", trigger=bt.Trigger.available_now()
    )
    query.await_termination()


write_once([1, 2])
write_once([9, 9])   # a *different* batch 0, into the same directory
print(bt.read.parquet(shared).to_pydict())
# {'v': [1, 2]}
```

The second query's rows are gone. Its batch 0 collided with the first query's
`part-batch00000.parquet`, and `resume` skipped it. Two rules follow, and they are not
optional:

1. One query, one output directory. Never point a second streaming write at a path another
   query owns.
2. A file sink without a `checkpoint` has no idea what batch numbering it is continuing.
   If the source has advanced, low-numbered batches get dropped on the floor. Checkpoint
   your file writes.

Delta does not have this failure mode, because its key is `(app_id, batch_id)` and not a
filename. If you can write to a lakehouse table, do.

## Sources that cannot replay

Recovery can only seek a source that implements snapshot/seek. Brokers do (Kafka offsets,
Kinesis sequence numbers, Pulsar message ids), as does the Delta stream reader and the
`rate` generator.

:::{important}
An arbitrary `from_batches` iterator does **not**. Hand one a
`checkpoint` and no offsets are recorded, so a restart quietly re-reads the source from
the beginning. Nothing warns you. The output survived it in the runs above only because the
file sink deduped by position.
:::

`files_incremental` is a third case: it never uses the offset log at all. It tracks
already-ingested files in its own `state_dir`, which is what makes it exactly-once per
file, and which is separate state you must also keep.

:::{warning}
One more, from `api/streaming.py`: on a spot/preemptible cluster, a node-local checkpoint
path gets a warning, because a reclaimed node takes the checkpoint with it and the restart
you were counting on cannot happen. Put the checkpoint on object storage.
:::

## See also

- {doc}`Kafka to the lake <kafka-etl>`: the ingestion job this write terminates.
- {doc}`Windowed aggregation <windowed-aggregation>`: the stateful query whose snapshots live in
  the same checkpoint directory.
- {doc}`Stream join <stream-join>`: the one query shape that cannot reach any of these sinks.
- {doc}`Writing data <../../user-guide/writing-data>`: the batch write surface and Delta
  commits.
- {doc}`Streaming <../../user-guide/streaming>`: triggers, checkpoints, and the query handle.
- {doc}`Delta Lake integration <../../integrations/delta-lake>`: the `txn` action that makes the
  strongest sink strong.
- {doc}`Fault tolerance <../../architecture/fault-tolerance>`: the recovery model the offset and
  commit logs implement.
- {doc}`Deduplication <../data-engineering/deduplication>`: what to do when the sink you are
  stuck with cannot dedup at all.
