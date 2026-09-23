# Retries and idempotency

This page covers the one part of orchestration that is Batcher's job rather than the scheduler's: making a task land the same rows however many times it runs.

A scheduler's retry is worth nothing if the second run doubles the output. Every option below is a property of how the write is spelled, so the decision is made once, in the pipeline, and every scheduler inherits it.

```python
import batcher as bt

day = bt.from_pydict(
    {"id": [1, 2, 3], "region": ["eu", "us", "eu"], "amount": [10.0, 20.0, 30.0]}
)
```

## A file sink has no append

Batcher refuses `mode="append"` to a plain file sink rather than quietly rewriting the output, and the error says what to do instead:

```python
try:
    day.write.parquet("sink.parquet", mode="append")
except bt.PlanError as exc:
    print(str(exc).split(":")[0])
# write()
```

A plain directory of files has no transaction log, so there is nothing to add a commit to. The two honest answers are to replace the whole output, or to use a sink that does have a log.

## Overwrite a partition, not a table

`mode="overwrite"` is idempotent by construction: the second run produces exactly what the first did. Scoped to the partition a task owns, it is also cheap.

```python
day.write.parquet("warehouse/day", mode="overwrite", partition_by=["region"])
day.write.parquet("warehouse/day", mode="overwrite", partition_by=["region"])
print(bt.read.parquet("warehouse/day").count())
# 3
```

The write also drops a completion marker, so a reader can tell a finished output from one that a killed task left half-written:

```python
import os

print(sorted(os.listdir("warehouse/day")))
# ['_SUCCESS', 'region=eu', 'region=us']
```

For a backfill that must replace a slice of a larger table without touching the rest, `replace_where` deletes exactly the matching rows in the same commit that adds the new ones. {doc}`/user-guide/moving-data/lakehouse` covers it.

## A keyed upsert on a transactional table

A Delta or Iceberg table does have a log, so an append is a real commit and runs twice exactly as often as you call it:

```python
day.write.delta("lake/events", mode="append")
day.write.delta("lake/events", mode="append")
print(bt.read.delta("lake/events").count())
# 6
```

Adding `merge_on` turns the same call into an upsert keyed on a column, which is the shape a retry needs: a row that is already there is rewritten rather than added.

```python
day.write.delta("lake/upserted", mode="append")  # create the table
day.write.delta("lake/upserted", mode="append", merge_on="id")
day.write.delta("lake/upserted", mode="append", merge_on="id")
print(bt.read.delta("lake/upserted").count())
# 3
```

`merge_on` needs a table to merge into, so the first write creates it and later ones upsert. In a scheduled pipeline that is usually a one-off bootstrap rather than a branch in the task body.

## Choosing

The following table maps a task's shape to the write that makes it safe to retry. Rows run from the cheapest option to the most general.

| The task | Write it as | Safe to retry because |
| --- | --- | --- |
| Rebuilds one partition from its source | `mode="overwrite"` on that partition's path | The output is a function of the input |
| Backfills a slice of a bigger table | `replace_where=` on a transactional table | Delete and insert land in one commit |
| Applies a keyed batch of changes | `merge_on=` on a Delta or Iceberg table | A key already present is rewritten, not added |
| Appends immutable events | `mode="append"` plus a dedup key downstream | Nothing else can be, so move the problem to the reader |
| Streams continuously | A checkpoint directory | The sink records the batch id and commits it once |

The last row is the streaming case and is not an orchestration concern at all: a checkpointed streaming write records the query name and batch id with the data, so a replayed batch commits nothing. {doc}`/user-guide/moving-data/streaming/index` covers it.

## What a failed task leaves behind

A write publishes atomically: files land under a temporary name and are moved into place at the end, so a killed task leaves no half-written file where a reader will find it. What it can leave is a *partial* output when a task writes several partitions and dies between them, which is what `_SUCCESS` exists to detect.

Deadlines are worth wiring up for the same reason. With `BATCHER_DEADLINE_SECONDS` set, Batcher drains and finishes its writes instead of being cut mid-run, so a timeout produces a complete output rather than one you have to inspect. {doc}`/integrations/compute/schedulers` covers what it reads.

## See also

- {doc}`/user-guide/moving-data/writing-data`: save modes, partitioned layouts, file sizing, and resume.
- {doc}`/user-guide/moving-data/lakehouse`: `MERGE INTO`, `replace_where`, change feeds, and time travel.
- {doc}`/cookbook/data-engineering/maintenance/partition-backfill`: a backfill worked end to end.
- {doc}`index`: the schedulers this makes safe.
