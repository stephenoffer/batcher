# CDC pipeline

A change-data-capture connector (Debezium, a Delta change feed, a Snowflake stream)
does not hand you the current state of a table. It hands you the log of what happened
to it: inserts, updates, deletes, each stamped with a sequence number (an LSN, a commit
version, a transaction id). That log is delivered at-least-once and it is not in order.

:::{warning}
Treat a change log as a snapshot and you will corrupt the target. The rows are not
duplicates to be dropped and they are not in the order the changes happened. The last row
in the file is routinely the oldest change in it.
:::

Here is a batch off the wire for one customer. The connector redelivered the `lsn=20`
change after the `lsn=30` one:

```python
import os
import tempfile

import batcher as bt

work = tempfile.mkdtemp()

feed = bt.from_pydict(
    {
        "id": [7, 7, 7],
        "email": ["ann@old.io", "ann@new.io", "ann@old.io"],
        "op": ["INSERT", "UPDATE", "UPDATE"],
        "lsn": [10, 30, 20],
    }
)
```

## What breaks

The obvious pipeline collapses the batch to one row per key and upserts it. Collapse by
*arrival order* (the order the rows sit in the file) and the redelivered row wins:

```python
naive = os.path.join(work, "naive.parquet")

by_arrival = feed.with_row_index("arrived").distinct(
    subset=["id"], keep="last", order_by="arrived"
)
by_arrival.select("id", "email").write.merge(naive, on="id")

print(bt.read.parquet(naive).to_pydict())
# {'id': [7], 'email': ['ann@old.io']}
```

The customer changed their email an hour ago and your warehouse just changed it back.
Nothing errored. Nothing will error tomorrow either, when the next batch arrives and
it happens again.

Skipping the collapse does not help: a `merge` whose source has two rows for one key
raises a cardinality violation, because there is no single row to update from. Which is
the right error, and it tells you the real question is *which* row wins.

## Sequence, not arrival

`ds.scd.apply_changes` is the CDC-aware upsert (Delta Live Tables spells it
`APPLY CHANGES INTO`). You give it the key, the column that sequences the changes, and
a predicate that identifies a delete:

```python
customers = os.path.join(work, "customers.parquet")

feed.scd.apply_changes(
    customers,
    keys="id",
    sequence_by="lsn",
    deletes=bt.col("op") == "DELETE",
    columns=["id", "email"],  # `op` drives the delete predicate; it is not stored
)

print(bt.read.parquet(customers).to_pydict())
# {'id': [7], 'email': ['ann@new.io'], 'lsn': [30]}
```

`lsn=30` wins over both `lsn=10` and the redelivered `lsn=20`, regardless of where they
sat in the batch. `lsn` is *stored in the target*. That is not clutter, it is
the whole mechanism: it lets the next batch tell a fresh change from a stale one.

## The next batch

Now a batch containing an update that lost a race upstream (`lsn=15`, older than the
`lsn=30` already applied), plus a customer who is created and deleted inside the same
batch:

```python
batch2 = bt.from_pydict(
    {
        "id": [7, 8, 8],
        "email": ["ann@stale.io", "bob@x.io", "bob@x.io"],
        "op": ["UPDATE", "INSERT", "DELETE"],
        "lsn": [15, 40, 50],
    }
)
batch2.scd.apply_changes(
    customers,
    keys="id",
    sequence_by="lsn",
    deletes=bt.col("op") == "DELETE",
    columns=["id", "email"],
)

print(bt.read.parquet(customers).sort("id").to_pydict())
# {'id': [7], 'email': ['ann@new.io'], 'lsn': [30]}
```

Two things happened. Ann's `lsn=15` update was discarded, because the target already
holds `lsn=30` for that key. And Bob never landed at all: within the batch only the
greatest-`lsn` change per key survives, so his `DELETE` at 50 beat his `INSERT` at 40,
and a delete for a key that is not in the target is a tombstone that changes nothing.

Three rules, and they are worth stating plainly:

| Rule | What it decides | Consequence |
|---|---|---|
| Within a batch, greatest `sequence_by` per key wins | which of several changes to one key survives the collapse | redeliveries and reordering inside a batch stop mattering |
| Across batches, a change applies only if its sequence is at least the one stored for that key | whether an arriving change is fresh or stale | a stale change is dropped, not applied |
| A `deletes` match removes the row | what a delete does to the target | the row leaves; nothing is written in its place |

## Replay is a no-op

Which means re-applying a batch you already applied does nothing:

```python
batch2.scd.apply_changes(
    customers,
    keys="id",
    sequence_by="lsn",
    deletes=bt.col("op") == "DELETE",
    columns=["id", "email"],
)

print(bt.read.parquet(customers).sort("id").to_pydict())
# {'id': [7], 'email': ['ann@new.io'], 'lsn': [30]}
```

That is what makes the pipeline restartable. A worker dies mid-load, you re-run from the
last committed offset, and the overlap costs you I/O and nothing else.

## The sharp edge

:::{important}
Idempotent is not commutative. A delete here is physical: the row leaves the target, so
there is no stored sequence left to compare a later change against. Replay an *old insert*
for a key that was since deleted and it comes back from the dead, carrying whatever values
it had before the delete.
:::

The practical rule: feed batches in non-decreasing sequence order, which is what a CDC
reader gives you anyway. And treat a full replay of a feed that contains deletes as a
table rebuild, not a resume. Delta Live Tables behaves identically here, and warns about it
for the same reason.

The other constraint is that a Parquet target is copy-on-write and single-writer: each
apply rewrites the files its keys can reach and atomically replaces them. One CDC
consumer per target table. If you need concurrent writers, point the same call at a
Delta table, where the commit is a real transaction.

## Wiring the real feed

The feed itself comes off Kafka or a Delta change feed. Only the source line changes, and the
`apply_changes` call is the same one you ran above.

::::{tab-set}

:::{tab-item} Debezium on Kafka

```python
# docs: skip
import batcher as bt

changes = bt.read.kafka("dbserver.public.customers", bootstrap_servers="broker:9092")
for batch in changes.iter_batches():
    bt.from_arrow(batch).scd.apply_changes(
        "s3://lake/customers",
        keys="id",
        sequence_by="lsn",
        deletes=bt.col("op") == "d",  # Debezium spells delete "d"
        columns=["id", "email"],
    )
```
:::

:::{tab-item} A Delta change feed

When the upstream is itself a Delta table, read its change feed from the last version you
processed:

```python
# docs: skip
changes = bt.read.read_change_feed("s3://lake/customers_raw", starting_version=42)
```
:::

::::

:::{tip}
Store the version or offset you last applied next to the target, not in the job's memory.
The apply is idempotent, so overlapping a few versions on restart is free. A gap is
not, and a gap is what you get when the bookmark lives only in a dead process.
:::

## See also

- [Slowly changing dimensions](slowly-changing-dimensions.md): keep the history
  instead of overwriting it.
- [Deduplication](deduplication.md): the same collapse-by-sequence trick, on a feed
  with no deletes.
- [Late-arriving data](late-arriving-data.md): the other reason yesterday's numbers move.
- [Lakehouse tables](../../user-guide/lakehouse.md): the transactional target.
- [Kafka](../../integrations/kafka.md): the source connector and its offsets.
- [Delta Lake](../../integrations/delta-lake.md): change feeds, and what a commit buys you.
- [Dataset API](../../api/dataset.md): `ds.scd.apply_changes` and the rest of the accessor.
