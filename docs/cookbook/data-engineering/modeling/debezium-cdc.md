# Debezium change data capture

A Debezium connector turns every insert, update and delete in a source database into a message, and it does not hand you flat rows. Each message is a JSON envelope: the row as it was (`before`), the row as it is now (`after`), what happened (`op`), and where in the database's log it happened (`source.lsn`). This recipe decodes that envelope and applies it to a target table with {py:meth}`ds.scd.apply_changes <batcher.api.dataset.scd.DatasetSCD.apply_changes>`, the same call {doc}`/user-guide/moving-data/lakehouse` uses for a flat change feed.

The feed below is what a Kafka topic holds after a snapshot read of customer 1, an insert of customer 2, a move, and a delete. It has the two things a real feed has and a tidy example usually leaves out: a `null` message, the tombstone Kafka's log compaction expects after a delete, and a redelivered update.

```python
import os
import tempfile

import batcher as bt

feed = bt.from_pydict(
    {
        "value": [
            '{"payload": {"op": "r", "before": null, "after": {"id": 1, "city": "NYC"}, "source": {"lsn": 10}}}',
            '{"payload": {"op": "c", "before": null, "after": {"id": 2, "city": "LA"}, "source": {"lsn": 11}}}',
            '{"payload": {"op": "u", "before": {"id": 1, "city": "NYC"}, "after": {"id": 1, "city": "SF"}, "source": {"lsn": 12}}}',
            '{"payload": {"op": "d", "before": {"id": 2, "city": "LA"}, "after": null, "source": {"lsn": 13}}}',
            None,
            '{"payload": {"op": "u", "before": {"id": 1, "city": "NYC"}, "after": {"id": 1, "city": "SF"}, "source": {"lsn": 12}}}',
        ]
    }
)
```

## Decode the envelope

The key comes from `after` for an insert or an update and from `before` for a delete, because a delete has no `after`. Coalescing the two gives every change its key. The tombstone carries no change at all, so it is filtered out before anything reads it.

```python
value = bt.col("value")
changes = feed.filter(value.is_not_null()).select(
    value.json.extract_string("$.payload.op").alias("op"),
    bt.coalesce(
        value.json.extract_int("$.payload.after.id"),
        value.json.extract_int("$.payload.before.id"),
    ).alias("id"),
    value.json.extract_string("$.payload.after.city").alias("city"),
    value.json.extract_int("$.payload.source.lsn").alias("lsn"),
)
print(changes.to_pydict())
# {'op': ['r', 'c', 'u', 'd', 'u'], 'id': [1, 2, 1, 2, 1],
#  'city': ['NYC', 'LA', 'SF', None, 'SF'], 'lsn': [10, 11, 12, 13, 12]}
```

## Apply it

The log sequence number orders the changes, so it is the `sequence_by` column, and Debezium's `"d"` marks a delete. Only `id` and `city` are stored: `op` decides what happens to a row and is not part of it.

```python
target = os.path.join(tempfile.mkdtemp(), "customers.parquet")
changes.scd.apply_changes(
    target,
    keys="id",
    sequence_by="lsn",
    deletes=bt.col("op") == "d",
    columns=["id", "city"],
)
print(bt.read.parquet(target).select("id", "city").to_pydict())
# {'id': [1], 'city': ['SF']}
```

Worked by hand, that is the answer. Customer 1 was read at `lsn` 10 and moved at 12, and the redelivered move collapses into the first because only the greatest `lsn` per key survives a batch. Customer 2 was inserted at 11 and deleted at 13. The snapshot read (`"r"`) is applied like an insert, which is what it is.

## Requirements and limitations

The envelope here is Debezium's JSON converter with schemas enabled, which is why the change sits under `payload`. With `value.converter.schemas.enable=false` the fields are at the top level, so drop the `payload.` prefix from every path. An Avro-encoded topic decodes through the broker source's codec instead; see {doc}`/integrations/streams/payload-formats`.

The guarantees and the one caveat are `apply_changes`' own. Batches must arrive in non-decreasing `lsn` order, and replaying an old insert for a key that was since deleted resurrects it, because a physical delete leaves no sequence behind to compare against. {doc}`/user-guide/moving-data/lakehouse` states both in full.

## See also

- {doc}`/user-guide/moving-data/lakehouse`: `apply_changes` on a flat feed, and the rules that make it safe against a feed you don't control.
- {doc}`/integrations/streams/kafka`: reading the topic the connector writes to.
- {doc}`Slowly changing dimensions <slowly-changing-dimensions>`: keeping history instead of only the latest row.
