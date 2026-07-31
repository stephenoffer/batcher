# Deduplication

Every at-least-once delivery path eventually hands you the same event twice. Kafka
redelivers after a consumer rebalance, an upstream job retries a failed partition, a
producer never got the ack and sent again. The duplicate is the same *event*, but it is
almost never the same *row*: the second copy carries a different ingest timestamp, a
different source partition, maybe a different batch id.

That distinction is what breaks the reflex fix.

```python
import batcher as bt

events = bt.from_pydict(
    {
        "event_id": ["e1", "e1", "e2", "e3", "e3"],
        "user": ["ann", "ann", "bob", "cat", "cat"],
        "amount": [10, 10, 20, 30, 30],
        "ingested_at": [100, 140, 110, 120, 121],
        "source": ["kafka-1", "kafka-2", "kafka-1", "kafka-1", "kafka-2"],
    }
)
```

`e1` and `e3` are each in there twice. And `distinct()` over the whole row does nothing
about it, because the rows are not identical:

```python
print(events.distinct().count())
# 5
```

:::{warning}
Five rows in, five rows out. `distinct()` compares whole rows, and the duplicate carries a
different `ingested_at` and a different `source`, so no two rows are equal and nothing is
removed. Sum `amount` for a daily revenue number and you are 40 over. This is the failure
that gets caught in a board meeting rather than in CI.
:::

## Find it before you fix it

`ds.dq.unique` counts the keys that occur more than once, which is the check to put in
front of anything that assumes a primary key:

```python
print(str(events.dq.unique("event_id").validate()))
# ValidationReport(violations: unique(event_id)=2)
```

Two duplicated keys. Compare `count()` against `n_unique()` for the same signal in one
number:

```python
print(events.count(), events.n_unique("event_id"))
# 5 3
```

## Deduplicate on the key, not on the row

Give `distinct` the columns that actually identify the event, and tell it which copy
survives. Which copy that is depends on what a repeated key *means* in your data.

::::{tab-set}

:::{tab-item} An immutable event: keep the first

```python
clean = events.distinct(subset=["event_id"], keep="first", order_by="ingested_at")
print(clean.sort("event_id").to_pydict())
# {'event_id': ['e1', 'e2', 'e3'], 'user': ['ann', 'bob', 'cat'], 'amount': [10, 20, 30],
#  'ingested_at': [100, 110, 120], 'source': ['kafka-1', 'kafka-1', 'kafka-1']}
```

`keep="first"` with `order_by="ingested_at"` keeps the earliest copy of each event.
That is the right choice for an immutable event: the first delivery is the original, and
the later ones are the replay.
:::

:::{tab-item} A mutable record: keep the last

```python
# docs: skip
latest = events.distinct(subset=["event_id"], keep="last", order_by="ingested_at")
```

`keep="last"` is the right choice for a *mutable* record, where the later row is a newer
version of the same entity rather than a copy of the same event. Ordering by a
monotonic sequence and keeping the last is the "latest state per key" pattern that the
{doc}`CDC pipeline </cookbook/data-engineering/cdc-pipeline>` leans on.
:::

::::

| A repeat of the key is a... | Meaning | `keep` | Order by |
|---|---|---|---|
| replay | the same event delivered twice | `"first"` | ingest time, offset, arrival sequence |
| update | a newer version of the same entity | `"last"` | the version's sequence (LSN, `updated_at`) |

:::{important}
Get this backwards and you will silently keep the wrong row. There is no error either way,
and the row count is identical whichever you choose, so the mistake survives every check
that only counts. Decide deliberately: is a repeat of this key a *replay* or an *update*?
:::

## Ties

`order_by` decides the survivor. If two copies tie on it, the winner is whichever one
the engine reaches first, and that can change between a single-node run and a
distributed one. Do not leave it to chance: order by something that cannot tie. A
sequence number, an offset, or a compound `order_by=["ingested_at", "source"]` all work.

## Keeping the losers

Dropping a duplicate silently is fine right up until someone asks why the counts do not
reconcile. A row-number window numbers the copies instead of discarding them, so you can
route the losers somewhere:

```python
ranked = events.window(
    partition_by=["event_id"],
    order_by=["ingested_at"],
    functions={"copy": "row_number"},
)
keep = ranked.filter(bt.col("copy") == 1).drop("copy")
replays = ranked.filter(bt.col("copy") > 1)

print(keep.count())
# 3
print(replays.select("event_id", "source").sort("event_id").to_pydict())
# {'event_id': ['e1', 'e3'], 'source': ['kafka-2', 'kafka-2']}
```

Both replays came from `kafka-2`. That is a fact worth knowing, and `distinct` would
have thrown it away. Write `replays` to a side table and you have a duplicate-rate
metric for free.

The cost is a second pass over the data plus the ranking window. Use `distinct` when you
only want the answer, and the window when you want the evidence.

## Deduplicating a stream

On an unbounded source you cannot remember every key you have ever seen. State grows
forever and the job dies at 3am on a Sunday.

`drop_duplicates_within_watermark` bounds it: the seen-key set is dropped once the
event-time watermark passes it, so memory is proportional to the lateness you allow, not
to the lifetime of the job.

```python
import datetime

t0 = datetime.datetime(2024, 1, 1)
minute = datetime.timedelta(minutes=1)

stream = bt.from_pydict(
    {"event_id": ["e1", "e1", "e2"], "ts": [t0, t0 + minute, t0 + 2 * minute]}
)
deduped = stream.drop_duplicates_within_watermark(["event_id"], event_time="ts", lateness="10m")
print(deduped.sort("event_id").to_pydict())
# {'event_id': ['e1', 'e2'],
#  'ts': [datetime.datetime(2024, 1, 1, 0, 0), datetime.datetime(2024, 1, 1, 0, 2)]}
```

Over a bounded source (as here) it is exact, and reduces to a plain `distinct`. Over a
real stream it is exact *within the watermark*: a duplicate that arrives more than 10
minutes late will get through. That is the trade you are making when you bound the
state, and it is the same one Spark's `dropDuplicatesWithinWatermark` makes. Pick a
lateness from your actual delivery-delay distribution, not from a round number that
looked nice.

## Picking the tool

Four tools deduplicate, and they differ in what they hand back and what they cost. Match
the row to the question you actually need answered:

| Tool | Gives you | Costs | Reach for it when |
|---|---|---|---|
| `distinct(subset=..., keep=..., order_by=...)` | one row per key | one hash pass | you want the answer and nothing else |
| a `row_number` window | the survivors *and* the losers | a pass plus the ranking window | you need a duplicate-rate metric or a dead-letter table |
| `drop_duplicates_within_watermark` | one row per key, bounded state | exactness beyond the lateness bound | the source is unbounded and the job must not grow forever |

## See also

- {doc}`CDC pipeline </cookbook/data-engineering/cdc-pipeline>`: duplicates plus deletes plus reordering.
- {doc}`Quality gates </cookbook/data-engineering/quality-gates>`: fail the run when the duplicate rate spikes.
- {doc}`Late-arriving data </cookbook/data-engineering/late-arriving-data>`: what the watermark is really doing.
- {doc}`Distinct and dedup </user-guide/transform/distinct-and-dedup>`: every form of `distinct`.
- {doc}`Window functions </user-guide/analyze/window-functions>`: `row_number` and its friends.
- {doc}`Kafka </integrations/streams/kafka>`: where the at-least-once delivery comes from.
- {doc}`Dataset API </api/relational/dataset>`: `distinct`, `n_unique`, and the `dq` accessor.
