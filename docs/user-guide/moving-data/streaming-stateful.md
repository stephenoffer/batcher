# Stateful streaming operators

This page describes the streaming operators that **remember something between
micro-batches**: watermark deduplication, the stream-stream interval join, arbitrary keyed
state, and the union that interleaves two streams. Windows and watermarks are on
{doc}`streaming`, because they are the mechanism these lean on rather than an operator.

They share one problem and one answer. The problem is that state over an unbounded input is
unbounded unless something releases it; the answer is always a *bound you choose* — a
watermark, an interval, a TTL. An operator here with no bound set is one whose memory grows
for the life of the query.

Every example on this page starts from the same imports and clock:

```python
import datetime as dt

import pyarrow as pa

import batcher as bt
from batcher import col

base = dt.datetime(2024, 1, 1)
print(base.year)
# 2024
```

## Deduplication within a watermark

{py:meth}`drop_duplicates_within_watermark <batcher.Dataset.drop_duplicates_within_watermark>` keeps the first row per key seen inside the
watermark window, forgetting keys the watermark has passed so memory stays bounded.
Over a bounded source it is exact deduplication:

```python
records = bt.from_pydict({
    "id": ["x", "y", "x", "z"],
    "ts": [base, base, base + dt.timedelta(minutes=1), base],
    "v": [1, 2, 3, 4],
})
deduped = records.drop_duplicates_within_watermark(["id"], event_time="ts",
                                                   lateness="1h")
print(sorted(deduped.to_pydict()["id"]))  # ['x', 'y', 'z'] — the second 'x' dropped
```

## Stream-to-stream joins

{py:meth}`join_stream <batcher.Dataset.join_stream>` joins two streams on keys **and** an event-time interval
(`|left_time - right_time| <= within`). The time bound is what lets buffered state be
evicted by the watermark, keeping a two-stream join in bounded memory. Bounded
sources run it as a plain join plus the interval filter:

```python
impressions = bt.from_pydict({"ad": ["a", "b"], "shown": [base, base]})
clicks2 = bt.from_pydict({"ad": ["a"], "clicked": [base + dt.timedelta(minutes=2)]})

attributed = impressions.join_stream(
    clicks2, on="ad", left_time="shown", right_time="clicked", within="5m"
)
print(attributed.to_pydict()["ad"])  # ['a'] — clicked within 5 minutes of shown
```

`how=` takes `"inner"` (the default), `"left"`, `"right"`, and `"full"`. An unmatched row
is emitted null-padded at the moment the watermark guarantees no partner can still arrive
for it — which is the only moment that statement is decidable about an unbounded stream,
and why the interval is required rather than optional:

```python
unclicked = impressions.join_stream(
    clicks2, on="ad", left_time="shown", right_time="clicked", within="5m", how="left"
)
print(sorted(unclicked.to_pydict()["ad"]))  # ['a', 'b'] — 'b' with null click columns
```

A joined stream writes to a sink like any other streaming query:

```python
# docs: skip
attributed.write.delta("lake/attribution", trigger=bt.Trigger.processing_time("30 seconds"))
```

`checkpoint=` is the one thing it refuses. A join's state is two buffered sides and two
watermarks, none of it addressable by a source offset, so there is nothing to resume from
— and accepting the argument would restart from an empty join on every restart while
looking exactly like exactly-once recovery. The sink's own idempotency still applies, so a
replayed micro-batch does not duplicate rows.

## Arbitrary keyed state

Some shapes have no algebra: sessionization with custom rules, a running fraud score, a
per-device state machine, "alert when this key has been silent for ten minutes". None of
those is an aggregate, and every stream processor grows an escape hatch for them.
{py:meth}`transform_with_state <batcher.Dataset.transform_with_state>` is Batcher's, and
it is Spark's `transformWithState` bargain: your function owns one key's state, and the
engine owns when it is called, checkpointed, and expired.

```python
import pyarrow as pa

def running_total(key, rows, state):
    total = (state or {"total": 0})["total"] + sum(rows.column("v").to_pylist())
    return {"user": [key[0]], "total": [total]}, {"total": total}

events = bt.from_pydict({"user": ["a", "b", "a"], "v": [1, 2, 3]})
totals = events.transform_with_state(
    running_total,
    group_by="user",
    output_columns=["user", "total"],
    state_ttl="1 hour",
)
print(sorted(zip(*[totals.to_pydict()[c] for c in ("user", "total")], strict=True)))
# [('a', 4), ('b', 2)]
```

`fn(key, rows, state)` is called once per key **per micro-batch**, with that key's rows as
an Arrow `RecordBatch`. It returns `(rows_out, state_out)`: what to emit, and the state to
keep. Returning `None` for the state forgets the key. A key with no rows in a micro-batch
is not called, which is what keeps a million-key state from becoming a million Python calls
per trigger.

State must be a flat mapping of scalars, because the whole key space is checkpointed as one
Arrow batch — so a query resumes with its state intact. Keep a large payload elsewhere and
hold a reference to it in state.

:::{warning}
`state_ttl` is what bounds the memory. Without one, a key is remembered for the life of the
query, which is correct only if the key space is. An unbounded key space and no TTL ends in
a `ResourceError` against `memory.streaming_state_max_bytes` — loudly, but hours later.
:::

:::{note}
There is no distributed implementation yet. Its mergeable form is a shuffle by the group
keys, so each key's state lives on exactly one worker and the partitions' key sets are
disjoint; the distributed runner does not do that shuffle, so `distributed=True` is refused
rather than quietly answered by one machine.
:::

## Unioning streams

`union` works over streams. Because a UNION ALL is a multiset union and makes no ordering
claim, the branches are **interleaved** rather than concatenated: one batch from each in
turn, so an unbounded first branch cannot shut the second one out.

```python
import pyarrow as pa

feed_schema = pa.schema([("v", pa.int64())])

def eu():
    for i in (0, 1):
        yield pa.record_batch({"v": [i]}, schema=feed_schema)

def us():
    for i in (10, 11):
        yield pa.record_batch({"v": [i]}, schema=feed_schema)

both = bt.from_batches(eu, feed_schema, bounded=False).union(
    bt.from_batches(us, feed_schema, bounded=False)
)
print(sorted(x for b in both.iter_batches() for x in b.to_pydict()["v"]))
# [0, 1, 10, 11]
```

A union of *bounded* inputs still concatenates in order, because order is free there. A
`distinct=True` union over streams is refused: a global dedup is exactly the
whole-relation state a stream does not have.

:::{note}
A branch parked on an idle source delays the others, because pulling from it is a blocking
read. That is the same property {py:meth}`join_stream <batcher.Dataset.join_stream>` has,
and for the same reason: one driver thread, and the source decides when its read returns.
:::


## See also

- {doc}`streaming`: sources, sinks, triggers, output modes, windows, and checkpoints.
- {doc}`streaming-monitoring`: the state each of these operators reports per micro-batch.
- {doc}`/cookbook/streaming/stream-join`: the interval join end to end.
- {doc}`/configuration/options`: `memory.streaming_state_max_bytes`, the cap they share.
