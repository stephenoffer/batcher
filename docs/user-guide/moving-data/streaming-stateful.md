# Stateful streaming operators

This page describes the streaming operators that **remember something between
micro-batches**: watermark deduplication, the stream-stream interval join, the stream-static
join, session windows, arbitrary keyed state, and the union that interleaves two streams. Windows and watermarks are on
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

A deduplicated stream writes to a sink like any other streaming query, which is usually
the point of deduplicating one:

```python
# docs: skip
deduped.write.delta("lake/silver/events", trigger=bt.Trigger.processing_time("30 seconds"))
```

`checkpoint=` is refused here. The seen-key set is not a source offset, so a restart would
resume with an empty one while the offset log said otherwise, which looks exactly like
exactly-once recovery and is not. The sink's own idempotency still applies.

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

## Stream-to-static joins

Enriching a stream from a table that does not move is an ordinary
{py:meth}`join <batcher.Dataset.join>`. The static side is read once, before the first micro-batch, and every
micro-batch joins against the whole of it:

```python
catalogue = bt.from_pydict({"ad": ["a", "b"], "campaign": ["spring", "fall"]})

ad_schema = pa.schema([("ad", pa.string()), ("shown", pa.timestamp("us"))])


def ad_feed():
    yield pa.record_batch({"ad": ["a", "c"], "shown": [base, base]}, schema=ad_schema)


live_ads = bt.from_batches(ad_feed, ad_schema, bounded=False)

for batch in live_ads.join(catalogue, on="ad", how="left").iter_batches():
    print(sorted(str(c) for c in batch.to_pydict()["campaign"]))
# ['None', 'spring']
```

The state it holds is the dimension table, and it is a snapshot rather than a
subscription: a long-running query serves the table as it stood when it started. That is
deliberate, because a dimension that changed mid-stream would let two rows of the same
micro-batch disagree about the same key. Restart the query to pick up a new one.

A join type that would need the *static* side to be complete is refused rather than run,
since an unmatched static row could only be emitted once the stream ended. Inner works
either way round, left outer with the stream on the left, right outer with the stream on
the right, and semi/anti with the stream on the left. The full outer and the two mirrored
outers raise, with a message naming the combinations that do work.

## Session windows

{py:meth}`session_window <batcher.Dataset.session_window>` groups events per key into runs with no gap longer than the
timeout. Over a stream it is stateful for a reason no fixed window is: a session's end is
not knowable in advance, because every event extends the session it lands in and an event
between two sessions merges them. So its rows are held until the watermark passes its last
event plus the gap, and only then aggregated:

```python
visit_schema = pa.schema([("user", pa.string()), ("ts", pa.timestamp("us")),
                          ("pages", pa.int64())])


def visit_feed():
    yield pa.record_batch(
        {"user": ["u1", "u1"], "ts": [base, base + dt.timedelta(minutes=2)],
         "pages": [1, 1]},
        schema=visit_schema,
    )
    yield pa.record_batch(
        {"user": ["u1"], "ts": [base + dt.timedelta(hours=4)], "pages": [1]},
        schema=visit_schema,
    )


visits = bt.from_batches(visit_feed, visit_schema, bounded=False)

for batch in visits.session_window(
    "ts", "30m", partition_by=["user"], pages=col("pages").sum()
).iter_batches():
    print(batch.to_pydict()["pages"])
# [2]
# [1]
```

The buffer holds rows for sessions still open, which is the live key space times the gap
rather than the length of the stream. A source whose event time stalls never closes a
session, so the retained rows are checked against `memory.streaming_state_max_bytes` and a
{py:exc}`ResourceError <batcher.ResourceError>` names the stall instead of the process dying on memory.

A late event cannot reopen a session already emitted; it is dropped, exactly as a late row
is dropped from a closed window. Use `with_watermark` to buy a straggler room, and expect
every session to close that much later in exchange.

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


## When state outgrows memory

`memory.streaming_state_max_bytes` caps what one streaming operator may hold. A **windowed**
aggregate that reaches it moves its oldest windows to disk and keeps running. Everything else
still stops with a `ResourceError`.

The asymmetry is not an oversight, it is what the operators make possible. A watermark only
moves forward, so a windowed aggregate evicts its windows in increasing order — a window
written to disk is read back exactly once, when it closes, and never searched for. That is an
ordered run of files, which is cheap. A running aggregate with no watermark finalizes every
group on every micro-batch, so its state has no cold end to shed; making that spill needs a
keyed store with point lookups, which Batcher does not have.

You do not configure any of this. Spilling starts when the cap is reached and stops when
resident state is back under it, splitting at the median window start so the newest windows —
the ones incoming rows land in — stay in memory:

```python
# docs: skip
with bt.option_context("memory.streaming_state_max_bytes", 512 << 20):
    query = (
        events.with_watermark("ts", "1 hour")
        .group_by(w=bt.window(col("ts"), "5 minutes"), user=col("user"))
        .agg(total=col("amount").sum())
        .write.delta("lake/rollups", checkpoint="s3://lake/_ckpt")
    )
```

Spilled rows are still reported as state: `num_rows_total` in the progress record counts what
is on disk as well as what is resident, because a spilling query has *moved* state, not shed
it. Watch `memory_used_bytes` against `num_rows_total` to see the split.

`memory_used_bytes` also counts what `update` output mode retains beside the aggregate. That
mode diffs each result against a copy of the one before it, so a query in `update` holds the
aggregate **twice** — and the cap counts both. If a query that used to run now reaches the
cap sooner, this is why: it was always using that memory, and the budget was reporting half
of it.

A `ResourceError` still happens, and it now means something narrower than it used to: the
**newest** windows alone exceed the cap. No amount of disk fixes that — it is a key space too
wide for the envelope, or a watermark that has stopped closing anything.

### What spilling costs

Latency, on the micro-batch that spills and on the one that reads a run back. Nothing else:
the answer is unchanged, which
`tests/integration/test_streaming_state_spill.py::test_a_spilled_run_matches_the_unspilled_answer_exactly`
pins against the same query run entirely in memory.

Spilled runs go to `memory.spill_dir` when you set one, and to the engine's usual scratch
location otherwise — the same disk every other spill uses. They are scratch: a restart rebuilds
them from the checkpoint, and the query deletes them when it stops.

## How state is checkpointed

A stateful query with a `checkpoint=` location persists its state so a restart resumes
instead of recomputing. What it writes per micro-batch depends on the operator.

A **running aggregate** — a `group_by(...).agg(...)` with no watermark — records a
*changelog*: the partial aggregate that micro-batch folded in, rather than the whole state.
It can, because an aggregate's `combine` is associative and commutative, so combining a
snapshot with every partial recorded after it reconstructs exactly the state the whole
snapshot would have held. Recovery replays the chain.

That matters because this is the operator whose state only grows. Nothing closes a group, so
a query that has accumulated ten million of them was rewriting ten million rows on every
trigger — a checkpoint whose cost rises for the life of the query, with the flush on the
critical path of every epoch. A changelog entry costs the *batch's* distinct group count
instead:

| Micro-batches | Whole snapshot per epoch | Changelog | Reduction |
|---|---|---|---|
| 50 | 201,118 B | 58,398 B | 3.4x |
| 100 | 721,618 B | 138,706 B | 5.2x |
| 200 | 2,722,618 B | 416,186 B | 6.5x |
| 400 | 10,564,618 B | 1,287,946 B | 8.2x |

The reduction grows with the run because the two costs scale differently: writing the whole
state every epoch is quadratic in the number of epochs, and writing a changelog is linear.
The figures are bytes written to the checkpoint, one group per row on the `rate` source,
from `benchmarks/scenarios/streaming/state_checkpoint.py` — run it to reproduce them. They
are a write-volume measurement, not a wall-clock one: the flush is on the critical path of
every epoch, so fewer bytes is less latency, but how much depends entirely on what the
checkpoint is written to.

A **windowed** aggregate records one too, for a different reason. It *removes* state, which
is normally what disqualifies an operator: a changelog says what went in and has no way to
say what came out, so replaying one would resurrect the windows eviction already emitted. It
qualifies because its removal is not arbitrary. Eviction drops every window whose start is at
or below a threshold, on a totally ordered axis, so what it removes is always a **prefix** —
and a prefix is described by its upper bound. That single integer rides in each entry, and
replay combines the partials and re-applies it.

| Micro-batches | Whole snapshot per epoch | Changelog | Reduction |
|---|---|---|---|
| 40 | 1,092,754 B | 367,754 B | 3.0x |
| 80 | 4,040,834 B | 908,834 B | 4.4x |

**Deduplication and keyed state still snapshot whole.** Their TTL expires arbitrary keys at
arbitrary times, so there is no bound that describes what went; they would need a real
tombstone per key, which this changelog has no way to carry.

`streaming.checkpoint_delta_interval` bounds how many changelog entries accumulate before a
whole snapshot is written again, which is what bounds recovery: a longer chain writes less
and replays more. Set it to `0` to disable the changelog entirely and snapshot whole state
every epoch.

```python
# docs: skip
with bt.option_context("streaming.checkpoint_delta_interval", 25):
    query = (
        events.group_by("user")
        .agg(total=col("amount").sum())
        .write.for_each_batch(upsert, checkpoint="s3://lake/_ckpt", output_mode="update")
    )
```

A changelog entry is only written when it is genuinely smaller than the state — at least
twice as small. A stream whose every micro-batch touches every group gets whole snapshots,
because for that shape a chain would write more, not less. You do not have to know which
shape you have.

A windowed aggregate that has spilled writes a **multi-part** snapshot: its resident state and
each spilled run go into one file, streamed, so the checkpoint never pulls a state larger than
the memory cap back into memory to persist it. Recovery combines the parts — the same
`combine` that makes the changelog sound makes the split into parts invisible. On an object
store the snapshot buffers rather than streams, because a PUT needs the whole object.

A **distributed** streaming query still snapshots whole state. The driver combines the
workers' partials, so the same changelog would apply, but the distributed streaming path has
no automated coverage and a state backend is not something to change without a recorded
cluster run. The single-node behaviour is unaffected either way.

## See also

- {doc}`streaming`: sources, sinks, triggers, output modes, windows, and checkpoints.
- {doc}`streaming-monitoring`: the state each of these operators reports per micro-batch.
- {doc}`/cookbook/streaming/stream-join`: the interval join end to end.
- {doc}`/configuration/options`: `memory.streaming_state_max_bytes`, the cap they share.
