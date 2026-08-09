# Joining two streams

Ad impressions on one topic, clicks on another. You want the click attributed to the
impression that caused it.

:::{warning}
Write it as a plain equi-join and the build side is an unbounded stream: every impression
ever seen is retained, because a click for it could theoretically arrive next year. The job
runs for a week and then dies on memory.
:::

The fix is not a bigger heap. It is admitting that a click ten hours after its impression
is not an attribution, and telling the engine so, in the join itself.

## The interval is the contract

{py:meth}`join_stream <batcher.Dataset.join_stream>` joins on equality keys **and** an event-time interval:
`|left_time - right_time| <= within`. That bound is the whole reason the join can run
forever. Once the watermark passes `impression_time + within`, no future click can match
that impression, so it is evicted from the buffer.

::::{tab-set}
:::{tab-item} Bounded inputs

Develop it on bounded inputs, where it runs as an inner join plus the interval filter:

```python
import datetime as dt

import batcher as bt

base = dt.datetime(2024, 1, 1)
minute = dt.timedelta(minutes=1)

impressions = bt.from_pydict({
    "ad": ["a1", "a2", "a3"],
    "shown": [base, base + 5 * minute, base + 10 * minute],
})
clicks = bt.from_pydict({
    "ad": ["a1", "a3"],
    "clicked": [base + 2 * minute, base + 200 * minute],
})

attributed = impressions.join_stream(
    clicks, on="ad", left_time="shown", right_time="clicked", within="30m"
)
print(attributed.to_pydict()["ad"])
# ['a1']
```

`a2` was never clicked. `a3` was clicked three hours later, well outside the 30-minute
window, so it does not count as an attribution. The interval is a business rule, and it is
the same rule that bounds your memory. Pick it deliberately.
:::

:::{tab-item} Both sides unbounded

Swap in two unbounded sources and the call is unchanged. The engine buffers each side,
evicts by the watermark, and yields matches as they form:

```python
import pyarrow as pa

left_schema = pa.schema([("ad", pa.string()), ("shown", pa.timestamp("us"))])
right_schema = pa.schema([("ad", pa.string()), ("clicked", pa.timestamp("us"))])


def impression_feed():
    yield pa.record_batch({"ad": ["a1", "a2"], "shown": [base, base + 5 * minute]},
                          schema=left_schema)
    yield pa.record_batch({"ad": ["a3"], "shown": [base + 10 * minute]}, schema=left_schema)


def click_feed():
    yield pa.record_batch({"ad": ["a1"], "clicked": [base + 2 * minute]}, schema=right_schema)
    yield pa.record_batch({"ad": ["a3"], "clicked": [base + 200 * minute]},
                          schema=right_schema)


left = bt.from_batches(impression_feed, left_schema, bounded=False)
right = bt.from_batches(click_feed, right_schema, bounded=False)

joined = left.join_stream(
    right, on="ad", left_time="shown", right_time="clicked", within="30m"
)
for batch in joined.iter_batches():
    print(batch.to_pydict()["ad"])
# ['a1']
```
:::

:::{tab-item} Two Kafka topics

```python
# docs: skip
impressions = bt.read.kafka("impressions", bootstrap_servers="broker-1:9092")
clicks = bt.read.kafka("clicks", bootstrap_servers="broker-1:9092")

attributed = impressions.join_stream(
    clicks,
    on="ad_id",
    left_time="shown_at",
    right_time="clicked_at",
    within="30m",
    lateness="5m",   # grace before buffered rows are evicted
)
for batch in attributed.iter_batches():
    publish(batch)
```
:::
::::

`lateness` is extra grace on top of `within` before state is evicted. It buys tolerance
for a straggler; it costs you exactly that much more buffer.

| Knob | What it does | What it costs |
| --- | --- | --- |
| `within` | the event-time interval a match must fall inside | it is a business rule first: too wide and a click ten hours later counts as an attribution |
| `lateness` | grace on top of `within` before buffered rows are evicted | exactly that much more buffer |
| `memory.streaming_state_max_bytes` | the cap the retained buffers are checked against | nothing, because it turns an OOM into a {py:exc}`ResourceError <batcher.ResourceError>` that names the stall |

## The limits, before you build on this operator

:::{important}
**A stream-stream join has no checkpoint.** It writes to a sink like any other streaming
query — `joined.write.delta(..., trigger=...)` runs — but passing `checkpoint=` is refused
rather than accepted, because the join's state is two buffered sides and two watermarks,
none of it addressable by a source offset. A restart therefore begins with an empty join
and re-reads from wherever the sources start. The sink's own idempotency still applies, so
a replayed micro-batch does not duplicate rows; what you do not get is resumption.
:::

`how=` takes `"inner"` (the default), `"left"`, `"right"`, and `"full"`. An outer join is
how you answer "impressions with no click within 30 minutes": the unmatched impression is
emitted with null click columns, once, at the moment the watermark guarantees no click can
still arrive for it.

```python
# docs: skip
unattributed = impressions.join_stream(
    clicks, on="ad", left_time="shown", right_time="clicked", within="30m", how="left"
).filter(col("clicked").is_null())
```

:::{note}
**An outer row is emitted late, by design.** It cannot be emitted when the row arrives,
because a partner may still be coming; it is emitted when the row leaves the join buffer,
which is `within` plus the allowed `lateness` after its event time. A wider interval buys
more matches and delays every unmatched row by the same amount. This is Spark's rule too,
and it is why an outer stream-stream join needs a time bound rather than merely accepting
one.
:::

## Joining a stream to a table that does not move

The other join anyone writes: clicks against a product catalogue, events against a device
registry, impressions against a campaign dimension. Only one side streams, so no interval
is needed. Write it as an ordinary {py:meth}`join <batcher.Dataset.join>`:

```python
campaigns = bt.from_pydict({
    "ad": ["a1", "a2", "a3"],
    "campaign": ["spring", "fall", "spring"],
})

enriched = left.join(campaigns, on="ad", how="left")
for batch in enriched.iter_batches():
    print(sorted(batch.to_pydict()["campaign"]))
# ['fall', 'spring']
# ['spring']
```

The static side is read once, before the first micro-batch, and every micro-batch joins
against the whole of it. That is exact rather than approximate: an equi-join is per-row on
the stream side, so no stream row's result depends on another's, and joining batch by batch
gives the same rows as joining the whole stream at once.

Its memory bound is the dimension table itself. A table small enough to enrich a stream
against is small enough to hold, and one that is not wants a different design.

:::{warning}
**The dimension is a snapshot, not a subscription.** A query serves the table as it stood
when the query started, for as long as the query runs. That is deliberate and it matches
Spark: a dimension that changed mid-stream would otherwise let two rows of the same
micro-batch disagree about the same key. Restart the query to pick up a new one.
:::

### Which join types work

A side is *preserved* by an outer join when its unmatched rows must be emitted, and a
preserved row is only known to be unmatched once the **opposite** side is complete. The
static side is complete before the first micro-batch; the stream never is. So a join that
preserves the stream side works, and one that preserves the static side is refused with a
message saying so:

| Join type | Stream on the left | Stream on the right |
|---|---|---|
| `inner` | Works | Works |
| `left` | Works | Refused |
| `right` | Refused | Works |
| `full` | Refused | Refused |
| `semi`, `anti` | Works | Refused |

```python
try:
    list(left.join(campaigns, on="ad", how="full").iter_batches())
except Exception as exc:
    print(type(exc).__name__)
# PlanError
```

Batcher draws this line in exactly the place Spark does, so a ported job either runs or
fails on the same shapes it already failed on.

## When the buffer grows anyway

State is bounded by the watermark advancing, and the watermark advances on event time.
One side going idle stalls it: the other side's rows keep arriving and nothing ever
evicts. Batcher checks the retained buffers against `memory.streaming_state_max_bytes`
and raises a `ResourceError` naming the stall rather than letting the process OOM.

:::{dropdown} Setting the state budget explicitly

```python
from batcher.config import Config, MemoryConfig, config_context

tight = Config().replace(memory=MemoryConfig(streaming_state_max_bytes=256 << 20))
with config_context(tight):
    print(tight.memory.streaming_state_budget_bytes())
# 268435456
```
:::

When that error fires, ask which side stopped producing, not whether to raise the cap. A
one-sided stream is a broken pipeline, and a bigger budget only delays the diagnosis.

## See also

- {doc}`Late data and watermarks </cookbook/streaming/late-data-watermarks>`: how the watermark that evicts this
  buffer is computed.
- {doc}`Windowed aggregation </cookbook/streaming/windowed-aggregation>`: the other operator this watermark bounds.
- {doc}`Exactly-once sinks </cookbook/streaming/exactly-once-sink>`: what you would have had, if this join could
  reach a sink.
- {doc}`Joins </user-guide/analyze/joins>`: the bounded join surface, including the outer joins
  the streaming path does not have.
- {doc}`Streaming </user-guide/moving-data/streaming>`: sources, sinks, watermarks, and the query handle.
- {doc}`Join algorithms </architecture/deep-dives/operators/join-algorithms>`: what the bounded join does with the
  build side that the streaming one cannot.
- {doc}`Kafka integration </integrations/streams/kafka>`: the two topics above.
- {doc}`Multi-source join </cookbook/data-engineering/modeling/multi-source-join>`: the batch recipe for the same
  attribution question.
