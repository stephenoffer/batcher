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

`join_stream` joins on equality keys **and** an event-time interval:
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
| `memory.streaming_state_max_bytes` | the cap the retained buffers are checked against | nothing, because it turns an OOM into a `ResourceError` that names the stall |

## The limits, before you build on this operator

:::{important}
**You cannot write a stream-stream join to a sink.** A streaming write takes a single
source today. `joined.write.delta(...)` raises a `PlanError` that says so. The only way to
consume it is `iter_batches()`, and whatever you do with those batches (including a
transactional write) is your code, on your restart semantics. This is the sharpest edge in
Batcher's streaming story and it is worth knowing before you build a topology around it.
:::

:::{note}
**Inner join only.** No outer/left variants, so an impression that is never clicked never
appears in the output. If you need "impressions with no click within 30 minutes", the interval
join will not give it to you.
:::

:::{important}
**A stream cannot be joined to a static dimension table with `join`.** That plan has to
materialize, and the engine refuses on an unbounded source rather than hanging:

```python
# docs: skip
stream.join(dim_table, on="ad")   # PlanError: plan must materialize
```
:::

Do the enrichment inside the batch instead. `map_batches` receives a whole Arrow batch, so
the lookup is one vectorized Arrow join per micro-batch, against a table you hold in
memory:

```python
dim = pa.table({"ad": ["a1", "a2", "a3"], "campaign": ["spring", "fall", "spring"]})


def enrich(batch):
    table = pa.Table.from_batches([batch]).join(dim, keys="ad").combine_chunks()
    return table.to_batches()[0]


enriched = left.map_batches(enrich, output_columns=["ad", "shown", "campaign"])
for batch in enriched.iter_batches():
    print(sorted(batch.to_pydict()["campaign"]))
# ['fall', 'spring', 'spring']
```

:::{warning}
Refresh `dim` yourself if it changes. The engine will not reload it for you, and a
long-running query will happily serve a stale dimension forever.
:::

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
