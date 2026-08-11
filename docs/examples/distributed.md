# Distributed and streaming

This page covers the scripts that check the mergeable algebra, the shuffle, and the streaming
paths.

## One algebra, two schedules

A stateful operator is built as `partial`, `combine`, `finalize`, so running it on one core
and across a cluster is the same algebra with a different schedule. There is no second
distributed semantics, which is what makes the equivalence assertion meaningful rather than
decorative.

```python
import batcher as bt
from batcher import col

lineitem = bt.from_pydict(
    {
        "l_returnflag": ["A", "N", "A", "R", "N"],
        "l_quantity": [17, 36, 8, 28, 24],
    }
)

query = (
    lineitem.group_by("l_returnflag")
    .agg(lines=bt.count(), qty=col("l_quantity").sum())
    .sort("l_returnflag")
)

single = query.collect(num_partitions=1)
many = query.collect(num_partitions=8)

assert single.schema == many.schema
assert single.to_pydict() == many.to_pydict()
```

The claim to state carefully is about floating point. The multiset of rows, every column name
and every column type are exact. Floating-point reductions are identical *up to
reassociation*: `combine` is associative in exact arithmetic, IEEE addition is not, and the
partition count changes the summation order. Compensated summation bounds that error to near
the last bits; it does not remove it. The scripts assert integers exactly and floats to a
relative tolerance for that reason.

## Running against a cluster

Bringing up a cluster takes longer than the whole rest of the suite, so these default to
single node and still exercise the mergeable path across several local partitions. Opt in
when you have a cluster to point at:

```bash
python examples/dist/mergeable_equivalence.py --distributed
BATCHER_EXAMPLES_DISTRIBUTED=1 python -m pytest tests/docs/test_examples.py -q -k dist
```

CI installs no Ray, so a green PR gate says nothing about the distributed path. A recorded
cluster run in `benchmarks/BENCHMARK_RESULTS.md` is the only evidence it works.

## Streaming

Batch is the bounded case of streaming over Arrow batches, so the same operators serve both.
A tumbling window is a truncation of the timestamp used as a group key, and a session window
is the gaps-and-islands pattern: mark the rows that start a run, then take a running sum of
those marks as the session id.

## Every script on this page

The table below lists the distributed and streaming scripts in path order.

<!-- library-table: dist,streams -->
| Script | Shows |
| --- | --- |
| `examples/dist/broadcast_versus_shuffle.py` | Two ways to join across a cluster, and the size that decides between them |
| `examples/dist/end_to_end_distributed.py` | A full pipeline checked for single-node/distributed equivalence |
| `examples/dist/fault_tolerance.py` | What survives a worker failure, and what the mergeable algebra guarantees |
| `examples/dist/mergeable_equivalence.py` | The contract that makes distribution safe: partial, combine, finalize |
| `examples/dist/multi_node_aggregation.py` | A grouped aggregate across partitions, checked against the single-node answer |
| `examples/dist/partitioning.py` | Partition count: what it changes, and what it must not |
| `examples/dist/scaling_characteristics.py` | How a query's cost moves as the partition count changes |
| `examples/dist/shuffle_and_joins.py` | What a distributed join costs: the shuffle |
| `examples/dist/spill_and_memory.py` | Bounded memory: what spilling buys and what it costs |
| `examples/dist/transport_and_backpressure.py` | Moving bulk data between workers, and why it needs flow control |
| `examples/streams/deduplication_in_stream.py` | Dropping duplicate events without keeping every key forever |
| `examples/streams/exactly_once_semantics.py` | What makes a restart safe: idempotent writes and a durable position |
| `examples/streams/incremental_accumulation.py` | Accumulating state across batches with a mergeable aggregate |
| `examples/streams/late_data_and_watermarks.py` | Late-arriving events, and the watermark that decides when a window closes |
| `examples/streams/micro_batches.py` | Batch is the bounded case of streaming: the same operators over a batch at a time |
| `examples/streams/session_windows.py` | Session windows: grouping events by a gap rather than by a clock |
| `examples/streams/trigger_and_output_modes.py` | Trigger and output mode: how often a streaming query fires, and what it emits |
| `examples/streams/windowed_aggregation.py` | Time windows over an event stream, computed as a grouped aggregate |
<!-- /library-table -->
