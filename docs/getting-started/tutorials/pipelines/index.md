# Data pipelines

These tutorials build a working data pipeline from the first write to the last read: a transactional Delta table, a continuous query over an unbounded stream, and the seeded test data to exercise both. Each page runs as written on a laptop.

They assume you can write a {py:class}`Dataset <batcher.Dataset>` chain, which {doc}`/getting-started/tutorials/foundations/index` covers. They don't depend on each other.

The following table lists the three tutorials:

| Tutorial | What you build |
|---|---|
| {doc}`Building a lakehouse <building-a-lakehouse>` | The three medallion layers on a real Delta table, with upserts, time travel, an idempotent backfill, and file skipping |
| {doc}`A streaming pipeline <streaming-pipeline>` | A continuous pipeline over an unbounded source that dedupes, windows by event time, and checkpoints |
| {doc}`Synthetic data generation <synthetic-data-generation>` | Seeded test datasets built in memory, so the rest of your work needs no fixtures on disk |

The operators don't change between a batch job and a continuous one. Only the source, the sink, and the trigger do:

```python
import batcher as bt

orders = bt.from_pydict({"day": ["mon", "mon", "tue"], "amount": [10.0, 5.0, 7.0]})
daily = orders.group_by("day").agg(revenue=bt.col("amount").sum()).sort("day")
print(daily.to_pydict())
# {'day': ['mon', 'tue'], 'revenue': [15.0, 7.0]}
```

Point the same `group_by` at a stream and write it with a trigger, and it becomes a continuous query.

## See also

- {doc}`/user-guide/moving-data/index`: the reference behind every reader and writer used here.
- {doc}`/cookbook/data-engineering/index`: the same problems as focused recipes rather than walkthroughs.
- {doc}`../ml/index`: the tutorials that put a model on top of these pipelines.

```{toctree}
:hidden:

building-a-lakehouse
streaming-pipeline
synthetic-data-generation
```
