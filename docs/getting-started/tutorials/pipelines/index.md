# Data pipelines

These tutorials build a working data pipeline from the first write to the last read. You land raw files into a transactional Delta table, run a continuous query over an unbounded stream, and generate the test data to exercise both. Each page runs as written on a laptop, in a temp directory or in memory.

They assume you can already write a {py:class}`Dataset <batcher.Dataset>` chain, which {doc}`/getting-started/tutorials/foundations/index` covers. They don't depend on each other, so start with the one that matches what you're building.

The following table lists the three tutorials and what each one builds:

| Tutorial | What you build |
|---|---|
| {doc}`Building a lakehouse <building-a-lakehouse>` | The three medallion layers on a real Delta table, with upserts, time travel, an idempotent backfill, and file skipping |
| {doc}`A streaming pipeline <streaming-pipeline>` | A continuous pipeline over an unbounded source that dedupes, windows by event time, and checkpoints |
| {doc}`Synthetic data generation <synthetic-data-generation>` | Seeded test datasets built in memory, so the rest of your work needs no fixtures on disk |

The lakehouse and streaming tutorials share one idea worth noticing. The operators don't change between a batch job and a continuous one. Only the source, the sink, and the trigger do.

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
