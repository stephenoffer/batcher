# Data pipelines

These tutorials build a working data pipeline end to end. They assume you can already write
a {py:class}`Dataset <batcher.Dataset>` chain, which {doc}`/tutorials/foundations/index` covers, and they are independent
of each other, so take the one that matches what you are building.

| Tutorial | What you build |
|---|---|
| {doc}`Building a lakehouse <building-a-lakehouse>` | The three medallion layers on a real Delta table, from raw files to a transactional gold table |
| {doc}`A streaming pipeline <streaming-pipeline>` | A continuous pipeline over an unbounded source: dedupe, window by event time, and checkpoint |
| {doc}`Synthetic data generation <synthetic-data-generation>` | Test datasets built in memory, so the rest of your work needs no fixtures on disk |

## See also

- {doc}`/user-guide/moving-data/index`: the reference behind every reader and writer used here.
- {doc}`/cookbook/data-engineering/index`: the same problems as focused recipes rather than walkthroughs.

```{toctree}
:hidden:

building-a-lakehouse
streaming-pipeline
synthetic-data-generation
```
