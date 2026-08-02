# Keeping tables healthy

These recipes cover the work a table needs after it exists: removing what arrived twice,
rejecting what should never have arrived, repairing a day you got wrong, and stopping a
million small files from accumulating.

| Recipe | The problem |
|---|---|
| {doc}`Deduplication <deduplication>` | The same event delivered twice by an at-least-once path |
| {doc}`Quality gates <quality-gates>` | Rejecting, dropping, or quarantining rows that break a contract |
| {doc}`Partition backfill <partition-backfill>` | Replacing one day's rows without touching the days either side |
| {doc}`File compaction <file-compaction>` | A file per micro-batch, and the read cost that follows |

## See also

- {doc}`/user-guide/trust/data-quality`: the guide behind the quality-gate recipe.
- {doc}`/cookbook/data-engineering/ingest/index`: the arrival path these problems come from.

```{toctree}
:hidden:

deduplication
quality-gates
partition-backfill
file-compaction
```
