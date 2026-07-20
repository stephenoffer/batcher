# Data engineering recipes

Pipelines that move and reconcile data. The hard part is rarely the transformation. It is
everything around it. The source moved. The file arrived twice. Yesterday's numbers changed
overnight and nobody knows why.

Each recipe starts from the failure, shows the code that avoids it, and says what it costs.

:::{tip}
Every recipe on this page runs as written. The code blocks are executed on every docs
build, so if a snippet claims a result, that result was produced by the engine and not by
a hopeful author.
:::

## Getting data in

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`download;1.1em` Incremental ingest
:link: incremental-ingest
:link-type: doc
Read only what is new, without a bookmark file that drifts.
:::

:::{grid-item-card} {octicon}`git-merge;1.1em` CDC pipeline
:link: cdc-pipeline
:link-type: doc
Apply a change feed in the order the changes happened, not the order they arrived.
:::

:::{grid-item-card} {octicon}`broadcast;1.1em` Late-arriving data
:link: late-arriving-data
:link-type: doc
The event that shows up on Thursday and belongs to Tuesday.
:::

:::{grid-item-card} {octicon}`plug;1.1em` Multi-source join
:link: multi-source-join
:link-type: doc
Reconciling records that share no clean key.
:::
::::

## Keeping the table correct

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`filter;1.1em` Deduplication
:link: deduplication
:link-type: doc
Exactly-once is a property you build. You do not get it for free.
:::

:::{grid-item-card} {octicon}`stack;1.1em` Schema evolution
:link: schema-evolution
:link-type: doc
The column that changed type under you, and the default that hides it.
:::

:::{grid-item-card} {octicon}`check;1.1em` Quality gates
:link: quality-gates
:link-type: doc
Fail the pipeline, not the dashboard.
:::

:::{grid-item-card} {octicon}`versions;1.1em` Slowly changing dimensions
:link: slowly-changing-dimensions
:link-type: doc
Keeping history without keeping duplicates.
:::
::::

## Maintaining it

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`pencil;1.1em` Partition backfill
:link: partition-backfill
:link-type: doc
Rewrite one day atomically and leave the rest alone.
:::

:::{grid-item-card} {octicon}`database;1.1em` File compaction
:link: file-compaction
:link-type: doc
The small-files problem, and when it is actually worth fixing.
:::
::::

## Where to start

If you are building a pipeline from nothing, the order that tends to work is: get the data
in ([incremental ingest](incremental-ingest.md)), put a gate in front of the write
([quality gates](quality-gates.md)), make the write idempotent
([deduplication](deduplication.md)), and only then worry about the table's shape over time
([schema evolution](schema-evolution.md), [file compaction](file-compaction.md)).

:::{seealso}
- [Lakehouse tables](../../user-guide/lakehouse.md): the transactional target most of these
  recipes write to.
- [Reading data](../../user-guide/reading-data.md) and
  [Writing data](../../user-guide/writing-data.md): the reader and the sink, in full.
- [Delta Lake](../../integrations/delta-lake.md) and [Kafka](../../integrations/kafka.md):
  the systems the recipes talk to, and what each guarantees.
- [Building a lakehouse](../../tutorials/building-a-lakehouse.md): these pieces assembled
  into one pipeline, end to end.
- [Data engineer learning path](../../learning-paths/data-engineer.md): a reading order.
:::

```{toctree}
:hidden:

incremental-ingest
cdc-pipeline
deduplication
schema-evolution
partition-backfill
quality-gates
slowly-changing-dimensions
late-arriving-data
file-compaction
multi-source-join
```
