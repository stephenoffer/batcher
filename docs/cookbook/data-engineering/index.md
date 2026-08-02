# Data engineering

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

The arrival path, from a source you do not control. See {doc}`ingest/index`.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`download;1.1em` Incremental ingest
:link: /cookbook/data-engineering/ingest/incremental-ingest
:link-type: doc
Read only what is new, without a bookmark file that drifts.
:::

:::{grid-item-card} {octicon}`git-merge;1.1em` CDC pipeline
:link: /cookbook/data-engineering/ingest/cdc-pipeline
:link-type: doc
Apply a change feed in the order the changes happened, not the order they arrived.
:::

:::{grid-item-card} {octicon}`broadcast;1.1em` Late-arriving data
:link: /cookbook/data-engineering/ingest/late-arriving-data
:link-type: doc
The event that shows up on Thursday and belongs to Tuesday.
:::
::::

## Shaping the tables

Turning what arrived into tables people can query. See {doc}`modeling/index`.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`plug;1.1em` Multi-source join
:link: /cookbook/data-engineering/modeling/multi-source-join
:link-type: doc
Reconciling records that share no clean key.
:::

:::{grid-item-card} {octicon}`stack;1.1em` Schema evolution
:link: /cookbook/data-engineering/modeling/schema-evolution
:link-type: doc
The column that changed type under you, and the default that hides it.
:::

:::{grid-item-card} {octicon}`versions;1.1em` Slowly changing dimensions
:link: /cookbook/data-engineering/modeling/slowly-changing-dimensions
:link-type: doc
Keeping history without keeping duplicates.
:::
::::

## Keeping tables healthy

The work a table needs after it exists. See {doc}`maintenance/index`.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`filter;1.1em` Deduplication
:link: /cookbook/data-engineering/maintenance/deduplication
:link-type: doc
Exactly-once is a property you build. You do not get it for free.
:::

:::{grid-item-card} {octicon}`check;1.1em` Quality gates
:link: /cookbook/data-engineering/maintenance/quality-gates
:link-type: doc
Fail the pipeline, not the dashboard.
:::

:::{grid-item-card} {octicon}`pencil;1.1em` Partition backfill
:link: /cookbook/data-engineering/maintenance/partition-backfill
:link-type: doc
Rewrite one day atomically and leave the rest alone.
:::

:::{grid-item-card} {octicon}`database;1.1em` File compaction
:link: /cookbook/data-engineering/maintenance/file-compaction
:link-type: doc
The small-files problem, and when it is actually worth fixing.
:::
::::

## Where to start

{doc}`/cookbook/data-engineering/ingest/etl-pipeline` is the whole arc in one page: raw records in, deduplicated and rolled
up, Parquet out. Read it first if you want the shape before the details.

After that, if you are building a pipeline from nothing, the order that tends to work is:
get the data in ({doc}`incremental ingest </cookbook/data-engineering/ingest/incremental-ingest>`), put a gate in front of
the write ({doc}`quality gates </cookbook/data-engineering/maintenance/quality-gates>`), make the write idempotent
({doc}`deduplication </cookbook/data-engineering/maintenance/deduplication>`), and only then worry about the table's shape over
time ({doc}`schema evolution </cookbook/data-engineering/modeling/schema-evolution>`, {doc}`file compaction </cookbook/data-engineering/maintenance/file-compaction>`).

## See also

- {doc}`Lakehouse tables </user-guide/moving-data/lakehouse>`: the transactional target most of these
  recipes write to.
- {doc}`Reading data </user-guide/moving-data/reading-data>` and
  {doc}`Writing data </user-guide/moving-data/writing-data>`: the reader and the sink, in full.
- {doc}`Delta Lake </integrations/lakehouse/delta-lake>` and {doc}`Kafka </integrations/streams/kafka>`:
  the systems the recipes talk to, and what each guarantees.
- {doc}`Building a lakehouse </tutorials/pipelines/building-a-lakehouse>`: these pieces assembled
  into one pipeline, end to end.
- {doc}`Data engineer learning path </tutorials/paths/data-engineer>`: a reading order.

```{toctree}
:hidden:

ingest/index
modeling/index
maintenance/index
```
