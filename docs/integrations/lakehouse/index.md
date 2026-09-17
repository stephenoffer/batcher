# Lakehouse tables

Batcher reads and writes the open table formats natively, with no JVM and no Spark cluster in the path. A Delta or Iceberg write is one atomic commit however many workers produced it, every read can time travel to an earlier version, and every read skips the files the table's own metadata proves can't match.

Delta Lake is the most complete: merge, replace-where, change data feed, compaction, vacuum, and Delta Sharing reads. Iceberg adds native upserts, snapshot expiry, and the Puffin statistics other engines publish. Hudi tables written by your existing Spark or Flink jobs read in parallel, file slice by file slice.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`versions;1.1em` Delta Lake
:link: /integrations/lakehouse/delta-lake
:link-type: doc
Time travel, `MERGE INTO`, replace-where, change data feed, compact, and vacuum.
:::

:::{grid-item-card} {octicon}`versions;1.1em` Apache Iceberg
:link: /integrations/lakehouse/iceberg
:link-type: doc
REST, Glue, Hive, and SQL catalogs. Snapshot time travel, manifest pruning, and upserts.
:::

:::{grid-item-card} {octicon}`versions;1.1em` Apache Hudi
:link: /integrations/lakehouse/hudi
:link-type: doc
Snapshot, time-travel, and incremental reads of the tables your ingest jobs already write.
:::

::::

For the task-oriented guide to upserts, slowly changing dimensions, CDC, and table maintenance across formats, see {doc}`/user-guide/moving-data/lakehouse`.

```{toctree}
:hidden:

delta-lake
iceberg
hudi
```
