# Lakehouse tables

Batcher reads and writes the open table formats natively, with no JVM and no Spark cluster in the path. A Delta or Iceberg write is one atomic commit however many workers produced it, every read can time travel, and every read skips the files the table's metadata rules out:

```python
import os
import tempfile

import batcher as bt

table = os.path.join(tempfile.mkdtemp(), "orders")
bt.from_pydict({"id": [1, 2], "amount": [10, 20]}).write.delta(table)
bt.from_pydict({"id": [3], "amount": [30]}).write.delta(table, mode="append")

print(bt.read.delta(table).sort("id").to_pydict())
# {'id': [1, 2, 3], 'amount': [10, 20, 30]}
print(bt.read.delta(table, version=0).count())
# 2
```

Delta Lake is the most complete: merge, replace-where, change data feed, compaction, vacuum, and Delta Sharing reads. Iceberg adds native upserts, snapshot expiry, and Puffin statistics. Hudi tables written by your Spark or Flink jobs read in parallel, file slice by file slice.

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
