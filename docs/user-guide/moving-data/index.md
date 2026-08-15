# Move data

Get data in and out: the readers and writers, the storage layer underneath them, and unbounded sources.

::::{grid} 1 2 2 4
:gutter: 3

:::{grid-item-card} {octicon}`download;1.1em` Reading data
:link: /user-guide/moving-data/reading-data
:link-type: doc
Files, object storage, databases, streams.
:::

:::{grid-item-card} {octicon}`upload;1.1em` Writing data
:link: /user-guide/moving-data/writing-data
:link-type: doc
Files, lakehouse tables, sinks.
:::

:::{grid-item-card} {octicon}`plug;1.1em` Custom connectors
:link: /user-guide/moving-data/custom-connectors
:link-type: doc
Plug in your own source or sink format.
:::

:::{grid-item-card} {octicon}`cloud;1.1em` Cloud storage
:link: /user-guide/moving-data/cloud-storage
:link-type: doc
S3, GCS, Azure, on-prem.
:::

:::{grid-item-card} {octicon}`stack;1.1em` Lakehouse
:link: /user-guide/moving-data/lakehouse
:link-type: doc
Delta, Iceberg, and Hudi tables.
:::

:::{grid-item-card} {octicon}`broadcast;1.1em` Streaming
:link: /user-guide/moving-data/streaming
:link-type: doc
Windows, watermarks, exactly-once.
:::

:::{grid-item-card} {octicon}`clock;1.1em` When a stream emits
:link: /user-guide/moving-data/streaming-emission
:link-type: doc
Which shapes emit as rows arrive.
:::

:::{grid-item-card} {octicon}`pulse;1.1em` Monitoring a stream
:link: /user-guide/moving-data/streaming-monitoring
:link-type: doc
Progress, state, late rows, listeners.
:::

:::{grid-item-card} {octicon}`database;1.1em` Stateful streaming
:link: /user-guide/moving-data/streaming-stateful
:link-type: doc
Dedup, interval joins, keyed state, union.
:::
::::

```{toctree}
:hidden:

reading-data
reading-databases
writing-data
custom-connectors
cloud-storage
lakehouse
streaming
streaming-emission
streaming-stateful
streaming-monitoring
```
