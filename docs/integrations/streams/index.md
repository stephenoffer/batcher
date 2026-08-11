# Streams

Unbounded sources. The engine treats batch as the bounded special case of streaming, so the
operators are the same ones you already use.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`broadcast;1.1em` Kafka
:link: /integrations/streams/kafka
:link-type: doc
One reader per topic partition. No sink; publish back through {py:meth}`for_each_batch <batcher.api.io_namespace.writer.Writer.for_each_batch>`.
:::

:::{grid-item-card} {octicon}`broadcast;1.1em` Kinesis
:link: /integrations/streams/kinesis
:link-type: doc
A split per shard, and an exact resume from the stored sequence number.
:::

:::{grid-item-card} {octicon}`broadcast;1.1em` Pulsar
:link: /integrations/streams/pulsar
:link-type: doc
Partitions are a number you declare, not one the broker tells you.
:::

:::{grid-item-card} {octicon}`broadcast;1.1em` Pub/Sub
:link: /integrations/streams/pubsub
:link-type: doc
A single reader, however big the cluster. Ack deadlines make the duplicates.
:::

:::{grid-item-card} {octicon}`broadcast;1.1em` Event Hubs
:link: /integrations/streams/eventhubs
:link-type: doc
Native AMQP, or the Kafka protocol endpoint that actually resumes.
:::

::::

```{toctree}
:hidden:

kafka
kinesis
pulsar
pubsub
eventhubs
```
