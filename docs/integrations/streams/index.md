# Streams

Batcher reads the message brokers your platform already runs as unbounded datasets, and the operators you use on a table work on a stream unchanged. Kafka, Kinesis, Pulsar, Pub/Sub, and Event Hubs all deliver the same six-column schema, so a decode, a join, or a windowed aggregate written against one broker moves to another by changing the source line.

Every broker reader advances its position only after a micro-batch is published, so a crash replays a batch rather than dropping one. A checkpoint on local disk or object storage lets a restarted query pick up where it stopped. Readers take Spark's `max_offsets_per_trigger` and `max_bytes_per_trigger` by name, and they decode Avro, JSON, and Protobuf payloads in the source, Confluent Schema Registry framing included. Kafka, Kinesis, Pulsar, and Event Hubs split per partition or shard, so ingest spreads across the cluster.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`broadcast;1.1em` Kafka
:link: /integrations/streams/kafka
:link-type: doc
One reader per partition, bounded offset-range backfills, and a sink that publishes back to a topic.
:::

:::{grid-item-card} {octicon}`broadcast;1.1em` Kinesis
:link: /integrations/streams/kinesis
:link-type: doc
One reader per shard, exact resume from the stored sequence number, and resharding followed live.
:::

:::{grid-item-card} {octicon}`broadcast;1.1em` Pulsar
:link: /integrations/streams/pulsar
:link-type: doc
One reader per partition, with acknowledgements held until the micro-batch is published.
:::

:::{grid-item-card} {octicon}`broadcast;1.1em` Pub/Sub
:link: /integrations/streams/pubsub
:link-type: doc
Pull a subscription with ack-after-publish delivery and message attributes as headers.
:::

:::{grid-item-card} {octicon}`broadcast;1.1em` Event Hubs
:link: /integrations/streams/eventhubs
:link-type: doc
The native AMQP reader, or the Kafka endpoint with no Azure SDK at all.
:::

:::{grid-item-card} {octicon}`file-code;1.1em` Payload formats
:link: /integrations/streams/payload-formats
:link-type: doc
Avro, JSON, and Protobuf decoded in the source, with Confluent Schema Registry support.
:::

::::

For triggers, watermarks, output modes, and checkpoints, which apply to every broker here, see {doc}`/user-guide/moving-data/streaming/index`.

```{toctree}
:hidden:

kafka
kinesis
pulsar
pubsub
eventhubs
payload-formats
```
