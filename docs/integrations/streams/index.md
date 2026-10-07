# Streams

Batcher reads message brokers as unbounded datasets, and the operators you use on a table work on a stream unchanged. Kafka, Kinesis, Pulsar, Pub/Sub, and Event Hubs all deliver the same six-column schema, so moving a pipeline to another broker means changing the source line:

```python
# docs: skip
import batcher as bt

clicks = bt.read.kafka("clicks", bootstrap_servers="broker-1:9092")
# clicks = bt.read.kinesis("clicks", region="us-east-1")
# clicks = bt.read.pulsar("clicks", service_url="pulsar://pulsar:6650")

query = clicks.write.delta(
    "lake/bronze/clicks",
    trigger=bt.Trigger.processing_time("30 seconds"),
    checkpoint="/var/lib/batcher/ckpt/clicks",
)
```

A reader advances its position only after a micro-batch is published, so a crash replays a batch rather than dropping one, and the checkpoint lets a restarted query resume where it stopped. Readers take Spark's `max_offsets_per_trigger` and `max_bytes_per_trigger` by name, decode Avro, JSON, and Protobuf in the source, and split per partition or shard so ingest spreads across the cluster.

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

:::{grid-item-card} {octicon}`upload;1.1em` Broker sinks
:link: /integrations/streams/sinks
:link-type: doc
Publish a stream back to any of the five brokers, with ordering and retry guarantees stated per broker.
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
sinks
payload-formats
```
