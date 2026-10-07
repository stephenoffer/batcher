# Broker sinks

This page describes the contract every message-broker sink shares. {py:meth}`ds.write.kafka <batcher.api.io_namespace.writer.Writer.kafka>`, {py:meth}`ds.write.pulsar <batcher.api.io_namespace.writer.Writer.pulsar>`, {py:meth}`ds.write.kinesis <batcher.api.io_namespace.writer.Writer.kinesis>`, {py:meth}`ds.write.pubsub <batcher.api.io_namespace.writer.Writer.pubsub>`, and {py:meth}`ds.write.eventhubs <batcher.api.io_namespace.writer.Writer.eventhubs>` publish each row of each micro-batch as one message, and they read the same columns, refuse the same mistakes, and report a micro-batch written only once the broker has acknowledged all of it.

:::{warning}
The Pulsar, Kinesis, Pub/Sub, and Event Hubs sinks are not yet verified against a live broker; see tests/PENDING_VERIFICATION.md. Their unit tests drive fake clients that model the documented client calls.
:::

## The column contract

A sink reads a fixed set of columns and ignores every other one, so the six-column schema a broker source delivers can be written straight back after a `select`. `value` is required and is the message payload. `key`, `topic`, `partition`, and `headers` are read where the broker has somewhere to put them. Payload columns may be string or binary. A string column is UTF-8 encoded on the way out, and a column of any other type is refused at the first micro-batch with its name and type. `value_format=` and `key_format=` serialize a typed column through the same codecs the sources decode with, as {doc}`payload-formats` describes.

The following table lists which columns each sink reads:

| Sink | `value` | `key` | `topic` | `partition` | `headers` |
| --- | --- | --- | --- | --- | --- |
| Kafka | payload, null is a tombstone | message key | per-row topic | partition number | headers |
| Pulsar | payload | partition key | per-row topic | not read | properties, as text |
| Kinesis | record data | partition key | per-row stream | not read | not read |
| Pub/Sub | message data | ordering key, with `ordered=True` | per-row topic path | not read | attributes, as text |
| Event Hubs | event body | partition key | not read | partition id | application properties |

Only Kafka has a tombstone record, so the other sinks refuse a null `value` rather than send an empty message.

## Delivery, ordering, and retries

Every broker sink is at-least-once. A micro-batch that fails to publish fails the query, and the restarted query replays it from the checkpoint, so a consumer can see a record twice and never misses one. What differs between brokers is the scope in which order holds and what the broker offers to recognize a retried record.

The following table states those guarantees per sink. Each sink declares the same statements in code as its `capabilities` attribute, a `SinkCapabilities` record:

| Sink | Ordering | Batching | Retry identity |
| --- | --- | --- | --- |
| Kafka | per partition; a key hashes to one partition | librdkafka client batches, flushed per micro-batch | `enable_idempotence=True` dedups the client's own retries |
| Pulsar | per partition; the key routes to one partition | producer batching, flushed per micro-batch | broker-side deduplication on `producer_name=` plus derived sequence ids |
| Kinesis | per partition key, only with `ordered=True` | `PutRecords` of at most 500 records or 5 MiB, failed records resent | none |
| Pub/Sub | per ordering key, only with `ordered=True` | the publisher client's batches, every future awaited | none |
| Event Hubs | per partition; the key routes to one partition | `EventDataBatch` per routing, split at the size limit | none |

Where the broker offers no deduplication, `dedup_ids=<writer name>` stamps every record with a `batcher-dedup-id` header, attribute, or property holding `<writer name>:<batch id>:<row>`. A replayed micro-batch carries the same ids for the same rows when the pipeline emits the batch's rows in the same order, so a consumer can drop a duplicate by that id. Kinesis records have no metadata to carry one, so the Kinesis sink refuses `dedup_ids=`.

## Publish a stream

The following example stands a rate source in for a real stream and writes it to Kafka. Swap the last call for any of the other four sinks:

```python
# docs: skip
import batcher as bt

events = bt.read.rate_micro_batch(100, num_rows=1000).select(
    value=bt.col("value").cast("string"), key=(bt.col("value") % 8).cast("string")
)
query = events.write.kafka("numbers", bootstrap_servers="broker:9092", dedup_ids="numbers-writer")
# query = events.write.pulsar("numbers", service_url="pulsar://broker:6650", producer_name="numbers")
# query = events.write.kinesis("numbers", region="us-east-1")
# query = events.write.pubsub("projects/<project>/topics/numbers", ordered=True)
# query = events.write.eventhubs("numbers", connection_str="env:EVENTHUBS_CONN")
query.await_termination()
```

Credentials follow the sources: a connection string, token, or key can be a secret reference such as `env:NAME`, resolved on the worker that opens the client.

## See also

- {doc}`kafka`, {doc}`pulsar`, {doc}`kinesis`, {doc}`pubsub`, {doc}`eventhubs`: each broker's reader and sink options.
- {doc}`/user-guide/moving-data/streaming/index`: triggers, output modes, and checkpoints.
- {doc}`/integrations/orchestration/retries-and-idempotency`: making a consumer idempotent.
