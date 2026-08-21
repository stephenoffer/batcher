# Payload formats

A broker delivers a message as opaque bytes. This page describes how to turn those bytes
into typed columns with `value_format`, including Avro and Protobuf payloads written
against a Confluent Schema Registry.

Every broker source shares one message schema, in which `value` and `key` are `binary`.
Naming a format decodes them in the source itself, so the stream's real schema is known
before a message is polled and every expression over the payload type-checks at plan time:

```python
# docs: skip
orders = bt.read.kafka(
    "orders",
    bootstrap_servers="broker-1:9092",
    value_format="avro",
    schema_registry="http://schema-registry:8081",
)
orders.schema()  # value: struct<user: string, amount: int64>
```

The same options work on every broker source, because the decode belongs to the shared
broker base rather than to any one client: Kafka, Kinesis, Pulsar, Pub/Sub, and Event Hubs
all take them.

## Why decode in the source

You can decode a payload with `map_batches` and a hand-written function, and before these
formats existed that was the only option. It costs three things.

The engine cannot report the stream's schema, because the shape of the payload is inside a
Python callback the optimizer cannot see. `Dataset.schema()` answers `binary`, so nothing
downstream can be type-checked until rows arrive. A projection cannot be pushed into the
decode either, so a query reading one field of a fifty-field record still pays for all
fifty. And a malformed record raises from inside user code, where the engine cannot tell it
from a bug in the callback and has no policy to apply.

Naming a format moves the decode to the source boundary, which is where the other
Arrow-less formats already sit. It runs once per micro-batch over the whole column, never
per row.

## The formats

Batcher ships five. Each is named by the string you pass to `value_format` or `key_format`.

| Format | Decodes to | Reader schema comes from |
|---|---|---|
| `avro` | `struct` | `value_schema=`, or a registry subject |
| `json` | `struct` | `value_schema=`, or a registry subject |
| `protobuf` | `struct` | `value_schema=` (a generated message class) |
| `string` | `string` | nothing to declare |
| `bytes` | `binary` | nothing to declare |

`string` is worth naming even though it looks like a cast. A source declared
`value_format="string"` reports `string` in its schema, so `col("value").str.contains(...)`
resolves at plan time rather than failing on the first micro-batch.

### Avro

Point `value_schema` at a schema, as a dict, JSON text, or a path to a `.avsc` file:

```python
# docs: skip
orders = bt.read.kafka(
    "orders",
    bootstrap_servers="broker-1:9092",
    value_format="avro",
    value_schema="schemas/order.avsc",
)
```

Avro maps to the same Arrow types the file reader produces, logical types included: `date`
becomes `date32`, `timestamp-micros` becomes `timestamp[us]`, and `decimal` keeps its
precision and scale. A union of two or more non-null branches becomes a struct with one
`memberN` field per branch, exactly as the Avro file reader does.

### JSON

Declare the document shape:

```python
# docs: skip
events = bt.read.kafka(
    "events",
    bootstrap_servers="broker-1:9092",
    value_format="json",
    value_schema={"user": "string", "amount": "int64", "ts": "timestamp[us]"},
)
```

The schema is required, not merely recommended, and the reason is the same one that makes
declaring it useful: the plan is built before the first message is polled, so a type
discovered on the first poll arrives after every expression that needed it. Inferring would
not fail loudly either — the plan would carry an empty struct, the decode would produce real
fields, and the batch would be coerced back on the way out.

Parsing is pyarrow's own JSON reader over the batch, so it is the same C++ path a JSON file
takes and it does not run per row. A field the schema does not mention is ignored rather
than rejected, which is what lets a producer add fields without stopping the consumer.

### Protobuf

Pass the generated message class:

```python
# docs: skip
import orders_pb2

orders = bt.read.kafka(
    "orders",
    bootstrap_servers="broker-1:9092",
    value_format="protobuf",
    value_schema=orders_pb2.Order,
)
```

A Schema Registry stores `.proto` text, which cannot become a descriptor without running
`protoc`, so the generated class is required even when a registry supplies the framing.

Needs `pip install 'batcher-engine[protobuf]'`.

## Schema Registry

A payload written by a Confluent serializer is not a bare record. It carries five bytes of
framing: a zero magic byte and a big-endian schema id. Decoding it as if it were bare does
not fail, it returns plausible garbage, because those five bytes are consumed as field data.

Pass `schema_registry` and the framing is handled:

```python
# docs: skip
orders = bt.read.kafka(
    "orders",
    bootstrap_servers="broker-1:9092",
    value_format="avro",
    schema_registry="http://schema-registry:8081",
)
```

The subject defaults to `"{topic}-value"` for the value column and `"{topic}-key"` for the
key, which is the TopicNameStrategy every Confluent serializer registers under. Override it
with `value_subject=` or `key_subject=` for a topic using a different strategy. A registry
behind basic auth takes `schema_registry_auth="user:password"`, the Confluent
`basic.auth.user.info` spelling.

### What happens when a producer evolves its schema

The framing carries a schema id per message, so a producer mid-rollout writes two versions
into the same topic and both land in one micro-batch. Batcher resolves the *reader* schema
once, from the subject's latest version, and reads every record against it using Avro's own
schema resolution. A record written under an older version arrives with the new fields
null; a record written under a newer one is projected down.

That is what keeps the column a single Arrow type. Decoding each record against its own
writer schema would give a batch whose type changed between rows, which nothing downstream
can concatenate.

Pin the reader schema explicitly, with `value_schema=` alongside `schema_registry=`, when
you want a consumer to stay on a fixed column set while producers move ahead of it.

## Malformed records

`value_decode_mode` decides what a record that will not decode costs, matching Spark's
`FAILFAST` and `PERMISSIVE`:

| Mode | Behavior |
|---|---|
| `"fail"` (default) | Raise, naming the row within the batch. |
| `"permissive"` | Null that row's payload and keep the rest of the batch. |

```python
# docs: skip
events = bt.read.kafka(
    "events",
    bootstrap_servers="broker-1:9092",
    value_format="avro",
    schema_registry="http://schema-registry:8081",
    value_decode_mode="permissive",
)
```

Failing is the default deliberately. A stream that silently nulls every record after a
producer changes format is a stream that reports success while delivering nothing, and
nothing in the progress record distinguishes it from an idle topic. Reach for permissive
when you know the tail of the topic holds legacy records, and filter the nulls explicitly.

A null payload is not a decode failure. Kafka's tombstone record has a null `value` and
means the key was deleted, so it survives the decode as a null in every mode.

## Writing

The write side takes the same options, so a pipeline that reads Avro off one topic and
writes Avro to another names the format once per side:

```python
# docs: skip
query = (
    bt.read.kafka(
        "orders-raw",
        bootstrap_servers="broker-1:9092",
        value_format="avro",
        schema_registry="http://schema-registry:8081",
    )
    .filter(bt.col("value").struct.field("amount") > 100)
    .write.kafka(
        "orders-large",
        bootstrap_servers="broker-1:9092",
        value_format="avro",
        schema_registry="http://schema-registry:8081",
    )
)
```

The output is framed for the registry, so a Confluent deserializer reads it with no shim.
Framing needs the writer schema's registered id, so encode from a subject rather than from
an inline `value_schema`, or drop `schema_registry` to write bare payloads.

## Requirements and limitations

- Avro needs `pip install 'batcher-engine[avro]'`; Protobuf needs `[protobuf]`. JSON,
  `string`, and `bytes` need nothing beyond the base install.
- The registry client resolves a subject's latest version once, at query start, and does not
  re-poll it. A reader schema that changed mid-query would change the stream's Arrow schema
  in flight, which no downstream operator can absorb. Restart the query to pick up a new
  reader schema.
- Protobuf needs the generated message class even with a registry, for the reason above.
- A JSON payload needs `value_schema=` or a registry subject; it is never inferred, for the
  reason above.
- JSON Schema documents from a registry are translated only for the
  object-with-`properties` shape a message payload uses. `oneOf` and cross-document `$ref`
  are refused rather than approximated, since a silently wrong column type is worse than an
  explicit `value_schema=`.

## See also

- {doc}`Kafka </integrations/streams/kafka>`: the connector these options are set on.
- {doc}`Streaming pipelines </tutorials/pipelines/streaming-pipeline>`: triggers, watermarks, and checkpoints.
- {doc}`Reading data </user-guide/moving-data/reading-data>`: the Avro file reader, which shares this type mapping.
