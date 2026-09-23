# Payload formats

This page describes how to turn broker payloads into typed columns with `value_format`, including Avro and Protobuf payloads written against a Confluent Schema Registry. A broker delivers each message as opaque bytes, and naming the wire format makes Batcher decode them in the source, one column per micro-batch.

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
orders.schema  # value: struct<user: string, amount: int64>
```

The same options work on every broker source, because the decode belongs to the shared broker base rather than to any one client. Kafka, Kinesis, Pulsar, Pub/Sub, and Event Hubs all take them, and the Kafka sink takes them for encoding.

## Why decode in the source

You can decode a payload with `map_batches` and a hand-written function. It costs three things.

The engine can't report the stream's schema, because the shape of the payload is inside a Python callback the optimizer can't see. {py:obj}`Dataset.schema <batcher.Dataset.schema>` answers `binary`, so nothing
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
not fail loudly either. The plan would carry an empty struct, the decode would produce real
fields, and the batch would be coerced back on the way out.

Parsing is pyarrow's own JSON reader over the batch, so it is the same C++ path a JSON file
takes and it does not run per row. A field the schema does not mention is ignored rather
than rejected, which is what lets a producer add fields without stopping the consumer.

One message is always one row. An empty or whitespace-only message decodes to a null, and a message holding two documents is a malformed record rather than two rows. Batcher checks the batch parse against the number of messages it was given, and when they disagree, or the batch will not parse, it re-parses that batch one message at a time so a bad message affects only its own row.

With `schema_registry=`, JSON payloads are read as a Confluent JSON Schema serializer writes them: the five-byte framing is checked and stripped on decode and written on encode, as for Avro. A JSON topic whose producer does not frame its messages takes `value_schema=` without a registry.

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

### When the registry fails

A registry that is unreachable, or answers with an error, fails the query in every decode mode, including `"permissive"`. That failure says nothing about the message being decoded, and nulling on it would turn an outage into a stream of empty records. A schema id the registry answers "not found" for is different: the id came out of the payload, so that record is malformed, and `"permissive"` nulls it. Batcher looks each unknown id up once per micro-batch, not once per message.

Without `schema_registry=`, an Avro record must use every byte of its payload. A Confluent-framed payload read as bare Avro leaves bytes over, so it is reported as malformed, with a hint to pass the registry, instead of decoding into plausible garbage.

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

The mode applies per record for every format. Under `"permissive"`, one bad message in a JSON batch nulls that message and nothing else. You can see this on the codec itself, which is what the source runs on each micro-batch:

```python
import pyarrow as pa
from batcher.io.formats.streaming.codecs import resolve_codec

codec = resolve_codec("json", schema={"user": "string", "amount": "int64"}, mode="permissive")
payloads = pa.array(
    [b'{"user":"u1","amount":10}', b"{not json", b"   ", b'{"user":"u4","amount":4}', None]
)
decoded = codec.decode(payloads)
print(decoded.to_pylist())
# [{'user': 'u1', 'amount': 10}, None, None, {'user': 'u4', 'amount': 4}, None]
assert len(decoded) == len(payloads)
```

Failing is the default deliberately. A stream that silently nulls every record after a
producer changes format is a stream that reports success while delivering nothing, and
nothing in the progress record distinguishes it from an idle topic. Reach for permissive
when you know the tail of the topic holds legacy records, and filter the nulls explicitly.

A null payload is not a decode failure. Kafka's tombstone record has a null `value` and
means the key was deleted, so it survives the decode as a null in every mode.

Both of those behaviours are visible without a broker. Standing a batch of the broker column
contract in for the source, a malformed payload and a tombstone each null their own row and
leave the rest of the batch intact:

```python
import batcher as bt
import pyarrow as pa
from batcher import col

schema = pa.schema(
    [
        ("key", pa.binary()),
        ("value", pa.binary()),
        ("partition", pa.int64()),
        ("offset", pa.int64()),
        ("timestamp", pa.int64()),
        ("topic", pa.string()),
    ]
)
batch = pa.record_batch(
    {
        "key": [b"u1", b"u2", b"u3", None],
        "value": [
            b'{"user":"u1","amount":10}',
            b"not json at all",  # malformed
            b'{"user":"u3","amount":7}',
            None,  # tombstone
        ],
        "partition": [0, 0, 1, 1],
        "offset": [11, 12, 4, 5],
        "timestamp": [1700000000000] * 4,
        "topic": ["orders"] * 4,
    },
    schema=schema,
)
orders = bt.from_batches(lambda: iter([batch]), schema)

decoded = orders.select(
    user=col("value").cast("string").json.extract_string("$.user"),
    amount=col("value").cast("string").json.extract_int("$.amount"),
)
print(decoded.to_pydict())
# {'user': ['u1', None, 'u3', None], 'amount': [10, None, 7, None]}
```

Two rows decoded, two nulled, and no exception. Filtering the nulls is then the explicit step
the paragraph above asks for:

```python
print(decoded.filter(col("user").is_not_null()).count())
# 2
```

That is the ad-hoc shape. Against a real topic, name `value_format` instead and the source
decodes in the reader, so the schema is known before a message is polled rather than
recovered per expression.

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

- Avro needs `pip install 'batcher-engine[avro]'`, and Protobuf needs `[protobuf]`. JSON,
  `string`, and `bytes` need nothing beyond the base install.
- The registry client resolves a subject's latest version once, at query start, and does not
  re-poll it. A reader schema that changed mid-query would change the stream's Arrow schema
  in flight, which no downstream operator can absorb. Restart the query to pick up a new
  reader schema.
- Protobuf needs the generated message class even with a registry, for the reason above.
- A JSON payload needs `value_schema=` or a registry subject. It's never inferred, for the reason above.
- JSON Schema documents from a registry are translated only for the
  object-with-`properties` shape a message payload uses. A property using `$ref`, `oneOf`,
  `anyOf`, `allOf`, or `not`, or a `type` list naming two non-null types, is refused with the
  property named, since a silently wrong column type is worse than an explicit
  `value_schema=`.
- A Protobuf codec decodes one message of its `.proto`. A framed payload whose message index
  names a different message is a malformed record, not a decode of the wrong fields.

## See also

- {doc}`Kafka </integrations/streams/kafka>`: the connector these options are most often set on, and the one sink that encodes with them.
- {doc}`/integrations/streams/index`: every broker source that accepts these options.
- {doc}`Streaming pipelines </getting-started/tutorials/pipelines/streaming-pipeline>`: triggers, watermarks, and checkpoints.
- {doc}`Reading data </user-guide/moving-data/reading-data>`: the Avro file reader, which shares this type mapping.
