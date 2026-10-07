# Airbyte

This page covers {py:meth}`bt.read.airbyte <batcher.api.io_namespace.reader.Reader.airbyte>`, which reads one stream of an Airbyte source connector and keeps the connector's STATE checkpoints.

:::{warning}
Not yet verified against a live Airbyte connector; see [`tests/PENDING_VERIFICATION.md`](https://github.com/stephenoffer/batcher/blob/main/tests/PENDING_VERIFICATION.md). The protocol handling is tested against a fake message stream and a fake connector program only.
:::

| | |
| --- | --- |
| **Read** | {py:meth}`bt.read.airbyte(stream, image= or command=) <batcher.api.io_namespace.reader.Reader.airbyte>` |
| **Extra** | none; Docker for `image=` |
| **Splits** | one, because a connector run is one ordered message stream |

## Run a connector

An Airbyte source is a program that answers `discover` and `read` with newline-delimited JSON messages. Batcher speaks that protocol directly. It runs `discover` for the stream's JSON schema, builds a configured catalog for the one stream, and runs `read`:

```python
# docs: skip
import batcher as bt

users = bt.read.airbyte(
    "users",
    image="airbyte/source-faker:6",
    config={"count": 1000, "seed": 1},
    state="s3://bucket/state/faker-users.json",
)
```

Pass `command=` instead of `image=` to run a connector executable directly, such as the one a PyAirbyte-installed connector provides. A `config` value may be a secret reference. It's resolved when the connector starts and written to a 0600 file in a private temporary directory that is removed when the connector exits. With `image=` that directory is bind-mounted into the container, so the Docker daemon must run on the same host.

## Ordering and checkpoints

Records keep the order the connector emitted them in. In the Airbyte protocol a STATE message means every record before it is covered. Batcher flushes the buffered records as a batch at each STATE and accepts the state only after that batch has been taken by the consumer. Once the whole read is consumed, the last accepted state is written to `state=`, and the next read passes it back with `--state`. With `auto_commit=False` the state is staged until you call `bt.io.Incremental(state=...).commit()` after your write succeeds.

Records after the last STATE are covered by no checkpoint, so the next read delivers them again. That is Airbyte's own at-least-once contract. Per-stream, global, and legacy state are kept in the form the connector expects back. Incremental sync is chosen when `state=` is given and the stream supports it, and `sync_mode=` overrides that.

A TRACE message of type ERROR fails the read with its message, and so does a non-zero exit, with the end of the connector's stderr. A connector that dies part-way never yields a table that looks complete, and the state isn't advanced.

## Types

The stream's JSON schema maps `integer` to int64, `number` to float64, `boolean` to bool, and `string` to string. An `object` or `array` field becomes a string column of JSON text, which the `.json` accessor reads. Pass `schema=` to override the mapping.

## See also

- {doc}`/integrations/apis/http-json`: the `Incremental` state document.
