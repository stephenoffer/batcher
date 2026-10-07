# Flight SQL

{py:func}`batcher.integrations.flightsql.serve` starts an Arrow Flight SQL service over a {py:class}`Session <batcher.Session>`, so a remote client sends SQL and reads Arrow back over gRPC. It's a pilot covering statements, bound parameters, streamed results, cancellation, and token authentication.

:::{warning}
Not yet verified against a live Flight SQL client such as the ADBC or JDBC driver; see tests/PENDING_VERIFICATION.md. The tests run a localhost round trip with a pyarrow Flight client, and check the protobuf encoding against the protobuf runtime.
:::

## Start a server

`serve` returns a running pyarrow Flight server. Its `port` attribute is the bound port, `serve()` blocks until it stops, and `shutdown()` stops it. Every client runs against the one session, so a table registered on it is visible to all of them. The `flightsql` extra names pyarrow with Flight, which pyarrow's PyPI wheels already include.

```python
import batcher as bt
from batcher.integrations import flightsql

session = bt.Session()
session.register("orders", bt.from_pydict({"id": [1, 2, 3]}))
server = flightsql.serve(session, "grpc://127.0.0.1:0", auth="s3cret-token")
print(server.port > 0)
# True
server.shutdown()
```

`auth` is one shared bearer token. Every call must send it as an `authorization: Bearer <token>` header, and the server compares it in constant time. It may be a secret reference such as `env:NAME` or `file:PATH`, resolved once at startup. With `auth=None` the server accepts every call, which is only appropriate on a trusted interface. Over anything but localhost, listen on a `grpc+tls://` location and pass `tls_certificates=`, because a bearer token over plaintext gRPC is readable on the path.

## What a client can do

The service answers these Flight SQL commands:

| Command | Flight call | Effect |
|---|---|---|
| `CommandStatementQuery` | `GetFlightInfo`, then `DoGet` | Runs a query and streams the result |
| `CommandStatementUpdate` | `DoPut` | Runs a statement for its effect |
| `CreatePreparedStatement`, `ClosePreparedStatement` | `DoAction` | Holds a statement with `?` placeholders |
| `CommandPreparedStatementQuery` | `DoPut` binds, then `GetFlightInfo` and `DoGet` run | Runs a prepared query with its bound row |
| `CommandPreparedStatementUpdate` | `DoPut` | Runs a prepared statement once per bound row |
| `CancelFlightInfo` | `DoAction` | Stops a result stream at its next batch |

Parameters bind through `Session.sql`'s `params=`: the bound batch's first row fills the `?` placeholders by position, as typed values. A result streams from `Dataset.iter_batches`, one Arrow batch per Flight message. A cancelled stream ends with a cancellation error, never with a short result that looks complete. Each `DoGet` ticket is read once.

## Requirements and limitations

There are no transactions. `BeginTransaction` and any command carrying a `transaction_id` are refused. An update reports a record count of `-1`, Flight SQL's "unknown".

Catalog metadata commands such as `GetSqlInfo`, `GetTables`, and `GetCatalogs` are refused as unimplemented. Query `information_schema` instead. A client that requires `GetSqlInfo` on connect won't work with the pilot. A prepared statement doesn't report its parameter or result schema when it's created.

Cancellation is checked between batches, so a query computing its first batch notices only once that batch is ready.

## See also

- {doc}`dbapi`: the in-process adapter for Python clients.
- {doc}`/api/operations/sql-clients`: the reference for `serve`.
