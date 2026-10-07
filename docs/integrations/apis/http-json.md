# HTTP JSON APIs

This page covers {py:meth}`bt.read.http_json <batcher.api.io_namespace.reader.Reader.http_json>`, which reads any paginated JSON API as a lazy dataset, one Arrow batch per page.

| | |
| --- | --- |
| **Read** | {py:meth}`bt.read.http_json(url, pagination=, records_path=) <batcher.api.io_namespace.reader.Reader.http_json>` |
| **Extra** | none: the standard library's `urllib` plus pyarrow |
| **Splits** | one, because a paged API is walked in order |
| **Pushdown** | projection after each page; filters run in the engine |

## Read a paged API

The examples on this page run against a small local server so you can execute them. It serves ten records, four per page, and returns the next cursor in the body:

```python
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import batcher as bt

RECORDS = [
    {"id": i, "status": "open" if i % 3 else "closed", "updated_at": f"2024-05-{i + 1:02d}T00:00:00Z"}
    for i in range(10)
]


class Api(BaseHTTPRequestHandler):
    def do_GET(self):
        query = {k: v[-1] for k, v in parse_qs(urlsplit(self.path).query).items()}
        since = query.get("since")
        rows = [r for r in RECORDS if since is None or r["updated_at"] >= since]
        start = int(query.get("cursor", 0))
        nxt = start + 4 if start + 4 < len(rows) else None
        body = json.dumps({"data": rows[start : start + 4], "meta": {"next": nxt}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


server = ThreadingHTTPServer(("127.0.0.1", 0), Api)
threading.Thread(target=server.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{server.server_address[1]}/tickets"
```

Name the paging style with a typed option and say where each page keeps its records:

```python
tickets = bt.read.http_json(
    url,
    pagination=bt.io.CursorPagination(cursor_path="meta.next"),
    records_path="data",
)
print(tickets.schema.names)
# ['id', 'status', 'updated_at']
print(tickets.filter(bt.col("status") == "closed").sort("id").to_pydict()["id"])
# [0, 3, 6, 9]
```

Four pagination styles cover most APIs. Each is a frozen option object, so it pickles to a worker and states its contract in its signature:

| Option | The next page comes from | Stops when |
| --- | --- | --- |
| {py:class}`CursorPagination(cursor_path=, param=) <batcher.io.CursorPagination>` | a cursor in the body, sent back as a query parameter | the cursor is missing or empty, or `has_more_path` reads false |
| {py:class}`NextLinkPagination(path=None) <batcher.io.NextLinkPagination>` | the `Link: <...>; rel="next"` header, or a next-URL field in the body | there is no next link |
| {py:class}`OffsetPagination(limit=) <batcher.io.OffsetPagination>` | `offset` advanced by `limit` | a page is short, or the offset reaches `total_path` |
| {py:class}`PagePagination(size=) <batcher.io.PagePagination>` | `page` advanced by one | a page is empty or short, or `total_pages_path` is reached |

A field path is dotted (`"meta.next"`). Use a tuple for a key that itself contains a dot, such as `("@odata.nextLink",)`.

## Schema

Pass `schema=` to declare the columns. A declared schema is the contract: fields beyond it aren't read, and an ISO-8601 string in a field declared as a timestamp, date, or time is parsed by Arrow. Without `schema=`, Batcher infers the schema from the first page's records and holds every later page to it. A later page carrying a field the first page didn't have, or a value in a field the first page held only nulls in, fails the read with a message naming the field, rather than dropping or retyping it.

```python
import pyarrow as pa

typed = bt.read.http_json(
    url,
    pagination=bt.io.CursorPagination(cursor_path="meta.next"),
    records_path="data",
    schema=pa.schema([("id", pa.int64()), ("updated_at", pa.timestamp("s", tz="UTC"))]),
)
print(typed.schema)
# id: int64
# updated_at: timestamp[s, tz=UTC]
```

Inferring the schema costs one extra request for the first page.

## Auth, retries, and the concurrency cap

`auth=` takes a provider that holds a secret *reference*, resolved when the request is sent:

- {py:class}`bt.io.BearerToken("env:API_TOKEN") <batcher.io.BearerToken>` sends `Authorization: Bearer <token>`.
- {py:class}`bt.io.OAuth2ClientCredentials(token_url=, client_id=, client_secret=) <batcher.io.OAuth2ClientCredentials>` fetches a token with the client-credentials grant, caches it in the process until shortly before it expires, and fetches a fresh one once if the API answers 401.

A value in `headers=` may also be a secret reference, such as `{"X-Api-Key": "env:API_KEY"}`. Auth headers and resolved secret headers aren't forwarded across a redirect to another host, and error messages print the URL without its query string.

{py:class}`bt.io.RetryPolicy <batcher.io.RetryPolicy>` controls retries. By default a 429, 500, 502, 503, or 504, or a network error, is retried up to five attempts in all. The wait is the server's `Retry-After` when it sends one, else exponential backoff with full jitter. Set `rate_limit_reset_header=` for an API that reports its quota reset as epoch seconds. A server asking for a wait longer than `max_retry_after` fails the read rather than parking a worker.

`max_concurrency=` bounds the requests in flight from this process to one host, shared by every source pointed at it.

## Resume where the last read stopped

{py:class}`bt.io.Incremental <batcher.io.Incremental>` keeps a small JSON state document, on local disk or an object store, recording the watermark, the last accepted page cursor, and the dedup tokens still inside the lookback window. Give it a `cursor_field` to resume by watermark:

```python
import os
import tempfile
from datetime import timedelta

inc = bt.io.Incremental(
    state=os.path.join(tempfile.mkdtemp(), "tickets.json"),
    cursor_field="updated_at",
    key="id",
    lookback=timedelta(days=1),
    param="since",
)
pages = bt.io.CursorPagination(cursor_path="meta.next")
first = bt.read.http_json(url, pagination=pages, records_path="data", incremental=inc)
print(len(first.to_pydict()["id"]))
# 10
print(inc.load()["watermark"])
# 2024-05-10T00:00:00Z

RECORDS.append({"id": 10, "status": "open", "updated_at": "2024-05-11T00:00:00Z"})
again = bt.read.http_json(url, pagination=pages, records_path="data", incremental=inc)
print(again.to_pydict()["id"])
# [10]
```

The second read sent `since=2024-05-09T00:00:00Z`, the watermark minus the lookback, and got records 8, 9, and 10 back. Records 8 and 9 had already been delivered with the same `updated_at`, so they were dropped, and only the new record came through. A record updated inside the lookback window arrives as a new version and does pass. A record that shares the boundary value isn't lost, because the bound is inclusive.

Leave `cursor_field` unset to resume by page cursor instead. The next read starts at the last page the previous one accepted and re-reads that page, so give a `key` to drop the overlap.

The new state is staged only after every page has been consumed, so a read that fails part-way leaves the previous state in force and loses nothing. With the default `auto_commit=True` the staged state is committed at that point. A one-shot `read` then `write` drains the read before the sink commits, so pass `auto_commit=False` and call `inc.commit()` after the write succeeds when a failed write must not advance the state.

```python
server.shutdown()
```

## Requirements and limitations

- A paged read is one split, so it runs on one worker.
- The concurrency cap is per process. Several workers, or several processes, reading the same API each get their own cap, so there is no shared cluster-wide quota. Size `max_concurrency` and the number of concurrent reads with that in mind.
- Dedup tokens are kept for the lookback window, so a very wide lookback over a busy API keeps a correspondingly large state document.
- A string cursor field is compared as a UTC timestamp when every value parses as ISO-8601 with an offset, and as text otherwise.
- Only top-level temporal fields in a declared schema are parsed from strings.

## See also

- {doc}`/integrations/apis/graphql` and {doc}`/integrations/apis/saas`: readers built on this one.
- {doc}`/user-guide/trust/secrets`: the secret reference schemes.
- {doc}`/user-guide/moving-data/streaming/index`: running a read under a trigger and checkpoint.
