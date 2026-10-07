# HTTP APIs and SaaS

Batcher reads a paginated HTTP JSON API as a lazy dataset with {py:meth}`bt.read.http_json <batcher.api.io_namespace.reader.Reader.http_json>`. The SaaS connectors on these pages are built on the same request loop: GraphQL, GitHub, Salesforce, Google Sheets, SharePoint and OneDrive, and a bridge to Airbyte source connectors.

```python
# docs: skip
import batcher as bt

orders = bt.read.http_json(
    "https://api.example.com/v1/orders",
    pagination=bt.io.CursorPagination(cursor_path="meta.next_cursor"),
    records_path="data",
    auth=bt.io.BearerToken("env:ORDERS_API_TOKEN"),
)
```

Each page becomes one Arrow batch, and every page is held to one schema, so a page that doesn't fit fails the read instead of quietly changing a column. Requests retry 429 and 5xx answers with backoff and honor `Retry-After`. Credentials are secret references such as `env:NAME`, resolved per request and kept out of the plan, logs, and reprs. A {py:class}`bt.io.Incremental <batcher.io.Incremental>` makes any of these reads resumable from a durable watermark or page cursor.

A paged API can only be walked in order, because page N holds the address of page N+1. Every reader here is therefore one split that runs on one worker. That is the shape of the API rather than a setting to raise.

The generic source and its options need nothing beyond the base install. Google Application Default Credentials need the `gsheets` extra. The Airbyte bridge needs Docker or the connector's own executable.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`globe;1.1em` HTTP JSON
:link: /integrations/apis/http-json
:link-type: doc
Cursor, next-link, offset, and page pagination, auth, retries, and incremental resume.
:::

:::{grid-item-card} {octicon}`git-merge;1.1em` GraphQL
:link: /integrations/apis/graphql
:link-type: doc
A query paged by a Relay cursor, where an error never passes as a short table.
:::

:::{grid-item-card} {octicon}`cloud;1.1em` SaaS connectors
:link: /integrations/apis/saas
:link-type: doc
GitHub, Salesforce Bulk API 2.0, Google Sheets, and SharePoint or OneDrive.
:::

:::{grid-item-card} {octicon}`plug;1.1em` Airbyte
:link: /integrations/apis/airbyte
:link-type: doc
Any Airbyte source connector, with its STATE checkpoints kept.
:::

::::

## See also

- {doc}`/user-guide/trust/secrets`: the secret reference schemes every `auth=` accepts.
- {doc}`/user-guide/moving-data/custom-connectors`: the `Source` protocol these readers implement.
- {doc}`/api/relational/io`: the reader reference.

```{toctree}
:hidden:

http-json
graphql
saas
airbyte
```
