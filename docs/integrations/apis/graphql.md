# GraphQL

This page covers {py:meth}`bt.read.graphql <batcher.api.io_namespace.reader.Reader.graphql>`, which reads the results of a GraphQL query as a lazy dataset, paging a cursor variable.

:::{warning}
Not yet verified against a live GraphQL service; see `tests/PENDING_VERIFICATION.md`. The request and response shapes are tested against a local fake server only.
:::

| | |
| --- | --- |
| **Read** | {py:meth}`bt.read.graphql(url, query, records_path=) <batcher.api.io_namespace.reader.Reader.graphql>` |
| **Extra** | none |
| **Splits** | one, because the cursor of page N+1 is in page N |

## Read a connection

The query is POSTed as `{"query": ..., "variables": ...}`. Point `records_path` at the list of records inside `data`, and `page_info_path` at a Relay `pageInfo` object. While `hasNextPage` is true, the next request sends `endCursor` back as the variable named by `cursor_variable`, which defaults to `after`:

```python
# docs: skip
import batcher as bt

issues = bt.read.graphql(
    "https://api.github.com/graphql",
    """
    query($owner: String!, $name: String!, $after: String) {
      repository(owner: $owner, name: $name) {
        issues(first: 100, after: $after) {
          nodes { number title createdAt }
          pageInfo { endCursor hasNextPage }
        }
      }
    }
    """,
    variables={"owner": "apache", "name": "arrow"},
    records_path="repository.issues.nodes",
    page_info_path="repository.issues.pageInfo",
    auth=bt.io.BearerToken("env:GITHUB_TOKEN"),
)
```

The query must declare the cursor variable. Without `page_info_path` the reader takes one page. `schema=`, `headers=`, `auth=`, `retry=`, `max_pages=`, and `incremental=` work as they do for {doc}`/integrations/apis/http-json`. An `Incremental` with no `cursor_field` resumes from the last accepted `endCursor`.

## Errors fail the read

GraphQL reports failure inside a successful HTTP response. A response can carry status 200, a partial `data` object, and a non-empty `errors` list at once. A reader that looks only at `data` returns a table silently missing whatever the failed fields held.

So any entry in `errors` fails the read with every message and path, even when `data` came with it, and so does a response with no `data`. An error on page five fails the read rather than returning the first four pages as if they were everything. There is no option to accept a partial result, because the reader can't know whether the missing part mattered.

## See also

- {doc}`/integrations/apis/http-json`: the options shared by every API reader.
