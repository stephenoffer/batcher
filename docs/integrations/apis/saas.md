# SaaS connectors

This page covers four readers and one writer built on {doc}`the HTTP JSON source </integrations/apis/http-json>`: GitHub, Salesforce, Google Sheets, and SharePoint or OneDrive through Microsoft Graph. Each inherits its retries, its secret-reference auth, and the `Incremental` state document.

:::{warning}
None of these connectors has been verified against the live service yet; see `tests/PENDING_VERIFICATION.md`. Each is implemented against the service's documented REST API and tested against a local fake that speaks those request and response shapes.
:::

The following table summarizes what each one reads and how it resumes:

| Connector | Reads | Extra | Incremental |
| --- | --- | --- | --- |
| {py:meth}`bt.read.github <batcher.api.io_namespace.reader.Reader.github>` | issues, pull requests, releases | none | `since` on issues, client-side elsewhere |
| {py:meth}`bt.read.salesforce <batcher.api.io_namespace.reader.Reader.salesforce>` | an sObject, by Bulk API 2.0 query job | none | `SystemModstamp >=` in the SOQL |
| {py:meth}`bt.read.google_sheets <batcher.api.io_namespace.reader.Reader.google_sheets>` | a range of a sheet | `gsheets` for default credentials | none |
| {py:meth}`bt.read.sharepoint <batcher.api.io_namespace.reader.Reader.sharepoint>` | drive items and their bytes | none | the Graph delta link |

## GitHub

{py:meth}`bt.read.github(repo, resource) <batcher.api.io_namespace.reader.Reader.github>` reads `"issues"`, `"pulls"`, or `"releases"` from the REST API, 100 records a page, following the `Link` header:

```python
# docs: skip
import batcher as bt

issues = bt.read.github(
    "apache/arrow",
    "issues",
    token="env:GITHUB_TOKEN",
    incremental=bt.io.Incremental(state="state/arrow-issues.json"),
)
```

The schema is declared per resource from GitHub's documented fields, so a repository whose first page has no milestones doesn't infer a null column. GitHub's issues endpoint also lists pull requests, and `pull_request` is non-null on those rows. A 403 or 429 whose `x-ratelimit-remaining` reads `0` waits until `x-ratelimit-reset`, and a secondary limit's `Retry-After` is honored. The token is sent as an unredirected header and appears in no log line, repr, or identity.

An `Incremental` defaults to the cursor field `updated_at` (`published_at` for releases) and the key `id`. On issues it sends `since`, so the server returns only what changed. The pulls and releases endpoints take no such parameter, so there the filter runs on the client, which is correct but reads every page. Updated records arrive as new versions and records already delivered are dropped, so an update doesn't duplicate a row. Use `base_url=` for GitHub Enterprise Server.

## Salesforce

{py:meth}`bt.read.salesforce(sobject, instance_url=, schema=) <batcher.api.io_namespace.reader.Reader.salesforce>` runs a Bulk API 2.0 query job. It creates the job, polls it until `JobComplete`, and reads the CSV results page by page, following the `Sforce-Locator` header:

```python
# docs: skip
import pyarrow as pa

accounts = bt.read.salesforce(
    "Account",
    instance_url="https://acme.my.salesforce.com",
    schema=pa.schema(
        [
            ("Id", pa.string()),
            ("Name", pa.string()),
            ("AnnualRevenue", pa.float64()),
            ("SystemModstamp", pa.timestamp("ms", tz="UTC")),
        ]
    ),
    auth=bt.io.OAuth2ClientCredentials(
        token_url="https://acme.my.salesforce.com/services/oauth2/token",
        client_id="<consumer key>",
        client_secret="env:SF_CLIENT_SECRET",
    ),
    include_deleted=True,
    incremental=bt.io.Incremental(state="state/accounts.json"),
)
```

The schema is required, because it *is* the `SELECT` list as well as the column types. Bulk results are CSV text, and inferring types from text is how an Id or a postal code turns into a number. Each page is parsed as strings and cast to the declared types, and a value that doesn't fit fails the read naming the field.

`include_deleted=True` runs `queryAll`, which also returns deleted and archived records, and adds the `IsDeleted` column so a downstream merge can apply the deletes. An `Incremental` defaults to the cursor field `SystemModstamp` and the key `Id`, adds `SystemModstamp >= <watermark>` to the query, and drops the boundary records already delivered. `where=` adds your own SOQL condition. A `Failed` or `Aborted` job fails the read with the job's error message.

## Google Sheets

{py:meth}`bt.read.google_sheets(spreadsheet_id, range) <batcher.api.io_namespace.reader.Reader.google_sheets>` reads an A1 range in one request:

```python
# docs: skip
budget = bt.read.google_sheets("1AbC...", "Budget!A1:F")
```

The first row names the columns unless you pass `header=False`, which names them `c0`, `c1`, and so on, or `header=[...]`. Values are read with `value_render="UNFORMATTED_VALUE"` by default, so numbers and booleans arrive typed and an empty cell is null. A column whose cells mix types fails with its name. Declare `schema=`, or read with `value_render="FORMATTED_VALUE"` to take every cell as text.

{py:meth}`ds.write.google_sheets(spreadsheet_id, range) <batcher.api.io_namespace.writer.Writer.google_sheets>` writes the other way, `batch_rows` rows per request:

```python
# docs: skip
summary.write.google_sheets("1AbC...", "Report!A1", mode="overwrite", batch_rows=500)
```

The overwrite scope is the range you name and nothing else. `mode="overwrite"` clears exactly that range, then writes the header and rows into it without shifting cells outside it. `mode="append"` adds rows, without a header, after the table the range holds. Timestamps and decimals are written as text, NaN and infinity are refused, and an overwrite refuses to run from several workers because each would clear the last one's rows.

Authenticate with a {py:class}`bt.io.BearerToken <batcher.io.BearerToken>` holding an access token. A `cmd:` reference can run `gcloud auth print-access-token`. With no `auth=`, Google Application Default Credentials are used through `google-auth`, which needs `pip install 'batcher-engine[gsheets]'`.

## SharePoint and OneDrive

{py:meth}`bt.read.sharepoint(drive_id= or site_id=) <batcher.api.io_namespace.reader.Reader.sharepoint>` lists a document library with the Microsoft Graph `delta` function, one row per drive item:

```python
# docs: skip
docs = bt.read.sharepoint(
    site_id="contoso.sharepoint.com,<site-guid>,<web-guid>",
    auth=bt.io.OAuth2ClientCredentials(
        token_url="https://login.microsoftonline.com/<tenant>/oauth2/v2.0/token",
        client_id="<app id>",
        client_secret="env:GRAPH_SECRET",
        scope="https://graph.microsoft.com/.default",
    ),
    state="state/contracts-delta.json",
    folder="/Contracts",
)
```

Each row carries `id`, `name`, `parent_path`, `parent_id`, `size`, `last_modified`, `etag`, `ctag`, `mime_type`, `web_url`, `is_folder`, and `deleted`. With `state=`, the delta link from the end of the walk is kept, and the next read returns only what changed since. A rename arrives as the same `id` under a new `name`, a change arrives with a new `etag`, and a delete arrives with `deleted = true`, so a merge on `id` reconciles a downstream copy. `folder=` keeps the items under one path, along with every delete, because a deleted item often arrives without a path.

`include_content=True` adds a `content` column with each file's bytes, one request per file. Graph answers with a redirect to a pre-authenticated download URL, and the bearer token isn't sent to that host. An expired delta link answers HTTP 410, which fails the read and tells you to delete the state document to re-list the drive.

## Requirements and limitations

- Each reader is one split and runs on one worker.
- The request concurrency cap is per process, with no cluster-wide quota.
- The Salesforce reader doesn't infer a schema; every column is declared.
- The SharePoint reader walks the drive root's delta, which is the form SharePoint libraries support, and filters by folder on the client.
- Google Sheets reads only a single range per call.

## See also

- {doc}`/integrations/apis/http-json`: retries, auth providers, and the `Incremental` state.
- {doc}`/user-guide/trust/secrets`: the secret reference schemes.
