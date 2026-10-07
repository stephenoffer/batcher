"""`salesforce`: an sObject read with a Salesforce Bulk API 2.0 query job.

The documented Bulk API 2.0 query flow, over the shared `http.transport.HttpClient`:

1. ``POST /services/data/v{api}/jobs/query`` with ``{"operation": "query"|"queryAll",
   "query": <SOQL>}`` creates a job;
2. ``GET .../jobs/query/{id}`` is polled until ``state`` is ``JobComplete`` (``Failed`` and
   ``Aborted`` fail the read with the job's ``errorMessage``);
3. ``GET .../jobs/query/{id}/results`` returns CSV, page by page, following the
   ``Sforce-Locator`` response header until it reads ``null``.

**The schema is explicit.** The caller's Arrow schema *is* the field list -- the SOQL
``SELECT`` is built from its names -- and the column types. Bulk results are CSV text, and
inferring types from text is how an 18-character Id that happens to be numeric-looking, or
a ZIP code, turns into a number. Each page is parsed as strings with Arrow's CSV reader and
cast column by column to the declared types; a value that does not fit fails the read
naming the field.

**Deleted records.** ``include_deleted=True`` runs ``queryAll``, which also returns deleted
and archived records, and adds the ``IsDeleted`` field so a downstream merge can apply the
deletes rather than keep stale rows.

**Incremental by SystemModstamp.** An `Incremental` (cursor field ``SystemModstamp``, key
``Id`` unless set otherwise) adds ``SystemModstamp >= <watermark - lookback>`` to the
query's ``WHERE`` and deduplicates the overlap, so a resumed extraction picks up every
record modified since the accepted watermark -- deletes included under ``queryAll`` -- and
`progress()` reports the job and the last accepted result locator.
"""

from __future__ import annotations

import io
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pyarrow as pa

from batcher._internal.errors import BackendError, FormatError, PlanError
from batcher.io.formats.base import SOURCES
from batcher.io.formats.http.auth import AuthProvider
from batcher.io.formats.http.options import RetryPolicy
from batcher.io.formats.http.state import Incremental, IncrementalRun
from batcher.io.formats.http.transport import HttpClient, redact_url

__all__ = ["SalesforceSource", "soql_datetime"]

_TERMINAL_FAILURES = ("Failed", "Aborted")

#: Indirection a test replaces so job polling does not sleep.
_sleep = time.sleep


def soql_datetime(value: Any) -> str:
    """`value` as a SOQL dateTime literal, truncated down to the second.

    Truncating *down* keeps the bound inclusive of everything at or after the watermark;
    the records it re-reads are removed by the dedup key.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.saas.salesforce import soql_datetime
            >>> soql_datetime("2024-05-01T10:20:30.456Z")
            '2024-05-01T10:20:30Z'
    """
    moment = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if moment.tzinfo is not None:
        moment = moment.astimezone(UTC)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _quote_ident(name: str) -> str:
    if not all(part.replace("_", "").isalnum() for part in name.split(".")):
        raise PlanError(f"salesforce field/object name {name!r} is not a SOQL identifier")
    return name


@SOURCES.register("salesforce")
class SalesforceSource:
    """An sObject read through a Bulk API 2.0 query job.

    Args:
        sobject: The sObject name, such as ``"Account"``.
        instance_url: The org's instance URL.
        schema: The fields to select and their Arrow types.
        auth: A `BearerToken` or `OAuth2ClientCredentials`.
        where: A SOQL condition ANDed into the query.
        include_deleted: Run ``queryAll`` and add ``IsDeleted``.
        incremental: An `Incremental`; resumes by ``SystemModstamp`` keyed on ``Id``.
        api_version: The REST API version.
        max_records: Records per results page (Salesforce chooses when None).
        poll_interval: Seconds between job status polls, doubling up to 30.
        job_timeout: Seconds to wait for the job before failing the read.
        retry: A `RetryPolicy`.
    """

    format_name = "salesforce"
    continues_across_passes = True

    def __init__(
        self,
        sobject: str,
        *,
        instance_url: str,
        schema: pa.Schema,
        auth: AuthProvider | None = None,
        where: str | None = None,
        include_deleted: bool = False,
        incremental: Incremental | None = None,
        api_version: str = "62.0",
        max_records: int | None = None,
        poll_interval: float = 2.0,
        job_timeout: float = 3600.0,
        retry: RetryPolicy | None = None,
    ) -> None:
        if schema is None or len(schema) == 0:
            raise PlanError("salesforce needs schema=: its field names are the SELECT list")
        if include_deleted and "IsDeleted" not in schema.names:
            schema = schema.append(pa.field("IsDeleted", pa.bool_()))
        self._sobject = _quote_ident(sobject)
        for name in schema.names:
            _quote_ident(name)
        self._schema = schema
        self._instance = instance_url.rstrip("/")
        self._auth = auth
        self._where = where
        self._include_deleted = include_deleted
        self._incremental = _salesforce_incremental(incremental)
        if self._incremental is not None:
            for name in (self._incremental.cursor_field, *self._incremental.keys):
                if name not in schema.names:
                    raise PlanError(
                        f"salesforce incremental field {name!r} must be in schema= so it is "
                        "selected and can be compared"
                    )
        self._api = api_version
        self._max_records = max_records
        self._poll = poll_interval
        self._job_timeout = job_timeout
        self._retry = retry
        self._progress: dict[str, Any] = {"job": None, "records": 0, "cursor": None}

    # ---- the query --------------------------------------------------------------
    def soql(self, run: IncrementalRun | None = None) -> str:
        """The SOQL the job runs, with the incremental bound when `run` carries one."""
        conditions = [f"({self._where})"] if self._where else []
        if run is not None and self._incremental is not None and run.lower_bound is not None:
            conditions.append(
                f"{self._incremental.cursor_field} >= {soql_datetime(run.lower_bound)}"
            )
        query = f"SELECT {', '.join(self._schema.names)} FROM {self._sobject}"
        return f"{query} WHERE {' AND '.join(conditions)}" if conditions else query

    def _base(self) -> str:
        return f"{self._instance}/services/data/v{self._api}/jobs/query"

    def _client(self) -> HttpClient:
        return HttpClient(auth=self._auth, retry=self._retry)

    def _run_job(self, client: HttpClient, soql: str) -> str:
        created = client.request(
            "POST",
            self._base(),
            json_body={
                "operation": "queryAll" if self._include_deleted else "query",
                "query": soql,
            },
        ).json()
        job_id = str(created["id"])
        self._progress["job"] = job_id
        wait, deadline = self._poll, time.monotonic() + self._job_timeout
        while True:
            info = client.get(f"{self._base()}/{job_id}").json()
            state = info.get("state")
            if state == "JobComplete":
                return job_id
            if state in _TERMINAL_FAILURES:
                raise BackendError(
                    f"Salesforce query job {job_id} {state.lower()}: "
                    f"{info.get('errorMessage') or 'no error message'}"
                )
            if time.monotonic() > deadline:
                raise BackendError(
                    f"Salesforce query job {job_id} still {state!r} after {self._job_timeout:.0f}s"
                )
            _sleep(wait)
            wait = min(wait * 2, 30.0)

    def _results(self, client: HttpClient, job_id: str) -> Iterator[tuple[bytes, str | None]]:
        locator: str | None = None
        while True:
            response = client.get(
                f"{self._base()}/{job_id}/results",
                {"locator": locator, "maxRecords": self._max_records},
                headers={"Accept": "text/csv"},
            )
            yield response.body, locator
            locator = response.headers.get("sforce-locator")
            if not locator or locator == "null":
                return

    def _parse(self, body: bytes, where: str) -> pa.RecordBatch:
        import pyarrow.csv as pacsv

        if not body.strip():
            return pa.RecordBatch.from_pylist([], schema=self._schema)
        names = self._schema.names
        table = pacsv.read_csv(
            io.BytesIO(body),
            convert_options=pacsv.ConvertOptions(
                column_types=dict.fromkeys(names, pa.string()),
                strings_can_be_null=True,
                null_values=[""],
                include_columns=names,
                include_missing_columns=True,
            ),
        )
        columns = []
        for field in self._schema:
            column = table.column(field.name)
            try:
                columns.append(column.combine_chunks().cast(field.type))
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
                raise FormatError(
                    f"{where}: Salesforce field {field.name!r} holds a value that does not "
                    f"parse as {field.type}: {exc}"
                ) from exc
        return pa.RecordBatch.from_arrays(columns, schema=self._schema)

    # ---- the Source surface -------------------------------------------------------
    def schema(self) -> pa.Schema:
        """The declared schema (plus ``IsDeleted`` under ``include_deleted``)."""
        return self._schema

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        """Every results page as a batch."""
        return list(self.iter_batches(projection))

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        """Run the job and yield one batch per results page."""
        run = IncrementalRun(self._incremental) if self._incremental is not None else None
        client = self._client()
        self._progress = {"job": None, "records": 0, "cursor": None}
        job_id = self._run_job(client, self.soql(run))
        where = redact_url(f"{self._base()}/{job_id}/results")
        for body, locator in self._results(client, job_id):
            batch = self._parse(body, where)
            if run is not None:
                batch = run.filter(batch)
            if batch.num_rows:
                yield batch.select(projection) if projection is not None else batch
            self._progress["records"] += batch.num_rows
            self._progress["cursor"] = locator
            if run is not None:
                run.accept(f"{job_id}:{locator}")
        if run is not None:
            run.finish()

    def progress(self) -> dict[str, Any]:
        """The job id, the records accepted, and the last accepted results locator.

        Returns:
            ``{"job": str, "records": int, "cursor": <locator>}``.
        """
        return dict(self._progress)

    def row_count(self) -> int | None:
        """Unknown until the job has run."""
        return None

    def identity(self) -> str:
        """The org, object and selected fields; never the token."""
        return f"salesforce:{redact_url(self._instance)}:{self._sobject}:{len(self._schema)}"

    def governed_name(self) -> str:
        """The sObject name, the table a governance policy names."""
        return self._sobject

    def splits(self, target_size: int | None = None) -> list[Any]:  # noqa: ARG002
        """One split: one query job produces the whole result."""
        from batcher.io.splits import WholeSourceSplit

        return [WholeSourceSplit(self)]

    def confirm(self) -> None:
        """Commit a staged incremental state once its epoch is published."""
        if self._incremental is not None:
            self._incremental.commit()


def _salesforce_incremental(spec: Incremental | None) -> Incremental | None:
    if spec is None:
        return None
    return Incremental(
        state=spec.state,
        cursor_field=spec.cursor_field or "SystemModstamp",
        key=spec.key or "Id",
        lookback=spec.lookback,
        start=spec.start,
        param=None,  # the bound goes into the SOQL, not a query parameter
        auto_commit=spec.auto_commit,
    )
