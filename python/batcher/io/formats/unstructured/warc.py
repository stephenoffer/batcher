"""WARC source — web-archive records (ISO 28500) as Arrow rows.

A WARC file is a flat sequence of records, each a version line, RFC-822-style headers, a
blank line, and exactly ``Content-Length`` bytes of payload. It is how Common Crawl and
every web-scale crawler ship their output, which makes it the front door of an LLM
pretraining corpus: fetch the crawl, filter to `response` records, pull the HTML out, and
hand it to `.str.strip_html()` / `.str.chunk()` / the embedding path.

Read-only, and no third-party dependency — the format is simple enough that parsing it is
smaller than a wrapper around a library would be, and `gzip` in the standard library
already handles the per-record gzip members a `.warc.gz` concatenates.

Records are parsed at batch granularity, never row-at-a-time into the query: the payload
lands as a `binary` column and every downstream question about it is a Rust expression.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
from collections.abc import Iterator
from typing import IO, Any

import pyarrow as pa

from batcher._internal.errors import FormatError
from batcher.io.base import FileSource
from batcher.io.formats.base import SOURCES

__all__ = ["WARC_SCHEMA", "WarcSource"]

#: The named headers get their own typed column; everything else is carried as JSON in
#: `warc_headers` rather than being dropped. A crawl's own extension headers
#: (`WARC-Block-Digest`, `WARC-IP-Address`, an operator's custom field) are exactly what a
#: provenance or dedup pass needs, and `col("warc_headers").json.extract_string(...)`
#: reads them without this source having to know their names.
WARC_SCHEMA = pa.schema(
    [
        ("path", pa.string()),
        ("warc_record_id", pa.string()),
        ("warc_type", pa.string()),
        ("warc_date", pa.timestamp("us")),
        ("warc_target_uri", pa.string()),
        ("warc_content_length", pa.int64()),
        ("warc_content_type", pa.string()),
        ("warc_identified_payload_type", pa.string()),
        ("warc_headers", pa.string()),
        ("warc_content", pa.large_binary()),
    ]
)

#: Header name (lowercased) → the column it populates. Lowercased because the spec makes
#: header names case-insensitive and crawlers differ in how they spell them.
_TYPED_HEADERS = {
    "warc-record-id": "warc_record_id",
    "warc-type": "warc_type",
    "warc-date": "warc_date",
    "warc-target-uri": "warc_target_uri",
    "content-length": "warc_content_length",
    "content-type": "warc_content_type",
    "warc-identified-payload-type": "warc_identified_payload_type",
}

#: Records per emitted batch. A WARC is unbounded — a Common Crawl segment is ~1 GB of
#: payload — so records stream out in batches rather than accumulating.
_RECORDS_PER_BATCH = 1024


def _parse_date(raw: str) -> dt.datetime | None:
    """Parse a `WARC-Date` into a tz-naive UTC datetime, or `None` if unparseable.

    The spec says UTC ISO 8601, and crawlers emit it with a `Z`, with an explicit offset,
    and with or without fractional seconds. Anything that will not parse yields null
    rather than failing the record: a bad timestamp on one of a billion records is not a
    reason to lose the crawl.
    """
    try:
        parsed = dt.datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    # Batcher timestamps are tz-naive UTC instants; normalize then drop the zone.
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(dt.UTC).replace(tzinfo=None)
    return parsed


def _read_headers(stream: IO[bytes]) -> dict[str, str] | None:
    """Read one record's version line and headers; `None` at clean end of stream.

    Returns the headers keyed by their lowercased names. A continuation line (leading
    whitespace, RFC 822 folding) appends to the previous header rather than being dropped.
    """
    # Skip the blank lines that separate records, then the `WARC/1.x` version line.
    while True:
        line = stream.readline()
        if not line:
            return None  # clean end of stream
        if line.strip():
            break
    if not line.upper().startswith(b"WARC/"):
        raise FormatError(
            f"not a WARC record: expected a 'WARC/1.x' version line, got {line[:32]!r}"
        )
    headers: dict[str, str] = {}
    last: str | None = None
    while True:
        line = stream.readline()
        if not line or line in (b"\r\n", b"\n"):
            break  # blank line ends the header block
        text = line.decode("utf-8", errors="replace").rstrip("\r\n")
        if text[:1] in (" ", "\t") and last is not None:
            headers[last] += " " + text.strip()
            continue
        name, _, value = text.partition(":")
        last = name.strip().lower()
        headers[last] = value.strip()
    return headers


def iter_records(stream: IO[bytes]) -> Iterator[tuple[dict[str, str], bytes]]:
    """Yield `(headers, payload)` for each record in an uncompressed WARC stream.

    Payload length comes from `Content-Length` rather than from scanning for a delimiter,
    because a record's bytes are arbitrary and may contain anything a delimiter search
    would trip over — a `\\r\\n\\r\\n` inside an HTTP response body is ordinary.
    """
    while True:
        headers = _read_headers(stream)
        if headers is None:
            return
        try:
            length = int(headers.get("content-length", "0"))
        except ValueError as exc:
            raise FormatError(
                f"WARC record has a non-integer Content-Length: {headers.get('content-length')!r}"
            ) from exc
        payload = stream.read(length) if length > 0 else b""
        if len(payload) != length:
            raise FormatError(
                f"truncated WARC record: Content-Length said {length} bytes, got {len(payload)}"
            )
        yield headers, payload


def _open_records(fh: IO[Any], path: str) -> IO[bytes]:
    """Wrap `fh` so the caller reads uncompressed WARC bytes either way.

    `.warc.gz` is conventionally *per record* gzip members concatenated, not one member
    over the whole file, which is what lets a reader seek to a record boundary.
    `gzip.GzipFile` reads a multi-member stream transparently, so the same wrapper serves
    both that and a single-member file.
    """
    return gzip.GzipFile(fileobj=fh) if path.endswith(".gz") else fh  # type: ignore[return-value]


@SOURCES.register("warc")
class WarcSource(FileSource):
    """Web-archive (WARC) files, one Arrow row per record.

    Produces `WARC_SCHEMA`: the named WARC headers as typed columns, every other header
    as JSON in ``warc_headers``, and the record payload as ``warc_content``. One file is
    one split — a WARC has no index, so a reader cannot start in the middle of one.
    """

    suffix = (".warc", ".warc.gz")
    format_name = "warc"

    __slots__ = ()

    def _read_schema(self, fh: IO[Any]) -> pa.Schema:  # noqa: ARG002 (fixed schema)
        return WARC_SCHEMA

    def _read_by_path(self, path: str, projection: list[str] | None) -> list[pa.RecordBatch] | None:
        """Read by path so `read()` and `iter_batches()` share one reader.

        Also the only way the ``path`` column can be right: a filesystem handle carries no
        usable name, so a handle-only read of a *directory* would stamp every record with
        the directory rather than the file it came from.
        """
        return list(self._iter_file(path, projection))

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        """Read from an open handle — the `FileSource` fallback when no path is available."""
        name = getattr(fh, "name", self._path)
        return list(self._batches(_open_records(fh, name), name, projection))

    def _iter_file(self, path: str, projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        """Stream one file's records in bounded batches."""
        with self._fs.open(path) as fh:
            yield from self._batches(_open_records(fh, path), path, projection)

    def _batches(
        self, stream: IO[bytes], path: str, projection: list[str] | None
    ) -> Iterator[pa.RecordBatch]:
        """Parse `stream` into `_RECORDS_PER_BATCH`-sized batches.

        A malformed record is tolerated under ``on_error="skip"`` the way a corrupt file
        is: the file stops contributing at that point and is recorded in `corrupt_files()`,
        because a WARC is a *stream* — once a record's framing is wrong, the reader's
        position is wrong too and every record after it would be garbage.
        """
        rows: list[dict[str, Any]] = []
        records = iter_records(stream)
        while True:
            try:
                headers, payload = next(records)
            except StopIteration:
                break
            except (FormatError, OSError, EOFError) as exc:
                self._errors.tolerate(path, exc, format_name=self.format_name)
                break
            rows.append(_row(headers, payload, path))
            if len(rows) >= _RECORDS_PER_BATCH:
                yield _batch(rows, projection)
                rows = []
        if rows:
            yield _batch(rows, projection)


def _row(headers: dict[str, str], payload: bytes, path: str) -> dict[str, Any]:
    """One record's headers and payload as a row of `WARC_SCHEMA`."""
    row: dict[str, Any] = dict.fromkeys(WARC_SCHEMA.names)
    row["path"] = path
    row["warc_content"] = payload
    extra: dict[str, str] = {}
    for name, value in headers.items():
        column = _TYPED_HEADERS.get(name)
        if column is None:
            extra[name] = value
        elif column == "warc_date":
            row[column] = _parse_date(value)
        elif column == "warc_content_length":
            row[column] = int(value) if value.strip().isdigit() else None
        else:
            row[column] = value
    # Always a JSON object, never null, so `.json` accessors on it need no null guard.
    row["warc_headers"] = json.dumps(extra, sort_keys=True)
    return row


def _batch(rows: list[dict[str, Any]], projection: list[str] | None) -> pa.RecordBatch:
    columns = {name: [row[name] for row in rows] for name in WARC_SCHEMA.names}
    batch = pa.RecordBatch.from_pydict(columns, schema=WARC_SCHEMA)
    return batch.select(projection) if projection is not None else batch
