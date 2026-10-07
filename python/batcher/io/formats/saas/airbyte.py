"""`airbyte`: one stream of an Airbyte source connector, read through the Airbyte protocol.

An Airbyte source is a program -- a Docker image, or the executable a PyAirbyte-installed
connector provides -- that answers ``discover`` and ``read`` with newline-delimited JSON
*messages* on stdout. This bridge speaks that protocol directly, so it needs no Python
dependency: it runs ``discover --config`` for the stream's JSON schema, builds a configured
catalog for the one stream, runs ``read --config --catalog [--state]`` and consumes the
messages in order.

**Ordering and checkpoint meaning are preserved.** In the protocol a STATE message means
"every record emitted before me is covered". `AirbyteMessages` therefore flushes the
buffered records as a batch at each STATE, and only *after* the consumer has taken that
batch does it accept the state. The accepted state is staged once the read is fully
consumed and committed to ``state=`` (or left for an explicit `Incremental.commit` with
``auto_commit=False``), so the next read passes it back with ``--state`` and resumes
exactly where the connector said it may. Records after the last STATE are covered by no
checkpoint and are read again next time -- which is Airbyte's own at-least-once contract,
not a loss. Per-stream (``STREAM``), ``GLOBAL`` and legacy state are all kept in the form
the connector expects back.

**Errors are not end-of-stream.** A ``TRACE`` message of type ``ERROR`` fails the read with
its message, and so does a non-zero exit (with the tail of stderr), so a connector that dies
half-way never yields a table that looks complete.

**Types.** The stream's JSON schema maps ``integer`` (and ``airbyte_type: integer``) to
int64, ``number`` to float64, ``boolean`` to bool and ``string`` to string; an ``object`` or
``array`` field becomes a JSON-text string column, readable with the ``.json`` accessor.
``schema=`` overrides the mapping.

**Secrets.** ``config`` values may be secret references; they are resolved when the
connector starts and written to a 0600 file in a private temporary directory that is
removed when it exits. With ``image=`` that directory is bind-mounted into the container,
so the Docker daemon must run on the same host.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import tempfile
from collections.abc import Iterable, Iterator
from typing import Any

import pyarrow as pa

from batcher._internal.errors import BackendError, PlanError
from batcher._internal.logging import get_logger
from batcher.io.formats.base import SOURCES
from batcher.io.formats.http.records import PageBuilder
from batcher.io.formats.http.state import Incremental

__all__ = ["AirbyteMessages", "AirbyteSource", "json_schema_to_arrow", "parse_messages"]

_LOG = get_logger("io")
_BATCH_ROWS = 10_000


def parse_messages(lines: Iterable[str | bytes]) -> Iterator[dict]:
    """The Airbyte messages in a connector's stdout, skipping lines that are not JSON objects.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.saas.airbyte import parse_messages
            >>> [m["type"] for m in parse_messages(['{"type": "LOG"}', "not json"])]
            ['LOG']
    """
    for line in lines:
        text = line.decode("utf-8", "replace") if isinstance(line, bytes) else line
        text = text.strip()
        if not text:
            continue
        try:
            message = json.loads(text)
        except ValueError:
            _LOG.debug("airbyte: skipping a non-JSON line from the connector")
            continue
        if isinstance(message, dict):
            yield message


def _arrow_type(spec: dict) -> tuple[pa.DataType, bool]:
    """The Arrow type for one JSON-schema property, and whether it is JSON text."""
    kinds = spec.get("type", "string")
    kinds = [kinds] if isinstance(kinds, str) else list(kinds)
    kind = next((k for k in kinds if k != "null"), "string")
    if spec.get("airbyte_type") == "integer" or kind == "integer":
        return pa.int64(), False
    if kind == "number":
        return pa.float64(), False
    if kind == "boolean":
        return pa.bool_(), False
    if kind == "string":
        return pa.string(), False
    return pa.string(), True


def json_schema_to_arrow(json_schema: dict) -> pa.Schema:
    """The Arrow schema for an Airbyte stream's JSON schema.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.saas.airbyte import json_schema_to_arrow
            >>> json_schema_to_arrow(
            ...     {"properties": {"id": {"type": ["null", "integer"]}, "tags": {"type": "array"}}}
            ... ).types
            [DataType(int64), DataType(string)]
    """
    props = json_schema.get("properties") or {}
    return pa.schema([pa.field(name, _arrow_type(spec)[0]) for name, spec in props.items()])


def _json_fields(json_schema: dict) -> frozenset[str]:
    props = json_schema.get("properties") or {}
    return frozenset(n for n, spec in props.items() if _arrow_type(spec)[1])


class AirbyteMessages:
    """Turn an ordered Airbyte message stream into batches, tracking the accepted state.

    Args:
        stream: The stream whose records are kept.
        schema: The batches' schema.
        json_fields: Fields held as JSON text.
        batch_rows: The most records in one batch.
    """

    __slots__ = ("_builder", "_json", "_rows", "_stream", "accepted", "legacy")

    def __init__(
        self,
        stream: str,
        schema: pa.Schema,
        *,
        json_fields: frozenset[str] = frozenset(),
        batch_rows: int = _BATCH_ROWS,
    ) -> None:
        self._stream = stream
        self._builder = PageBuilder(schema, declared=True)
        self._json = json_fields
        self._rows = batch_rows
        #: Accepted per-stream / global state messages, keyed so a later one replaces.
        self.accepted: dict[str, dict] = {}
        #: Accepted legacy state blob, for connectors on the pre-per-stream protocol.
        self.legacy: Any = None

    def _encode(self, data: dict) -> dict:
        if not self._json:
            return data
        return {
            k: json.dumps(v) if k in self._json and v is not None and not isinstance(v, str) else v
            for k, v in data.items()
        }

    def _flush(self, buffer: list[dict]) -> pa.RecordBatch:
        return self._builder.batch(buffer, where=f"airbyte stream {self._stream!r}")

    def _accept(self, state: dict) -> None:
        kind = state.get("type", "LEGACY")
        if kind == "STREAM":
            desc = (state.get("stream") or {}).get("stream_descriptor") or {}
            self.accepted[f"stream:{desc.get('namespace') or ''}:{desc.get('name')}"] = state
        elif kind == "GLOBAL":
            self.accepted["global"] = state
        else:
            self.legacy = state.get("data")

    def batches(self, messages: Iterable[dict]) -> Iterator[pa.RecordBatch]:
        """Yield the stream's records in order; accept each STATE once its records are taken.

        Args:
            messages: The decoded messages, in the order the connector emitted them.

        Yields:
            Batches of the stream's records.

        Raises:
            BackendError: On a ``TRACE`` message of type ``ERROR``.
        """
        buffer: list[dict] = []
        for message in messages:
            kind = message.get("type")
            if kind == "RECORD":
                record = message.get("record") or {}
                if record.get("stream") == self._stream:
                    buffer.append(self._encode(record.get("data") or {}))
                    if len(buffer) >= self._rows:
                        yield self._flush(buffer)
                        buffer = []
            elif kind == "STATE":
                if buffer:
                    yield self._flush(buffer)
                    buffer = []
                # Reached only once the consumer took every record before this message.
                self._accept(message.get("state") or {})
            elif kind == "TRACE":
                trace = message.get("trace") or {}
                if trace.get("type") == "ERROR":
                    error = trace.get("error") or {}
                    raise BackendError(
                        f"airbyte connector reported an error: "
                        f"{error.get('message') or error.get('internal_message') or trace}"
                    )
            elif kind == "LOG":
                log = message.get("log") or {}
                _LOG.debug("airbyte %s: %s", log.get("level"), log.get("message"))
        if buffer:
            yield self._flush(buffer)

    def state_document(self) -> dict:
        """The accepted state as it is persisted: per-stream/global messages or legacy data."""
        if self.accepted:
            return {"messages": list(self.accepted.values())}
        return {"legacy": self.legacy}


def _resolve_config(value: Any) -> Any:
    from batcher.io.credentials import is_secret_ref, resolve_secret

    if isinstance(value, dict):
        return {k: _resolve_config(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_config(v) for v in value]
    if isinstance(value, str) and is_secret_ref(value):
        return resolve_secret(value, what="airbyte config value")
    return value


@SOURCES.register("airbyte")
class AirbyteSource:
    """One stream of an Airbyte source connector.

    Args:
        stream: The stream to read.
        image: A connector Docker image (``"airbyte/source-faker:6"``).
        command: A connector executable and its leading arguments, instead of `image`.
        config: The connector's configuration; values may be secret references.
        state: Where the accepted STATE is kept between reads; None reads in full.
        sync_mode: ``"incremental"`` or ``"full_refresh"``; incremental when `state` is
            given and the stream supports it.
        schema: The Arrow schema, overriding the one mapped from the stream's JSON schema.
        auto_commit: Commit the state when the read is consumed (see `Incremental`).
        docker: The Docker executable.
    """

    format_name = "airbyte"
    continues_across_passes = True

    def __init__(
        self,
        stream: str,
        *,
        image: str | None = None,
        command: list[str] | None = None,
        config: dict[str, Any] | None = None,
        state: str | None = None,
        sync_mode: str | None = None,
        schema: pa.Schema | None = None,
        auto_commit: bool = True,
        docker: str = "docker",
    ) -> None:
        if (image is None) == (command is None):
            raise PlanError("airbyte needs exactly one of image= or command=")
        if sync_mode not in (None, "incremental", "full_refresh"):
            raise PlanError(f"sync_mode must be 'incremental' or 'full_refresh', got {sync_mode!r}")
        self._stream = stream
        self._image = image
        self._command = list(command or [])
        self._config = dict(config or {})
        self._state = Incremental(state=state, auto_commit=auto_commit) if state else None
        self._sync_mode = sync_mode
        self._declared = schema
        self._docker = docker
        self._catalog_stream: dict | None = None

    # ---- running the connector ---------------------------------------------------
    def _argv(self, workdir: str, *args: str) -> list[str]:
        if self._image is not None:
            return [
                self._docker,
                "run",
                "--rm",
                "-i",
                "-v",
                f"{workdir}:{workdir}",
                self._image,
                *args,
            ]
        return [*self._command, *args]

    @contextlib.contextmanager
    def _workdir(self) -> Iterator[str]:
        with tempfile.TemporaryDirectory(prefix="batcher-airbyte-") as workdir:
            os.chmod(workdir, 0o700)
            self._write(workdir, "config.json", _resolve_config(self._config))
            yield workdir

    @staticmethod
    def _write(workdir: str, name: str, document: Any) -> str:
        path = os.path.join(workdir, name)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(document, fh)
        return path

    def _run(self, argv: list[str]) -> Iterator[dict]:
        """Start the connector and yield its messages; raise on a non-zero exit."""
        with tempfile.TemporaryFile() as stderr:
            process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=stderr)
            try:
                assert process.stdout is not None
                yield from parse_messages(process.stdout)
                code = process.wait()
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
            if code != 0:
                stderr.seek(0)
                tail = stderr.read()[-800:].decode("utf-8", "replace")
                raise BackendError(f"airbyte connector exited with status {code}: {tail}")

    def _stream_spec(self) -> dict:
        if self._catalog_stream is None:
            with self._workdir() as workdir:
                config = os.path.join(workdir, "config.json")
                streams: list[dict] = []
                for message in self._run(self._argv(workdir, "discover", "--config", config)):
                    if message.get("type") == "CATALOG":
                        streams = (message.get("catalog") or {}).get("streams") or []
                    elif message.get("type") == "TRACE":
                        list(AirbyteMessages(self._stream, pa.schema([])).batches([message]))
            match = [s for s in streams if s.get("name") == self._stream]
            if not match:
                raise BackendError(
                    f"airbyte connector has no stream {self._stream!r}; it offers "
                    f"{sorted(s.get('name', '?') for s in streams)}"
                )
            self._catalog_stream = match[0]
        return self._catalog_stream

    def _configured_catalog(self, spec: dict) -> dict:
        supported = spec.get("supported_sync_modes") or ["full_refresh"]
        mode = self._sync_mode or (
            "incremental"
            if self._state is not None and "incremental" in supported
            else "full_refresh"
        )
        return {
            "streams": [
                {
                    "stream": spec,
                    "sync_mode": mode,
                    "destination_sync_mode": "append",
                    "cursor_field": spec.get("default_cursor_field") or [],
                    "primary_key": spec.get("source_defined_primary_key") or [],
                }
            ]
        }

    # ---- the Source surface --------------------------------------------------------
    def schema(self) -> pa.Schema:
        """The declared schema, or the one mapped from the stream's JSON schema."""
        if self._declared is not None:
            return self._declared
        return json_schema_to_arrow(self._stream_spec().get("json_schema") or {})

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        """Every record of the stream, as batches."""
        return list(self.iter_batches(projection))

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        """Run ``read`` and yield the stream's records; stage the accepted state at the end."""
        spec = self._stream_spec()
        schema = self.schema()
        reader = AirbyteMessages(
            self._stream, schema, json_fields=_json_fields(spec.get("json_schema") or {})
        )
        with self._workdir() as workdir:
            args = [
                "read",
                "--config",
                os.path.join(workdir, "config.json"),
                "--catalog",
                self._write(workdir, "catalog.json", self._configured_catalog(spec)),
            ]
            prior = self._state.load() if self._state is not None else None
            if prior:
                saved = prior.get("messages") if "messages" in prior else prior.get("legacy")
                if saved:
                    args += ["--state", self._write(workdir, "state.json", saved)]
            for batch in reader.batches(self._run(self._argv(workdir, *args))):
                yield batch.select(projection) if projection is not None else batch
        if self._state is not None and (reader.accepted or reader.legacy is not None):
            self._state._stage(reader.state_document())

    def row_count(self) -> int | None:
        """Unknown: the connector streams its records."""
        return None

    def identity(self) -> str:
        """The connector and stream; never a config value."""
        return f"airbyte:{self._image or ' '.join(self._command[:1])}:{self._stream}"

    def splits(self, target_size: int | None = None) -> list[Any]:  # noqa: ARG002
        """One split: a connector run is one ordered message stream."""
        from batcher.io.splits import WholeSourceSplit

        return [WholeSourceSplit(self)]

    def confirm(self) -> None:
        """Commit a staged state once its epoch is published."""
        if self._state is not None:
            self._state.commit()
