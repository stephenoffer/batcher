"""Data sinks — persisting query results.

Sinks write an Arrow table to storage. Kept behind a small protocol + registry so
new formats (and partitioned / streaming writers) slot in uniformly. The
per-format writers live one-per-file under `io/formats/` and register into the
`SINKS` registry; this module re-exports them and owns the `Sink` protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pyarrow as pa

from batcher._internal.errors import FormatError
from batcher.io.formats import SINKS, CSVSink, JSONSink, ParquetSink
from batcher.io.manifest import WriteManifest, WrittenFile

__all__ = [
    "SINKS",
    "CSVSink",
    "JSONSink",
    "ParquetSink",
    "Sink",
    "check_write_options",
    "table_sink_kwargs",
]


def table_sink_kwargs(fmt: str, path: str) -> dict[str, object]:
    """The constructor kwargs a *table*-format sink needs but a file sink does not.

    A file sink takes its destination per call (``write(table, path)``); a table format
    may need it at construction, because staging and commit both address the table rather
    than a file. Iceberg is the one that does today: the write `path` **is** the table
    identifier, and it also wants a per-write token so the staged files of one write share
    a name prefix and cannot clobber a file an earlier snapshot still references.

    This lives here rather than in the caller because there are now two callers — the
    batch write path and the streaming table sink — and the second one did not know about
    the first's special case, so a streaming Iceberg write failed at the first micro-batch
    with a bare ``TypeError`` from inside the sink's constructor.

    Args:
        fmt: The registered sink format.
        path: The write destination, which for a table format is its identifier.

    Returns:
        Kwargs to pass to ``SINKS.get(fmt)(...)``; empty for a format that needs none.
    """
    if fmt != "iceberg":
        return {}
    import uuid

    return {"identifier": path, "write_token": uuid.uuid4().hex[:12]}


def check_write_options(fmt: str, opts: dict[str, object]) -> None:
    """Reject a write keyword the sink for `fmt` does not accept, by name.

    Reading a mistyped option already fails with the format, the misspelling and a
    "did you mean" (`OptionSpec` on the way in, `FileSource` for the base keywords).
    Writing one did not: it travelled all the way down and surfaced as
    ``DeltaSink.__init__() got an unexpected keyword argument 'schema_mode'`` — an error
    naming a class the caller never typed and cannot import, with no hint that the option
    they wanted is spelled `merge_schema`. `table_sink_kwargs` above records the same
    failure from the other side.

    Two things make checking here rather than at each construction site worth it. There
    are seven construction sites (batch, streaming, distributed, per-worker), so a check
    at any one of them leaves the rest raw. And a distributed write builds its sink
    *inside a Ray worker*, so the `TypeError` arrived as a remote-task traceback after the
    cluster had already been provisioned — for a typo knowable before any of it started.

    A `**kwargs` in the signature means two opposite things, and the distinction is what
    decides whether a name can be judged here at all:

    * On a `FileSink` subclass it is the **forwarding idiom** — ``super().__init__(**kwargs)``
      passing `filesystem=` / `storage_options=` up, as the base's own comment requires. The
      accepted set is then exactly the union over the constructor chain, so it is knowable,
      and reading only the leaf's signature is what let a misspelled `compression` through.
    * On a connector-backed sink it is genuine **passthrough** of open-ended driver
      keywords, which are not knowable here. Those are skipped, mirroring
      `OptionSpec(passthrough=True)` on the read side.

    Args:
        fmt: The registered sink format the write is going to.
        opts: The write keywords the caller passed, already stripped of the ones the
            writer itself consumes.

    Raises:
        FormatError: If a keyword matches no parameter of the sink's constructor chain.
    """
    import inspect

    from batcher._internal.errors import unknown_value
    from batcher.io.base.sink import FileSink

    sink_cls = SINKS.get(fmt)
    if not isinstance(sink_cls, type):
        return
    # A sink that declares its own write vocabulary knows more than a signature does — it
    # knows the aliases (`sep` for `delimiter`) and the deliberately-absent options and why.
    # Defer to it, which also moves its error to the same early point as everyone else's.
    spec = getattr(sink_cls, "write_spec", None)
    if spec is not None:
        spec.resolve(dict(opts))
        return
    names: set[str] = set()
    forwards = False
    for klass in sink_cls.__mro__:
        if klass is object:
            # `object.__init__` is `(self, /, *args, **kwargs)`, so counting it would read
            # every sink as an open-ended passthrough and check nothing at all.
            break
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        try:
            params = inspect.signature(init).parameters
        except (TypeError, ValueError):  # unintrospectable — judge no name against it
            return
        names.update(
            n
            for n, p in params.items()
            if n != "self"
            and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        )
        forwards |= any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    if forwards and not issubclass(sink_cls, FileSink):
        return  # open-ended driver keywords: not this layer's to judge
    accepted = tuple(sorted(names))
    for key in opts:
        if key not in accepted:
            raise unknown_value(
                FormatError,
                f"{fmt} write option",
                key,
                accepted,
                label="Accepted options",
                hint="see the writer's docstring for what each option does.",
            )


@runtime_checkable
class Sink(Protocol):
    """A writer that persists Arrow tables to storage.

    `write` produces a single file; `write_partitioned` writes one shard of a
    (possibly Hive-partitioned) directory write; `commit` finalizes a write
    atomically from the collected manifest (a no-op for plain file sinks).

    Examples:
        .. doctest::

            >>> from batcher.io import ParquetSink, Sink
            >>> isinstance(ParquetSink(), Sink)
            True
    """

    def write(self, table: pa.Table, path: str) -> WrittenFile:
        """Write the whole table to a single file at `path`, atomically.

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import ParquetSink
                >>> table = pa.table({"x": [1, 2, 3]})
                >>> ParquetSink().write(table, "out.parquet").rows  # doctest: +SKIP
                3

        Args:
            table: The rows to persist.
            path: Destination file URI.

        Returns:
            The file that was written, with its row count and size.
        """
        ...

    def write_partitioned(
        self,
        table: pa.Table,
        path: str,
        *,
        partition_by: list[str] | None = None,
        file_index: int = 0,
    ) -> list[WrittenFile]:
        """Write `table` under directory `path` as one shard of a directory write.

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import ParquetSink
                >>> table = pa.table({"c": ["a", "b"], "x": [1, 2]})
                >>> sink = ParquetSink()
                >>> len(sink.write_partitioned(table, "out", partition_by=["c"]))  # doctest: +SKIP
                2

        Args:
            table: The rows to persist.
            path: Destination directory URI.
            partition_by: Columns to encode as Hive ``col=value`` directories.
                They are dropped from the data, since the path carries them.
            file_index: This shard's index, which names its part files so
                concurrent writers never collide.

        Returns:
            One entry per file written by this shard.
        """
        ...

    def commit(self, manifest: WriteManifest, path: str) -> None:
        """Finalize a write from the manifest every shard contributed to.

        Examples:
            .. doctest::

                >>> from batcher.io import ParquetSink, WriteManifest
                >>> ParquetSink().commit(WriteManifest(), "out")  # a no-op for file sinks

        Args:
            manifest: Every file the write produced, merged across shards.
            path: The write's destination root.
        """
        ...
