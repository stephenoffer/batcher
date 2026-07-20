"""ASAM MDF4 (``.mf4``) — the measurement format vehicle data is logged in.

MDF is the ASAM standard every automotive OEM and test fleet records to: CAN/CAN-FD and
LIN signals (already decoded against their DBC by the logger), plus ECU-internal and
analogue sensor channels. An ADAS validation corpus is millions of these files.

**The shape problem, and why this reader is long-format.** An MDF file is a set of
*channel groups*, and each group has its own raster — the powertrain group at 100 Hz, the
chassis group at 4 Hz, a diagnostic group at 1 Hz. There is no single wide table: joining
them into one would either resample (inventing data) or pad with nulls at every raster
boundary. So a row here is one **sample of one signal**:

    {signal, timestamp, value, unit}

which gives one uniform schema for the whole file however many rasters it has, and makes
the query pattern identical to the MCAP one next door — filter to the signals you want,
then `join_asof` on `timestamp` to put them on a common clock. Resampling stays an
explicit choice the caller makes, rather than something the reader did silently.

`timestamp` is **absolute** (the file's `start_time` plus the channel's offset), which is
what lets an MDF measurement be as-of joined against an MCAP log from the same drive.

`value` is `float64`: CAN signals are physical quantities, and a single numeric column is
what keeps the schema uniform. Non-numeric channels (string or byte-array diagnostics)
have no place in that column and are skipped — `signals()` lists what is actually
readable, so the omission is discoverable rather than silent.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import IO, Any, ClassVar

import pyarrow as pa

from batcher._internal.optional import require
from batcher.io.base import FileSource
from batcher.io.formats.base import SOURCES
from batcher.plan.source_stats import SourceStatistics

__all__ = ["MDF_SCHEMA", "MDFSource"]

MDF_SCHEMA = pa.schema(
    [
        ("signal", pa.string()),
        ("timestamp", pa.timestamp("ns")),
        ("value", pa.float64()),
        ("unit", pa.string()),
    ]
)

# Samples per emitted batch. A single channel of a long measurement runs to millions of
# samples, so even one signal is streamed rather than materialized whole.
_SAMPLES_PER_BATCH = 65_536

# NumPy kinds that fit the `value` column: float, signed and unsigned integer, and bool.
_NUMERIC_KINDS = frozenset("fiub")

_NANOS_PER_SECOND = 1_000_000_000


def _require_asammdf() -> Any:
    """Import and return the `asammdf.MDF` class, or raise `BackendError`."""
    return require(
        'asammdf', 'MDF', feature='MDF support', provides='asammdf', extra='robotics'
    )


@SOURCES.register("mdf")
class MDFSource(FileSource):
    """One or more ASAM MDF4 measurements, one Arrow row per signal sample.

    Produces ``{signal, timestamp, value, unit}`` in long format, so a file's several
    sampling rasters share one schema. `signals` restricts the read to named channels —
    the analogue of naming topics on an MCAP log, and the usual case, since a measurement
    carries thousands of channels and a query wants a handful.
    """

    suffix = ".mf4"
    format_name = "mdf"
    supports_predicate: ClassVar[bool] = False

    __slots__ = ("_signals",)

    def __init__(
        self,
        path: str,
        *,
        signals: list[str] | None = None,
        schema_mode: str = "strict",
        files: list[str] | None = None,
        on_error: str = "raise",
    ) -> None:
        super().__init__(path, schema_mode=schema_mode, files=files, on_error=on_error)
        self._signals = list(signals) if signals is not None else None

    def _reader_kwargs(self) -> dict[str, object]:
        """`signals` changes which rows a split produces, so a worker must rebuild it."""
        base = super()._reader_kwargs()
        return {**base, "signals": list(self._signals)} if self._signals is not None else base

    def _read_schema(self, fh: IO[Any]) -> pa.Schema:  # noqa: ARG002 (fixed schema)
        return MDF_SCHEMA

    def schema(self) -> pa.Schema:
        """The long-format sample schema, which is fixed and needs no file access."""
        return MDF_SCHEMA

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        """Read from an open handle — the template's fallback when no path is available."""
        return list(self._iter_handle(fh, projection))

    def _iter_file(self, path: str, projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        with self._fs.open(path) as fh:
            yield from self._iter_handle(fh, projection)

    def _iter_handle(self, fh: IO[Any], projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        mdf_cls = _require_asammdf()
        wanted = set(self._signals) if self._signals is not None else None
        with mdf_cls(fh) as mdf:
            start_ns = _epoch_nanos(mdf.start_time)
            for channel in mdf.iter_channels():
                name = channel.name
                if wanted is not None and name not in wanted:
                    continue
                samples = channel.samples
                if samples.dtype.kind not in _NUMERIC_KINDS:
                    # A byte-array or string diagnostic channel has no `float64` value;
                    # `signals()` reports what is readable, so this is discoverable.
                    continue
                yield from self._channel_batches(channel, name, start_ns, projection)

    def _channel_batches(
        self, channel: Any, name: str, start_ns: int, projection: list[str] | None
    ) -> Iterator[pa.RecordBatch]:
        import numpy as np

        unit = channel.unit or ""
        values = channel.samples.astype("float64", copy=False)
        # Channel offsets are seconds from the measurement start; make them absolute so an
        # MDF measurement is as-of joinable against a log from the same drive.
        stamps = (np.asarray(channel.timestamps, dtype="float64") * _NANOS_PER_SECOND).astype(
            "int64"
        ) + start_ns
        for begin in range(0, len(values), _SAMPLES_PER_BATCH):
            end = begin + _SAMPLES_PER_BATCH
            chunk = values[begin:end]
            batch = pa.RecordBatch.from_arrays(
                [
                    pa.array([name] * len(chunk), pa.string()),
                    pa.array(stamps[begin:end], pa.timestamp("ns")),
                    pa.array(chunk, pa.float64()),
                    pa.array([unit] * len(chunk), pa.string()),
                ],
                schema=MDF_SCHEMA,
            )
            yield batch.select(projection) if projection is not None else batch

    def signals(self) -> list[str]:
        """Every readable (numeric) channel name across the source's files, sorted.

        The discovery step before a query names the handful it wants — a measurement
        carries thousands of channels. Non-numeric channels are omitted, which is exactly
        the set this reader skips, so the list is what a read will actually produce.

        Examples:
            .. doctest::

                >>> from batcher.io import MDFSource  # doctest: +SKIP
                >>> MDFSource("s3://fleet/2026-07-18/drive.mf4").signals()  # doctest: +SKIP
                ['EngineRPM', 'SteeringAngle', 'VehicleSpeed']

        Returns:
            The distinct numeric channel names.
        """
        mdf_cls = _require_asammdf()
        found: set[str] = set()
        for path in self._files():
            with self._fs.open(path) as fh, mdf_cls(fh) as mdf:
                for group in mdf.groups:
                    for channel in group.channels:
                        # The time channel is the raster, not a signal.
                        if channel.name != "time":
                            found.add(channel.name)
        return sorted(found)

    def statistics(self) -> SourceStatistics:
        """A row estimate from the channel-group metadata, without reading samples.

        `cycles_nr` x the group's channel count is exact when every channel is numeric and
        unrestricted; a `signals` restriction or a skipped non-numeric channel makes it an
        upper bound, so it is reported as **not** exact rather than used to answer a count.

        Returns:
            The statistics, with an inexact row count.
        """
        mdf_cls = _require_asammdf()
        total = 0
        for path in self._files():
            with self._fs.open(path) as fh, mdf_cls(fh) as mdf:
                for group in mdf.groups:
                    signals = max(0, len(group.channels) - 1)  # less the time channel
                    total += int(group.channel_group.cycles_nr) * signals
        return SourceStatistics(row_count=total, exact_rows=False)


def _epoch_nanos(start: Any) -> int:
    """A measurement's `start_time` as nanoseconds since the Unix epoch.

    A naive `start_time` is read as UTC rather than local time: a fleet's measurements are
    compared across time zones, and guessing the recorder's zone would shift a drive by
    hours with nothing in the data to reveal it.
    """
    import datetime as dt

    if start is None:
        return 0
    if start.tzinfo is None:
        start = start.replace(tzinfo=dt.timezone.utc)
    return int(start.timestamp() * _NANOS_PER_SECOND)
