"""The per-*row* error policy: what a reader does with one record it cannot parse.

Sibling to `_tolerance.py`, which owns the per-*file* policy, and deliberately separate
from it. The two failures look alike and want opposite responses:

* A file whose bytes are unreadable is usually infrastructure — a truncated upload, a
  zero-byte object. Dropping it loses an unknown number of rows, which is why `on_error`
  keeps an audit trail of paths.
* A single malformed record inside a file that is otherwise fine is usually the producer
  upstream. Dropping the *file* to be rid of it discards every good row in it, which at
  corpus scale is total loss reported as a warning. That is the failure this module exists
  to remove.

`mode` is pandas' vocabulary (`on_bad_lines`), because that is the spelling most users
arrive with and it maps onto Spark's `FAILFAST`/`DROPMALFORMED` without inventing a third.

Neutral by design: it names no format. The CSV reader hands it to pyarrow as an
``invalid_row_handler``; the JSON reader, whose parser has no such hook, calls `record`
directly. A format added later gets the same three modes and the same metric for free.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from contextvars import ContextVar

from batcher._internal.errors import ConfigError
from batcher._internal.logging import get_logger, log_kv

__all__ = ["BAD_ROW_MODES", "BadRowPolicy", "bad_row_handler", "measuring"]

_LOG = get_logger(__name__)

#: True while the current thread/task is re-reading a source to *measure* it rather than to
#: produce its rows. See `measuring`.
_MEASURING: ContextVar[bool] = ContextVar("batcher_bad_rows_measuring", default=False)


@contextlib.contextmanager
def measuring() -> Iterator[None]:
    """Suppress drop counting for a read whose output is a statistic, not the answer.

    The engine re-reads a bounded prefix of a source after a query to refresh the column
    sketches Kyber learns from. That read meets the same malformed records the data read
    did, so without this the drop count is inflated by however many bad records happen to
    fall in the sample — a metric whose whole purpose is telling an operator how much data
    was quietly lost, itself quietly wrong.

    Scoped rather than parameterized because the re-read reaches the policy through five
    layers of generic source machinery that has no business knowing why it is being called.

    Yields:
        Nothing; drops inside the block are tolerated exactly as before, just not counted.
    """
    token = _MEASURING.set(True)
    try:
        yield
    finally:
        _MEASURING.reset(token)


#: How a record the parser rejects is handled. ``"error"`` aborts the read (the default,
#: and Spark's ``FAILFAST``); ``"warn"`` drops the record and logs it; ``"skip"`` drops it
#: silently, counted on the metrics export (Spark's ``DROPMALFORMED``).
BAD_ROW_MODES = frozenset({"error", "warn", "skip"})


class BadRowPolicy:
    """Decides whether one unparseable record aborts a read, and counts the ones dropped.

    Tolerance is per record and local to the bytes it was found in, so it means the same
    thing on a byte-range split as on a whole-file read — which is what lets a distributed
    read of a messy corpus return what the single-node read of it returns.
    """

    #: Cap on per-record warnings from one policy. A file that is malformed throughout
    #: would otherwise emit one log line per record, which is itself a way to lose a job.
    WARN_LIMIT = 20

    #: How much of a dropped record is quoted in the warning. Enough to recognize it in the
    #: source file, short enough that a wide row does not fill the log.
    TEXT_PREVIEW = 120

    __slots__ = ("_format", "_mode", "_observe", "_path", "_warned", "dropped")

    def __init__(
        self, mode: str, path: str = "", *, format_name: str = "", observe: bool = True
    ) -> None:
        if mode not in BAD_ROW_MODES:
            raise ConfigError(f"on_bad_lines must be one of {sorted(BAD_ROW_MODES)}, got {mode!r}")
        self._mode = mode
        self._path = path
        self._format = format_name
        # Schema inference re-parses the same bytes the read is about to. It must *tolerate*
        # a bad record — otherwise inference aborts and the tolerance flag never gets a
        # chance to apply — but it must not count or announce one, or every record in the
        # inference window is reported twice and the malformed-row metric reads double.
        # Read once, here, rather than at each drop. pyarrow invokes an
        # `invalid_row_handler` from its own parse threads, and a `ContextVar` set on the
        # driver is not visible there — so the flag has to be captured where the policy is
        # built, which is always the thread that opened the read.
        self._observe = observe and not _MEASURING.get()
        self._warned = 0
        self.dropped = 0

    @property
    def mode(self) -> str:
        """The configured mode, so a split can rebuild this policy on a worker."""
        return self._mode

    def record(
        self,
        *,
        line: int | None = None,
        text: str = "",
        expected_columns: int | None = None,
        actual_columns: int | None = None,
    ) -> None:
        """Count one dropped record, and log it under ``"warn"``.

        Args:
            line: The record's position, where the parser reports one. pyarrow's CSV
                reader leaves it unset whenever it parsed the block on a worker thread,
                which it does by default, so `text` is what usually locates the record.
            text: The raw record, quoted up to `TEXT_PREVIEW` characters.
            expected_columns: Field count the header declared, for a CSV ragged row.
            actual_columns: Field count the record actually had.
        """
        if not self._observe:
            return
        self.dropped += 1
        if self._mode == "warn" and self._warned < self.WARN_LIMIT:
            self._warned += 1
            log_kv(
                _LOG,
                logging.WARNING,
                "skipping malformed row",
                format=self._format,
                path=self._path,
                line=line,
                expected_columns=expected_columns,
                actual_columns=actual_columns,
                row=text[: self.TEXT_PREVIEW],
                suppressed_after=(self.WARN_LIMIT if self._warned == self.WARN_LIMIT else None),
            )
        _publish_malformed(self._format)

    def __call__(self, row: object) -> str:
        """Handle one invalid row in pyarrow's ``invalid_row_handler`` protocol.

        Args:
            row: pyarrow's `InvalidRow`, carrying the expected and actual column counts,
                the line number, and the raw text.

        Returns:
            ``"skip"`` to drop the row, or ``"error"`` to let the parse abort.
        """
        if self._mode == "error":
            return "error"
        self.record(
            line=getattr(row, "number", None),
            text=str(getattr(row, "text", "") or ""),
            expected_columns=getattr(row, "expected_columns", None),
            actual_columns=getattr(row, "actual_columns", None),
        )
        return "skip"


def _publish_malformed(format_name: str) -> None:
    """Announce one dropped record on the event bus, so a fleet can alert on quiet loss.

    One event per record, carrying ``count=1``: `observe` sums the increments, so a reader
    of the metric never has to reconcile two sources for the same total.

    Nothing identifying the record is carried: a malformed line is user data, and a metrics
    label built from it would be both unbounded and potentially sensitive. The warning log
    above carries it for whoever needs it.
    """
    from batcher._internal import events

    events.publish(
        events.MALFORMED,
        name=format_name,
        count=1,
        reason="unparseable_record",
        source=format_name,
    )


def bad_row_handler(
    mode: str, path: str = "", *, format_name: str = "", observe: bool = True
) -> BadRowPolicy | None:
    """The policy for `mode`, or None when a bad record must abort the parse.

    Returning None rather than a policy that answers ``"error"`` matters for the pyarrow
    path: its default error carries the offending text and line number, and routing that
    through a handler which merely re-raises would replace it with a generic one.

    Args:
        mode: One of `BAD_ROW_MODES`.
        path: The file being read, for the warning log.
        format_name: The reader's format, for the log record and the metric label.
        observe: Whether drops are counted and announced. False on the schema-inference
            pass, which re-reads the bytes the read will.

    Returns:
        A `BadRowPolicy`, or None under ``"error"``.

    Examples:
        .. doctest::

            >>> from batcher.io.base._bad_rows import bad_row_handler
            >>> bad_row_handler("error") is None
            True
            >>> bad_row_handler("skip").mode
            'skip'
    """
    policy = BadRowPolicy(mode, path, format_name=format_name, observe=observe)
    return None if mode == "error" else policy
