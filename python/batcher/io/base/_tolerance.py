"""The per-file error policy a `FileSource` read applies to an unreadable file.

Separated from the read spine because it is a *policy* decision, not a read mechanism:
the spine asks "may I drop this file?" and this module owns the answer, the audit trail,
and the warning. Keeping it here also means the two call sites in `FileSource` (`read`
and `iter_batches`) share one implementation rather than each growing their own
`try`/`except`, which is how the two paths would drift apart.

A corpus at scale always contains a few unreadable members — a truncated upload, a
zero-byte object, a JPEG whose trailer never arrived. Aborting a 10,000-file read for one
of them is the wrong default *at scale* and the right one for a single file you just
wrote, which is why the mode is explicit rather than inferred.
"""

from __future__ import annotations

import logging

from batcher._internal.errors import ConfigError, FormatError, SchemaError
from batcher._internal.errors import IOError as BatcherIOError
from batcher._internal.logging import get_logger, log_kv

__all__ = ["ON_ERROR_MODES", "ErrorPolicy"]

_LOG = get_logger(__name__)

# "raise" (the default) keeps the historical all-or-nothing behavior. "skip" drops the
# offending file, records it, and carries on.
ON_ERROR_MODES = frozenset({"raise", "skip"})


class ErrorPolicy:
    """Decides whether an unreadable file aborts the read, and remembers the ones dropped."""

    __slots__ = ("_mode", "_seen", "_skipped")

    def __init__(self, mode: str) -> None:
        if mode not in ON_ERROR_MODES:
            raise ConfigError(f"on_error must be one of {sorted(ON_ERROR_MODES)}, got {mode!r}")
        self._mode = mode
        # The list keeps failure order (what `skipped()` promises); the set answers "already
        # recorded?" in O(1). A list membership test would be O(files skipped) per skip, and
        # a corpus that is largely unreadable is exactly when this runs most — quadratic on
        # the one path whose whole purpose is surviving a bad corpus at scale.
        self._skipped: list[str] = []
        self._seen: set[str] = set()

    @property
    def mode(self) -> str:
        """The configured mode, so a split can rebuild this policy on a worker.

        `FileSource._reader_kwargs` reads this to thread `on_error` to the per-file
        reader a worker reconstructs. Without it the rebuilt reader silently falls back
        to `"raise"` and the user's explicit tolerance is void on every split path.
        """
        return self._mode

    def tolerate(self, path: str, exc: Exception, *, format_name: str) -> bool:
        """Record and swallow a per-file failure, or raise it as a typed error.

        A `SchemaError` is never tolerated, whatever the mode. `on_error` answers "this
        file's *bytes* are unreadable, may I drop it?", and a file whose schema disagrees
        with the source's is perfectly readable — dropping it would silently delete rows
        the user has every reason to think were read. That is the same silent data loss
        strict-mode conformance exists to catch, so it must not be reintroduced here.

        Args:
            path: The file that could not be read.
            exc: Why it could not be read.
            format_name: The reader's format, for the log record.

        Returns:
            True, meaning the caller should drop this file and continue. The failure is
            raised rather than returned when it must propagate, so a caller never has to
            re-raise (and cannot forget to).

        Raises:
            SchemaError: If `exc` is one — it is about the data, not the file's health.
            FormatError: Under ``on_error="raise"``, wrapping `exc` with the path, the
                format, and the flag that would tolerate it.
        """
        if isinstance(exc, SchemaError):
            raise exc
        if self._mode != "skip":
            if isinstance(exc, BatcherIOError):
                # The format already diagnosed this in its own vocabulary (the CSV reader's
                # invalid-UTF-8 message, say). Wrapping it would bury the specific advice
                # under a generic one, so the better error is the one already raised.
                raise exc
            raise FormatError(
                f"could not read {path!r} as {format_name}: {type(exc).__name__}: {exc}. "
                "Pass on_error='skip' to drop unreadable files and read the rest "
                "(source.corrupt_files() then lists what was dropped)."
            ) from exc
        if path in self._seen:
            # One unreadable file is met more than once per query — schema inference, the
            # footer row count, split planning and the read itself each touch it — and
            # `corrupt_files()` answers "which files were dropped", not "how many times did
            # a drop happen". Recording it once also keeps the warning to one line per file.
            return True
        self._seen.add(path)
        self._skipped.append(path)
        log_kv(
            _LOG,
            logging.WARNING,
            "skipping unreadable file",
            path=path,
            format=format_name,
            error=f"{type(exc).__name__}: {exc}",
        )
        _publish_skip(format_name, exc)
        return True

    def skipped(self) -> list[str]:
        """The paths dropped so far, in failure order."""
        return list(self._skipped)


def _publish_skip(format_name: str, exc: Exception) -> None:
    """Announce one dropped input on the event bus, so a fleet can alert on it.

    `corrupt_files()` answers "what did *this* source drop", which requires already
    suspecting that something was dropped and holding the source object to ask. The warning
    log answers it for a human reading a terminal. Neither reaches a metrics backend, and
    silent data loss is exactly the condition that has to reach one: a job that quietly read
    98% of its corpus produces a plausible answer and no error.

    The path is deliberately *not* carried. A metrics label built from a path is unbounded
    cardinality, and a path can itself be sensitive; the exception type is the bounded fact
    worth counting, and the path is already in the warning above for whoever needs it.
    """
    from batcher._internal import events

    events.publish(
        events.SKIPPED,
        name=format_name,
        count=1,
        reason=type(exc).__name__,
        source=format_name,
    )
