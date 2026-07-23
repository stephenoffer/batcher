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

from batcher._internal.errors import ConfigError
from batcher._internal.logging import get_logger, log_kv

__all__ = ["ON_ERROR_MODES", "ErrorPolicy"]

_LOG = get_logger(__name__)

# "raise" (the default) keeps the historical all-or-nothing behavior. "skip" drops the
# offending file, records it, and carries on.
ON_ERROR_MODES = frozenset({"raise", "skip"})


class ErrorPolicy:
    """Decides whether an unreadable file aborts the read, and remembers the ones dropped."""

    __slots__ = ("_mode", "_skipped")

    def __init__(self, mode: str) -> None:
        if mode not in ON_ERROR_MODES:
            raise ConfigError(f"on_error must be one of {sorted(ON_ERROR_MODES)}, got {mode!r}")
        self._mode = mode
        self._skipped: list[str] = []

    @property
    def mode(self) -> str:
        """The configured mode, so a split can rebuild this policy on a worker.

        `FileSource._reader_kwargs` reads this to thread `on_error` to the per-file
        reader a worker reconstructs. Without it the rebuilt reader silently falls back
        to `"raise"` and the user's explicit tolerance is void on every split path.
        """
        return self._mode

    def tolerate(self, path: str, exc: Exception, *, format_name: str) -> bool:
        """Record and swallow a per-file failure, or report that it must propagate.

        Args:
            path: The file that could not be read.
            exc: Why it could not be read.
            format_name: The reader's format, for the log record.

        Returns:
            True when the caller should drop this file and continue; False when the
            exception must be re-raised.
        """
        if self._mode != "skip":
            return False
        self._skipped.append(path)
        log_kv(
            _LOG,
            logging.WARNING,
            "skipping unreadable file",
            path=path,
            format=format_name,
            error=f"{type(exc).__name__}: {exc}",
        )
        return True

    def skipped(self) -> list[str]:
        """The paths dropped so far, in failure order."""
        return list(self._skipped)
