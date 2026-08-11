"""Normalize a user's time-travel timestamp into the form a table-format client accepts.

Every lakehouse format offers "read the table as it was at time T", and every client wants
that T as a **fully-qualified RFC-3339 instant** — delta-rs's `load_as_version` rejects
anything else outright. What a user actually writes is `"2026-08-05"`, or
`"2026-08-05 14:30:00"`, or whatever `datetime.now().isoformat()` produced. All three are
the natural spellings, all three are what Spark's `timestampAsOf` accepts, and all three
used to fail here with delta-rs's own message — ``Failed to parse datetime string:
premature end of input`` — which names neither the argument at fault nor a form that works.

So the conversion happens once, here, at the connector boundary: parse the spellings a
person writes, qualify a naive one, and hand the client the instant it requires. A value
that genuinely cannot be a timestamp is rejected *with the accepted forms listed*, before
any table is opened.

## Naive timestamps are local time

A timestamp with no offset (``"2026-08-05 14:30:00"``) is read in the **driver's local
timezone**, which is what Delta Lake and Spark both do with `timestampAsOf` and what a
person writing a wall-clock time means. The alternative — reading it as UTC — silently
shifts every such query by the local offset, which surfaces as "time travel returned the
wrong version" rather than as an error.

That does make the resolved version depend on the driver's timezone. It is confined to the
driver: time travel resolves to a concrete version number at plan time, and *that* is what
travels to the workers. Pass an explicit offset (``"2026-08-05T14:30:00Z"``) when a query
must resolve identically regardless of where it runs.
"""

from __future__ import annotations

import datetime as _dt

from batcher._internal.errors import BackendError

__all__ = ["normalize_timestamp"]

#: Spelled out in the error, so a rejected value carries its own fix.
_ACCEPTED = (
    "a datetime/date object, 'YYYY-MM-DD', 'YYYY-MM-DD HH:MM:SS', "
    "'YYYY-MM-DDTHH:MM:SS', or any of those with a '+HH:MM'/'Z' offset"
)


def normalize_timestamp(value: str | _dt.datetime | _dt.date, *, argument: str) -> str:
    """A time-travel timestamp as the RFC-3339 string a lakehouse client accepts.

    A value that already carries a UTC offset passes through unchanged in meaning; a naive
    one is qualified with the driver's local offset (see the module docstring).

    Args:
        value: The timestamp as the caller wrote it.
        argument: The parameter name to name in an error message (e.g. ``"timestamp"``).

    Returns:
        The same instant as an RFC-3339 string with an explicit offset.

    Raises:
        BackendError: If `value` is not a timestamp in any accepted spelling.
    """
    moment = _as_datetime(value, argument)
    if moment.tzinfo is None:
        # `astimezone()` on a naive datetime reads it as local time and attaches the local
        # offset — exactly the Delta/Spark rule, and the reason it is not `replace(tzinfo=utc)`.
        moment = moment.astimezone()
    return moment.isoformat()


def _as_datetime(value: str | _dt.datetime | _dt.date, argument: str) -> _dt.datetime:
    """`value` as a `datetime`, whatever spelling it arrived in."""
    if isinstance(value, _dt.datetime):
        return value
    if isinstance(value, _dt.date):
        # A bare date means the start of that day, which is what "as of 2026-08-05" says.
        return _dt.datetime(value.year, value.month, value.day)
    if not isinstance(value, str):
        raise BackendError(
            f"{argument} must be a timestamp, got {type(value).__name__}; accepts {_ACCEPTED}"
        )
    text = value.strip()
    try:
        # `fromisoformat` covers every spelling above from Python 3.11 on, including the
        # space separator and a trailing 'Z' — so there is no format table to keep in step.
        return _dt.datetime.fromisoformat(text)
    except ValueError as exc:
        raise BackendError(
            f"{argument}={value!r} is not a timestamp Batcher can read; accepts {_ACCEPTED}"
        ) from exc
