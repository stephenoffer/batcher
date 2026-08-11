"""Dropping the unparseable line from a newline-delimited JSON buffer.

The CSV reader gets row-level tolerance for free: pyarrow's CSV parser takes an
``invalid_row_handler`` and answers per row. The JSON reader has no such hook — one bad
line aborts the whole read — so the same guarantee has to be built, and the shape it takes
matters more than it looks.

The obvious build is to validate each line with `json.loads`, which is exactly the per-row
Python `.claude/rules/architecture.md` forbids: a 10 M-line file would be ten million
interpreter round trips on a path whose whole purpose is surviving a large messy corpus.

So this does the opposite. pyarrow's error already names the offending row, and locating
that row in the buffer is a vectorized newline scan. Drop the line, hand the buffer back,
and let Arrow C++ re-parse. A file with *k* bad lines costs *k + 1* parses and no per-row
Python at all — and *k* is small in every case this exists for, because a corpus where most
lines are unparseable is not a tolerance problem, it is the wrong format.

Only lines Arrow cannot *parse* are dropped. A line that parses but whose types disagree
with the emerging schema is the other failure — inference having been shown too little —
and its fix is `schema=`, not deletion. Dropping it would silently delete the rows that
were about to tell you the schema is wrong, which is the failure mode this whole area
exists to remove.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pyarrow as pa

from batcher.io.base._bad_rows import BadRowPolicy

__all__ = ["MAX_DROPPED_LINES", "read_json_records"]

#: pyarrow reports the failing record as ``... in row N``, zero-based within the buffer it
#: was handed. That index is the only machine-readable part of the message, and without it
#: there is nothing to drop.
_ROW_RE = re.compile(r"in row (\d+)")

#: A type disagreement reads as a parse error but is not one. Arrow spells it by naming the
#: column and the two types, so these two fragments separate "this line is not JSON" from
#: "this line is JSON that does not fit the schema so far".
_SCHEMA_MARKERS = ("changed from", "column(")

#: Ceiling on retries for one buffer. Each drop costs a re-parse, so a buffer that is mostly
#: unparseable would otherwise turn a read into a quadratic grind with no error to explain
#: it. Hitting the cap re-raises Arrow's own message, which is the honest outcome: at this
#: density the file is not newline-delimited JSON.
MAX_DROPPED_LINES = 1000

#: ASCII newline, the record separator this format is defined by.
_NEWLINE = 0x0A


def _is_unparseable(exc: Exception) -> bool:
    """Whether `exc` says a line is not JSON, rather than that it does not fit the schema."""
    lowered = str(exc).lower()
    return not any(marker in lowered for marker in _SCHEMA_MARKERS)


def _without_line(data: bytes, index: int) -> bytes | None:
    """`data` with its zero-based line `index` removed, or None when there is no such line.

    The newline scan is `numpy`, not a Python loop or a `bytes.split` — splitting a
    100 MB window would allocate a million small `bytes` objects to delete one of them.
    """
    offsets = np.flatnonzero(np.frombuffer(data, dtype=np.uint8) == _NEWLINE)
    start = 0 if index == 0 else int(offsets[index - 1]) + 1
    if start >= len(data):
        return None
    end = int(offsets[index]) + 1 if index < len(offsets) else len(data)
    return data[:start] + data[end:]


def read_json_records(
    source: bytes | Any, parse_options: Any, policy: BadRowPolicy | None
) -> pa.Table:
    """Parse newline-delimited JSON from `source`, dropping bad lines per `policy`.

    The strict parse is attempted first and unchanged, so a clean file pays nothing at all:
    no extra copy, no scan, the same single `read_json` call it made before tolerance
    existed. Only a buffer Arrow has already refused is materialized and retried.

    Args:
        source: The bytes to parse, or a seekable binary handle over them. A handle is
            read into memory only when the strict parse has already failed.
        parse_options: The `pyarrow.json.ParseOptions` the caller would have used, or None.
        policy: The bad-row policy, or None to parse strictly (the default).

    Returns:
        The parsed table.

    Raises:
        pyarrow.ArrowInvalid: When `policy` is None, when the handle cannot be rewound,
            when the failure is a schema disagreement rather than an unparseable line, or
            when more than `MAX_DROPPED_LINES` lines would have to go.
    """
    import pyarrow.json as pajson

    kwargs = {} if parse_options is None else {"parse_options": parse_options}
    reader = pa.BufferReader(source) if isinstance(source, bytes) else source
    try:
        return pajson.read_json(reader, **kwargs)
    except pa.ArrowInvalid as first:
        if policy is None or not _is_unparseable(first):
            raise
        data = source if isinstance(source, bytes) else _rewound_bytes(source)
        if data is None:
            # A consumed, non-seekable stream. Retrying from here would parse a truncated
            # file and return it as a success, which is worse than the error being raised.
            raise
        return _drop_and_retry(data, kwargs, policy, first)


def _rewound_bytes(handle: Any) -> bytes | None:
    """The whole of `handle` from the start, or None when it cannot be rewound."""
    try:
        handle.seek(0)
    except (AttributeError, OSError, ValueError):
        return None
    return handle.read()


def _drop_and_retry(
    data: bytes, kwargs: dict[str, Any], policy: BadRowPolicy, first: pa.ArrowInvalid
) -> pa.Table:
    """Re-parse `data` without the lines Arrow rejects, one at a time."""
    import pyarrow.json as pajson

    exc = first
    while True:
        match = _ROW_RE.search(str(exc))
        if match is None or not _is_unparseable(exc) or policy.dropped >= MAX_DROPPED_LINES:
            raise exc
        index = int(match.group(1))
        trimmed = _without_line(data, index)
        if trimmed is None or trimmed == data:
            raise exc
        data = trimmed
        policy.record(line=index)
        try:
            return pajson.read_json(pa.BufferReader(data), **kwargs)
        except pa.ArrowInvalid as again:
            exc = again
