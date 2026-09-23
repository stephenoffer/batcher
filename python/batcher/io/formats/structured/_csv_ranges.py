"""Where a CSV file may be cut into byte ranges without cutting a record in half.

A byte range is aligned to the next newline and parsed on its own, which is only right when
that newline ends a record. A quoted field may hold a newline (``1,"x\\n2,y"``), and a range
that starts after it parses the rest of the field as a record of its own: rows that are not
in the file, with keys the file never held. The single-node read parses the file whole and
gets it right, so this was a distributed-only wrong answer.

A newline ends a record exactly when it sits outside every quoted field, which is when the
number of quote characters before it is even. A doubled quote inside a field (``""``) counts
twice and leaves the parity unchanged, which is what makes the count exact for standard CSV.
It stops being exact under an escape character (``\\"`` does not toggle anything), so a file
whose escape byte appears at all is not cut.

The count needs every byte before the cut. That is a full scan, done here with vectorized
numpy over fixed-size blocks, and it is only worth doing where reading the file is cheap for
the planner: a local file. A remote file is read as one split rather than paying a second
full download on the driver. Correctness wins over the parallelism the ranges would buy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from batcher.io.stats.file_identity import FileMetaCache, file_identity

__all__ = ["record_aligned_starts"]

_BLOCK_BYTES = 8 << 20
_STARTS = FileMetaCache(4096)


def record_aligned_starts(
    fs: Any, path: str, size: int, chunk: int, quote: str | None, escape: str | None
) -> list[int] | None:
    """Byte offsets, about `chunk` apart, at which a record of `path` begins, or None.

    Args:
        fs: The filesystem the file was listed through.
        path: The CSV file.
        size: Its size in bytes.
        chunk: The rough range size wanted.
        quote: The quote character, or None when quoting is disabled.
        escape: The escape character, or None when there is none.

    Returns:
        Ascending offsets starting at 0, each the first byte of a record. None when no
        offset can be proven to start a record, so the caller reads the file whole.
    """
    identity = file_identity(path, fs)
    key = None if identity is None else (identity, chunk, quote, escape)
    if key is not None:
        hit = _STARTS.get(key)
        if hit is not None:
            return list(hit) if hit else None
    starts = _scan(fs, path, size, chunk, quote, escape)
    if key is not None:
        _STARTS.put(key, tuple(starts or ()), weight=1)
    return starts


def _scan(
    fs: Any, path: str, size: int, chunk: int, quote: str | None, escape: str | None
) -> list[int] | None:
    quote_byte = _byte(quote)
    escape_byte = _byte(escape)
    if (quote and quote_byte is None) or (escape and escape_byte is None):
        return None  # a multi-byte quote or escape: the count below cannot see it
    targets = list(range(chunk, size, chunk))
    starts = [0]
    parity = 0
    offset = 0
    ti = 0
    with fs.open(path) as fh:
        while True:
            buf = fh.read(_BLOCK_BYTES)
            if not buf:
                break
            arr = np.frombuffer(buf, dtype=np.uint8)
            if escape_byte is not None and bool((arr == escape_byte).any()):
                return None
            quotes = (
                np.cumsum(arr == quote_byte, dtype=np.int64) if quote_byte is not None else None
            )
            newlines = np.flatnonzero(arr == 10)
            if quotes is None:
                safe = newlines + offset + 1
            else:
                safe = newlines[((parity + quotes[newlines]) & 1) == 0] + offset + 1
                parity = (parity + int(quotes[-1])) & 1
            while ti < len(targets) and targets[ti] < offset + len(buf):
                at = int(np.searchsorted(safe, targets[ti]))
                if at == safe.size:
                    break  # no record starts after this target in this block; try the next
                start = int(safe[at])
                if starts[-1] < start < size:
                    starts.append(start)
                while ti < len(targets) and targets[ti] <= start:
                    ti += 1
            offset += len(buf)
    return starts if len(starts) > 1 else None


def _byte(char: str | None) -> int | None:
    if not char:
        return None
    encoded = char.encode()
    return encoded[0] if len(encoded) == 1 else None
