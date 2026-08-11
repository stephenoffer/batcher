"""MIME sniffing for file payloads — magic bytes first, extension as the fallback.

An unstructured corpus routinely carries files whose extension lies (a ``.jpg`` that is
really a PNG) or that have no extension at all (an object key from an upload pipeline, a
content-addressed blob store, a WARC extraction). So the ``mime`` column cannot come from
the filename: a file is what its bytes say it is, and the extension is a hint of last
resort.

Shared by every reader that hands back opaque payloads — the media sources and the
binary-blob source — rather than living beside one of them. They face the same corpus and
must not disagree about what a file is.

The magic-number table itself is **not here**. It lives in `bc-expr`, where the
`.str.mime_type()` expression also reads it, and this module reaches it through the engine.
That is deliberate: a reader with its own copy would be a second answer to "what is this
file", and the two would drift the first time a format was added to one of them — silently,
on a column whose whole job is to route rows to different branches.

What does live here is the fallback the engine cannot do. It sees bytes and not filenames,
so an unrecognized payload comes back null and only this layer, which has the path, can try
the extension before giving up on `application/octet-stream`.
"""

from __future__ import annotations

import mimetypes
from typing import IO, Any

from batcher._internal.native import engine_or_none

__all__ = ["MAGIC_PEEK_BYTES", "OCTET_STREAM", "read_header", "sniff_mime"]

# How many leading bytes are enough to sniff a media type by magic number and to read a
# format header. Kept small so metadata extraction stays header-only.
MAGIC_PEEK_BYTES = 4096

#: The answer for a payload nothing recognizes. Named because several readers compare
#: against it, and a literal repeated is a literal that will diverge.
OCTET_STREAM = "application/octet-stream"


def sniff_mime(path: str, data: bytes) -> str:
    """Best-effort MIME type from magic bytes, falling back to the extension.

    The magic-number reading is the engine's — the same table `.str.mime_type()` uses —
    so there is one answer to "what is this file" rather than one per caller. Only the
    extension fallback lives here, because the engine sees bytes and not filenames.

    Args:
        path: The file's path, used for the extension fallback.
        data: Its leading bytes (at least `MAGIC_PEEK_BYTES` where available).

    Returns:
        The MIME type, or ``application/octet-stream`` when nothing recognizes it.
    """
    engine = engine_or_none()
    if engine is not None:
        sniffed = engine.sniff_mime(data[:MAGIC_PEEK_BYTES])
        if sniffed is not None:
            return sniffed
    guessed, _ = mimetypes.guess_type(path)
    return guessed or OCTET_STREAM


def read_header(fh: IO[Any]) -> bytes:
    """Read just the leading header bytes from an open handle.

    Args:
        fh: An open binary file handle positioned at the start.

    Returns:
        The leading `MAGIC_PEEK_BYTES` bytes, or fewer for a shorter file.
    """
    return fh.read(MAGIC_PEEK_BYTES)
