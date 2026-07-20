"""MIME sniffing for media files — magic bytes first, extension as the fallback.

A media corpus routinely carries files whose extension lies (a ``.jpg`` that is really a
PNG, an extensionless object key from an upload pipeline), so the ``mime`` column cannot
come from the filename alone. Magic bytes win where they are recognized because a file is
what its bytes say it is.

Separate from `media.py` because it is a self-contained lookup with its own table of
format signatures — the part that grows as new media formats are recognized, and the part
worth testing directly against crafted headers.
"""

from __future__ import annotations

import mimetypes
from typing import IO, Any

__all__ = ["MAGIC_PEEK_BYTES", "read_header", "sniff_mime"]

# How many leading bytes are enough to sniff a media type by magic number and to read a
# format header. Kept small so metadata extraction stays header-only.
MAGIC_PEEK_BYTES = 4096

# Magic-number prefixes for media types stdlib `mimetypes` may miss by extension.
_MAGIC_PREFIXES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
    (b"RIFF", "image/webp"),  # disambiguated below by the WEBP/WAVE tag at [8:12]
    (b"OggS", "audio/ogg"),
    (b"fLaC", "audio/flac"),
    (b"\x1aE\xdf\xa3", "video/x-matroska"),
)


def sniff_mime(path: str, data: bytes) -> str:
    """Best-effort MIME type from magic bytes, falling back to the extension.

    Args:
        path: The file's path, used for the extension fallback.
        data: Its leading bytes.

    Returns:
        The MIME type, or ``application/octet-stream`` when nothing recognizes it.
    """
    head = data[:MAGIC_PEEK_BYTES]
    for prefix, mime in _MAGIC_PREFIXES:
        if head.startswith(prefix):
            if prefix == b"RIFF":
                # RIFF is a container tag, not a format — the format is at [8:12].
                tag = head[8:12]
                if tag == b"WEBP":
                    return "image/webp"
                if tag == b"WAVE":
                    return "audio/x-wav"
                if tag == b"AVI ":
                    return "video/x-msvideo"
                continue
            return mime
    guessed, _ = mimetypes.guess_type(path)
    return guessed or "application/octet-stream"


def read_header(fh: IO[Any]) -> bytes:
    """Read just the leading header bytes from an open handle."""
    return fh.read(MAGIC_PEEK_BYTES)
