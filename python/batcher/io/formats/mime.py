"""MIME sniffing for file payloads — magic bytes first, extension as the fallback.

An unstructured corpus routinely carries files whose extension lies (a ``.jpg`` that is
really a PNG) or that have no extension at all (an object key from an upload pipeline, a
content-addressed blob store, a WARC extraction). So the ``mime`` column cannot come from
the filename: a file is what its bytes say it is, and the extension is a hint of last
resort.

Shared by every reader that hands back opaque payloads — the media sources and the
binary-blob source — rather than living beside one of them. They face the same corpus and
must not disagree about what a file is.

The table below is deliberately not "every format": it is the formats an unstructured
pipeline routes on. Three families need more than a prefix match and get their own
readers:

* **ISO base media** (MP4, MOV, HEIC, AVIF, 3GP) puts a four-byte box length *before* the
  ``ftyp`` marker, so the signature is at offset 4 and the actual format is the brand at
  offset 8. A prefix table structurally cannot see it, which left the most common video
  container in the world — and every photo a recent phone takes — as
  ``application/octet-stream`` whenever the key carried no extension.
* **Zip** is the container for the whole Office and EPUB family, which share one magic
  number and differ only in what is stored inside.
* **Matroska and WebM** share their magic and differ only in an EBML DocType string.
"""

from __future__ import annotations

import mimetypes
from typing import IO, Any

__all__ = ["MAGIC_PEEK_BYTES", "OCTET_STREAM", "read_header", "sniff_mime"]

# How many leading bytes are enough to sniff a media type by magic number and to read a
# format header. Kept small so metadata extraction stays header-only.
MAGIC_PEEK_BYTES = 4096

#: The answer for a payload nothing recognizes. Named because several readers compare
#: against it, and a literal repeated is a literal that will diverge.
OCTET_STREAM = "application/octet-stream"

# Magic-number prefixes. Everything here is decided by the leading bytes alone; the
# families that are not are handled by the readers below.
_MAGIC_PREFIXES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
    (b"II*\x00", "image/tiff"),  # little-endian TIFF
    (b"MM\x00*", "image/tiff"),  # big-endian TIFF
    (b"%PDF-", "application/pdf"),
    (b"OggS", "audio/ogg"),
    (b"fLaC", "audio/flac"),
    (b"ID3", "audio/mpeg"),  # MP3 carrying an ID3v2 tag
    (b"\xff\xfb", "audio/mpeg"),  # bare MPEG-1 Layer III frame sync
    (b"\xff\xf3", "audio/mpeg"),
    (b"\xff\xf2", "audio/mpeg"),
    (b"\x00\x00\x01\xba", "video/mpeg"),  # MPEG program stream
    (b"\x1f\x8b", "application/gzip"),
    (b"BZh", "application/x-bzip2"),
    (b"\xfd7zXZ\x00", "application/x-xz"),
    (b"\x28\xb5\x2f\xfd", "application/zstd"),
    (b"Obj\x01", "application/avro"),
    (b"PAR1", "application/vnd.apache.parquet"),
    (b"ARROW1", "application/vnd.apache.arrow.file"),
    (b"SQLite format 3\x00", "application/vnd.sqlite3"),
)

# ISO base media brands. The brand at offset 8 is what separates a still from a video, and
# reading it wrong routes a photograph into a video decoder. Matched as a prefix because
# the field is space-padded to four bytes and versioned (`3gp4`, `3gp5`, ...).
_ISO_BRANDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("heic", "heix", "hevc", "hevx", "heim", "heis", "hevm", "hevs"), "image/heic"),
    (("mif1", "msf1"), "image/heif"),
    (("avif", "avis"), "image/avif"),
    (("qt  ",), "video/quicktime"),
    (("3gp", "3g2"), "video/3gpp"),
    (("M4A", "M4B"), "audio/mp4"),
    (("crx",), "image/x-canon-cr3"),
)

# Zip-container formats, keyed by a marker in the archive's leading bytes. EPUB is exact
# by construction (the spec requires an uncompressed `mimetype` entry first); the Office
# formats are matched by their first stored part's name, which every producer writes early
# enough to fall inside the peeked header.
_ZIP_MARKERS: tuple[tuple[bytes, str], ...] = (
    (b"mimetypeapplication/epub+zip", "application/epub+zip"),
    (b"word/", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    (b"ppt/", "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
    (b"xl/", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
)


def sniff_mime(path: str, data: bytes) -> str:
    """Best-effort MIME type from magic bytes, falling back to the extension.

    Args:
        path: The file's path, used for the extension fallback.
        data: Its leading bytes (at least `MAGIC_PEEK_BYTES` where available).

    Returns:
        The MIME type, or ``application/octet-stream`` when nothing recognizes it.
    """
    head = data[:MAGIC_PEEK_BYTES]
    for reader in (_riff, _iso_base_media, _matroska, _zip, _prefix):
        found = reader(head)
        if found is not None:
            return found
    guessed, _ = mimetypes.guess_type(path)
    return guessed or OCTET_STREAM


def _prefix(head: bytes) -> str | None:
    """The plain prefix table."""
    for prefix, mime in _MAGIC_PREFIXES:
        if head.startswith(prefix):
            return mime
    return None


def _riff(head: bytes) -> str | None:
    """RIFF is a container tag, not a format — the format is four bytes at offset 8."""
    if not head.startswith(b"RIFF"):
        return None
    return {
        b"WEBP": "image/webp",
        b"WAVE": "audio/x-wav",
        b"AVI ": "video/x-msvideo",
    }.get(head[8:12])


def _iso_base_media(head: bytes) -> str | None:
    """MP4 and its relatives: ``ftyp`` at offset 4, the brand at offset 8."""
    if head[4:8] != b"ftyp":
        return None
    brand = head[8:12].decode("ascii", "replace")
    for brands, mime in _ISO_BRANDS:
        if any(brand.startswith(b) for b in brands):
            return mime
    # `isom`, `mp41`, `mp42`, `dash`, `avc1` and the rest of the long tail are all MP4.
    return "video/mp4"


def _matroska(head: bytes) -> str | None:
    """Matroska and WebM share the EBML magic and differ only in their DocType.

    Reporting every WebM as ``video/x-matroska`` is not wrong so much as useless: routing
    on the mime type is exactly where the distinction is wanted.
    """
    if not head.startswith(b"\x1aE\xdf\xa3"):
        return None
    # The DocType string sits in the EBML header, well inside the peeked bytes.
    return "video/webm" if b"webm" in head[:64] else "video/x-matroska"


def _zip(head: bytes) -> str | None:
    """Zip carries the whole Office and EPUB family under one magic number."""
    if not head.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
        return None
    for marker, mime in _ZIP_MARKERS:
        if marker in head:
            return mime
    return "application/zip"


def read_header(fh: IO[Any]) -> bytes:
    """Read just the leading header bytes from an open handle.

    Args:
        fh: An open binary file handle positioned at the start.

    Returns:
        The leading `MAGIC_PEEK_BYTES` bytes, or fewer for a shorter file.
    """
    return fh.read(MAGIC_PEEK_BYTES)
