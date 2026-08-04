"""What a file is, decided from its bytes rather than its name.

The `mime` column is what an unstructured pipeline routes on — send the images to a vision
model, the audio to a transcriber, the PDFs to a text extractor — so getting it wrong does
not raise, it silently sends a row to the wrong branch or to none at all.

The corpora this matters for are exactly the ones where the name is least informative:
object keys from an upload pipeline, content-addressed blob stores, WARC extractions.
Every fixture here is therefore built with **no usable extension**, because a test that
names its file `x.mp4` proves only that `mimetypes` works.
"""

from __future__ import annotations

import pytest

from batcher.io.formats.mime import OCTET_STREAM, sniff_mime

pytestmark = pytest.mark.unit


def _iso(brand: bytes, extra: bytes = b"") -> bytes:
    """An ISO base media header: a box length, `ftyp`, then the brand."""
    return b"\x00\x00\x00\x20ftyp" + brand + b"\x00\x00\x02\x00" + extra


#: The Office family shares one long prefix; naming it keeps the table readable.
_OOXML = "application/vnd.openxmlformats-officedocument."

#: `(name, leading bytes, expected mime)`. The name carries no extension throughout.
_CASES = [
    ("png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 16, "image/png"),
    ("jpeg", b"\xff\xd8\xff\xe0\x00\x10JFIF", "image/jpeg"),
    ("gif", b"GIF89a" + b"\x00" * 16, "image/gif"),
    ("webp", b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/webp"),
    ("wav", b"RIFF\x00\x00\x00\x00WAVEfmt ", "audio/x-wav"),
    ("avi", b"RIFF\x00\x00\x00\x00AVI LIST", "video/x-msvideo"),
    ("tiff_le", b"II*\x00\x08\x00\x00\x00", "image/tiff"),
    ("tiff_be", b"MM\x00*\x00\x00\x00\x08", "image/tiff"),
    ("pdf", b"%PDF-1.7\n%\xe2\xe3\xcf\xd3", "application/pdf"),
    ("flac", b"fLaC\x00\x00\x00\x22", "audio/flac"),
    ("ogg", b"OggS\x00\x02\x00\x00", "audio/ogg"),
    ("mp3_id3", b"ID3\x03\x00\x00\x00", "audio/mpeg"),
    ("mp3_bare", b"\xff\xfb\x90\x00", "audio/mpeg"),
    ("gzip", b"\x1f\x8b\x08\x00", "application/gzip"),
    ("zstd", b"\x28\xb5\x2f\xfd\x00", "application/zstd"),
    ("parquet", b"PAR1" + b"\x00" * 16, "application/vnd.apache.parquet"),
    ("sqlite", b"SQLite format 3\x00", "application/vnd.sqlite3"),
    # ISO base media — the family a prefix table structurally cannot see, because the
    # signature is at offset 4 rather than 0.
    ("mp4", _iso(b"isom"), "video/mp4"),
    ("mp4_avc", _iso(b"mp42"), "video/mp4"),
    ("mov", _iso(b"qt  "), "video/quicktime"),
    ("heic", _iso(b"heic"), "image/heic"),
    ("heif", _iso(b"mif1"), "image/heif"),
    ("avif", _iso(b"avif"), "image/avif"),
    ("3gp", _iso(b"3gp4"), "video/3gpp"),
    ("m4a", _iso(b"M4A "), "audio/mp4"),
    # Zip is one magic number for a whole family.
    ("docx", b"PK\x03\x04\x14\x00\x00\x00word/document.xml", _OOXML + "wordprocessingml.document"),
    ("xlsx", b"PK\x03\x04\x14\x00\x00\x00xl/workbook.xml", _OOXML + "spreadsheetml.sheet"),
    (
        "pptx",
        b"PK\x03\x04\x14\x00\x00\x00ppt/presentation.xml",
        _OOXML + "presentationml.presentation",
    ),
    (
        "epub",
        b"PK\x03\x04" + b"\x00" * 26 + b"mimetypeapplication/epub+zip",
        "application/epub+zip",
    ),
    ("zip", b"PK\x03\x04\x14\x00\x00\x00stuff/thing.txt", "application/zip"),
    # Matroska and WebM share their magic and differ only in the DocType.
    ("webm", b"\x1aE\xdf\xa3\x01\x00\x00\x00\x00\x00\x00\x23B\x82\x84webm", "video/webm"),
    ("mkv", b"\x1aE\xdf\xa3\x01\x00\x00\x00\x00\x00\x00\x23B\x82\x88matroska", "video/x-matroska"),
]


@pytest.mark.parametrize(("label", "head", "expected"), _CASES, ids=[c[0] for c in _CASES])
def test_a_payload_is_identified_without_an_extension(label, head, expected):
    """The name is `blob/<label>` — no extension, so only the bytes can answer."""
    assert sniff_mime(f"s3://bucket/blob/{label}", head) == expected


def test_the_bytes_win_over_a_lying_extension():
    """A `.jpg` that is really a PNG is common enough to be the motivating case."""
    assert sniff_mime("photo.jpg", b"\x89PNG\r\n\x1a\n" + b"\x00" * 8) == "image/png"


def test_the_extension_is_the_fallback_not_the_answer():
    """Nothing in the magic table places a CSV, so the name still gets its turn."""
    assert sniff_mime("data.csv", b"a,b\n1,2\n") == "text/csv"


def test_an_unrecognisable_payload_with_no_extension_is_octet_stream():
    assert sniff_mime("s3://bucket/2f9c1a", b"\x00\x01\x02\x03") == OCTET_STREAM


def test_a_short_or_empty_payload_does_not_raise():
    """A zero-length object is a real thing in a blob store, and slicing past the end of
    `bytes` yields a short slice rather than raising — so the readers must not assume a
    minimum length."""
    for head in (b"", b"\xff", b"RIFF", b"PK", b"\x00\x00\x00\x20ftyp"):
        assert isinstance(sniff_mime("s3://bucket/key", head), str)


def test_the_binary_source_sniffs_the_bytes_it_already_read(tmp_path):
    """`bt.read.binary` had the payload in hand and was still reading the filename.

    This source exists for corpora of extensionless objects, so trusting the extension
    left its whole reason for existing reporting `application/octet-stream`.
    """
    import batcher as bt

    (tmp_path / "no_extension_here").write_bytes(_iso(b"heic"))
    (tmp_path / "also_nameless").write_bytes(b"%PDF-1.4\n")
    got = bt.read.binary(str(tmp_path)).sort("uri").to_pydict()["mime"]
    assert got == ["application/pdf", "image/heic"]
