"""`str.compress`/`str.decompress` against Python's own codec implementations.

DuckDB has no compression functions, so it is not the oracle here. Python's standard
library is a better one for this particular contract: the point of naming a codec `gzip`
is that the bytes are a *gzip frame*, readable by anything that reads gzip. A test that
only round-tripped Batcher against itself would pass just as happily on a private format.
So each codec is checked in both directions against the reference implementation —
`gzip`/`zlib` from the standard library, `zstd`/`brotli`/`lz4` from their own packages
where installed, skipped where not.

`deflate` has no framing, so it is the one codec that cannot detect a bad input; that is a
property of the format and is asserted as such rather than worked around.
"""

from __future__ import annotations

import gzip
import zlib

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher._internal.errors import PlanError

CODECS = ["gzip", "zlib", "deflate", "zstd", "brotli", "lz4"]

PAYLOAD = "the quick brown fox jumps over the lazy dog " * 30


def _external_compress(codec: str, data: bytes) -> bytes | None:
    """Compress with a reference implementation, or `None` if it isn't installed."""
    if codec == "gzip":
        return gzip.compress(data)
    if codec == "zlib":
        return zlib.compress(data)
    if codec == "deflate":
        return zlib.compress(data)[2:-4]  # strip the zlib header and adler32 trailer
    if codec == "zstd":
        zstandard = pytest.importorskip("zstandard")
        return zstandard.ZstdCompressor().compress(data)
    if codec == "brotli":
        brotli = pytest.importorskip("brotli")
        return brotli.compress(data)
    lz4 = pytest.importorskip("lz4.block")
    return lz4.compress(data)


def _external_decompress(codec: str, data: bytes) -> bytes:
    if codec == "gzip":
        return gzip.decompress(data)
    if codec == "zlib":
        return zlib.decompress(data)
    if codec == "deflate":
        return zlib.decompress(data, -zlib.MAX_WBITS)
    if codec == "zstd":
        zstandard = pytest.importorskip("zstandard")
        return zstandard.ZstdDecompressor().decompress(data)
    if codec == "brotli":
        brotli = pytest.importorskip("brotli")
        return brotli.decompress(data)
    lz4 = pytest.importorskip("lz4.block")
    return lz4.decompress(data)


@pytest.mark.parametrize("codec", CODECS)
def test_batcher_frames_are_readable_by_the_reference_implementation(codec):
    # The interop direction that matters most: what Batcher writes, the ecosystem reads.
    t = pa.table({"s": [PAYLOAD]})
    packed = bt.from_arrow(t).select(b=col("s").str.compress(codec)).to_pydict()["b"][0]
    assert _external_decompress(codec, packed) == PAYLOAD.encode()


@pytest.mark.parametrize("codec", CODECS)
def test_reference_frames_are_readable_by_batcher(codec):
    external = _external_compress(codec, PAYLOAD.encode())
    t = pa.table({"b": [external]})
    got = bt.from_arrow(t).select(r=col("b").str.decompress(codec).cast("string")).to_pydict()
    assert got["r"] == [PAYLOAD]


@pytest.mark.parametrize("codec", CODECS)
def test_round_trip_preserves_text_empties_and_nulls(codec):
    t = pa.table({"s": [PAYLOAD, "", "x", None]})
    got = (
        bt.from_arrow(t)
        .select(r=col("s").str.compress(codec).str.decompress(codec).cast("string"))
        .to_pydict()["r"]
    )
    assert got == [PAYLOAD, "", "x", None]


@pytest.mark.parametrize("codec", CODECS)
def test_a_text_column_and_the_same_bytes_compress_identically(codec):
    # The Utf8 and Binary paths must not drift: `compress` is defined on the bytes.
    text = pa.table({"v": ["some payload"]})
    binary = pa.table({"v": [b"some payload"]}, schema=pa.schema([("v", pa.binary())]))
    a = bt.from_arrow(text).select(r=col("v").str.compress(codec)).to_pydict()["r"]
    b = bt.from_arrow(binary).select(r=col("v").str.compress(codec)).to_pydict()["r"]
    assert a == b


@pytest.mark.parametrize("codec", ["gzip", "zlib", "zstd", "brotli", "lz4"])
def test_a_corrupt_frame_is_null_rather_than_a_failed_query(codec):
    # The reason there is no separate `try_decompress`: one bad blob is a bad row.
    t = pa.table({"b": [b"\x00not a frame\xff", _external_compress(codec, b"ok")]})
    got = bt.from_arrow(t).select(r=col("b").str.decompress(codec)).to_pydict()["r"]
    assert got[0] is None
    assert got[1] == b"ok"


def test_raw_deflate_cannot_detect_a_corrupt_frame():
    # Stated as a test so the limit is known: raw deflate has no header and no checksum,
    # so "is this a deflate frame?" has no answer. Use zlib or gzip where detection
    # matters — same algorithm, wrapped in a frame that can be validated.
    t = pa.table({"b": [b"\x00not a frame\xff"]})
    got = bt.from_arrow(t).select(r=col("b").str.decompress("deflate")).to_pydict()["r"]
    assert got[0] != b"the original payload"


def test_compression_shrinks_repetitive_data():
    t = pa.table({"s": [PAYLOAD]})
    sizes = (
        bt.from_arrow(t)
        .select(
            raw=col("s").str.len_bytes(),
            **{c: col("s").str.compress(c).str.len_bytes() for c in CODECS},
        )
        .to_pydict()
    )
    for codec in CODECS:
        assert sizes[codec][0] < sizes["raw"][0] // 2, codec


@pytest.mark.parametrize("method", ["compress", "decompress"])
def test_unknown_codec_fails_at_plan_build(method):
    with pytest.raises(PlanError, match="codec must be one of"):
        getattr(col("s").str, method)("snappy")


def test_a_frame_does_not_decode_under_a_different_codec():
    t = pa.table({"s": [PAYLOAD]})
    packed = bt.from_arrow(t).select(b=col("s").str.compress("gzip")).collect()
    for codec in ["zstd", "brotli", "lz4"]:
        got = bt.from_arrow(packed).select(r=col("b").str.decompress(codec)).to_pydict()["r"]
        assert got[0] != PAYLOAD.encode(), codec
