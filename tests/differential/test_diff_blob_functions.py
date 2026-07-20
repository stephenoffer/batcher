"""Differential tests for byte-oriented functions over BLOB (Binary) columns.

Each test pins a defect where a function defined on the *raw bytes* of a BLOB was
routed through a Binary->Utf8 cast that nulled (or zeroed) any row whose bytes were
not valid UTF-8. DuckDB's ``hex``/``md5``/``sha256``/``base64``/``octet_length`` over a
BLOB operate on the bytes regardless of textual validity, so the engine must too.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

# A column mixing non-UTF-8 bytes, empty, valid-text, and NULL blobs.
_BLOBS = [b"\xde\xad\xbe\xef", b"\xff", b"", b"abc", None]


def _tbl() -> pa.Table:
    return pa.table({"b": pa.array(_BLOBS, pa.binary()), "row": list(range(len(_BLOBS)))})


def test_hex_of_blob_uses_raw_bytes(duck):
    """hex(BLOB '\\xDE\\xAD\\xBE\\xEF') is 'DEADBEEF', not NULL (non-UTF-8 was dropped)."""
    t = _tbl()
    out = bt.from_arrow(t).select(v=bt.col("b").str.hex(), row=bt.col("row")).collect()
    duck.register("t", t)
    assert_same(out, duck.sql("SELECT hex(b) AS v, row FROM t"))


def test_octet_length_of_blob_counts_all_bytes(duck):
    """octet_length of a non-UTF-8 BLOB is its byte count, not NULL/0."""
    t = _tbl()
    out = bt.from_arrow(t).select(v=bt.col("b").str.octet_length(), row=bt.col("row")).collect()
    duck.register("t", t)
    assert_same(out, duck.sql("SELECT octet_length(b) AS v, row FROM t"))


def test_md5_of_blob_hashes_raw_bytes(duck):
    """md5(BLOB) digests the raw bytes, matching DuckDB, not NULL on non-UTF-8 rows."""
    t = _tbl()
    out = bt.from_arrow(t).select(v=bt.col("b").str.md5(), row=bt.col("row")).collect()
    duck.register("t", t)
    assert_same(out, duck.sql("SELECT md5(b) AS v, row FROM t"))


def test_sha256_of_blob_hashes_raw_bytes(duck):
    """sha256(BLOB) digests the raw bytes, matching DuckDB."""
    t = _tbl()
    out = bt.from_arrow(t).select(v=bt.col("b").str.sha256(), row=bt.col("row")).collect()
    duck.register("t", t)
    assert_same(out, duck.sql("SELECT sha256(b) AS v, row FROM t"))


def test_base64_of_blob_encodes_raw_bytes(duck):
    """base64(BLOB) encodes the raw bytes, matching DuckDB, not NULL on non-UTF-8 rows."""
    t = _tbl()
    out = bt.from_arrow(t).select(v=bt.col("b").str.base64(), row=bt.col("row")).collect()
    duck.register("t", t)
    assert_same(out, duck.sql("SELECT base64(b) AS v, row FROM t"))
