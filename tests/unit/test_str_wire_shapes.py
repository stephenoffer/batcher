"""The Python half of the `.str` wire shapes pinned by `crates/bc-expr/tests/strings_wire.rs`.

Each new tag or slot is a two-sided contract: the `to_ir()` dict built here must be the
exact JSON the Rust test deserializes and evaluates. The shapes are written out literally
on both sides, so a rename on either one fails a test instead of reaching a user.
"""

from __future__ import annotations

import pytest

import batcher as bt

pytestmark = pytest.mark.unit

_COL = {"e": "col", "name": "s"}


def _str(fn: str, **slots: object) -> dict:
    return {"e": "str", "fn": fn, "input": _COL, **slots}


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (lambda s: s.normalize(), _str("normalize", pattern="NFC")),
        (lambda s: s.normalize("NFKD"), _str("normalize", pattern="NFKD")),
        (lambda s: s.casefold(), _str("casefold")),
        (lambda s: s.len_chars(unit="grapheme"), _str("length_grapheme")),
        (lambda s: s.len_chars(unit="byte"), _str("octet_length")),
        (lambda s: s.len_chars(), _str("len")),
        (
            lambda s: s.substr(2, 1, unit="grapheme"),
            _str("substring_grapheme", start=2, length=1),
        ),
        (
            lambda s: s.extract_groups(r"(?P<k>\w)=(\d)"),
            _str("regexp_extract_groups", pattern=r"(?P<k>\w)=(\d)"),
        ),
        (
            lambda s: s.extract_groups("(a)", missing="null"),
            _str("regexp_extract_groups_or_null", pattern="(a)"),
        ),
        (
            lambda s: s.chunk(3, 1, offsets=True),
            _str("chunk_offsets", pattern="char", start=1, length=3),
        ),
        (lambda s: s.chunk(3, 1), _str("chunk", pattern="char", start=1, length=3)),
        (
            lambda s: s.parse_filename(separator="forward"),
            _str("parse_filename", pattern="forward"),
        ),
        (lambda s: s.parse_filename(), _str("parse_filename")),
        (lambda s: s.decompress("gzip"), _str("decompress", pattern="gzip")),
        (
            lambda s: s.decompress("gzip", max_output_bytes=99),
            _str("decompress", pattern="gzip", length=99),
        ),
        (
            lambda s: s.replace_all(".", "x", literal=True),
            _str("replace", pattern=".", replacement="x"),
        ),
    ],
)
def test_wire_shape(build, expected):
    assert build(bt.col("s").str).to_ir() == expected


def test_a_per_row_parameter_lowers_to_str_dyn():
    assert bt.col("s").str.levenshtein(bt.col("t")).to_ir() == {
        "e": "str_dyn",
        "fn": "levenshtein",
        "input": _COL,
        "pattern": {"e": "col", "name": "t"},
    }
