"""Per-row `.str` parameters, path separators, bounded decompression and chunk offsets.

The `.str` namespace takes an expression wherever the engine already evaluates a parameter
per row (`StrFuncDyn`), building through the same neutral helper the SQL front-end uses.
Each case is held against DuckDB computing the same function with the parameter supplied
row by row, including a NULL parameter (which nulls the row, as SQL requires).

Comparisons are positional throughout: a projection over `from_pydict` keeps row order.
"""

from __future__ import annotations

import gzip
import zlib

import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.func_nodes import StrFunc, StrFuncDyn

pytestmark = pytest.mark.differential

_ROWS = {
    "s": ["a-b-c", "x|y|z", "a-b-c", "", None, "kitten", "same"],
    "d": ["-", "|", None, "-", "-", "t", "same"],
    "n": [2, 3, 1, 1, 1, None, 1],
    "t": ["a-b", "x|y", "abc", "", "q", "sitting", None],
}


def _duck_rows(duck, sql: str, *cols: str) -> list:
    rows = zip(*(_ROWS[c] for c in cols), strict=True)
    return [duck.execute(sql, list(r)).fetchone()[0] for r in rows]


def _got(expr) -> list:
    return bt.from_pydict(_ROWS).select(r=expr).to_pydict()["r"]


@pytest.mark.parametrize(
    ("build", "sql", "cols"),
    [
        (lambda s: s.split_part(bt.col("d"), bt.col("n")), "SELECT split_part(?, ?, ?)", "sdn"),
        # DuckDB's `string_split(s, NULL)` returns `[s]`; a NULL parameter nulls the row
        # here, as it does in `bt.sql`'s `string_split` -- so the NULL row is pinned apart.
        (
            lambda s: s.split(bt.col("d")),
            "SELECT CASE WHEN ?2 IS NOT NULL THEN string_split(?1, ?2) END",
            "sd",
        ),
        (lambda s: s.replace(bt.col("d"), "+"), "SELECT replace(?, ?, '+')", "sd"),
        (lambda s: s.contains(bt.col("d")), "SELECT contains(?::VARCHAR, ?::VARCHAR)", "sd"),
        (lambda s: s.starts_with(bt.col("t")), "SELECT starts_with(?, ?)", "st"),
        (lambda s: s.ends_with(bt.col("d")), "SELECT suffix(?, ?)", "sd"),
        (lambda s: s.levenshtein(bt.col("t")), "SELECT levenshtein(?, ?)", "st"),
        (lambda s: s.jaro_similarity(bt.col("t")), "SELECT jaro_similarity(?, ?)", "st"),
        (lambda s: s.substr(bt.col("n"), 2), "SELECT substring(?, ?, 2)", "sn"),
        (lambda s: s.substr(1, bt.col("n")), "SELECT substring(?, 1, ?)", "sn"),
    ],
)
def test_a_per_row_parameter_matches_duckdb(duck, build, sql, cols):
    got = _got(build(bt.col("s").str))
    want = _duck_rows(duck, sql, *cols)
    if "jaro" in sql:
        got = [None if v is None else round(v, 12) for v in got]
        want = [None if v is None else round(v, 12) for v in want]
    assert got == want


def test_slice_with_a_per_row_offset_matches_python_slicing():
    texts = ["abcdef", "abcdef", "abcdef", "xy", None]
    offsets = [0, 2, -3, -1, 1]
    ds = bt.from_pydict({"s": texts, "o": offsets})
    got = ds.select(r=bt.col("s").str.slice(bt.col("o"), 2)).to_pydict()["r"]
    expected = []
    for s, o in zip(texts, offsets, strict=True):
        start = None if s is None else (o if o >= 0 else len(s) + o)
        expected.append(None if s is None else s[start : start + 2])
    assert got == expected


def test_a_literal_expression_stays_on_the_constant_kernel():
    """`lit("-")` is a constant, so the plan should not group rows by it."""
    assert isinstance(bt.col("s").str.split(bt.lit("-")), StrFunc)
    assert isinstance(bt.col("s").str.split(bt.col("d")), StrFuncDyn)


def test_column_against_column_hamming_raises_on_unequal_length_like_duckdb():
    ds = bt.from_pydict({"a": ["abc", "abd"], "b": ["abd", "ab"]})
    with pytest.raises(Exception, match="equal length"):
        ds.select(r=bt.col("a").str.hamming(bt.col("b"))).to_pydict()


def test_a_regex_replacement_refuses_a_per_row_pattern():
    with pytest.raises(PlanError, match="literal=True"):
        bt.col("s").str.replace_all(bt.col("d"), "x")


# --- parse_*(separator=) ------------------------------------------------------------------

_PATHS = ["/a/b\\c.txt", "C:\\dir\\f.txt", "a/b/", "/", "a\\b/c", "noslash", "", None]
_DUCK_SEPARATOR = {"both": "both_slash", "forward": "forward_slash", "backslash": "backslash"}


@pytest.mark.parametrize("separator", ["both", "forward", "backslash"])
@pytest.mark.parametrize("fn", ["parse_filename", "parse_dirname", "parse_dirpath", "parse_path"])
def test_path_separator_matches_duckdb(duck, separator, fn):
    ds = bt.from_pydict({"p": _PATHS})
    got = ds.select(r=getattr(bt.col("p").str, fn)(separator=separator)).to_pydict()["r"]
    sep = _DUCK_SEPARATOR[separator]
    sql = f"SELECT {fn}(?, false, ?)" if fn == "parse_filename" else f"SELECT {fn}(?, ?)"
    want = [duck.execute(sql, [p, sep]).fetchone()[0] for p in _PATHS]
    assert got == want


def test_the_default_separator_keeps_the_wire_shape():
    assert "pattern" not in bt.col("p").str.parse_filename().to_ir()


# --- decompress(max_output_bytes=) --------------------------------------------------------


def test_decompress_bound_against_stdlib():
    small = b"hello world"
    bomb = b"\x00" * 1_000_000
    frames = [gzip.compress(small), gzip.compress(bomb), b"not gzip", None]
    ds = bt.from_pydict({"z": frames})
    capped = bt.col("z").str.decompress("gzip", max_output_bytes=1024)
    unbounded = bt.col("z").str.decompress("gzip")
    out = ds.select(capped=capped, unbounded=unbounded).to_pydict()
    assert out["capped"] == [gzip.decompress(frames[0]), None, None, None]
    assert out["unbounded"] == [small, bomb, None, None]


@pytest.mark.parametrize("codec", ["gzip", "zlib", "zstd", "brotli", "lz4", "deflate"])
def test_the_bound_is_exact_for_every_codec(codec):
    payload = "x" * 4096
    ds = bt.from_pydict({"s": [payload]}).select(z=bt.col("s").str.compress(codec))
    at = ds.select(r=bt.col("z").str.decompress(codec, max_output_bytes=4096)).to_pydict()
    under = ds.select(r=bt.col("z").str.decompress(codec, max_output_bytes=4095)).to_pydict()
    assert at["r"] == [payload.encode()]
    assert under["r"] == [None]
    if codec == "zlib":
        assert zlib.decompress(ds.to_pydict()["z"][0]) == payload.encode()


def test_a_negative_bound_is_a_plan_error():
    with pytest.raises(PlanError, match="max_output_bytes"):
        bt.col("z").str.decompress("gzip", max_output_bytes=-1)


# --- chunk(offsets=True) ------------------------------------------------------------------

_DOCS = ["alpha beta gamma. delta! epsilon\nzeta", "h\u00e9llo\u2192w\u00f6rld", "ab", "", None]


@pytest.mark.parametrize("boundary", ["char", "word", "sentence", "line"])
@pytest.mark.parametrize(("size", "overlap"), [(4, 0), (5, 2), (9, 3), (1, 0)])
def test_chunk_offsets_locate_each_chunk_in_its_source(boundary, size, overlap):
    ds = bt.from_pydict({"d": _DOCS})
    out = ds.select(
        plain=bt.col("d").str.chunk(size, overlap, boundary),
        located=bt.col("d").str.chunk(size, overlap, boundary, offsets=True),
    ).to_pydict()
    for doc, plain, located in zip(_DOCS, out["plain"], out["located"], strict=True):
        if doc is None:
            assert plain is None and located is None
            continue
        assert [c["text"] for c in located] == plain
        for c in located:
            assert doc[c["start"] : c["start"] + len(c["text"])] == c["text"]
        starts = [c["start"] for c in located]
        assert starts == sorted(starts)
