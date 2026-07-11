"""`.str.chunk(size, overlap)` — the RAG document splitter — matches a DuckDB oracle.

DuckDB has no chunking function, so the oracle is built from primitives it *does* have
(`range` + `substring`, both character-based on VARCHAR). Expressing the contract twice,
once in Rust and once in SQL, is what makes the agreement meaningful: the SQL says
"chunk *i* is the `size` characters starting at ``i * (size - overlap)``, and there are
just enough chunks to reach the end", which is the definition, independent of the
implementation.

The properties that a retrieval pipeline actually depends on — every character lands in
some chunk, consecutive chunks share exactly `overlap` characters, and no chunk splits a
Unicode codepoint — are pinned separately, because a chunker that agrees with an oracle
on ASCII can still corrupt UTF-8.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.plan.expr_ir import col

pytestmark = pytest.mark.differential

_DOCS = ["abcdefg", "abcdef", "ab", "a", "", "héllo→wörld", "the quick brown fox jumps"]


def _oracle_sql(size: int, overlap: int) -> str:
    """Chunking spelled out in DuckDB: `n` chunks of `size` chars, stride `size-overlap`.

    The chunk count is ``ceil((len - size) / stride) + 1`` — just enough starts for one
    chunk to reach the end — and 1 when the text is no longer than a chunk.
    """
    stride = size - overlap
    return f"""
        SELECT CASE WHEN length(d) = 0 THEN []
          ELSE list_transform(
            range(0, CASE WHEN length(d) <= {size} THEN 1
                     ELSE cast(ceil((length(d) - {size})::DOUBLE / {stride}) AS INT) + 1 END),
            i -> substring(d, i * {stride} + 1, {size}))
        END AS r FROM docs
    """


@pytest.mark.parametrize("doc", _DOCS)
@pytest.mark.parametrize("size", [1, 2, 3, 4, 5, 7])
def test_chunk_matches_duckdb_for_every_overlap(duck, doc, size):
    duck.execute("CREATE OR REPLACE TABLE docs AS SELECT ? AS d", [doc])
    ds = bt.from_pydict({"d": [doc]})
    for overlap in range(size):
        got = ds.select(r=col("d").str.chunk(size, overlap)).to_pydict()["r"][0]
        expected = duck.sql(_oracle_sql(size, overlap)).fetchone()[0]
        assert list(got) == list(expected), f"size={size} overlap={overlap} doc={doc!r}"


def test_null_yields_null_and_empty_yields_an_empty_list():
    got = bt.from_pydict({"d": ["ab", None, ""]}).select(r=col("d").str.chunk(5)).to_pydict()["r"]
    assert got == [["ab"], None, []]


@pytest.mark.parametrize("size", [2, 3, 4, 5])
def test_every_character_is_covered_exactly_once_beyond_the_overlap(size):
    """Rebuild the text from its chunks: first chunk in full, then each later chunk
    minus the `overlap` characters it repeats."""
    text = "the quick brown fox jumps over the lazy dog"
    ds = bt.from_pydict({"d": [text]})
    for overlap in range(size):
        chunks = ds.select(r=col("d").str.chunk(size, overlap)).to_pydict()["r"][0]
        rebuilt = chunks[0] + "".join(c[overlap:] for c in chunks[1:])
        assert rebuilt == text, f"size={size} overlap={overlap}"


def test_consecutive_chunks_share_exactly_the_overlap():
    chunks = (
        bt.from_pydict({"d": ["abcdefghij"]}).select(r=col("d").str.chunk(4, 2)).to_pydict()["r"][0]
    )
    for prev, nxt in pairwise(chunks):
        assert prev[-2:] == nxt[:2]


def test_chunking_never_splits_a_codepoint():
    """A byte-wise slice of this text would cut a 2- or 3-byte character in half."""
    text = "héllo→wörld"
    chunks = bt.from_pydict({"d": [text]}).select(r=col("d").str.chunk(3)).to_pydict()["r"][0]
    assert chunks == ["hél", "lo→", "wör", "ld"]
    assert "".join(chunks) == text


def test_no_redundant_tail_chunk():
    """A final start whose chunk sits wholly inside the previous one is not emitted."""
    got = bt.from_pydict({"d": ["abcdefg"]}).select(r=col("d").str.chunk(4, 2)).to_pydict()["r"][0]
    assert got == ["abcd", "cdef", "efg"]  # not [..., "efg", "g"]


def test_chunk_then_explode_is_one_row_per_chunk():
    """The RAG ingest shape: split a document into rows ready to embed."""
    out = (
        bt.from_pydict({"id": [1, 2], "doc": ["abcdef", "xyz"]})
        .with_columns(chunk=col("doc").str.chunk(4, overlap=1))
        .explode("chunk")
        .select("id", "chunk")
        .to_pydict()
    )
    assert out == {"id": [1, 1, 2], "chunk": ["abcd", "def", "xyz"]}


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (-1, 0), (3, 3), (3, 4), (3, -1)])
def test_invalid_frames_are_rejected_at_the_api_edge(size, overlap):
    with pytest.raises(PlanError, match=r"str\.chunk"):
        col("d").str.chunk(size, overlap)
