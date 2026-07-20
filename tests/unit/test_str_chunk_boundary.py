"""`.str.chunk(boundary=...)` — chunking that does not cut words in half.

Fixed-size chunking is the default and stays the default, but it is the wrong tool for
retrieval: a chunk ending `…diagnosed with hyperten` embeds as something the query
`hypertension treatment` will not match, so the mid-word cut silently costs recall on
exactly the chunk that should have been the answer. Nothing in a result reveals it —
which is why the boundary modes are pinned here rather than left to inspection.

The kernel's own edge cases live in Rust (`bc-expr/src/eval/str/chunk.rs`); this file
covers the public surface and the two properties a caller relies on: no word is split,
and with no overlap nothing is lost.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.unit

_PROSE = "The patient was diagnosed with hypertension. Treatment began at once! Ok?"


def _chunks(text: str, size: int, **kw) -> list[str]:
    ds = bt.from_pydict({"doc": [text]})
    return ds.select(r=bt.col("doc").str.chunk(size, **kw)).to_pydict()["r"][0]


def test_char_boundary_is_the_default_and_unchanged() -> None:
    assert _chunks("abcdefg", 3) == ["abc", "def", "g"]
    assert _chunks("abcdefg", 3, boundary="char") == ["abc", "def", "g"]


def test_word_boundary_never_splits_a_word() -> None:
    got = _chunks(_PROSE, 20, boundary="word")
    words = set(_PROSE.split())

    for chunk in got:
        trimmed = chunk.strip()
        assert not trimmed or any(trimmed.endswith(w) for w in words), (
            f"chunk {chunk!r} ends mid-word"
        )


def test_sentence_boundary_uses_terminators_and_never_cuts_mid_word() -> None:
    """The cascade: a full stop when one fits the window, a word boundary when not.

    Falling straight back to an arbitrary character would defeat the request — the caller
    asked for readable chunks, and one mid-word cut is the chunk that then fails to match.
    """
    got = _chunks(_PROSE, 40, boundary="sentence")

    for chunk in got[:-1]:
        assert chunk.endswith((".", "!", "?", " ")), f"{chunk!r} was cut mid-word"
    # At least one chunk actually used a sentence terminator, or the mode did nothing.
    assert any(c.rstrip().endswith((".", "!", "?")) for c in got)


def test_line_boundary_ends_on_newlines() -> None:
    text = "first line\nsecond line\nthird line\nfourth"
    got = _chunks(text, 18, boundary="line")

    for chunk in got[:-1]:
        assert chunk.endswith("\n"), f"{chunk!r} does not end a line"


@pytest.mark.parametrize("boundary", ["char", "word", "sentence", "line"])
def test_no_overlap_is_lossless_in_every_mode(boundary: str) -> None:
    """A separator ends the chunk it belongs to, so no character is dropped."""
    assert "".join(_chunks(_PROSE, 16, boundary=boundary)) == _PROSE


def test_a_token_longer_than_the_chunk_is_still_emitted() -> None:
    """No boundary to back off to must mean a hard cut, not a dropped token or a hang."""
    text = "short " + "x" * 50 + " end"
    got = _chunks(text, 10, boundary="word")

    assert "".join(got) == text
    assert any(len(c) == 10 for c in got), "the long token was never hard-cut"


def test_overlap_still_works_with_a_boundary() -> None:
    got = _chunks(_PROSE, 24, overlap=6, boundary="word")

    assert len(got) > 1
    # Overlap means coverage, so every word still appears somewhere.
    joined = " ".join(got)
    for word in _PROSE.split():
        assert word.strip(".!?") in joined


def test_null_and_empty_are_unchanged() -> None:
    ds = bt.from_pydict({"doc": [None, ""]})
    got = ds.select(r=bt.col("doc").str.chunk(5, boundary="word")).to_pydict()["r"]

    assert got[0] is None
    assert got[1] == []


def test_non_ascii_text() -> None:
    text = "café naïve résumé déjà vu"
    assert "".join(_chunks(text, 10, boundary="word")) == text


def test_an_unknown_boundary_is_rejected_at_plan_time() -> None:
    """Caught in the control plane, before the engine sees it."""
    with pytest.raises(PlanError, match="boundary must be one of"):
        bt.col("doc").str.chunk(10, boundary="paragraph")


def test_the_rag_shape_chunk_then_explode_with_a_position() -> None:
    """The whole pipeline: split a document, keep the order, lose no document."""
    docs = bt.from_pydict({"id": [1, 2], "doc": [_PROSE, ""]})

    out = (
        docs.with_columns(chunks=bt.col("doc").str.chunk(24, boundary="word"))
        .explode("chunks", outer=True, index="pos")
        .to_pydict()
    )

    # The empty document survives as one null-chunk row rather than vanishing.
    assert 2 in out["id"]
    assert out["chunks"][out["id"].index(2)] is None
    # Chunks of document 1 are numbered from 0 in order.
    positions = [p for i, p in zip(out["id"], out["pos"], strict=True) if i == 1]
    assert positions == list(range(len(positions)))
