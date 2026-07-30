"""`list.lcs_length` against a textbook LCS, over generated sequences.

DuckDB has no longest-common-subsequence function, so the oracle here is a plain Python DP
written the obvious way — the reference the engine's rolling-row, row-encoded version has to
agree with. Random sequences over a small alphabet are what makes that comparison worth
running: they produce the repeats, the interleavings, and the near-misses a hand-written case
list would not think to include.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import batcher as bt

pytestmark = pytest.mark.property

# A small alphabet forces collisions, which is where an LCS is interesting: over distinct
# symbols almost every pair shares almost nothing and the property passes vacuously.
_TOKENS = st.sampled_from(["a", "b", "c", "d"])
_SEQUENCE = st.lists(_TOKENS, min_size=0, max_size=12)


def _lcs(left: list[str], right: list[str]) -> int:
    """The textbook full-table LCS length — the oracle."""
    table = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i, a in enumerate(left, start=1):
        for j, b in enumerate(right, start=1):
            table[i][j] = (
                table[i - 1][j - 1] + 1 if a == b else max(table[i - 1][j], table[i][j - 1])
            )
    return table[len(left)][len(right)]


def _engine(pairs: list[tuple[list[str], list[str]]]) -> list[float]:
    ds = bt.from_pydict({"a": [p[0] for p in pairs], "b": [p[1] for p in pairs]})
    return ds.select(n=bt.col("a").list.lcs_length(bt.col("b"))).to_pydict()["n"]


@settings(max_examples=150, deadline=None)
@given(left=_SEQUENCE, right=_SEQUENCE)
def test_the_engine_agrees_with_a_textbook_lcs(left, right):
    assert _engine([(left, right)]) == [float(_lcs(left, right))]


@settings(max_examples=60, deadline=None)
@given(left=_SEQUENCE, right=_SEQUENCE)
def test_the_length_is_symmetric(left, right):
    assert _engine([(left, right)]) == _engine([(right, left)])


@settings(max_examples=60, deadline=None)
@given(left=_SEQUENCE, right=_SEQUENCE)
def test_the_length_never_exceeds_either_sequence(left, right):
    got = _engine([(left, right)])[0]
    assert got <= min(len(left), len(right))


@settings(max_examples=60, deadline=None)
@given(sequence=_SEQUENCE)
def test_a_sequence_matches_itself_completely(sequence):
    assert _engine([(sequence, sequence)]) == [float(len(sequence))]


@settings(max_examples=40, deadline=None)
@given(pairs=st.lists(st.tuples(_SEQUENCE, _SEQUENCE), min_size=1, max_size=8))
def test_a_batch_is_scored_row_by_row(pairs):
    """The rolling DP buffers are reused across rows, so independence is the risk."""
    assert _engine(pairs) == [float(_lcs(a, b)) for a, b in pairs]
