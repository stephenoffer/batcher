"""Character n-gram overlap metrics (chrF-style) — language-agnostic generation scoring.

These score the overlap of two strings' character n-gram *sets*, so they are pinned to the exact
set arithmetic on hand-computable inputs: the intersection over each side's distinct n-grams
(precision, recall), their harmonic mean (F1), and intersection-over-union (Jaccard). The point of
the family is that it needs no whitespace, so a CJK case is included.
"""

from __future__ import annotations

import pytest

import batcher as bt

pytestmark = pytest.mark.unit


def _bigrams(s: str) -> set[str]:
    s = s.lower()
    return {s[i : i + 2] for i in range(len(s) - 1)} if len(s) >= 2 else {s}


def _agg(expr) -> float:
    ds = bt.from_pydict({"p": ["abcd"], "r": ["abce"]})
    return ds.agg(m=expr).to_pydict()["m"][0]


def test_precision_is_intersection_over_predicted() -> None:
    # bigrams("abcd") = {ab, bc, cd}; bigrams("abce") = {ab, bc, ce}; ∩ = {ab, bc}.
    p, r = _bigrams("abcd"), _bigrams("abce")
    expected = len(p & r) / len(p)
    assert _agg(bt.char_ngram_precision("p", "r", n=2)) == pytest.approx(expected)


def test_recall_is_intersection_over_reference() -> None:
    p, r = _bigrams("abcd"), _bigrams("abce")
    expected = len(p & r) / len(r)
    assert _agg(bt.char_ngram_recall("p", "r", n=2)) == pytest.approx(expected)


def test_f1_is_the_harmonic_mean_of_precision_and_recall() -> None:
    p, r = _bigrams("abcd"), _bigrams("abce")
    prec = len(p & r) / len(p)
    rec = len(p & r) / len(r)
    expected = 2 * prec * rec / (prec + rec)
    assert _agg(bt.char_ngram_f1("p", "r", n=2)) == pytest.approx(expected)


def test_jaccard_is_intersection_over_union() -> None:
    p, r = _bigrams("abcd"), _bigrams("abce")
    expected = len(p & r) / len(p | r)
    assert _agg(bt.char_ngram_jaccard("p", "r", n=2)) == pytest.approx(expected)


def test_identical_strings_score_one() -> None:
    ds = bt.from_pydict({"p": ["hello world"], "r": ["hello world"]})
    for metric in (bt.char_ngram_f1, bt.char_ngram_precision, bt.char_ngram_jaccard):
        assert ds.agg(m=metric("p", "r", n=3)).to_pydict()["m"][0] == pytest.approx(1.0)


def test_disjoint_strings_score_zero() -> None:
    ds = bt.from_pydict({"p": ["abcd"], "r": ["wxyz"]})
    assert ds.agg(m=bt.char_ngram_f1("p", "r", n=2)).to_pydict()["m"][0] == pytest.approx(0.0)


def test_case_and_whitespace_are_normalized() -> None:
    # Folding case and collapsing runs of spaces makes these identical.
    ds = bt.from_pydict({"p": ["Hello   World"], "r": ["hello world"]})
    assert ds.agg(m=bt.char_ngram_f1("p", "r", n=3)).to_pydict()["m"][0] == pytest.approx(1.0)


def test_works_without_whitespace_boundaries() -> None:
    # The whole point: no spaces, so a whitespace tokenizer would see one token.
    ds = bt.from_pydict({"p": ["東京都"], "r": ["東京市"]})
    # bigrams: {東京, 京都} vs {東京, 京市}; ∩ = {東京}; P = R = 1/2; F1 = 0.5.
    assert ds.agg(m=bt.char_ngram_f1("p", "r", n=2)).to_pydict()["m"][0] == pytest.approx(0.5)


def test_invalid_n_raises() -> None:
    from batcher._internal.errors import PlanError

    with pytest.raises(PlanError):
        bt.char_ngram_f1("p", "r", n=0)


def test_char_ngram_metrics_compose_with_group_by() -> None:
    ds = bt.from_pydict(
        {"lang": ["en", "en"], "p": ["cat", "dog"], "r": ["cat", "dig"]}
    )
    out = ds.group_by("lang").agg(f=bt.char_ngram_f1("p", "r", n=2)).to_pydict()
    assert out["lang"] == ["en"]
    assert len(out["f"]) == 1
