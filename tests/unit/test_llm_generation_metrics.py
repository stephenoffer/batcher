"""Generation-quality metrics for LLM output evaluation.

No rouge_score / sacrebleu is installed, so each metric is checked against a hand-written
reference that applies the same SQuAD normalization (lowercase, drop articles/punctuation) and
set-based token overlap the metric documents.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

import batcher as bt

pytestmark = pytest.mark.unit


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def _tok(s: str) -> set[str]:
    n = _norm(s)
    return set(n.split(" ")) if n else set()


@pytest.fixture(scope="module")
def corpus() -> tuple[list[str], list[str], bt.Dataset]:
    pred = ["the cat sat on the mat", "hello world", "yes", "abc def", "Paris"]
    ref = ["the cat sat on a mat", "hello there", "yes", "xyz", "paris."]
    ds = bt.from_pydict({"p": pred, "r": ref})
    return pred, ref, ds


def _agg(ds: bt.Dataset, expr) -> float:
    return ds.agg(m=expr).to_pydict()["m"][0]


def test_exact_match(corpus) -> None:
    pred, ref, ds = corpus
    expected = np.mean([1.0 if p == r else 0.0 for p, r in zip(pred, ref, strict=True)])
    assert _agg(ds, bt.exact_match("p", "r")) == pytest.approx(expected)


def test_normalized_exact_match(corpus) -> None:
    pred, ref, ds = corpus
    expected = np.mean(
        [1.0 if _norm(p) == _norm(r) else 0.0 for p, r in zip(pred, ref, strict=True)]
    )
    assert _agg(ds, bt.normalized_exact_match("p", "r")) == pytest.approx(expected)


def test_token_set_precision(corpus) -> None:
    pred, ref, ds = corpus
    vals = []
    for p, r in zip(pred, ref, strict=True):
        tp, tr = _tok(p), _tok(r)
        vals.append(len(tp & tr) / len(tp) if tp else 0.0)
    assert _agg(ds, bt.token_set_precision("p", "r")) == pytest.approx(np.mean(vals))


def test_token_set_recall(corpus) -> None:
    pred, ref, ds = corpus
    vals = []
    for p, r in zip(pred, ref, strict=True):
        tp, tr = _tok(p), _tok(r)
        vals.append(len(tp & tr) / len(tr) if tr else 0.0)
    assert _agg(ds, bt.token_set_recall("p", "r")) == pytest.approx(np.mean(vals))


def test_token_set_f1(corpus) -> None:
    pred, ref, ds = corpus
    vals = []
    for p, r in zip(pred, ref, strict=True):
        tp, tr = _tok(p), _tok(r)
        total = len(tp) + len(tr)
        vals.append(2 * len(tp & tr) / total if total else 0.0)
    assert _agg(ds, bt.token_set_f1("p", "r")) == pytest.approx(np.mean(vals))


def test_token_set_jaccard(corpus) -> None:
    pred, ref, ds = corpus
    vals = []
    for p, r in zip(pred, ref, strict=True):
        tp, tr = _tok(p), _tok(r)
        vals.append(len(tp & tr) / len(tp | tr) if (tp | tr) else 0.0)
    assert _agg(ds, bt.token_set_jaccard("p", "r")) == pytest.approx(np.mean(vals))


def test_length_ratio(corpus) -> None:
    pred, ref, ds = corpus
    vals = [len(p.split()) / len(r.split()) for p, r in zip(pred, ref, strict=True)]
    assert _agg(ds, bt.length_ratio("p", "r")) == pytest.approx(np.mean(vals))


def test_perfect_prediction_scores_one() -> None:
    ds = bt.from_pydict({"p": ["hello world"], "r": ["hello world"]})
    assert _agg(ds, bt.exact_match("p", "r")) == pytest.approx(1.0)
    assert _agg(ds, bt.token_set_f1("p", "r")) == pytest.approx(1.0)
    assert _agg(ds, bt.token_set_jaccard("p", "r")) == pytest.approx(1.0)


def test_metrics_compose_with_group_by() -> None:
    ds = bt.from_pydict({"model": ["a", "a", "b"], "p": ["x y", "z", "q"], "r": ["x y", "w", "q"]})
    out = ds.group_by("model").agg(em=bt.exact_match("p", "r")).sort("model").to_pydict()
    assert out["em"] == [0.5, 1.0]
