"""Prepare a small training corpus, then rerank retrieved candidates for diversity.

The corpus half runs the three steps between "we have text" and "we can train on it": mix
sources at declared weights, drop the documents that are not prose, and remove training
documents that quote the evaluation set. The retrieval half takes a retrieved candidate list
and reranks it with maximal marginal relevance, so the context does not hold one fact twice.

Everything here is a plan over the public `Dataset` API; the reranker runs as a batch UDF over
whole candidate lists, never per row.

    python examples/ml/corpus_and_retrieval.py
"""

from __future__ import annotations

import batcher as bt
from batcher import ml


def main() -> None:
    # --- Mixing: sample two sources at 3:1, whatever their sizes. -------------------------
    web = bt.from_pydict({"text": [f"web page number {i} about cooking" for i in range(800)]})
    code = bt.from_pydict({"text": [f"def function_{i}(x): return x" for i in range(800)]})
    mixed, report = ml.mix_corpora(
        {"web": web, "code": code}, {"web": 3, "code": 1}, total_rows=400, seed=0
    )
    print("realized weights", report.realized_weights)
    assert report.realized_weights == {"web": 0.75, "code": 0.25}
    assert mixed.count() == 400

    # --- Quality filtering: see what each rule removes before trusting it. ----------------
    docs = bt.from_pydict(
        {
            "text": [
                "A real sentence about something, written out properly.",
                "Another paragraph of ordinary prose that a model can learn from.",
                "buy now!!!",
                "1234 5678 9012 3456",
            ]
        }
    )
    thresholds = ml.QualityThresholds(min_words=3)
    share_passing = ml.quality_report(docs, "text", thresholds)
    print("share passing each rule", share_passing)
    assert share_passing["all"] == 0.5  # two of the four documents pass every rule
    kept = ml.quality_filter(docs, "text", thresholds).to_pydict()["text"]
    assert kept == [
        "A real sentence about something, written out properly.",
        "Another paragraph of ordinary prose that a model can learn from.",
    ]

    # --- Decontamination: a training document that quotes the eval set must go. ----------
    train = bt.from_pydict(
        {"text": ["what is the capital of france", "an unrelated training document"]}
    )
    evals = bt.from_pydict({"text": ["what is the capital of france"]})
    assert ml.contamination_rate(train, "text", evals, n=4) == 0.5
    clean = ml.decontaminate(train, "text", evals, n=4).to_pydict()["text"]
    assert clean == ["an unrelated training document"]

    # --- Ordering: group similar lengths so a padded batch wastes less. -------------------
    lengths = [1, 9, 2, 8, 3, 7, 4, 6]
    corpus = bt.from_pydict({"tokens": [[1] * n for n in lengths]})
    before = ml.padding_waste(corpus, "tokens", batch_size=2)
    grouped = ml.length_grouped_order(corpus, "tokens", batch_size=2)
    after = ml.padding_waste(grouped, "tokens", batch_size=2)
    print(f"padding waste {before:.3f} -> {after:.3f}")
    assert after < before
    assert grouped.count() == len(lengths)

    # --- Retrieval: MMR keeps one of two near-duplicate passages. -------------------------
    candidates = bt.from_pydict(
        {
            "vecs": [[[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]],
            "docs": [["paris is the capital", "paris is the capital (copy)", "lyon is a city"]],
            "scores": [[0.9, 0.89, 0.5]],
        }
    )
    rerank = ml.mmr_rerank_udf(
        embedding_column="vecs", score_column="scores", rerank_columns=("docs",), k=2
    )
    chosen = candidates.map_batches(rerank).to_pydict()["docs"][0]
    print("MMR picked", chosen)
    assert chosen == ["paris is the capital", "lyon is a city"]


if __name__ == "__main__":
    main()
