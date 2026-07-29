"""Comparing a generated answer against a reference, without a model.

These are the reference-based scores you can compute in the engine: exact match for
closed-form answers, token-set overlap for short free text, and character n-gram overlap
when wording varies but content should not. No embedding call, no GPU.

    python examples/metrics/text_overlap.py
"""

from __future__ import annotations

import batcher as bt


def main() -> None:
    graded = bt.from_pydict(
        {
            "prediction": [
                "Paris",
                "paris ",
                "The capital of France is Paris",
                "Berlin",
            ],
            "reference": ["Paris", "Paris", "Paris is the capital of France", "Paris"],
        }
    )

    scores = graded.select(
        exact=bt.exact_match("prediction", "reference"),
        # Case- and whitespace-insensitive, which is usually what you meant.
        normalized=bt.normalized_exact_match("prediction", "reference"),
        # Bag-of-tokens overlap: order-free credit for the same words.
        set_f1=bt.token_set_f1("prediction", "reference"),
        set_precision=bt.token_set_precision("prediction", "reference"),
        set_recall=bt.token_set_recall("prediction", "reference"),
        set_jaccard=bt.token_set_jaccard("prediction", "reference"),
        # Character n-grams: robust to morphology and typos.
        ngram_f1=bt.char_ngram_f1("prediction", "reference", n=3),
        ngram_jaccard=bt.char_ngram_jaccard("prediction", "reference", n=3),
        # Are answers systematically too long or too short?
        length_ratio=bt.length_ratio("prediction", "reference"),
    ).to_pydict()

    print(scores)

    # These are corpus aggregates: one row out.
    assert len(scores["exact"]) == 1
    # Only row 1 matches byte for byte, but row 2 matches once normalized.
    assert scores["exact"][0] == 0.25
    assert scores["normalized"][0] == 0.5
    # Row 3 says the same thing in a different order, so set overlap beats exact match.
    assert scores["set_f1"][0] > scores["exact"][0]
    assert 0.0 <= scores["set_jaccard"][0] <= 1.0
    assert 0.0 <= scores["ngram_f1"][0] <= 1.0
    assert scores["length_ratio"][0] > 0.0


if __name__ == "__main__":
    main()
