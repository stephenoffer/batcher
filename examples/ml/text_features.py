"""Turning raw text into model-ready features without a model.

Before reaching for an embedding, check whether cheap features answer the question. Length,
character mix, and token counts separate a lot of classes on their own, and they cost a
scan rather than a GPU.

    python examples/ml/text_features.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col, ml


def main() -> None:
    docs = bt.from_pydict(
        {
            "body": [
                "Hello there, how are you doing today?",
                "BUY NOW!!! LIMITED TIME OFFER!!!",
                "invoice 88213 total 4920.00 due 2024-03-01",
            ],
            "label": [0, 1, 0],
        }
    )

    # The featurizer builds a standard block of text statistics in one pass.
    featurized = (
        ml.TextStatFeaturizer(
            "body",
            features=("char_count", "word_count", "avg_word_length", "digit_ratio", "upper_ratio"),
        )
        .fit(docs)
        .transform(docs)
        .to_pydict()
    )

    print(sorted(featurized))
    assert any("char_count" in c for c in featurized)
    assert any("digit_ratio" in c for c in featurized)

    # The same thing hand-rolled, when you want a specific set.
    custom = docs.with_columns(
        length=col("body").str.len_chars(),
        words=col("body").str.word_count(),
        upper=col("body").str.uppercase_ratio(),
        digits=col("body").str.digit_ratio(),
        punct=col("body").str.punctuation_ratio(),
        exclamations=col("body").str.count_char("!"),
        tokens=col("body").str.estimate_tokens(),
    ).to_pydict()

    print(custom["upper"], custom["digits"])

    # The shouting row scores highest on uppercase and exclamation marks.
    assert custom["upper"][1] == max(custom["upper"])
    assert custom["exclamations"][1] == 6
    # The invoice row is the digit-heavy one.
    assert custom["digits"][2] == max(custom["digits"])

    # Which is already enough to separate the classes here.
    flagged = docs.with_columns(
        spam_score=col("body").str.uppercase_ratio() + col("body").str.count_char("!") / 10
    ).to_pydict()
    print("spam scores:", [round(v, 3) for v in flagged["spam_score"]])
    assert flagged["spam_score"][1] == max(flagged["spam_score"])

    # Normalize the text before any of this, so casing does not leak into the features
    # you did not intend it to.
    normalized = docs.select(
        clean=col("body").str.to_lowercase().str.remove_punctuation().str.normalize_whitespace()
    ).to_pydict()
    print(normalized["clean"][1])
    assert "!" not in normalized["clean"][1]
    assert normalized["clean"][1] == "buy now limited time offer"


if __name__ == "__main__":
    main()
