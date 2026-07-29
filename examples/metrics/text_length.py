"""Length and readability distribution over a text column.

Means hide the tail, which is where cost lives: a token budget is blown by the p99, not
the average. The quantile aggregates answer that directly, and the token estimates turn a
character count into the number you actually get billed for.

    python examples/metrics/text_length.py
"""

from __future__ import annotations

import batcher as bt


def main() -> None:
    docs = bt.from_pydict(
        {
            "text": [
                "Short one.",
                "A somewhat longer sentence with several more words in it.",
                "Tiny.",
                "The quick brown fox jumps over the lazy dog near the river bank at dawn.",
            ],
        }
    )

    shape = docs.select(
        shortest=bt.min_char_length("text"),
        longest=bt.max_char_length("text"),
        spread=bt.char_length_range("text"),
        median_chars=bt.char_length_quantile("text", 0.5),
        p90_chars=bt.char_length_quantile("text", 0.9),
        median_words=bt.word_count_quantile("text", 0.5),
        # Token accounting for an LLM stage.
        total_tokens=bt.total_token_estimate("text"),
        p90_tokens=bt.token_estimate_quantile("text", 0.9),
        over_budget=bt.token_budget_exceed_rate("text", budget=10),
        # Readability and word shape.
        chars_per_word=bt.mean_chars_per_word("text"),
        words_per_sentence=bt.mean_words_per_sentence("text"),
        long_words=bt.long_word_rate("text", min_length=6),
        paragraphs=bt.mean_paragraph_count("text"),
        readability=bt.automated_readability_index("text"),
    ).to_pydict()

    print(shape)

    assert shape["shortest"] == [5]  # "Tiny."
    assert shape["longest"][0] == max(len(v) for v in docs.to_pydict()["text"])
    assert shape["spread"][0] == shape["longest"][0] - shape["shortest"][0]
    assert shape["p90_chars"][0] >= shape["median_chars"][0]
    assert shape["median_words"][0] > 0
    assert shape["total_tokens"][0] > 0
    assert shape["p90_tokens"][0] > 0
    assert 0.0 <= shape["over_budget"][0] <= 1.0
    assert shape["chars_per_word"][0] > 0
    assert 0.0 <= shape["long_words"][0] <= 1.0

    # The budget check this exists for: how much of the corpus blows a 10-token budget?
    assert shape["over_budget"][0] > 0.0


if __name__ == "__main__":
    main()
