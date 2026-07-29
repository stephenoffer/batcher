"""Degeneracy detection: repetition, truncation, refusal, and empty output.

A model that has started looping produces text that is long and nearly information-free.
The character n-gram measures catch that reliably; ``truncation_rate`` and
``refusal_rate`` catch the two other common failure shapes.

    python examples/metrics/text_diversity.py
"""

from __future__ import annotations

import batcher as bt


def main() -> None:
    outputs = bt.from_pydict(
        {
            "text": [
                "A varied sentence carrying genuine information.",
                "the the the the the the the the the the",
                "",
                "I'm sorry, but I cannot help with that request.",
                "This sentence stops mid-",
            ],
        }
    )

    health = outputs.select(
        distinct_tokens=bt.distinct_token_ratio("text"),
        distinct_ngrams=bt.distinct_char_ngram_ratio("text", n=3),
        char_repetition=bt.char_repetition_rate("text", n=3),
        compression=bt.compression_ratio_proxy("text", n=3),
        repeated_lines=bt.repeated_line_rate("text"),
        empty=bt.empty_generation_rate("text"),
        refusals=bt.refusal_rate("text"),
        truncated=bt.truncation_rate("text"),
        mean_tokens=bt.mean_output_tokens("text"),
    ).to_pydict()

    print(health)

    # One of five rows is empty; one is a refusal.
    assert health["empty"][0] == 0.2
    assert health["refusals"][0] == 0.2
    assert 0.0 <= health["distinct_tokens"][0] <= 1.0
    assert 0.0 <= health["char_repetition"][0] <= 1.0
    assert health["mean_tokens"][0] > 0

    # The detector this exists for: a looping generation against a healthy one. Compare
    # them on the character n-gram measures, which separate the two cleanly.
    looped = bt.from_pydict({"text": ["the the the the the the the the the the"]})
    varied = bt.from_pydict({"text": ["A varied sentence carrying genuine information."]})

    def score(ds: bt.Dataset, metric) -> float:
        return ds.select(x=metric("text")).to_pydict()["x"][0]

    # Distinct n-grams collapse, and repetition and compression both spike.
    assert score(looped, bt.distinct_char_ngram_ratio) < score(varied, bt.distinct_char_ngram_ratio)
    assert score(looped, bt.char_repetition_rate) > score(varied, bt.char_repetition_rate)
    assert score(looped, bt.compression_ratio_proxy) > score(varied, bt.compression_ratio_proxy)
    print(
        score(looped, bt.distinct_char_ngram_ratio),
        score(varied, bt.distinct_char_ngram_ratio),
    )


if __name__ == "__main__":
    main()
