"""Cheap readability signals over a text column.

Sentence length and word length are the two inputs to every readability index worth having.
They are also the two that catch a corpus of machine-generated boilerplate, which reads as
uniformly as it was written.

    python examples/text_analytics/readability_signals.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    comments = tpch("orders").select("o_orderkey", "o_comment").head(5_000)

    signals = comments.select(
        "o_orderkey",
        words=col("o_comment").str.word_count(),
        sentences=col("o_comment").str.sentence_count(),
        avg_word=col("o_comment").str.avg_word_length(),
        avg_sentence=col("o_comment").str.avg_sentence_length(),
        long_words=col("o_comment").str.long_word_count(),
    ).with_columns(long_word_share=col("long_words") / col("words"))

    summary = signals.agg(
        mean_words=col("words").mean(),
        mean_avg_word=col("avg_word").mean(),
        mean_sentence=col("avg_sentence").mean(),
        mean_long_share=col("long_word_share").mean(),
        sd_words=bt.std(col("words")),
    ).to_pydict()
    print({name: round(value[0], 4) for name, value in summary.items()})

    values = signals.to_pydict()

    # Every signal is in a sane range.
    assert all(value > 0 for value in values["words"])
    assert all(value > 0 for value in values["avg_word"])
    assert all(0.0 <= value <= 1.0 for value in values["long_word_share"])

    # Long words are a subset of words.
    assert all(
        long <= total for long, total in zip(values["long_words"], values["words"], strict=True)
    )

    # TPC-H comments come from a fixed grammar, so the *typical* document is tightly
    # clustered even though individual outliers are not. The interquartile range is the
    # statistic that says so; the min-to-max spread is dominated by the shortest comment,
    # which is a single word and drags the floor down.
    word_length_spread = max(values["avg_word"]) - min(values["avg_word"])
    middle = signals.agg(q1=bt.q1(col("avg_word")), q3=bt.q3(col("avg_word"))).to_pydict()
    iqr = middle["q3"][0] - middle["q1"][0]
    print(f"average-word-length: full spread {word_length_spread:.3f}, IQR {iqr:.3f}")
    assert summary["sd_words"][0] > 0
    assert iqr < 1.5
    assert iqr < word_length_spread


if __name__ == "__main__":
    main()
