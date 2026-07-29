"""Counting structure in text: words, lines, sentences, and entities.

Counts are the other half of a corpus filter. Length in characters says little; length in
words, sentences, or paragraphs says whether a document is a fragment, a paragraph, or a
scraped page. The entity counts (urls, emails, hashtags, mentions) find rows that are
mostly links rather than prose.

    python examples/expressions/strings_counts.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    posts = bt.from_pydict(
        {
            "text": [
                "First line.\nSecond line here.",
                "Check https://a.example and https://b.example #deal @acme",
                "One sentence only",
            ],
        }
    )

    counted = posts.with_columns(
        words=col("text").str.word_count(),
        lines=col("text").str.line_count(),
        sentences=col("text").str.sentence_count(),
        newlines=col("text").str.newline_count(),
        spaces=col("text").str.space_count(),
        urls=col("text").str.url_count(),
        hashtags=col("text").str.hashtag_count(),
        mentions=col("text").str.mention_count(),
        # Words at least `min_length` characters long (default 5).
        long_words=col("text").str.long_word_count(min_length=6),
        avg_word=col("text").str.avg_word_length(),
    )

    result = counted.to_pydict()
    print(result)

    assert result["words"] == [5, 6, 3]
    assert result["lines"] == [2, 1, 1]
    assert result["newlines"] == [1, 0, 0]
    assert result["urls"][1] == 2
    assert result["hashtags"][1] == 1
    assert result["mentions"][1] == 1
    assert result["sentences"][0] == 2
    assert all(v > 0 for v in result["avg_word"])
    # "Second" is the only 6+ letter word on row 1.
    assert result["long_words"][0] == 1

    # The filter this exists for: keep prose, drop the link dump.
    prose = posts.filter(col("text").str.url_count() == 0).to_pydict()
    assert len(prose["text"]) == 2


if __name__ == "__main__":
    main()
