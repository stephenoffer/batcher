"""Keywords that distinguish one group from the rest.

A word's raw frequency tells you it is common, not that it is characteristic. The ratio of
its rate inside a group to its rate everywhere is what makes it a keyword — the same idea as
TF-IDF, computed as two group-bys and a join.

    python examples/text_analytics/topic_keywords.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lines = tpch("lineitem").select("l_shipmode", "l_comment").head(40_000)

    words = (
        lines.select("l_shipmode", word=col("l_comment").str.to_lowercase().str.split(" "))
        .explode("word")
        .filter(col("word").str.len_chars() > 3)
    )
    total_words = words.count()
    print("tokens:", total_words)

    # The global rate of each word.
    globally = (
        words.group_by("word")
        .agg(n=bt.count())
        .with_columns(global_rate=col("n") / total_words)
        .select("word", "global_rate")
    )

    # The rate within each ship mode.
    per_mode = words.group_by("l_shipmode").agg(mode_total=bt.count())
    within = (
        words.group_by("l_shipmode", "word")
        .agg(n=bt.count())
        .join(per_mode, on="l_shipmode")
        .with_columns(mode_rate=col("n") / col("mode_total"))
    )

    # The lift: how much more common the word is here than everywhere.
    keywords = (
        within.join(globally, on="word")
        .with_columns(lift=col("mode_rate") / col("global_rate"))
        # Ignore words too rare to be evidence of anything.
        .filter(col("n") >= 20)
        .sort("lift", descending=True)
        .limit(10)
    )

    result = keywords.to_pydict()
    for mode, word, lift, count in zip(
        result["l_shipmode"], result["word"], result["lift"], result["n"], strict=True
    ):
        print(f"  {mode:<9} {word:<16} lift {lift:.3f} ({count} occurrences)")

    assert result["lift"] == sorted(result["lift"], reverse=True)
    assert all(value >= 20 for value in result["n"])

    # A lift above one means the word is over-represented in its group, which is the
    # entire claim being made.
    assert result["lift"][0] > 1.0

    # And the rates are both proportions.
    assert all(0.0 < value <= 1.0 for value in result["mode_rate"])
    assert all(0.0 < value <= 1.0 for value in result["global_rate"])


if __name__ == "__main__":
    main()
