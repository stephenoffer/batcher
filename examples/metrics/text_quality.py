"""Corpus hygiene rates: what fraction of a text column looks broken.

Every metric here is an aggregate returning a rate in [0, 1], so one ``select`` gives you
a scorecard for a whole generation run. These are the numbers you watch between model
versions: a jump in ``empty_or_whitespace_rate`` is a broken prompt, not a worse model.

    python examples/metrics/text_quality.py
"""

from __future__ import annotations

import batcher as bt


def main() -> None:
    outputs = bt.from_pydict(
        {
            "text": [
                "A clean, ordinary answer.",
                "   ",
                "SHOUTING ALL THE WAY",
                "Trailing whitespace here   ",
                "Double  spaces  everywhere",
            ],
        }
    )

    scorecard = outputs.select(
        blank=bt.empty_or_whitespace_rate("text"),
        all_caps=bt.all_caps_rate("text"),
        trailing_ws=bt.trailing_whitespace_rate("text"),
        leading_ws=bt.leading_whitespace_rate("text"),
        double_space=bt.double_space_rate("text"),
        has_tab=bt.has_tab_rate("text"),
        blank_lines=bt.blank_line_rate("text"),
        repeated_punct=bt.repeated_punctuation_rate("text"),
        non_ascii=bt.non_ascii_rate("text"),
        urls=bt.url_rate("text"),
        code_blocks=bt.code_block_rate("text"),
        short=bt.short_output_rate("text", max_chars=10),
        long=bt.long_output_rate("text", min_chars=20),
        mean_words=bt.mean_word_length("text"),
        mean_sentences=bt.mean_sentence_count("text"),
    ).to_pydict()

    print(scorecard)

    # Every rate is a proportion of the five rows.
    for name, value in scorecard.items():
        if name.startswith("mean_"):
            continue
        assert 0.0 <= value[0] <= 1.0, name

    # One of five rows is whitespace-only; one shouts; one has doubled spaces.
    assert scorecard["blank"] == [0.2]
    assert scorecard["all_caps"] == [0.2]
    # Three rows contain a doubled space: the whitespace-only row and the
    # trailing-whitespace row count too, not just the obvious one.
    assert scorecard["double_space"] == [0.6]
    assert scorecard["trailing_ws"] == [0.4]
    assert scorecard["non_ascii"] == [0.0]
    assert scorecard["urls"] == [0.0]
    assert scorecard["code_blocks"] == [0.0]

    # The gate this exists for: fail the run if too much of it is blank.
    assert scorecard["blank"][0] <= 0.25


if __name__ == "__main__":
    main()
