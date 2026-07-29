"""Tone and writing-system rates: style drift and language mix.

Tone rates catch a model that has become hedging or sycophantic after a prompt change.
Script rates catch a corpus that is not the language you think it is, which is the usual
reason a "multilingual" eval quietly measures English.

    python examples/metrics/text_tone_and_script.py
"""

from __future__ import annotations

import batcher as bt


def main() -> None:
    replies = bt.from_pydict(
        {
            "text": [
                "It might possibly be around 40, perhaps.",
                "Thank you so much! Please let me know.",
                "What would you like to do next?",
                "I calculated the total as 42.",
                "Absolutely amazing!!!",
            ],
        }
    )

    tone = replies.select(
        hedging=bt.hedge_rate("text"),
        polite=bt.politeness_rate("text"),
        questions=bt.question_rate("text"),
        exclamations=bt.exclamation_rate("text"),
        first_person=bt.first_person_rate("text"),
        # Your own phrase list, for house-style checks.
        boilerplate=bt.contains_phrase_rate("text", ["as an AI", "I cannot"]),
    ).to_pydict()

    print(tone)

    for name, value in tone.items():
        assert 0.0 <= value[0] <= 1.0, name
    assert tone["questions"][0] == 0.2
    assert tone["boilerplate"][0] == 0.0
    assert tone["hedging"][0] > 0.0
    assert tone["polite"][0] > 0.0

    # Writing systems, over a deliberately mixed corpus.
    mixed = bt.from_pydict(
        {
            "text": [
                "Plain latin text",
                "Русский текст",
                "中文文本",
                "نص عربي",
                "emoji only 🎉🚀",
            ],
        }
    )
    scripts = mixed.select(
        latin_only=bt.latin_only_rate("text"),
        cyrillic=bt.cyrillic_rate("text"),
        cjk=bt.cjk_rate("text"),
        arabic=bt.arabic_rate("text"),
        emoji=bt.emoji_rate("text"),
    ).to_pydict()
    print(scripts)

    for name, value in scripts.items():
        assert 0.0 <= value[0] <= 1.0, name
    assert scripts["cyrillic"][0] > 0.0
    assert scripts["cjk"][0] > 0.0
    assert scripts["arabic"][0] > 0.0
    assert scripts["emoji"][0] > 0.0


if __name__ == "__main__":
    main()
