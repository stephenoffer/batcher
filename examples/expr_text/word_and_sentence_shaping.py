"""Truncating text to a budget, by characters, words or sentences.

Truncating mid-word is what makes a preview look broken; truncating mid-sentence is what
makes a summary read as if it were cut off, because it was. Pick the unit that matches what
the text is for.

    python examples/expr_text/word_and_sentence_shaping.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from batcher import col


def main() -> None:
    documents = bt.from_pydict(
        {
            "id": [1, 2],
            "body": [
                "The engine reads Arrow batches. It never touches a row in Python. "
                "That is the whole design.",
                "Short one.",
            ],
        }
    )

    shaped = documents.select(
        "id",
        by_chars=col("body").str.truncate_chars(30),
        by_words=col("body").str.truncate_words(5),
        by_sentences=col("body").str.truncate_sentences(1),
        first_sentence=col("body").str.first_sentence(),
        first_word=col("body").str.first_word(),
        last_word=col("body").str.last_word(),
    )
    result = shaped.to_pydict()
    for name, column in result.items():
        if name != "id":
            print(f"{name:<16} {column[0]!r}")

    # Character truncation respects the budget.
    assert all(len(value) <= 30 for value in result["by_chars"])

    # Word truncation keeps whole words.
    assert len(result["by_words"][0].split()) <= 5
    assert not result["by_words"][0].endswith(" ")

    # Sentence truncation ends at a boundary.
    assert result["by_sentences"][0].rstrip().endswith(".")
    assert result["first_sentence"][0] == "The engine reads Arrow batches."

    # A document already inside the budget is unchanged.
    assert result["by_words"][1] == documents.to_pydict()["body"][1]

    # First and last word.
    assert result["first_word"][0] == "The"
    assert result["last_word"][1].rstrip(".") == "one"


if __name__ == "__main__":
    main()
