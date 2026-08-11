"""Classifying text by shape before classifying it by meaning.

Questions, exclamations, all-caps and code blocks are all detectable without a model. Doing
so first is how you route a corpus cheaply — and how you notice that 30% of it is stack
traces before you pay to embed them.

    python examples/text_analytics/language_shape.py
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
            "id": [1, 2, 3, 4, 5],
            "text": [
                "Is this the right way to configure the reader?",
                "THIS IS URGENT AND ALSO VERY LOUD",
                "def main():\n    return 42\n",
                "A perfectly ordinary sentence about data.",
                "",
            ],
        }
    )

    shaped = documents.select(
        "id",
        question=col("text").str.is_question(),
        exclamation=col("text").str.is_exclamation(),
        all_caps=col("text").str.is_all_caps(),
        looks_code=col("text").str.looks_like_code(),
        blank=col("text").str.is_blank(),
        single_line=col("text").str.is_single_line(),
    )
    result = shaped.to_pydict()
    print(result)

    assert result["question"] == [True, False, False, False, False]
    assert result["all_caps"][1] is True
    assert result["looks_code"][2] is True
    assert result["blank"][4] is True

    # A multi-line document is not single-line, which is the cheapest structural signal.
    assert result["single_line"][2] is False
    assert result["single_line"][0] is True

    # Routing: keep only the prose.
    prose = documents.filter(
        ~col("text").str.is_blank()
        & ~col("text").str.looks_like_code()
        & ~col("text").str.is_all_caps()
    )
    print("prose documents:", prose.to_pydict()["id"])
    assert prose.to_pydict()["id"] == [1, 4]


if __name__ == "__main__":
    main()
