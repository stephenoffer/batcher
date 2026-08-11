"""Detecting non-ASCII content and mixed scripts.

A corpus that is supposed to be English and is 3% non-ASCII has something in it — a
translation, an encoding error, or a block of emoji. All three want routing rather than
embedding, and the ratio finds them without a model.

    python examples/expr_text/language_and_script.py
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
            "id": [1, 2, 3, 4],
            "text": [
                "A plain English sentence.",
                "Une phrase avec des accents: naive, cafe, eleve.",
                "Mixed content with an emoji and a symbol.",
                "1234567890 !!! ???",
            ],
        }
    )

    profiled = documents.select(
        "id",
        ascii_only=col("text").str.is_ascii_only(),
        non_ascii=col("text").str.non_ascii_count(),
        non_ascii_ratio=col("text").str.non_ascii_ratio(),
        digits=col("text").str.digit_ratio(),
        alpha=col("text").str.alpha_ratio(),
        punctuation=col("text").str.punctuation_ratio(),
    )
    result = profiled.to_pydict()
    print({name: column for name, column in result.items() if name != "id"})

    # Every ratio is a proportion.
    for name in ("non_ascii_ratio", "digits", "alpha", "punctuation"):
        assert all(0.0 <= value <= 1.0 for value in result[name]), name

    # The non-ASCII count and the ratio agree with the flag.
    assert all(
        (count == 0) == flag
        for count, flag in zip(result["non_ascii"], result["ascii_only"], strict=True)
    )

    # The numeric document is mostly digits and punctuation, not letters.
    assert result["alpha"][3] < 0.2
    assert result["digits"][3] > 0.3

    # And the prose documents are mostly letters.
    assert result["alpha"][0] > 0.6

    # Routing: keep the documents that look like prose.
    prose = documents.filter(
        (col("text").str.alpha_ratio() > 0.5) & (col("text").str.digit_ratio() < 0.1)
    )
    print("prose documents:", prose.to_pydict()["id"])
    assert 4 not in prose.to_pydict()["id"]

    # Removing non-ASCII, when the pipeline downstream cannot take it.
    stripped = documents.select("id", text=col("text").str.remove_non_ascii()).to_pydict()
    assert all(value.isascii() for value in stripped["text"])


if __name__ == "__main__":
    main()
