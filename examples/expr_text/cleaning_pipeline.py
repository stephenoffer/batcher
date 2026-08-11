"""Normalizing free text before anything downstream reads it.

Each step is a vectorized expression, so a five-step cleanup is still one pass and no
Python. The order matters: strip markup before collapsing whitespace, or the tags leave
gaps behind.

    python examples/expr_text/cleaning_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from batcher import col


def main() -> None:
    raw = bt.from_pydict(
        {
            "id": [1, 2, 3],
            "body": [
                "  <p>Visit https://example.com for   details!!!</p>  ",
                "Contact SALES@example.com — or call 555-0100.",
                "### Heading\n\nSome  text with **markdown** and a [link](http://x.io).",
            ],
        }
    )

    cleaned = raw.select(
        "id",
        text=col("body")
        .str.strip_html()
        .str.remove_urls()
        .str.remove_emails()
        .str.normalize_whitespace()
        .str.strip(),
    )

    result = cleaned.to_pydict()
    for value in result["text"]:
        print(repr(value))

    # No markup, no links, no addresses, and no double spaces survive.
    assert all("<" not in value and ">" not in value for value in result["text"])
    assert all("http" not in value for value in result["text"])
    assert all("@" not in value for value in result["text"])
    assert all("  " not in value for value in result["text"])
    assert all(value == value.strip() for value in result["text"])

    # A slug for a URL path, from the cleaned text.
    slugged = cleaned.select("id", slug=col("text").str.slugify()).to_pydict()
    print(slugged["slug"])
    assert all(
        value == "" or all(character.isalnum() or character == "-" for character in value)
        for value in slugged["slug"]
    )


if __name__ == "__main__":
    main()
