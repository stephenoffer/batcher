"""Handling text with structure: lines, paragraphs, code fences and markdown.

Documents scraped from the web carry markup that is noise to a model and signal to a router.
Detecting it costs nothing and decides which pipeline a document belongs in.

    python examples/expr_text/multiline_and_markdown.py
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
            "id": [1, 2, 3],
            "body": [
                "# Title\n\nA paragraph.\n\nAnother one with a [link](http://x.io).",
                "Plain single line of prose.",
                "Intro:\n\n```python\nprint(1)\n```\n\nOutro.",
            ],
        }
    )

    shaped = documents.select(
        "id",
        lines=col("body").str.line_count(),
        paragraphs=col("body").str.paragraph_count(),
        fences=col("body").str.code_fence_count(),
        single_line=col("body").str.is_single_line(),
        newlines=col("body").str.newline_count(),
    )
    result = shaped.to_pydict()
    print(result)

    assert result["single_line"] == [False, True, False]
    assert result["lines"][1] == 1
    assert result["fences"][2] >= 1
    assert result["paragraphs"][0] >= 2

    # Newlines and line count agree: n newlines means n+1 lines.
    assert all(
        lines == newlines + 1
        for lines, newlines in zip(result["lines"], result["newlines"], strict=True)
    )

    # Stripping the markup, for the documents that carry it.
    cleaned = documents.select(
        "id",
        text=col("body").str.remove_code_blocks().str.remove_markdown_links(),
    ).to_pydict()
    for value in cleaned["text"]:
        print(repr(value[:60]))
    assert "print(1)" not in cleaned["text"][2]
    assert "http://x.io" not in cleaned["text"][0]

    # Routing: code-bearing documents go one way, prose the other.
    prose = documents.filter(col("body").str.code_fence_count() == 0)
    assert set(prose.to_pydict()["id"]) == {1, 2}


if __name__ == "__main__":
    main()
