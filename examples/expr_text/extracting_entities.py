"""Pulling structured values out of free text.

URLs, emails, numbers and hashtags all have extractors that return list columns, so one
document can yield several. Exploding afterwards turns them into rows, which is the shape a
lookup table wants.

    python examples/expr_text/extracting_entities.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from batcher import col


def main() -> None:
    posts = bt.from_pydict(
        {
            "id": [1, 2, 3],
            "text": [
                "See https://a.example.com and https://b.example.com #release #v2",
                "Mail ada@example.com or grace@example.com about the 42 open issues",
                "Nothing structured in this one at all",
            ],
        }
    )

    extracted = posts.select(
        "id",
        urls=col("text").str.extract_urls(),
        emails=col("text").str.extract_emails(),
        numbers=col("text").str.extract_numbers(),
        hashtags=col("text").str.extract_hashtags(),
    )
    result = extracted.to_pydict()
    print(result)

    assert len(result["urls"][0]) == 2
    assert len(result["emails"][1]) == 2
    assert result["numbers"][1] == ["42"]
    assert len(result["hashtags"][0]) == 2

    # The document with nothing in it yields empty lists, not nulls.
    assert result["urls"][2] == []
    assert result["emails"][2] == []

    # Exploding turns the lists into a lookup table, and drops the empty rows.
    links = extracted.select("id", url=col("urls")).explode("url")
    print(links.to_pydict())
    assert links.count() == 2
    assert set(links.to_pydict()["id"]) == {1}

    # Counting without extracting, when the count is all you need.
    counted = posts.select(
        "id",
        url_count=col("text").str.url_count(),
        email_count=col("text").str.email_count(),
    ).to_pydict()
    assert counted["url_count"] == [2, 0, 0]
    assert counted["email_count"] == [0, 2, 0]


if __name__ == "__main__":
    main()
