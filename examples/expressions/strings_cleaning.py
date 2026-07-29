"""Cleaning scraped text: strip markup, URLs, emails, and stray punctuation.

This is the pre-processing pass in front of an embedding or LLM stage. Each call is one
columnar operator, so a chain of ten of them still reads the column once per operator in
Rust rather than materializing Python strings.

    python examples/expressions/strings_cleaning.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    scraped = bt.from_pydict(
        {
            "body": [
                "<p>Contact us at sales@acme.com</p>",
                "Great deal!!! See https://example.com/offer now",
                "* bullet item with 123 digits",
            ],
        }
    )

    cleaned = scraped.with_columns(
        no_html=col("body").str.remove_html_tags(),
        no_urls=col("body").str.remove_urls(),
        no_emails=col("body").str.remove_emails(),
        no_digits=col("body").str.remove_digits(),
        no_bullets=col("body").str.remove_bullets(),
        # Collapse runs of "!!!" to a single mark.
        calm=col("body").str.remove_repeated_punctuation(),
        # Collapse runs of whitespace to one space.
        tidy=col("body").str.remove_html_tags().str.normalize_whitespace(),
        # Redact rather than delete, so the row still shows something happened.
        masked_email=col("body").str.mask_emails(),
        masked_url=col("body").str.mask_urls("<link>"),
        # A URL-safe key derived from the text.
        slug=col("body").str.remove_html_tags().str.slugify(),
    )

    result = cleaned.to_pydict()
    print(result)

    assert result["no_html"][0] == "Contact us at sales@acme.com"
    assert "https://example.com/offer" not in result["no_urls"][1]
    assert "sales@acme.com" not in result["no_emails"][0]
    assert "123" not in result["no_digits"][2]
    assert result["masked_email"][0].endswith("[EMAIL]</p>")
    assert "<link>" in result["masked_url"][1]
    # A slug is lowercase, hyphenated, and free of punctuation.
    assert result["slug"][2] == "bullet-item-with-123-digits"
    assert " " not in result["slug"][0]


if __name__ == "__main__":
    main()
