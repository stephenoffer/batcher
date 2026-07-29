"""Pulling entities and leading fragments out of free text.

The ``extract_*`` family returns a *list column*, so one row can carry many matches and
you can ``explode`` it into one row per match. The ``first_*``/``last_*``/``truncate_*``
family returns a scalar string, which is what you want for a preview or a title.

One edge worth knowing: a match that ends a sentence keeps the trailing period, so
``"mail a@b.com."`` extracts ``"a@b.com."``. Strip it with ``.str.strip_chars(".")`` when
the value has to round-trip as a real address.

    python examples/expressions/strings_extraction.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    notes = bt.from_pydict(
        {
            "body": [
                "Ping ops@acme.com, sre@acme.com, or read https://docs.example/start here.",
                "Order 8812 shipped. Total was 249 dollars. Thanks!",
            ],
        }
    )

    pulled = notes.with_columns(
        emails=col("body").str.extract_emails(),
        urls=col("body").str.extract_urls(),
        numbers=col("body").str.extract_numbers(),
        # Scalar previews.
        first_sentence=col("body").str.first_sentence(),
        first_word=col("body").str.first_word(),
        last_word=col("body").str.last_word(),
        preview=col("body").str.truncate_chars(20),
        headline=col("body").str.truncate_words(3),
    )

    result = pulled.to_pydict()
    print(result)

    assert result["emails"][0] == ["ops@acme.com", "sre@acme.com"]
    assert result["emails"][1] == []
    assert result["urls"][0] == ["https://docs.example/start"]
    assert result["numbers"][1] == ["8812", "249"]
    assert result["first_word"] == ["Ping", "Order"]
    assert result["first_sentence"][1].startswith("Order 8812 shipped")
    assert len(result["preview"][0]) <= 20
    assert result["headline"][0] == "Ping ops@acme.com, sre@acme.com,"

    # A list column explodes to one row per match, which is how you build a lookup table.
    exploded = (
        notes.select(email=col("body").str.extract_emails())
        .explode("email")
        .filter(col("email").is_not_null())
        .to_pydict()
    )
    print(exploded)
    assert exploded["email"] == ["ops@acme.com", "sre@acme.com"]


if __name__ == "__main__":
    main()
