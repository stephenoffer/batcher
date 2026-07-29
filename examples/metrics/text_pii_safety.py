"""PII leak rates over a text column.

Run this over model output *and* over training data. On output it tells you whether the
model is emitting personal data; on input it tells you whether you are about to train on
it. Both are one aggregate pass, so it is cheap enough to run on every batch.

    python examples/metrics/text_pii_safety.py
"""

from __future__ import annotations

import batcher as bt


def main() -> None:
    generations = bt.from_pydict(
        {
            "text": [
                "Sure, here is the summary you asked for.",
                "You can reach me at ada@example.com",
                "Call +1-555-010-0100 during business hours",
                "Card 4111 1111 1111 1111 expires soon",
                "SSN 123-45-6789 on file",
            ],
        }
    )

    leaks = generations.select(
        any_pii=bt.pii_rate("text"),
        emails=bt.email_rate("text"),
        phones=bt.phone_rate("text"),
        cards=bt.credit_card_like_rate("text"),
        ssns=bt.ssn_like_rate("text"),
        # Your own denylist, when the built-ins do not cover a domain term.
        internal=bt.contains_any_rate("text", ["PROJECT-ORION", "acme-secret"]),
    ).to_pydict()

    print(leaks)

    for name, value in leaks.items():
        assert 0.0 <= value[0] <= 1.0, name

    # One row each of email, phone, card, SSN out of five.
    assert leaks["emails"][0] == 0.2
    assert leaks["phones"][0] == 0.2
    assert leaks["cards"][0] == 0.2
    assert leaks["ssns"][0] == 0.2
    # `pii_rate` is the combined contact-detail rate (email or phone), so the card and
    # SSN rows are counted by their own metrics rather than folded in here.
    assert leaks["any_pii"][0] == 0.4
    assert leaks["internal"][0] == 0.0

    # The gate this exists for: block a release that leaks anything.
    assert leaks["any_pii"][0] > 0.0  # this sample would be blocked


if __name__ == "__main__":
    main()
