"""Boolean text predicates: the screen in front of an expensive stage.

Running an LLM over a corpus costs money per row, so the cheapest win is not sending the
rows that cannot help. These predicates all return a boolean column and compose with
``&``/``|``, so a screen is one filter rather than a Python loop.

    python examples/expressions/strings_predicates.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    rows = bt.from_pydict(
        {
            "text": [
                "How do I reset my password?",
                "   ",
                '{"status": "ok", "count": 3}',
                "SEE OUR AMAZING OFFER",
                "Call +1-555-010-0100 or email a@b.com",
            ],
        }
    )

    flagged = rows.with_columns(
        blank=col("text").str.is_blank(),
        question=col("text").str.is_question(),
        shouting=col("text").str.is_all_caps(),
        jsonish=col("text").str.looks_like_json(),
        has_email=col("text").str.has_email(),
        has_phone=col("text").str.has_phone(),
        has_digits=col("text").str.has_digits(),
        ascii_only=col("text").str.is_ascii_only(),
        one_line=col("text").str.is_single_line(),
        short=col("text").str.is_short(max_chars=10),
        # Budget guards for an LLM stage.
        approx_tokens=col("text").str.estimate_tokens(),
        fits=col("text").str.fits_token_budget(budget=8),
    )

    result = flagged.to_pydict()
    print(result)

    assert result["blank"] == [False, True, False, False, False]
    assert result["question"][0] is True
    assert result["shouting"][3] is True
    assert result["jsonish"][2] is True
    assert result["has_email"][4] is True
    assert result["has_phone"][4] is True
    assert result["has_digits"] == [False, False, True, False, True]
    assert all(result["ascii_only"])
    assert all(result["one_line"])
    assert result["short"] == [False, True, False, False, False]
    # ~4 chars per token by default, so the estimate tracks length.
    assert result["approx_tokens"][0] > result["approx_tokens"][1]

    # The screen: real questions only, nothing blank or shouted.
    keep = rows.filter(
        col("text").str.is_question() & ~col("text").str.is_blank() & ~col("text").str.is_all_caps()
    ).to_pydict()
    assert keep["text"] == ["How do I reset my password?"]


if __name__ == "__main__":
    main()
