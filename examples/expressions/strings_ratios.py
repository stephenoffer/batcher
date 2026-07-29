"""Character-class ratios: cheap quality signals for a text corpus.

These are the filters that keep junk out of a training set. A row that is 60% digits is
probably a table dump; one that is 90% uppercase is probably a shouting header; one with
a high non-ASCII ratio may be the wrong language or mojibake. Each ratio is a float in
[0, 1] computed in one pass.

    python examples/expressions/strings_ratios.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    docs = bt.from_pydict(
        {
            "text": [
                "The quick brown fox jumps.",
                "8829 1042 7731 0090 4412",
                "BUY NOW LIMITED OFFER",
            ],
        }
    )

    scored = docs.with_columns(
        alpha=col("text").str.alpha_ratio(),
        digit=col("text").str.digit_ratio(),
        upper=col("text").str.uppercase_ratio(),
        lower=col("text").str.lowercase_ratio(),
        punct=col("text").str.punctuation_ratio(),
        space=col("text").str.whitespace_ratio(),
        alnum=col("text").str.alnum_ratio(),
        non_ascii=col("text").str.non_ascii_ratio(),
    )

    result = scored.to_pydict()
    print(result)

    # Every ratio is a proportion.
    for name in ("alpha", "digit", "upper", "lower", "punct", "space", "alnum", "non_ascii"):
        assert all(0.0 <= v <= 1.0 for v in result[name]), name

    # Row 1 is prose: mostly letters, no digits.
    assert result["digit"][0] == 0.0
    assert result["alpha"][0] > 0.7
    # Row 2 is a number dump: no letters at all.
    assert result["alpha"][1] == 0.0
    assert result["digit"][1] > 0.7
    # Row 3 shouts: all its letters are uppercase, none lowercase.
    assert result["lower"][2] == 0.0
    assert result["upper"][2] > 0.7
    # Plain ASCII throughout.
    assert result["non_ascii"] == [0.0, 0.0, 0.0]

    # The filter this exists for: drop the number dump and the shouting header.
    prose = docs.filter(
        (col("text").str.alpha_ratio() > 0.5) & (col("text").str.uppercase_ratio() < 0.5)
    ).to_pydict()
    assert prose["text"] == ["The quick brown fox jumps."]


if __name__ == "__main__":
    main()
