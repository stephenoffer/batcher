"""String padding and trimming: fixed-width keys and cleaning stray whitespace.

Padding matters when a join key is stored at different widths in two systems: an account
id written ``42`` in one export and ``000042`` in another will not join until one side is
padded. Trimming matters because a trailing space is invisible and breaks equality.

    python examples/expressions/strings_padding.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    accounts = bt.from_pydict(
        {
            "raw_id": ["42", "7", "1234"],
            "messy": ["  alpha  ", "  beta  ", "--gamma--"],
        }
    )

    cleaned = accounts.with_columns(
        # Zero-pad to a fixed width so the two systems' keys match.
        padded=col("raw_id").str.zfill(6),
        # The general forms: pad on either side with any fill character.
        left_pad=col("raw_id").str.lpad(5, "0"),
        right_pad=col("raw_id").str.rpad(5, "."),
        pad_start=col("raw_id").str.pad_start(4, "*"),
        pad_end=col("raw_id").str.pad_end(4, "*"),
        # Trim from both ends, or just one. Pass the character set explicitly when the
        # input may hold tabs or newlines: the no-argument form follows SQL `TRIM` and
        # removes spaces only, so `strip_chars(" \t\n")` is the portable spelling.
        trimmed=col("messy").str.strip_chars(" "),
        # Trim specific characters instead of whitespace.
        undashed=col("messy").str.strip_chars("-"),
        lead_only=col("messy").str.strip_chars_start(" "),
        trail_only=col("messy").str.strip_chars_end(" "),
    )

    result = cleaned.to_pydict()
    print(result)

    assert result["padded"] == ["000042", "000007", "001234"]
    assert result["left_pad"] == ["00042", "00007", "01234"]
    assert result["right_pad"] == ["42...", "7....", "1234."]
    assert result["pad_start"] == ["**42", "***7", "1234"]
    assert result["pad_end"] == ["42**", "7***", "1234"]
    assert result["trimmed"] == ["alpha", "beta", "--gamma--"]
    assert result["undashed"][2] == "gamma"
    assert result["lead_only"][0] == "alpha  "
    assert result["trail_only"][0] == "  alpha"


if __name__ == "__main__":
    main()
