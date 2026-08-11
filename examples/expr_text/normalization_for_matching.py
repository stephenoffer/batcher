"""Normalizing text so two spellings of the same thing compare equal.

Every join on a human-entered string needs this. Case, whitespace, punctuation and accents
are four independent axes, and normalizing three of them still leaves the join broken on the
fourth — so normalize on the way in and keep the original alongside.

    python examples/expr_text/normalization_for_matching.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from batcher import col


def main() -> None:
    names = bt.from_pydict(
        {
            "id": [1, 2, 3, 4],
            "raw": ["  Acme  Corp. ", "ACME CORP", "acme corp", "Beta Ltd"],
        }
    )

    normalized = names.with_columns(
        key=col("raw")
        .str.to_lowercase()
        .str.remove_punctuation()
        .str.normalize_whitespace()
        .str.strip_chars(" \t\n")
    )
    result = normalized.to_pydict()
    for raw, key in zip(result["raw"], result["key"], strict=True):
        print(f"  {raw!r:<20} -> {key!r}")

    # The three spellings of Acme collapse to one key; Beta stays separate.
    assert result["key"][0] == result["key"][1] == result["key"][2] == "acme corp"
    assert result["key"][3] != result["key"][0]

    # Which is what makes a group-by find them.
    grouped = normalized.group_by("key").agg(variants=bt.count()).sort("key").to_pydict()
    print(grouped)
    assert dict(zip(grouped["key"], grouped["variants"], strict=True)) == {
        "acme corp": 3,
        "beta ltd": 1,
    }

    # The original is still there, which matters because the key is lossy: you cannot
    # reconstruct "Acme Corp." from "acme corp".
    assert "raw" in normalized.columns
    assert normalized.select("raw").distinct().count() == 4

    # And a join on the key finds all three where a join on the raw value finds one.
    lookup = bt.from_pydict({"key": ["acme corp"], "tier": ["gold"]})
    matched = normalized.join(lookup, on="key")
    assert matched.count() == 3


if __name__ == "__main__":
    main()
