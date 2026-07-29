"""Fuzzy string matching against a reference value.

Edit distances count operations (lower is closer); the Jaro family returns a similarity in
[0, 1] (higher is closer). Pick by the error you expect: typos favour Levenshtein,
transposed characters favour Damerau.

One constraint to plan around: the comparison target is a **plan-time literal**, not
another column. That makes these a fast screen against a known value (a canonical name, a
search term), not a column-to-column record-linkage join. For linkage, blocking on a
phonetic or normalized key -- as at the bottom of this file -- is the shape that works.

    python examples/expressions/strings_similarity.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    candidates = bt.from_pydict(
        {"name": ["martha", "marhta", "matha", "globex inc"]},
    )

    scored = candidates.with_columns(
        levenshtein=col("name").str.levenshtein("martha"),
        damerau=col("name").str.damerau_levenshtein("martha"),
        jaro=col("name").str.jaro_similarity("martha"),
        jaro_winkler=col("name").str.jaro_winkler_similarity("martha"),
    ).to_pydict()

    print(scored)

    # An exact match: zero distance, perfect similarity.
    assert scored["levenshtein"][0] == 0
    assert scored["jaro"][0] == 1.0
    assert scored["jaro_winkler"][0] == 1.0

    # "marhta" is a single transposition, which Damerau counts as one operation and
    # plain Levenshtein counts as two.
    assert scored["damerau"][1] < scored["levenshtein"][1]

    # Jaro-Winkler boosts a shared prefix, so it never scores below plain Jaro.
    assert all(w >= j for w, j in zip(scored["jaro_winkler"], scored["jaro"], strict=True))

    # An unrelated string scores low.
    assert scored["jaro"][3] < 0.6

    # `hamming` is the fixed-width cousin: it compares position by position and refuses
    # unequal lengths, so reach for it only on codes of a known width.
    codes = bt.from_pydict({"c": ["ABC123", "ABC124", "XYZ999"]})
    ham = codes.select(d=col("c").str.hamming("ABC123")).to_pydict()
    print(ham)
    assert ham["d"] == [0, 1, 6]

    # The screen this exists for: keep the near-matches to a known value.
    close = candidates.filter(col("name").str.jaro_winkler_similarity("martha") > 0.9).to_pydict()
    print(close)
    # "matha" (a dropped letter) also clears 0.9 -- the threshold is a recall/precision
    # dial, not a correctness boundary.
    assert close["name"] == ["martha", "marhta", "matha"]

    # For linkage between two tables, block on a derived key and join on that. Soundex
    # gives names that sound alike the same code.
    left = bt.from_pydict({"who": ["Robert", "Rubin"]})
    right = bt.from_pydict({"whom": ["Rupert", "Smith"]})
    keyed_left = left.with_columns(key=col("who").str.soundex())
    keyed_right = right.with_columns(key=col("whom").str.soundex())
    linked = keyed_left.join(keyed_right, on="key").to_pydict()
    print(linked)
    # Robert and Rupert share a Soundex code, so they link.
    assert linked["who"] == ["Robert"]
    assert linked["whom"] == ["Rupert"]


if __name__ == "__main__":
    main()
