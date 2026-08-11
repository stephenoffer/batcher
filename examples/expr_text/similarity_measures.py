"""Comparing a column against a target string: edit distance and phonetic keys.

The target is a plan-time constant, not another column: these functions lower it into the
plan so it can be compiled once rather than re-read per row. Comparing two *columns* is a
different operation and needs a different tool.

Edit distance is absolute, so normalize it by length before thresholding across rows.
Jaro-Winkler is already scaled and favours a shared prefix, which makes it the better
default for names. Soundex ignores spelling entirely.

    python examples/expr_text/similarity_measures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from batcher import col


def main() -> None:
    names = bt.from_pydict({"name": ["Ashcroft", "Ashcraft", "Ashworth", "Beaumont", "Ashcroft"]})
    target = "Ashcroft"

    scored = names.select(
        "name",
        edits=col("name").str.levenshtein(target),
        damerau=col("name").str.damerau_levenshtein(target),
        jaro=col("name").str.jaro_similarity(target),
        winkler=col("name").str.jaro_winkler_similarity(target),
        key=col("name").str.soundex(),
    ).with_columns(
        # Normalized distance, so a threshold means the same thing for every length.
        relative=col("edits") / col("name").str.len_chars()
    )

    result = scored.to_pydict()
    for index in range(len(result["name"])):
        print(
            f"{result['name'][index]:>10} edits={result['edits'][index]} "
            f"winkler={result['winkler'][index]:.3f} key={result['key'][index]}"
        )

    # An exact match scores zero distance and perfect similarity.
    assert result["edits"][0] == 0
    assert abs(result["jaro"][0] - 1.0) < 1e-9
    assert abs(result["winkler"][0] - 1.0) < 1e-9

    # Similarities are bounded, and Winkler never scores a shared prefix below Jaro.
    assert all(0.0 <= value <= 1.0 for value in result["jaro"])
    assert all(
        winkler >= jaro - 1e-9
        for jaro, winkler in zip(result["jaro"], result["winkler"], strict=True)
    )

    # Damerau counts a transposition as one edit, so it never exceeds plain Levenshtein.
    assert all(
        damerau <= edits for edits, damerau in zip(result["edits"], result["damerau"], strict=True)
    )

    # Soundex collapses spelling variants that sound alike.
    assert result["key"][0] == result["key"][1]
    assert result["key"][0] != result["key"][3]

    # A fuzzy-match filter, on the normalized distance.
    close = scored.filter(col("relative") < 0.2)
    print("near matches:", close.to_pydict()["name"])
    assert "Beaumont" not in close.to_pydict()["name"]


if __name__ == "__main__":
    main()
