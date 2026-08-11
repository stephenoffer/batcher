"""Character-class ratios as a document-quality signal.

Ratios travel better than counts: they compare a two-word title against a paragraph
without normalizing first. These are the cheap filters that go in front of an expensive
model, which is the only place a heuristic like this belongs.

    python examples/expr_text/quality_ratios.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    comments = tpch("orders").select("o_orderkey", "o_comment").head(2_000)

    scored = comments.select(
        "o_orderkey",
        alpha=col("o_comment").str.alpha_ratio(),
        digits=col("o_comment").str.digit_ratio(),
        upper=col("o_comment").str.uppercase_ratio(),
        punctuation=col("o_comment").str.punctuation_ratio(),
        whitespace=col("o_comment").str.whitespace_ratio(),
        non_ascii=col("o_comment").str.non_ascii_ratio(),
    )

    result = scored.head(3).to_pydict()
    print({name: [round(value, 3) for value in column] for name, column in result.items()})

    # Every ratio is a proportion.
    for name, column in result.items():
        if name == "o_orderkey":
            continue
        assert all(0.0 <= value <= 1.0 for value in column), name

    # TPC-H comments are lowercase English prose: mostly letters and spaces, no digits.
    summary = scored.agg(
        mean_alpha=col("alpha").mean(),
        mean_digits=col("digits").mean(),
        mean_non_ascii=col("non_ascii").mean(),
    ).to_pydict()
    print({name: round(value[0], 4) for name, value in summary.items()})
    assert summary["mean_alpha"][0] > 0.7
    assert summary["mean_digits"][0] < 0.01
    assert summary["mean_non_ascii"][0] == 0.0


if __name__ == "__main__":
    main()
