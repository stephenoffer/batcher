"""Querying JSON held in a string column.

The typed extractors are the point: `extract_int` gives you an integer column, not a string
you have to cast. A path that is absent yields null rather than raising, so a malformed
document degrades to a missing value instead of killing the batch.

    python examples/expr_collections/json_columns.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from batcher import col


def main() -> None:
    events = bt.from_pydict(
        {
            "id": [1, 2, 3],
            "payload": [
                '{"user": "ada", "score": 91, "ratio": 0.75, "active": true}',
                '{"user": "bob", "score": 45, "ratio": 0.10, "active": false}',
                '{"user": "cy"}',
            ],
        }
    )

    parsed = events.select(
        "id",
        user=col("payload").json.extract_string("$.user"),
        score=col("payload").json.extract_int("$.score"),
        ratio=col("payload").json.extract_float("$.ratio"),
        active=col("payload").json.extract_bool("$.active"),
        missing=col("payload").json.extract_string("$.nope"),
    )

    result = parsed.to_pydict()
    print(result)

    assert result["user"] == ["ada", "bob", "cy"]
    assert result["score"][:2] == [91, 45]
    # The third document has no score: null, not zero and not an error.
    assert result["score"][2] is None
    assert result["missing"] == [None, None, None]

    # Typed extraction means the column is usable straight away.
    total = parsed.agg(t=col("score").sum()).to_pydict()["t"][0]
    assert total == 136

    # Structural queries over the document.
    shape = events.select(
        "id",
        keys=col("payload").json.keys(),
        has_score=col("payload").json.exists("$.score"),
    ).to_pydict()
    print(shape)
    assert shape["has_score"] == [True, True, False]
    assert "user" in shape["keys"][0]


if __name__ == "__main__":
    main()
