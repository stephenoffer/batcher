"""Regular expressions over a column: extract, replace, and count.

``extract`` pulls one capture group, ``extract_all`` returns a list column of every match,
and ``replace_all`` rewrites every occurrence. The pattern is compiled once per operator
rather than per row, which is the whole reason these live in the engine.

    python examples/expressions/strings_regex.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    logs = bt.from_pydict(
        {
            "line": [
                "GET /users/42 200 13ms",
                "POST /orders/8812 500 220ms",
                "GET /health 200 1ms",
            ],
        }
    )

    parsed = logs.with_columns(
        # Capture group 1 by default; pass `group=` for another.
        verb=col("line").str.extract(r"^([A-Z]+) "),
        status=col("line").str.extract(r" (\d{3}) ", group=1),
        # Every match, as a list column.
        numbers=col("line").str.extract_all(r"\d+"),
        # Rewrite every occurrence.
        anonymized=col("line").str.replace_all(r"/\d+", "/<id>"),
        # Count matches without materializing them.
        digit_runs=col("line").str.count_matches(r"\d+"),
    )

    result = parsed.to_pydict()
    print(result)

    assert result["verb"] == ["GET", "POST", "GET"]
    assert result["status"] == ["200", "500", "200"]
    assert result["numbers"][0] == ["42", "200", "13"]
    assert result["anonymized"][0] == "GET /users/<id> 200 13ms"
    assert result["anonymized"][2] == "GET /health 200 1ms"
    assert result["digit_runs"] == [3, 3, 2]


if __name__ == "__main__":
    main()
