"""SQL over Datasets.

``bt.sql`` runs a SQL query against one or more Datasets bound by keyword. It returns
a lazy Dataset, so SQL and the DataFrame API compose freely. The supported subset is
SELECT, WHERE, GROUP BY / HAVING, ORDER BY, LIMIT, INNER/LEFT JOIN, CASE, and CAST.

    python examples/sql.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    events = bt.from_pydict(
        {
            "user": ["a", "b", "a", "c", "b"],
            "kind": ["click", "view", "click", "view", "click"],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    # SQL query bound to the `events` table; HAVING filters the aggregated groups.
    clicks_by_user = bt.sql(
        """
        SELECT user, SUM(value) AS total
        FROM events
        WHERE kind = 'click'
        GROUP BY user
        HAVING SUM(value) > 1
        ORDER BY total DESC
        """,
        events=events,
    )

    # The SQL result is a Dataset — keep going with the DataFrame API.
    out = clicks_by_user.with_columns(total_rounded=col("total").round(1))

    result = out.to_pydict()
    print(result)

    # user a: 1+3=4 (click), user b: 5 (click); both > 1.
    assert result["user"] == ["b", "a"]
    assert result["total"] == [5.0, 4.0]


if __name__ == "__main__":
    main()
