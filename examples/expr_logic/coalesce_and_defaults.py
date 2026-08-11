"""Supplying defaults: coalesce, nullif, and the fallback chain.

`coalesce` takes the first non-null of several columns, which is how you express "use the
override if there is one, otherwise the default". `nullif` is the inverse and is how you
turn a sentinel value — an empty string, a -1 — into a real null.

    python examples/expr_logic/coalesce_and_defaults.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from batcher import col


def main() -> None:
    settings = bt.from_pydict(
        {
            "id": [1, 2, 3, 4, 5],
            # Row 5 has both, which is the row that makes precedence observable.
            "user_value": ["dark", "", "", "light", "user-wins"],
            "team_value": ["", "compact", "", "", "team-loses"],
        }
    )

    # Sentinels first: an empty string is not a null until you say so.
    normalized = settings.select(
        "id",
        user=bt.nullif(col("user_value"), bt.lit("")),
        team=bt.nullif(col("team_value"), bt.lit("")),
    )
    values = normalized.to_pydict()
    print(values)
    assert values["user"] == ["dark", None, None, "light", "user-wins"]
    assert values["team"] == [None, "compact", None, None, "team-loses"]

    # Then the fallback chain: user, then team, then a built-in default.
    resolved = normalized.with_columns(
        setting=bt.coalesce(col("user"), col("team"), bt.lit("default"))
    ).to_pydict()
    print(resolved["setting"])
    assert resolved["setting"] == ["dark", "compact", "default", "light", "user-wins"]

    # Nothing is null after a chain that ends in a literal.
    assert (
        normalized.with_columns(setting=bt.coalesce(col("user"), col("team"), bt.lit("default")))
        .filter(col("setting").is_null())
        .count()
        == 0
    )

    # The order of the chain is the precedence, and reversing it changes the answer.
    reversed_chain = normalized.with_columns(
        setting=bt.coalesce(col("team"), col("user"), bt.lit("default"))
    ).to_pydict()
    assert reversed_chain["setting"] != resolved["setting"]


if __name__ == "__main__":
    main()
