"""String case: normalizing capitalization before you compare or group.

Case folding is the cheapest way to stop a group_by splitting "ACME", "Acme", and "acme"
into three groups. Do it once in the projection, then group on the normalized column.

    python examples/expressions/strings_case.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    vendors = bt.from_pydict(
        {
            "name": ["ACME corp", "acme CORP", "globex inc", "Globex Inc"],
            "spend": [10, 20, 5, 15],
        }
    )

    normalized = vendors.with_columns(
        lower=col("name").str.to_lowercase(),
        upper=col("name").str.to_uppercase(),
        title=col("name").str.to_titlecase(),
        # `initcap` is the SQL spelling of title case.
        initcap=col("name").str.initcap(),
        # `capitalize` upper-cases only the first character.
        first_only=col("name").str.capitalize(),
    )

    result = normalized.to_pydict()
    print(result)

    assert result["lower"] == ["acme corp", "acme corp", "globex inc", "globex inc"]
    assert result["upper"] == ["ACME CORP", "ACME CORP", "GLOBEX INC", "GLOBEX INC"]
    assert result["title"] == ["Acme Corp", "Acme Corp", "Globex Inc", "Globex Inc"]
    assert result["initcap"] == result["title"]
    assert result["first_only"][1] == "Acme corp"

    # The point of all that: two vendors, not four.
    rolled = (
        vendors.with_columns(key=col("name").str.to_lowercase())
        .group_by("key")
        .agg(total=col("spend").sum())
        .sort("key")
        .to_pydict()
    )
    print(rolled)
    assert rolled["key"] == ["acme corp", "globex inc"]
    assert rolled["total"] == [30, 20]


if __name__ == "__main__":
    main()
