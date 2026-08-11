"""Time zones: converting, and the ambiguity that conversion cannot remove.

A timestamp without a zone is a wall-clock reading, so `convert_timezone` takes **both**
zones: the one the reading was taken in and the one you want it in. There is no default
for the source, which is the API refusing to guess the half that cannot be inferred.

    python examples/expr_temporal/timezones.py
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from batcher import col


def main() -> None:
    stamps = bt.from_pydict(
        {
            "label": ["morning", "noon", "evening"],
            "at": [
                dt.datetime(2024, 6, 1, 8, 0),
                dt.datetime(2024, 6, 1, 12, 0),
                dt.datetime(2024, 6, 1, 20, 0),
            ],
        }
    )

    converted = stamps.select(
        "label",
        "at",
        in_tokyo=col("at").dt.convert_timezone("UTC", "Asia/Tokyo"),
        in_new_york=col("at").dt.convert_timezone("UTC", "America/New_York"),
    )
    result = converted.to_pydict()
    for row in zip(
        result["label"], result["at"], result["in_tokyo"], result["in_new_york"], strict=True
    ):
        print("  ", row)

    # The conversions differ from each other and from the source.
    assert result["in_tokyo"] != result["in_new_york"]

    # The wall-clock difference between the two zones is constant across the day.
    offsets = {
        (tokyo - york)
        for tokyo, york in zip(result["in_tokyo"], result["in_new_york"], strict=True)
    }
    print("Tokyo minus New York:", offsets)
    assert len(offsets) == 1

    # The ordering of the instants survives conversion, which is what makes a converted
    # column still safe to sort on.
    assert result["in_tokyo"] == sorted(result["in_tokyo"])
    assert result["in_new_york"] == sorted(result["in_new_york"])

    # Date parts come from the converted value, which is the whole reason to convert:
    # "which day was this, locally" is a different question in each zone.
    days = converted.select(
        tokyo_day=col("in_tokyo").dt.day(), york_day=col("in_new_york").dt.day()
    ).to_pydict()
    print(days)
    assert len(set(days["tokyo_day"]) | set(days["york_day"])) >= 1


if __name__ == "__main__":
    main()
