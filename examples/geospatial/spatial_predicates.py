"""Point-in-polygon and the other spatial predicates.

A spatial join is a predicate join, so it cannot hash. The usual shape is to bound the
candidates cheaply — with an envelope or a grid key — and evaluate the exact predicate only
on what survives. This shows the exact predicate; the grid version is the next example.

    python examples/geospatial/spatial_predicates.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from batcher import col


def main() -> None:
    points = bt.from_pydict(
        {
            "name": ["inside", "edge", "outside", "far"],
            "lon": [5.0, 10.0, 15.0, 100.0],
            "lat": [5.0, 5.0, 5.0, 5.0],
        }
    ).with_columns(point=bt.st_point(col("lon"), col("lat")))

    # A 10x10 box with its lower-left corner at the origin.
    box = bt.from_pydict(
        {"zone": ["square"], "wkt": ["POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"]}
    ).with_columns(shape=bt.st_geom_from_text(col("wkt")))

    tested = points.cross_join(box).select(
        "name",
        "zone",
        within=bt.st_within(col("point"), col("shape")),
        contains=bt.st_contains(col("shape"), col("point")),
        intersects=bt.st_intersects(col("point"), col("shape")),
    )

    result = tested.to_pydict()
    print(result)

    # `within` and `contains` are the same relation from opposite sides.
    assert result["within"] == result["contains"]

    assert result["within"][0] is True
    assert result["within"][2] is False
    assert result["within"][3] is False

    # The spatial join: keep only the points inside the zone.
    inside = tested.filter(col("within")).select("name").to_pydict()
    print("inside the zone:", inside["name"])
    assert "outside" not in inside["name"]
    assert "inside" in inside["name"]


if __name__ == "__main__":
    main()
