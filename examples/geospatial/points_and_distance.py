"""Building point geometry from coordinate columns, and measuring between them.

Longitude first, then latitude — that is the order WKT, GeoJSON and PostGIS all use, and
reversing it puts your data in the wrong hemisphere without any error. The distance
functions are the reason to build geometry at all rather than keeping two floats.

    python examples/geospatial/points_and_distance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from batcher import col


def main() -> None:
    cities = bt.from_pydict(
        {
            "name": ["San Francisco", "London", "Sydney", "Tokyo"],
            "lon": [-122.4194, -0.1276, 151.2093, 139.6917],
            "lat": [37.7749, 51.5074, -33.8688, 35.6895],
        }
    )

    located = cities.with_columns(point=bt.st_point(col("lon"), col("lat")))

    described = located.select(
        "name",
        wkt=bt.st_as_text(col("point")),
        x=bt.st_x(col("point")),
        y=bt.st_y(col("point")),
    )
    result = described.to_pydict()
    print(result["wkt"][0])

    # x is longitude, y is latitude — round-tripping proves the order.
    assert all(
        abs(x - lon) < 1e-9 for x, lon in zip(result["x"], cities.to_pydict()["lon"], strict=True)
    )
    assert result["wkt"][0].startswith("POINT")

    # Great-circle distance between every city and San Francisco.
    origin = located.filter(col("name") == "San Francisco").select(col("point").alias("origin"))
    distances = (
        located.cross_join(origin)
        .select("name", km=bt.st_distance_sphere(col("point"), col("origin")) / 1000.0)
        .sort("km")
        .to_pydict()
    )
    for name, km in zip(distances["name"], distances["km"], strict=True):
        print(f"  {name:<15} {km:>9,.0f} km")

    assert distances["name"][0] == "San Francisco"
    assert distances["km"][0] < 1.0
    # Nothing on Earth is more than half the circumference away.
    assert all(value < 20_100 for value in distances["km"])
    assert distances["km"] == sorted(distances["km"])


if __name__ == "__main__":
    main()
