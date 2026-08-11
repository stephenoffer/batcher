"""Lines and polygons: length, area, centroid, and the relationships between them.

Measuring in degrees is almost never what you want — a degree of longitude is a different
distance at every latitude. The planar measures here are exact for projected coordinates and
a rough guide for geographic ones, which is worth knowing before you report a number.

    python examples/geospatial/linestrings_and_areas.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from batcher import col


def main() -> None:
    shapes = bt.from_pydict(
        {
            "name": ["unit square", "big square", "diagonal", "triangle"],
            "wkt": [
                "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
                "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
                "LINESTRING (0 0, 3 4)",
                "POLYGON ((0 0, 4 0, 0 3, 0 0))",
            ],
        }
    ).with_columns(shape=bt.st_geom_from_text(col("wkt")))

    measured = shapes.select(
        "name",
        area=bt.st_area(col("shape")),
        length=bt.st_length(col("shape")),
        centroid=bt.st_as_text(bt.st_centroid(col("shape"))),
        kind=bt.st_geometry_type(col("shape")),
    )
    result = measured.to_pydict()
    for row in zip(
        result["name"], result["area"], result["length"], result["centroid"], strict=True
    ):
        print(f"  {row[0]:<12} area={row[1]:>6.2f} length={row[2]:>6.2f} centroid={row[3]}")

    # Exact planar geometry.
    assert abs(result["area"][0] - 1.0) < 1e-9
    assert abs(result["area"][1] - 16.0) < 1e-9
    assert abs(result["area"][3] - 6.0) < 1e-9  # 4x3 right triangle

    # A line has length and no area; the 3-4-5 diagonal is 5 long.
    assert abs(result["area"][2]) < 1e-9
    assert abs(result["length"][2] - 5.0) < 1e-9

    # The unit square's centroid is its middle.
    assert "0.5" in result["centroid"][0]

    # A polygon four times as wide has sixteen times the area, which is the check that
    # catches an area computed as a perimeter.
    assert abs(result["area"][1] / result["area"][0] - 16.0) < 1e-9


if __name__ == "__main__":
    main()
