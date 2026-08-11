"""Envelopes: the cheap bound that makes an exact predicate affordable.

An envelope is the axis-aligned box around a geometry. Testing envelopes is cheap and never
misses a true match, so it is the filter you put in front of the exact predicate — which is
expensive and would otherwise run on every pair.

    python examples/geospatial/bounding_boxes.py
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
            "name": ["square", "triangle", "line"],
            "wkt": [
                "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
                "POLYGON ((10 10, 14 10, 12 14, 10 10))",
                "LINESTRING (0 0, 10 10)",
            ],
        }
    ).with_columns(shape=bt.st_geom_from_text(col("wkt")))

    boxed = shapes.select(
        "name",
        min_x=bt.st_xmin(col("shape")),
        max_x=bt.st_xmax(col("shape")),
        min_y=bt.st_ymin(col("shape")),
        max_y=bt.st_ymax(col("shape")),
        area=bt.st_area(col("shape")),
    )
    result = boxed.to_pydict()
    print(result)

    # The box always contains its geometry, so min never exceeds max.
    assert all(low <= high for low, high in zip(result["min_x"], result["max_x"], strict=True))
    assert all(low <= high for low, high in zip(result["min_y"], result["max_y"], strict=True))

    # The 4x4 square: exact bounds and exact area.
    assert result["min_x"][0] == 0.0 and result["max_x"][0] == 4.0
    assert abs(result["area"][0] - 16.0) < 1e-9

    # A line has no area but still has a box.
    assert abs(result["area"][2]) < 1e-9
    assert result["max_x"][2] == 10.0

    # The box is never smaller than the geometry it bounds: box area >= shape area.
    box_areas = [
        (high_x - low_x) * (high_y - low_y)
        for low_x, high_x, low_y, high_y in zip(
            result["min_x"], result["max_x"], result["min_y"], result["max_y"], strict=True
        )
    ]
    assert all(box >= area - 1e-9 for box, area in zip(box_areas, result["area"], strict=True))


if __name__ == "__main__":
    main()
