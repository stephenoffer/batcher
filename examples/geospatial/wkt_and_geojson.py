"""Geometry codecs: WKT, WKB and GeoJSON.

WKT is the readable one, WKB the compact one, GeoJSON the one a browser wants. All three
describe the same geometry, so a round trip through any of them must return what went in —
which is worth asserting, because a codec bug looks like a data bug.

    python examples/geospatial/wkt_and_geojson.py
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
            "name": ["point", "line", "polygon"],
            "wkt": [
                "POINT (1 2)",
                "LINESTRING (0 0, 3 4)",
                "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
            ],
        }
    ).with_columns(shape=bt.st_geom_from_text(col("wkt")))

    encoded = shapes.select(
        "name",
        "wkt",
        text=bt.st_as_text(col("shape")),
        geojson=bt.st_as_geojson(col("shape")),
        kind=bt.st_geometry_type(col("shape")),
    )
    result = encoded.to_pydict()
    for name, kind, geojson in zip(result["name"], result["kind"], result["geojson"], strict=True):
        print(f"  {name:<8} {kind:<12} {geojson[:48]}")

    # The WKT round trip returns what went in.
    assert all(
        original.replace(" ", "") == round_trip.replace(" ", "")
        for original, round_trip in zip(result["wkt"], result["text"], strict=True)
    )

    # The geometry type is recovered, not guessed.
    assert [value.upper() for value in result["kind"]] == [
        "POINT",
        "LINESTRING",
        "POLYGON",
    ]

    # GeoJSON is valid JSON carrying the same type.
    assert all(value.startswith("{") for value in result["geojson"])
    parsed = encoded.select(kind=col("geojson").json.extract_string("$.type")).to_pydict()
    assert [value.upper() for value in parsed["kind"]] == [
        "POINT",
        "LINESTRING",
        "POLYGON",
    ]


if __name__ == "__main__":
    main()
