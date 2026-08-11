"""Making a spatial join hashable by joining on a grid cell first.

The exact predicate cannot hash, so the pattern is: join on a coarse grid key, then evaluate
the exact predicate on the candidates that survive. The grid does the pruning; the predicate
does the deciding.

    python examples/geospatial/spatial_joins_with_grids.py
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
            "name": ["ferry", "pier", "market", "london"],
            "lon": [-122.3937, -122.3941, -122.4194, -0.1276],
            "lat": [37.7955, 37.7960, 37.7749, 51.5074],
        }
    ).with_columns(point=bt.st_point(col("lon"), col("lat")))

    stops = bt.from_pydict(
        {
            "stop": ["embarcadero", "civic", "westminster"],
            "lon": [-122.3970, -122.4180, -0.1281],
            "lat": [37.7930, 37.7790, 51.5010],
        }
    ).with_columns(stop_point=bt.st_point(col("lon"), col("lat")))

    # Coarse grid keys on both sides: a 5-character geohash is a few kilometres.
    left = points.with_columns(cell=bt.st_geohash(col("point"), 5)).select("name", "point", "cell")
    right = stops.with_columns(cell=bt.st_geohash(col("stop_point"), 5)).select(
        "stop", "stop_point", "cell"
    )

    # The grid join: an equality join, so it hashes.
    candidates = left.join(right, on="cell")
    print("candidate pairs:", candidates.count())
    assert candidates.count() > 0
    # Far fewer than the cross product, which is the whole point.
    assert candidates.count() < left.count() * right.count()

    # The exact predicate, on the candidates only.
    scored = candidates.select(
        "name",
        "stop",
        metres=bt.st_distance_sphere(col("point"), col("stop_point")),
    ).sort("name", "metres")

    result = scored.to_pydict()
    for name, stop, metres in zip(result["name"], result["stop"], result["metres"], strict=True):
        print(f"  {name:<8} -> {stop:<12} {metres:>8,.0f} m")

    # Everything paired is genuinely nearby, and London paired with nothing in California.
    assert all(value < 20_000 for value in result["metres"])
    assert "london" not in result["name"] or "westminster" in result["stop"]


if __name__ == "__main__":
    main()
