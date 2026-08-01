"""Turning positions into cell ids you can group, sort and join on.

Latitude and longitude are floats, so no two observations share a value and
`GROUP BY lat, lon` returns the input. A grid function turns a position into a discrete
id, and the engine then hashes, sorts, shuffles and joins that at full speed with no
spatial index at all. This is usually the cheapest useful geospatial thing you can do.

    python examples/geospatial/grid_keys.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def pickups() -> bt.Dataset:
    """Seven positions: two pairs that should co-locate, three singletons."""
    return bt.from_pydict(
        {
            "id": [1, 2, 3, 4, 5, 6, 7],
            "lon": [-122.4194, -122.4190, -122.2712, -74.0060, -74.0059, -0.1278, 151.2093],
            "lat": [37.7749, 37.7751, 37.8044, 40.7128, 40.7130, 51.5074, -33.8688],
        }
    )


def geohash_is_a_prefix_code_over_space() -> None:
    """Sharing a prefix means being close, which turns a region filter into a LIKE."""
    ds = pickups()
    hashed = ds.with_columns(
        cell=bt.geohash_encode(col("lon"), col("lat"), 6),
        coarse=bt.geohash_encode(col("lon"), col("lat"), 3),
    )
    print("--- one cell id per position ---")
    print(hashed.select("id", "cell", "coarse").to_pydict())

    print("--- and integers group, so this is an ordinary hash aggregate ---")
    print(hashed.group_by("cell").agg(n=bt.count()).sort("cell").to_pydict())

    print("--- hashes nest, so a rollup is substr rather than a recomputation ---")
    print(
        hashed.select(
            "id",
            same=col("cell").str.substr(1, 3) == col("coarse"),
        ).to_pydict()
    )

    print("--- decode a cell back to a position, or to the cell itself ---")
    cells = bt.from_pydict({"h": ["9q8yyk", "gcpvj0"]})
    print(
        cells.select(
            lon=bt.geohash_decode_lon(col("h")).round(4),
            lat=bt.geohash_decode_lat(col("h")).round(4),
            box=bt.st_as_text(bt.st_geom_from_geohash(col("h"))),
        ).to_pydict()
    )

    print("--- st_geohash takes a geometry, reducing it to its centroid ---")
    areas = bt.from_pydict(
        {"g": ["POLYGON((-122.42 37.77, -122.41 37.77, -122.41 37.78, -122.42 37.77))"]}
    )
    print(areas.select(cell=bt.st_geohash(col("g"), 6)).to_pydict())


def tiles_are_the_grid_maps_are_served_on() -> None:
    ds = pickups()
    tiled = ds.with_columns(
        x=bt.st_tile_x(col("lon"), col("lat"), 12),
        # y increases *southward*: row 0 is the top of the map, near 85 degrees north.
        y=bt.st_tile_y(col("lon"), col("lat"), 12),
        key=bt.st_quadkey(col("lon"), col("lat"), 12),
    )
    print("--- tile addresses at zoom 12 ---")
    print(tiled.select("id", "x", "y", "key").to_pydict())

    print("--- northern positions have smaller y ---")
    print(tiled.select("id", "lat", "y").sort("lat", descending=True).to_pydict())

    print("--- a quadkey nests by digit, like a geohash by character ---")
    zooms = ds.filter(col("id") == 1)
    print(
        zooms.select(
            z4=bt.st_quadkey(col("lon"), col("lat"), 4),
            z8=bt.st_quadkey(col("lon"), col("lat"), 8),
            z12=bt.st_quadkey(col("lon"), col("lat"), 12),
        ).to_pydict()
    )


def s2_cells_sort_spatially_and_are_near_equal_area() -> None:
    """The best general-purpose spatial key: fair across latitudes, and range-scannable."""
    ds = pickups()
    celled = ds.with_columns(
        fine=bt.st_s2_cell(col("lon"), col("lat"), 15),
        coarse=bt.st_s2_cell(col("lon"), col("lat"), 8),
    )
    print("--- cell ids at two levels ---")
    print(celled.select("id", "fine", "coarse").to_pydict())

    print("--- rolling up is a bit mask, not a recomputation ---")
    print(
        celled.select(
            "id",
            same=bt.st_s2_cell_parent(col("fine"), 8) == col("coarse"),
        ).to_pydict()
    )

    print("--- neighbouring positions land in the same coarse cell ---")
    print(celled.group_by("coarse").agg(n=bt.count()).sort("coarse").to_pydict())

    # Because the id is a Hilbert index, sorting by it clusters neighbours onto the same
    # pages: the two San Francisco rows end up adjacent, and Sydney is far away.
    print("--- sorting by cell id is sorting by locality ---")
    print(celled.select("id", "fine").sort("fine").to_pydict()["id"])


def hexagons_remove_the_grid_bias_of_squares() -> None:
    """All six neighbours are equidistant, which a square grid cannot offer."""
    # Project first: this bins whatever coordinates it is given, and it is not H3.
    ds = pickups()
    projected = ds.with_columns(
        m=bt.st_transform(bt.st_point(col("lon"), col("lat")), 4326, 3857)
    ).with_columns(x=bt.st_x(col("m")), y=bt.st_y(col("m")))

    binned = projected.with_columns(cell=bt.st_hex_bin(col("x"), col("y"), 500.0))
    print("--- 500 metre hexagons in Web Mercator ---")
    print(binned.group_by("cell").agg(n=bt.count()).sort("n", descending=True).to_pydict()["n"])

    print("--- recover a plottable centre from the group key ---")
    centres = binned.select(
        "id",
        cx=bt.st_hex_center_x(col("cell"), 500.0).round(1),
        cy=bt.st_hex_center_y(col("cell"), 500.0).round(1),
    )
    print(centres.to_pydict())


def main() -> None:
    geohash_is_a_prefix_code_over_space()
    tiles_are_the_grid_maps_are_served_on()
    s2_cells_sort_spatially_and_are_near_equal_area()
    hexagons_remove_the_grid_bias_of_squares()


if __name__ == "__main__":
    main()
