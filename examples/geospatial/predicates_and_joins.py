"""Spatial relationships, and the prefilter that makes a spatial join affordable.

The predicates are the join conditions and filter clauses of spatial work. Three things
this script is built to show, because each of them silently changes a result:

* `contains` and `covers` differ exactly on the boundary, and border rows are never rare.
* `st_intersects_extent` compares four numbers and is exact in the negative direction,
  which is what makes it the correct first stage of a join.
* `st_dwithin` is exact and cheaper than buffering one side and intersecting.

    python examples/geospatial/predicates_and_joins.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def the_relationship_family() -> None:
    """Every predicate over one polygon and a range of neighbours."""
    square = "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))"
    cases = bt.from_pydict(
        {
            "label": ["inside", "on edge", "adjacent", "overlapping", "far", "same"],
            "a": [square] * 6,
            "b": [
                "POINT(2 2)",
                "POINT(0 2)",
                "POLYGON((4 0, 8 0, 8 4, 4 4, 4 0))",
                "POLYGON((2 2, 6 2, 6 6, 2 6, 2 2))",
                "POINT(50 50)",
                "POLYGON((0 0, 0 4, 4 4, 4 0, 0 0))",
            ],
        }
    )
    print("--- the full predicate matrix ---")
    print(
        cases.select(
            "label",
            intersects=bt.st_intersects(col("a"), col("b")),
            disjoint=bt.st_disjoint(col("a"), col("b")),
            contains=bt.st_contains(col("a"), col("b")),
            covers=bt.st_covers(col("a"), col("b")),
            touches=bt.st_touches(col("a"), col("b")),
            overlaps=bt.st_overlaps(col("a"), col("b")),
            equals=bt.st_equals(col("a"), col("b")),
        ).to_pydict()
    )

    print("--- the mirrored spellings say the same thing the other way round ---")
    print(
        cases.select(
            "label",
            within=bt.st_within(col("b"), col("a")),
            covered_by=bt.st_covered_by(col("b"), col("a")),
        ).to_pydict()
    )

    # `crosses` needs the intersection to be smaller-dimensional than the operands, so a
    # road crossing a district crosses it while a road inside one does not.
    roads = bt.from_pydict(
        {
            "road": ["through", "inside"],
            "district": [square, square],
            "line": ["LINESTRING(-1 2, 5 2)", "LINESTRING(1 1, 3 3)"],
        }
    )
    print("--- crosses versus contains, for a line and an area ---")
    print(
        roads.select(
            "road",
            crosses=bt.st_crosses(col("line"), col("district")),
            contained=bt.st_within(col("line"), col("district")),
        ).to_pydict()
    )


def the_boundary_is_where_contains_and_covers_part() -> None:
    """If a spatial join drops rows sitting exactly on a border, this is usually why."""
    edge = bt.from_pydict(
        {
            "where": ["interior", "on the edge", "on a corner"],
            "poly": ["POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))"] * 3,
            "pt": ["POINT(2 2)", "POINT(0 2)", "POINT(4 4)"],
        }
    )
    print("--- covers accepts the boundary, contains does not ---")
    print(
        edge.select(
            "where",
            covers=bt.st_covers(col("poly"), col("pt")),
            contains=bt.st_contains(col("poly"), col("pt")),
        ).to_pydict()
    )


def a_join_with_a_cheap_prefilter() -> None:
    """Reject on boxes first; the exact predicate then runs on a fraction of the pairs."""
    regions = bt.from_pydict(
        {
            "region": ["west", "east", "triangle"],
            "shape": [
                "POLYGON((0 0, 5 0, 5 10, 0 10, 0 0))",
                "POLYGON((5 0, 10 0, 10 10, 5 10, 5 0))",
                "POLYGON((0 0, 4 0, 0 4, 0 0))",
            ],
        }
    )
    points = bt.from_pydict(
        {"pid": [1, 2, 3, 4], "at": ["POINT(1 1)", "POINT(7 3)", "POINT(3.5 3.5)", "POINT(99 99)"]}
    )

    pairs = points.join(regions, how="cross")
    box_pass = pairs.filter(bt.st_intersects_extent(col("shape"), col("at")))
    exact = box_pass.filter(bt.st_intersects(col("shape"), col("at")))

    print("--- how many pairs survive each stage ---")
    print(
        {
            "all pairs": pairs.count(),
            "box prefilter": box_pass.count(),
            "exact predicate": exact.count(),
        }
    )
    print(exact.select("pid", "region").sort("pid", "region").to_pydict())

    # The box test admits false positives, never false negatives. Point 3 is inside the
    # triangle's bounding box and outside the triangle, which is exactly the case the
    # exact predicate is there to remove.
    print("--- the pairs the box let through and the exact test rejected ---")
    disagree = box_pass.filter(~bt.st_intersects(col("shape"), col("at")))
    print(disagree.select("pid", "region").sort("pid", "region").to_pydict())

    # `st_contains_extent` is the box-level counterpart of `st_covers`.
    print("--- box containment ---")
    print(
        pairs.select(
            "pid",
            "region",
            box_contains=bt.st_contains_extent(col("shape"), col("at")),
        )
        .sort("pid", "region")
        .to_pydict()
    )


def proximity_without_buffering() -> None:
    """`st_dwithin` is exact; a buffer-and-intersect is neither exact nor faster."""
    stores = bt.from_pydict(
        {
            "store": ["a", "b", "c"],
            "at": [
                "POINT(-122.4194 37.7749)",
                "POINT(-122.4094 37.7749)",
                "POINT(-118.2437 34.0522)",
            ],
        }
    )
    home = "POINT(-122.4194 37.7749)"

    print("--- planar radius, in coordinate units ---")
    print(
        stores.select(
            "store",
            near=bt.st_dwithin(col("at"), bt.lit(home), 0.02),
        ).to_pydict()
    )

    print("--- geodesic radius, in metres, which is what a person means ---")
    print(
        stores.select(
            "store",
            within_1km=bt.st_dwithin_sphere(col("at"), bt.lit(home), 1000.0),
            within_2km=bt.st_dwithin_sphere(col("at"), bt.lit(home), 2000.0),
            metres=bt.st_distance_sphere(col("at"), bt.lit(home)).round(0),
        ).to_pydict()
    )


def main() -> None:
    the_relationship_family()
    the_boundary_is_where_contains_and_covers_part()
    a_join_with_a_cheap_prefilter()
    proximity_without_buffering()


if __name__ == "__main__":
    main()
