"""The OGC spatial predicates: does this geometry intersect, contain, touch that one.

Every one of these is a boolean over two geometries, which makes them the join
conditions and filter clauses of spatial SQL. Two properties hold across the whole set
and are worth internalizing:

**Touching counts as intersecting.** Two parcels sharing a fence line do intersect. A
predicate set that said otherwise would make every adjacency query return nothing.

**`contains` and `covers` differ exactly on the boundary.** A polygon *covers* a point
sitting on its edge; it does not *contain* it. If a query is dropping rows that sit
exactly on a border — and border cases are never rare in real data — that difference is
usually why. `covers` is also the cheaper of the two.

**Make the bounding box do the work.** `st_intersects` decodes both geometries and walks
their segments. `st_intersects_extent` compares four numbers. Putting the extent test
first, or materializing `st_xmin`/`st_xmax` columns and comparing those, is the single
biggest lever on a spatial join, because the cheap test is exact in the negative
direction: a pair whose boxes miss cannot possibly intersect.
"""

from __future__ import annotations

from batcher.plan.expr_ir.core import Expr
from batcher.plan.functions.geo._build import geo_call, geometry, value

__all__ = [
    "st_contains",
    "st_contains_extent",
    "st_covered_by",
    "st_covers",
    "st_crosses",
    "st_disjoint",
    "st_dwithin",
    "st_dwithin_sphere",
    "st_equals",
    "st_intersects",
    "st_intersects_extent",
    "st_overlaps",
    "st_touches",
    "st_within",
]


def st_intersects(a: Expr | str, b: Expr | str) -> Expr:
    """Whether two geometries share at least one point.

    The workhorse. Touching counts, so two polygons sharing only an edge intersect.
    Put `st_intersects_extent` in front of it on a large join.

    Args:
        a: The first geometry.
        b: The second geometry.

    Returns:
        True when the geometries share at least one point.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {
            ...         'a': [
            ...             'POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))',
            ...             'POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))',
            ...         ],
            ...         'b': ['POINT(1 1)', 'POINT(9 9)'],
            ...     }
            ... )
            >>> got = bt.st_intersects(bt.col("a"), bt.col("b"))
            >>> ds.select(v=got).to_pydict()
            {'v': [True, False]}
    """
    return geo_call("st_intersects", geometry(a), geometry(b))


def st_disjoint(a: Expr | str, b: Expr | str) -> Expr:
    """Whether two geometries share no point at all.

    Exactly the negation of `st_intersects`, and it costs the same — there is no
    cheap way to prove disjointness that is not also a way to prove intersection.

    Args:
        a: The first geometry.
        b: The second geometry.

    Returns:
        True when the geometries share no point.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {'a': ['POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))'], 'b': ['POINT(9 9)']}
            ... )
            >>> got = bt.st_disjoint(bt.col("a"), bt.col("b"))
            >>> ds.select(v=got).to_pydict()
            {'v': [True]}
    """
    return geo_call("st_disjoint", geometry(a), geometry(b))


def st_contains(a: Expr | str, b: Expr | str) -> Expr:
    """Whether one geometry contains another.

    Requires `b` to be inside `a` **and** to touch `a`'s interior, so a point on the
    edge is not contained. When boundary cases should count, use `st_covers`.

    `st_contains(a, b)` is `st_within(b, a)` by construction.

    Args:
        a: The containing geometry.
        b: The contained geometry.

    Returns:
        True when `b` lies in `a` and meets its interior.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {
            ...         'a': [
            ...             'POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))',
            ...             'POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))',
            ...         ],
            ...         'b': ['POINT(2 2)', 'POINT(0 2)'],
            ...     }
            ... )
            >>> got = bt.st_contains(bt.col("a"), bt.col("b"))
            >>> ds.select(v=got).to_pydict()
            {'v': [True, False]}
    """
    return geo_call("st_contains", geometry(a), geometry(b))


def st_within(a: Expr | str, b: Expr | str) -> Expr:
    """Whether one geometry lies within another.

    The mirror of `st_contains`, with the arguments the other way round. Which one to
    reach for is purely about which geometry the sentence is about.

    Args:
        a: The contained geometry.
        b: The containing geometry.

    Returns:
        True when `a` lies in `b` and meets its interior.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {'a': ['POINT(2 2)'], 'b': ['POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))']}
            ... )
            >>> got = bt.st_within(bt.col("a"), bt.col("b"))
            >>> ds.select(v=got).to_pydict()
            {'v': [True]}
    """
    return geo_call("st_within", geometry(a), geometry(b))


def st_covers(a: Expr | str, b: Expr | str) -> Expr:
    """Whether one geometry covers another, boundary included.

    `st_contains` without the interior requirement, so a point on the edge counts.
    This is usually the predicate a person means by 'inside', and it is cheaper than
    `st_contains`, which has to establish the interior condition as well.

    Args:
        a: The covering geometry.
        b: The covered geometry.

    Returns:
        True when every point of `b` lies in `a`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {'a': ['POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))'], 'b': ['POINT(0 2)']}
            ... )
            >>> got = bt.st_covers(bt.col("a"), bt.col("b"))
            >>> ds.select(v=got).to_pydict()
            {'v': [True]}
    """
    return geo_call("st_covers", geometry(a), geometry(b))


def st_covered_by(a: Expr | str, b: Expr | str) -> Expr:
    """Whether one geometry is covered by another, boundary included.

    The mirror of `st_covers`.

    Args:
        a: The covered geometry.
        b: The covering geometry.

    Returns:
        True when every point of `a` lies in `b`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {'a': ['POINT(0 2)'], 'b': ['POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))']}
            ... )
            >>> got = bt.st_covered_by(bt.col("a"), bt.col("b"))
            >>> ds.select(v=got).to_pydict()
            {'v': [True]}
    """
    return geo_call("st_covered_by", geometry(a), geometry(b))


def st_touches(a: Expr | str, b: Expr | str) -> Expr:
    """Whether two geometries meet without overlapping.

    The adjacency predicate: shared boundary, no shared area. This is how you find
    neighbouring parcels, bordering countries, or the polygons a road runs along.

    Args:
        a: The first geometry.
        b: The second geometry.

    Returns:
        True when the geometries meet but their interiors do not.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {
            ...         'a': [
            ...             'POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))',
            ...             'POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))',
            ...         ],
            ...         'b': [
            ...             'POLYGON((4 0, 8 0, 8 4, 4 4, 4 0))',
            ...             'POLYGON((2 2, 6 2, 6 6, 2 6, 2 2))',
            ...         ],
            ...     }
            ... )
            >>> got = bt.st_touches(bt.col("a"), bt.col("b"))
            >>> ds.select(v=got).to_pydict()
            {'v': [True, False]}
    """
    return geo_call("st_touches", geometry(a), geometry(b))


def st_crosses(a: Expr | str, b: Expr | str) -> Expr:
    """Whether two geometries cross rather than merely meet or nest.

    A road crossing a district boundary, two flight paths intersecting. Requires the
    intersection to be *smaller-dimensional* than the operands, so two lines running
    along each other overlap rather than cross, and a line inside a polygon does
    neither.

    Args:
        a: The first geometry.
        b: The second geometry.

    Returns:
        True when the interiors meet in something of lower dimension than the
        operands.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {
            ...         'a': ['LINESTRING(-1 2, 5 2)'],
            ...         'b': ['POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))'],
            ...     }
            ... )
            >>> got = bt.st_crosses(bt.col("a"), bt.col("b"))
            >>> ds.select(v=got).to_pydict()
            {'v': [True]}
    """
    return geo_call("st_crosses", geometry(a), geometry(b))


def st_overlaps(a: Expr | str, b: Expr | str) -> Expr:
    """Whether two geometries of the same dimension partially overlap.

    Partial overlap specifically: same dimension, shared interior of that dimension,
    and neither covering the other. Two polygons where one is inside the other do not
    overlap, they contain — which is what makes this the predicate for finding conflicts
    in a supposedly-disjoint coverage.

    Args:
        a: The first geometry.
        b: The second geometry.

    Returns:
        True when same-dimension geometries share interior of that dimension without
        either covering the other.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {
            ...         'a': ['POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))'],
            ...         'b': ['POLYGON((2 2, 6 2, 6 6, 2 6, 2 2))'],
            ...     }
            ... )
            >>> got = bt.st_overlaps(bt.col("a"), bt.col("b"))
            >>> ds.select(v=got).to_pydict()
            {'v': [True]}
    """
    return geo_call("st_overlaps", geometry(a), geometry(b))


def st_equals(a: Expr | str, b: Expr | str) -> Expr:
    """Whether two geometries occupy the same set of points.

    **Topological, not structural.** The same square written clockwise and
    counter-clockwise is equal, and so is one with an extra collinear vertex. That is
    usually what you want and it is *not* what comparing the WKB bytes gives you — if
    you need byte equality, compare `st_as_binary` instead.

    Args:
        a: The first geometry.
        b: The second geometry.

    Returns:
        True when the geometries occupy the same set of points.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {
            ...         'a': ['POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))'],
            ...         'b': ['POLYGON((0 0, 0 4, 4 4, 4 0, 0 0))'],
            ...     }
            ... )
            >>> got = bt.st_equals(bt.col("a"), bt.col("b"))
            >>> ds.select(v=got).to_pydict()
            {'v': [True]}
    """
    return geo_call("st_equals", geometry(a), geometry(b))


def st_dwithin(a: Expr | str, b: Expr | str, radius: Expr | float) -> Expr:
    """Whether two geometries are within a distance of each other.

    Exact, and cheaper than `st_distance(a, b) <= r` because it rejects on bounding
    boxes first. Prefer it over buffering one side and intersecting: that is both
    slower and, since `st_buffer` is approximate, less correct.

    The radius is in coordinate units. A negative radius raises rather than answering
    false, because it is a query bug rather than an empty result.

    Args:
        a: The first geometry.
        b: The second geometry.
        radius: The maximum distance, in coordinate units.

    Returns:
        True when the geometries are within `radius` of each other.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'a': ['POINT(0 0)'], 'b': ['POINT(3 4)']})
            >>> got = bt.st_dwithin(bt.col("a"), bt.col("b"), 5.0)
            >>> ds.select(v=got).to_pydict()
            {'v': [True]}
    """
    return geo_call("st_dwithin", geometry(a), geometry(b), value(radius))


def st_dwithin_sphere(a: Expr | str, b: Expr | str, metres: Expr | float) -> Expr:
    """Whether two lon/lat geometries are within a distance in metres.

    The proximity filter for un-projected data: 'every store within 5 km of this
    customer' without having to convert kilometres into degrees at a particular
    latitude. Measured vertex to vertex like `st_distance_sphere`, so it is exact for
    points and slightly generous for extended shapes.

    Args:
        a: The first lon/lat geometry.
        b: The second lon/lat geometry.
        metres: The maximum distance, in metres.

    Returns:
        True when the geometries are within `metres` of each other on the sphere.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {'a': ['POINT(-122.4194 37.7749)'], 'b': ['POINT(-122.4094 37.7749)']}
            ... )
            >>> got = bt.st_dwithin_sphere(bt.col("a"), bt.col("b"), 1000.0)
            >>> ds.select(v=got).to_pydict()
            {'v': [True]}
    """
    return geo_call("st_dwithin_sphere", geometry(a), geometry(b), value(metres))


def st_intersects_extent(a: Expr | str, b: Expr | str) -> Expr:
    """Whether two geometries' bounding boxes overlap.

    Four comparisons instead of a segment walk, and **exact in the negative
    direction**: a false here means the geometries certainly do not intersect. That is
    what makes it the correct first stage of a spatial join — it can produce false
    positives, which the exact predicate then removes, and never false negatives.

    Args:
        a: The first geometry.
        b: The second geometry.

    Returns:
        True when the geometries' bounding boxes overlap.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {'a': ['POLYGON((0 0, 4 0, 0 4, 0 0))'], 'b': ['POINT(3.5 3.5)']}
            ... )
            >>> box = bt.st_intersects_extent(bt.col("a"), bt.col("b"))
            >>> exact = bt.st_intersects(bt.col("a"), bt.col("b"))
            >>> got = bt.struct(box=box, exact=exact)
            >>> ds.select(v=got).to_pydict()
            {'v': [{'box': True, 'exact': False}]}
    """
    return geo_call("st_intersects_extent", geometry(a), geometry(b))


def st_contains_extent(a: Expr | str, b: Expr | str) -> Expr:
    """Whether one geometry's bounding box contains another's.

    The box-level counterpart of `st_covers`, with the same role as
    `st_intersects_extent`: a cheap prefilter that is exact in the negative direction.

    Args:
        a: The first geometry.
        b: The second geometry.

    Returns:
        True when `a`'s bounding box contains `b`'s.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {'a': ['POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))'], 'b': ['POINT(2 2)']}
            ... )
            >>> got = bt.st_contains_extent(bt.col("a"), bt.col("b"))
            >>> ds.select(v=got).to_pydict()
            {'v': [True]}
    """
    return geo_call("st_contains_extent", geometry(a), geometry(b))
