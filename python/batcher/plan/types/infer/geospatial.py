"""Output types for the `st_*` geometry and `quat_*`/`se3_*` rigid-body functions.

Both families answer from the function name alone, the way `scalars` does for `.str` and
`.dt`, so this module is a pure lookup with no operand analysis.

Neither had a rule at all. `GeoFunc` and `SpatialFunc` were the two `Expr` node kinds
`infer_type` had no arm for, so **every** query touching a geometry or a pose reported
`Dataset.schema` as all-`null` -- not just the geometry column, because
`Project.available_schema` is all-or-nothing by design and one uncertain column takes its
neighbours with it. An empty result was typed from that same schema, and the device tier
had nothing to hold its own answer against.

Every entry below was measured against the engine rather than read off a name: each
function was built through its public constructor with arguments of the right shape and its
output type recorded. `tests/differential/test_diff_geospatial_declared_types.py` re-derives
the same table from `GEO_FNS`/`SPATIAL_FNS` and fails on a member this file does not
classify, so a function added to either vocabulary cannot land untyped.
"""

from __future__ import annotations

import pyarrow as pa

__all__ = ["geofunc_type", "spatialfunc_type"]

#: Constructors, accessors and transforms that yield a **geometry**, which crosses as WKB.
_GEO_BINARY = frozenset(
    {
        "st_affine", "st_as_binary", "st_as_ewkb", "st_boundary", "st_buffer", "st_centroid",
        "st_closest_point", "st_collect", "st_convex_hull", "st_end_point", "st_envelope",
        "st_expand", "st_exterior_ring", "st_flip_coordinates", "st_force2d", "st_force3d",
        "st_force_polygon_ccw", "st_force_polygon_cw", "st_geom_from_geohash",
        "st_geom_from_geojson", "st_geom_from_text", "st_geom_from_wkb", "st_geometry_n",
        "st_interior_ring_n", "st_line_interpolate_point", "st_line_substring", "st_make_envelope",
        "st_make_line", "st_make_polygon", "st_point", "st_point_n", "st_point_on_surface",
        "st_point_z", "st_project", "st_remove_repeated_points", "st_reverse", "st_rotate",
        "st_scale", "st_segmentize", "st_set_srid", "st_shortest_line", "st_simplify",
        "st_snap_to_grid", "st_start_point", "st_transform", "st_translate"
    }
)  # fmt: skip

#: Measures and coordinate readers -- areas, lengths, distances, bearings, ordinates.
_GEO_DOUBLE = frozenset(
    {
        "geohash_decode_lat", "geohash_decode_lon", "st_area", "st_area_spheroid", "st_azimuth",
        "st_distance", "st_distance_sphere", "st_distance_spheroid", "st_hausdorff_distance",
        "st_hex_center_x", "st_hex_center_y", "st_length", "st_length_spheroid",
        "st_line_locate_point", "st_max_distance", "st_perimeter", "st_perimeter_spheroid", "st_x",
        "st_xmax", "st_xmin", "st_y", "st_ymax", "st_ymin", "st_z"
    }
)  # fmt: skip

#: The spatial predicates. Every one answers a yes/no question about one or two geometries.
_GEO_BOOL = frozenset(
    {
        "st_contains", "st_contains_extent", "st_covered_by", "st_covers", "st_crosses",
        "st_disjoint", "st_dwithin", "st_dwithin_sphere", "st_equals", "st_has_z", "st_intersects",
        "st_intersects_extent", "st_is_closed", "st_is_collection", "st_is_empty", "st_is_ring",
        "st_is_simple", "st_is_valid", "st_overlaps", "st_touches", "st_within"
    }
)  # fmt: skip

#: Counts, dimensions, and the integer grid cell identifiers.
_GEO_INT = frozenset(
    {
        "st_coord_dim", "st_dimension", "st_hex_bin", "st_num_geometries", "st_num_interior_rings",
        "st_num_points", "st_s2_cell", "st_s2_cell_parent", "st_srid", "st_tile_x", "st_tile_y",
        "st_utm_epsg", "st_utm_zone"
    }
)  # fmt: skip

#: The text renderings -- WKT/EWKT/GeoJSON/hex-WKB, the type name, and the string grid codes.
_GEO_STRING = frozenset(
    {
        "geohash_encode", "st_as_ewkt", "st_as_geojson", "st_as_hex_wkb", "st_as_text",
        "st_geohash", "st_geometry_type", "st_is_valid_reason", "st_quadkey"
    }
)  # fmt: skip


def geofunc_type(fn: str) -> pa.DataType | None:
    """The Arrow type an `st_*` geometry function produces, or ``None`` if unclassified."""
    if fn in _GEO_BINARY:
        return pa.binary()
    if fn in _GEO_DOUBLE:
        return pa.float64()
    if fn in _GEO_BOOL:
        return pa.bool_()
    if fn in _GEO_INT:
        return pa.int64()
    if fn in _GEO_STRING:
        return pa.string()
    return None


def spatialfunc_type(_fn: str) -> pa.DataType:
    """Float64, for every member of the rigid-body vocabulary.

    There is no table here because there is nothing to tabulate. `SPATIAL_FNS` names one
    entry per *output component* -- `quat_multiply_w`, `se3_transform_x` -- precisely so
    every function returns a single number rather than a struct, and rigid-body arithmetic
    is `f64` throughout (`bc_spatial` is `f64` and the standard library's transcendentals).
    All 42 members were measured and all 42 are `double`; the argument is unused, and named
    so, because taking it keeps the call site identical to its `geofunc_type` neighbour.
    """
    return pa.float64()
