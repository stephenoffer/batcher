# Geospatial reference

Every `ST_*` function, grouped by what it does. Names and semantics follow PostGIS, so a
ported query reads the same.

A geometry column is **WKB in a Binary column**. That is the encoding GeoParquet,
PostGIS and DuckDB spatial all store, so a column round-trips through any of them
unconverted. A
geometry that will not parse yields null rather than raising, because one corrupt row in
a hundred million must not abort a scan; `st_is_valid_reason` names every bad row and
why. A query bug, such as a negative radius or an unsupported EPSG code, does raise.

To learn these rather than look them up, start with
{doc}`/user-guide/analyze/geospatial`.

```{eval-rst}
.. currentmodule:: batcher
```

## Reading and writing geometry

Crossing between the WKB a geometry column holds and the text encodings people and other systems use. You rarely need `st_geom_from_text`: every function here accepts a text column directly and detects WKT, EWKT, GeoJSON or hex WKB by content.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   st_as_binary
   st_as_ewkb
   st_as_ewkt
   st_as_geojson
   st_as_hex_wkb
   st_as_text
   st_geom_from_geohash
   st_geom_from_geojson
   st_geom_from_text
   st_geom_from_wkb
```

## Building and deriving geometry

Turning coordinate columns into geometries, and reducing a geometry to a simpler shape. `st_envelope` and `st_convex_hull` are the two rungs of a filter ladder: the box is exact to compute and is what an index stores, the hull is tighter but costs a sort.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   st_boundary
   st_buffer
   st_centroid
   st_collect
   st_convex_hull
   st_envelope
   st_expand
   st_make_envelope
   st_make_line
   st_make_polygon
   st_point
   st_point_on_surface
   st_point_z
```

## Ordinates, bounds and counts

The cheap functions. `st_xmin` and its three siblings are the ones to know: materialize them once beside the geometry and every later region filter becomes four Float64 comparisons that push down to the scan.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   st_coord_dim
   st_dimension
   st_geometry_type
   st_num_geometries
   st_num_interior_rings
   st_num_points
   st_set_srid
   st_srid
   st_x
   st_xmax
   st_xmin
   st_y
   st_ymax
   st_ymin
   st_z
```

## Members, vertices and validity

Picking a geometry apart, and asking whether it is well formed. Run `st_is_valid_reason` over a new geometry column before trusting anything else here: an invalid polygon makes every areal predicate wrong, silently.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   st_end_point
   st_exterior_ring
   st_geometry_n
   st_has_z
   st_interior_ring_n
   st_is_closed
   st_is_collection
   st_is_empty
   st_is_ring
   st_is_simple
   st_is_valid
   st_is_valid_reason
   st_point_n
   st_start_point
```

## Areas, lengths, distances and bearings

Two families, and mixing them up is silent. The planar functions answer in coordinate units, which on EPSG:4326 means degrees. The `_sphere` and `_spheroid` functions answer in metres on the Earth.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   st_area
   st_area_spheroid
   st_azimuth
   st_distance
   st_distance_sphere
   st_distance_spheroid
   st_hausdorff_distance
   st_length
   st_length_spheroid
   st_max_distance
   st_perimeter
   st_perimeter_spheroid
```

## Spatial relationships

The OGC predicates, which are the join conditions and filter clauses of spatial SQL. Touching counts as intersecting; `contains` and `covers` differ exactly on the boundary; and `st_intersects_extent` in front of `st_intersects` is the single biggest lever on a spatial join.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   st_contains
   st_contains_extent
   st_covered_by
   st_covers
   st_crosses
   st_disjoint
   st_dwithin
   st_dwithin_sphere
   st_equals
   st_intersects
   st_intersects_extent
   st_overlaps
   st_touches
   st_within
```

## Moving, normalizing and reprojecting

Affine transforms that cannot invalidate a geometry, normalizations for a column that arrived with different conventions, and `st_transform`, which is how a measurement becomes metres.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   st_affine
   st_flip_coordinates
   st_force_2d
   st_force_3d
   st_force_polygon_ccw
   st_force_polygon_cw
   st_remove_repeated_points
   st_reverse
   st_rotate
   st_scale
   st_segmentize
   st_simplify
   st_snap_to_grid
   st_transform
   st_translate
```

## Positions along a chain

Linear referencing: the vocabulary route and network data is described in. Every function here takes a single chain, because a multi-chain geometry has no defined traversal order.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   st_closest_point
   st_line_interpolate_point
   st_line_locate_point
   st_line_substring
   st_project
   st_shortest_line
```

## Grids and cell ids

Turning a position into a discrete cell the engine can hash, sort, shuffle and join at full speed with no spatial index. Usually the cheapest useful geospatial thing you can do.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   geohash_decode_lat
   geohash_decode_lon
   geohash_encode
   st_geohash
   st_hex_bin
   st_hex_center_x
   st_hex_center_y
   st_quadkey
   st_s2_cell
   st_s2_cell_parent
   st_tile_x
   st_tile_y
   st_utm_epsg
   st_utm_zone
```

## See also

- {doc}`/user-guide/analyze/geospatial`: the guide that teaches these.
- {doc}`/api/relational/functions`: the non-spatial scalar, aggregate and window functions.
- {doc}`/cookbook/analytics/geospatial-binning`: snapping coordinates to a grid by hand,
  and why `floor` is the only correct way to do it.
