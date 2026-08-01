//! The geometry model — one `Geometry` value every codec and algorithm speaks.
//!
//! This mirrors the OGC Simple Features type hierarchy, which is what WKB, WKT and
//! GeoJSON all encode, so a round trip through any of the three is lossless. Two
//! modelling choices are load-bearing and worth stating rather than rediscovering:
//!
//! **Z is per-geometry, not per-coordinate.** OGC dimensionality is a property of the
//! geometry (`POINT Z`, not "a point some of whose coordinates have z"), and WKB
//! encodes it in the type code. Storing `Option<f64>` per coordinate would let a
//! `LineString` hold a mix that no encoding can represent. `has_z` on the root is the
//! honest shape, and `Coord::z` is meaningful only when it is set.
//!
//! **SRID rides on the root only.** PostGIS EWKB puts the SRID on the outermost
//! geometry and nowhere else; a collection whose members disagreed would not be
//! writable. `Geom` is that root wrapper: an SRID plus the geometry it labels.

use crate::error::{GeoError, GeoResult};

/// A single position. `z` is meaningful only when the owning geometry has `has_z`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Coord {
    /// Easting / longitude.
    pub x: f64,
    /// Northing / latitude.
    pub y: f64,
    /// Elevation, read only when the owning geometry is 3D.
    pub z: f64,
}

impl Coord {
    /// A 2D position (`z` defaulted to 0, unread unless the geometry is 3D).
    pub fn new(x: f64, y: f64) -> Self {
        Coord { x, y, z: 0.0 }
    }

    /// A 3D position.
    pub fn new_z(x: f64, y: f64, z: f64) -> Self {
        Coord { x, y, z }
    }

    /// True when either ordinate is NaN — the one input every predicate must reject,
    /// because NaN comparisons are false in both directions and would silently make a
    /// point "outside" every polygon including the one containing it.
    pub fn is_nan(&self) -> bool {
        self.x.is_nan() || self.y.is_nan()
    }
}

/// An ordered chain of positions. A ring is a closed `LineString`.
pub type LineString = Vec<Coord>;

/// A polygon: an exterior ring followed by zero or more interior rings (holes).
///
/// Ring orientation is *not* normalized on construction. WKB and GeoJSON disagree on
/// the canonical winding (GeoJSON right-hand rule versus PostGIS's usual CW exterior),
/// so forcing one on read would make a round trip lossy. Algorithms that need an
/// orientation compute it (`signed_area`) rather than assuming one.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Polygon {
    /// The outer boundary. Empty for an empty polygon.
    pub exterior: LineString,
    /// Holes, each a closed ring inside the exterior.
    pub interiors: Vec<LineString>,
}

/// An OGC Simple Features geometry.
#[derive(Debug, Clone, PartialEq)]
pub enum Geometry {
    /// A single position, or `None` for `POINT EMPTY`.
    Point(Option<Coord>),
    /// A chain of positions.
    LineString(LineString),
    /// A ringed area.
    Polygon(Polygon),
    /// A set of positions.
    MultiPoint(Vec<Option<Coord>>),
    /// A set of chains.
    MultiLineString(Vec<LineString>),
    /// A set of ringed areas.
    MultiPolygon(Vec<Polygon>),
    /// A heterogeneous set of geometries.
    GeometryCollection(Vec<Geometry>),
}

/// The OGC type code of a geometry, as used by WKB and by `st_geometry_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeomType {
    /// `POINT`
    Point = 1,
    /// `LINESTRING`
    LineString = 2,
    /// `POLYGON`
    Polygon = 3,
    /// `MULTIPOINT`
    MultiPoint = 4,
    /// `MULTILINESTRING`
    MultiLineString = 5,
    /// `MULTIPOLYGON`
    MultiPolygon = 6,
    /// `GEOMETRYCOLLECTION`
    GeometryCollection = 7,
}

impl GeomType {
    /// The uppercase OGC name (`"POINT"`), as `ST_GeometryType` reports it.
    pub fn name(self) -> &'static str {
        match self {
            GeomType::Point => "POINT",
            GeomType::LineString => "LINESTRING",
            GeomType::Polygon => "POLYGON",
            GeomType::MultiPoint => "MULTIPOINT",
            GeomType::MultiLineString => "MULTILINESTRING",
            GeomType::MultiPolygon => "MULTIPOLYGON",
            GeomType::GeometryCollection => "GEOMETRYCOLLECTION",
        }
    }

    /// The type code carried by a WKB header, or an error for an unknown code.
    pub fn from_code(code: u32) -> GeoResult<Self> {
        Ok(match code {
            1 => GeomType::Point,
            2 => GeomType::LineString,
            3 => GeomType::Polygon,
            4 => GeomType::MultiPoint,
            5 => GeomType::MultiLineString,
            6 => GeomType::MultiPolygon,
            7 => GeomType::GeometryCollection,
            other => {
                return Err(GeoError::parse(
                    "WKB",
                    format!("unknown geometry type code {other}"),
                ))
            }
        })
    }

    /// The topological dimension: 0 for points, 1 for lines, 2 for areas.
    ///
    /// A collection reports the maximum of its members, which is what PostGIS
    /// `ST_Dimension` does; the empty collection reports 0.
    pub fn dimension(self) -> i64 {
        match self {
            GeomType::Point | GeomType::MultiPoint => 0,
            GeomType::LineString | GeomType::MultiLineString => 1,
            GeomType::Polygon | GeomType::MultiPolygon => 2,
            GeomType::GeometryCollection => 0,
        }
    }
}

/// An axis-aligned bounding box.
///
/// Constructed only from a non-empty geometry, so the bounds are always real numbers;
/// an empty geometry yields `None` rather than the inverted-infinity sentinel that
/// makes every downstream comparison quietly true.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bbox {
    /// Minimum x.
    pub xmin: f64,
    /// Minimum y.
    pub ymin: f64,
    /// Maximum x.
    pub xmax: f64,
    /// Maximum y.
    pub ymax: f64,
}

impl Bbox {
    /// The box containing exactly `c`.
    pub fn from_coord(c: Coord) -> Self {
        Bbox {
            xmin: c.x,
            ymin: c.y,
            xmax: c.x,
            ymax: c.y,
        }
    }

    /// Grow to contain `c`.
    pub fn extend(&mut self, c: Coord) {
        self.xmin = self.xmin.min(c.x);
        self.ymin = self.ymin.min(c.y);
        self.xmax = self.xmax.max(c.x);
        self.ymax = self.ymax.max(c.y);
    }

    /// Grow to contain `other`.
    pub fn union(&mut self, other: Bbox) {
        self.xmin = self.xmin.min(other.xmin);
        self.ymin = self.ymin.min(other.ymin);
        self.xmax = self.xmax.max(other.xmax);
        self.ymax = self.ymax.max(other.ymax);
    }

    /// True when the two boxes share at least a boundary point.
    pub fn intersects(&self, other: &Bbox) -> bool {
        self.xmin <= other.xmax
            && other.xmin <= self.xmax
            && self.ymin <= other.ymax
            && other.ymin <= self.ymax
    }

    /// True when `other` lies entirely inside this box (boundary counts as inside).
    pub fn contains(&self, other: &Bbox) -> bool {
        self.xmin <= other.xmin
            && self.xmax >= other.xmax
            && self.ymin <= other.ymin
            && self.ymax >= other.ymax
    }

    /// True when `c` lies inside or on this box.
    pub fn contains_coord(&self, c: Coord) -> bool {
        c.x >= self.xmin && c.x <= self.xmax && c.y >= self.ymin && c.y <= self.ymax
    }

    /// Grow the box by `dx` horizontally and `dy` vertically on every side.
    pub fn expand(&self, dx: f64, dy: f64) -> Bbox {
        Bbox {
            xmin: self.xmin - dx,
            ymin: self.ymin - dy,
            xmax: self.xmax + dx,
            ymax: self.ymax + dy,
        }
    }

    /// The smallest distance between the two boxes, 0 when they intersect.
    ///
    /// The cheap lower bound on `st_distance`, which is what makes it usable as a
    /// spatial-join prefilter: a pair whose boxes are further apart than the radius
    /// cannot possibly satisfy the predicate, so the expensive test never runs.
    pub fn distance(&self, other: &Bbox) -> f64 {
        let dx = (other.xmin - self.xmax).max(self.xmin - other.xmax).max(0.0);
        let dy = (other.ymin - self.ymax).max(self.ymin - other.ymax).max(0.0);
        (dx * dx + dy * dy).sqrt()
    }

    /// The box as a closed 5-point counter-clockwise ring.
    pub fn to_ring(self) -> LineString {
        vec![
            Coord::new(self.xmin, self.ymin),
            Coord::new(self.xmax, self.ymin),
            Coord::new(self.xmax, self.ymax),
            Coord::new(self.xmin, self.ymax),
            Coord::new(self.xmin, self.ymin),
        ]
    }
}

/// A geometry plus the spatial reference system its coordinates are stated in.
///
/// SRID 0 means "unknown", matching PostGIS: it is not an assertion of WGS 84, and
/// the geodesic functions say so rather than assuming.
#[derive(Debug, Clone, PartialEq)]
pub struct Geom {
    /// The EPSG code, or 0 for unknown.
    pub srid: i32,
    /// True when coordinates carry a meaningful `z`.
    pub has_z: bool,
    /// The geometry itself.
    pub geometry: Geometry,
}

impl Geom {
    /// A 2D geometry with an unknown SRID.
    pub fn new(geometry: Geometry) -> Self {
        Geom {
            srid: 0,
            has_z: false,
            geometry,
        }
    }

    /// This geometry relabelled with `srid`. Coordinates are unchanged — this is
    /// `ST_SetSRID`, the assertion, not `ST_Transform`, the conversion.
    pub fn with_srid(mut self, srid: i32) -> Self {
        self.srid = srid;
        self
    }

    /// The OGC type code.
    pub fn geom_type(&self) -> GeomType {
        match &self.geometry {
            Geometry::Point(_) => GeomType::Point,
            Geometry::LineString(_) => GeomType::LineString,
            Geometry::Polygon(_) => GeomType::Polygon,
            Geometry::MultiPoint(_) => GeomType::MultiPoint,
            Geometry::MultiLineString(_) => GeomType::MultiLineString,
            Geometry::MultiPolygon(_) => GeomType::MultiPolygon,
            Geometry::GeometryCollection(_) => GeomType::GeometryCollection,
        }
    }

    /// True when the geometry holds no coordinates at all.
    pub fn is_empty(&self) -> bool {
        self.geometry.is_empty()
    }

    /// Every coordinate, in encoding order.
    pub fn coords(&self) -> Vec<Coord> {
        let mut out = Vec::new();
        self.geometry.collect_coords(&mut out);
        out
    }

    /// The number of coordinates, without materializing them.
    pub fn num_points(&self) -> usize {
        self.geometry.num_points()
    }

    /// The axis-aligned bounds, or `None` for an empty geometry.
    pub fn bbox(&self) -> Option<Bbox> {
        let mut out: Option<Bbox> = None;
        self.geometry.fold_bbox(&mut out);
        out
    }
}

impl Geometry {
    /// True when the geometry holds no coordinates at all.
    ///
    /// A collection of empty geometries is empty: emptiness is about coordinates, not
    /// about member count, which is what makes `ST_IsEmpty` agree with PostGIS on
    /// `GEOMETRYCOLLECTION(POINT EMPTY)`.
    pub fn is_empty(&self) -> bool {
        match self {
            Geometry::Point(p) => p.is_none(),
            Geometry::LineString(l) => l.is_empty(),
            Geometry::Polygon(p) => p.exterior.is_empty(),
            Geometry::MultiPoint(ps) => ps.iter().all(|p| p.is_none()),
            Geometry::MultiLineString(ls) => ls.iter().all(|l| l.is_empty()),
            Geometry::MultiPolygon(ps) => ps.iter().all(|p| p.exterior.is_empty()),
            Geometry::GeometryCollection(gs) => gs.iter().all(|g| g.is_empty()),
        }
    }

    /// Push every coordinate into `out`, in encoding order.
    pub fn collect_coords(&self, out: &mut Vec<Coord>) {
        match self {
            Geometry::Point(p) => out.extend(p.iter().copied()),
            Geometry::LineString(l) => out.extend_from_slice(l),
            Geometry::Polygon(p) => {
                out.extend_from_slice(&p.exterior);
                for r in &p.interiors {
                    out.extend_from_slice(r);
                }
            }
            Geometry::MultiPoint(ps) => out.extend(ps.iter().filter_map(|p| *p)),
            Geometry::MultiLineString(ls) => ls.iter().for_each(|l| out.extend_from_slice(l)),
            Geometry::MultiPolygon(ps) => {
                for p in ps {
                    Geometry::Polygon(p.clone()).collect_coords(out);
                }
            }
            Geometry::GeometryCollection(gs) => gs.iter().for_each(|g| g.collect_coords(out)),
        }
    }

    /// The coordinate count, without allocating.
    pub fn num_points(&self) -> usize {
        match self {
            Geometry::Point(p) => usize::from(p.is_some()),
            Geometry::LineString(l) => l.len(),
            Geometry::Polygon(p) => {
                p.exterior.len() + p.interiors.iter().map(|r| r.len()).sum::<usize>()
            }
            Geometry::MultiPoint(ps) => ps.iter().filter(|p| p.is_some()).count(),
            Geometry::MultiLineString(ls) => ls.iter().map(|l| l.len()).sum(),
            Geometry::MultiPolygon(ps) => ps
                .iter()
                .map(|p| p.exterior.len() + p.interiors.iter().map(|r| r.len()).sum::<usize>())
                .sum(),
            Geometry::GeometryCollection(gs) => gs.iter().map(|g| g.num_points()).sum(),
        }
    }

    /// Merge this geometry's bounds into `acc`.
    fn fold_bbox(&self, acc: &mut Option<Bbox>) {
        if let Geometry::GeometryCollection(gs) = self {
            gs.iter().for_each(|g| g.fold_bbox(acc));
            return;
        }
        let mut visit = |c: Coord| match acc {
            Some(b) => b.extend(c),
            None => *acc = Some(Bbox::from_coord(c)),
        };
        match self {
            Geometry::Point(p) => p.iter().for_each(|c| visit(*c)),
            Geometry::LineString(l) => l.iter().for_each(|c| visit(*c)),
            Geometry::Polygon(p) => {
                p.exterior.iter().for_each(|c| visit(*c));
                p.interiors.iter().flatten().for_each(|c| visit(*c));
            }
            Geometry::MultiPoint(ps) => ps.iter().flatten().for_each(|c| visit(*c)),
            Geometry::MultiLineString(ls) => ls.iter().flatten().for_each(|c| visit(*c)),
            Geometry::MultiPolygon(ps) => {
                for p in ps {
                    p.exterior.iter().for_each(|c| visit(*c));
                    p.interiors.iter().flatten().for_each(|c| visit(*c));
                }
            }
            Geometry::GeometryCollection(_) => unreachable!("handled above"),
        }
    }

    /// Rewrite every coordinate through `f`, preserving structure.
    ///
    /// The shared spine of every affine transform, projection, and coordinate
    /// normalization in the crate, so those are each one closure rather than one
    /// seven-arm match.
    pub fn map_coords(&self, f: &mut impl FnMut(Coord) -> Coord) -> Geometry {
        fn line(l: &LineString, f: &mut impl FnMut(Coord) -> Coord) -> LineString {
            l.iter().map(|c| f(*c)).collect()
        }
        fn poly(p: &Polygon, f: &mut impl FnMut(Coord) -> Coord) -> Polygon {
            Polygon {
                exterior: line(&p.exterior, f),
                interiors: p.interiors.iter().map(|r| line(r, f)).collect(),
            }
        }
        match self {
            Geometry::Point(p) => Geometry::Point(p.map(&mut *f)),
            Geometry::LineString(l) => Geometry::LineString(line(l, f)),
            Geometry::Polygon(p) => Geometry::Polygon(poly(p, f)),
            Geometry::MultiPoint(ps) => {
                Geometry::MultiPoint(ps.iter().map(|p| p.map(&mut *f)).collect())
            }
            Geometry::MultiLineString(ls) => {
                Geometry::MultiLineString(ls.iter().map(|l| line(l, f)).collect())
            }
            Geometry::MultiPolygon(ps) => {
                Geometry::MultiPolygon(ps.iter().map(|p| poly(p, f)).collect())
            }
            Geometry::GeometryCollection(gs) => {
                Geometry::GeometryCollection(gs.iter().map(|g| g.map_coords(f)).collect())
            }
        }
    }

    /// Every polygon in the geometry, flattening collections.
    ///
    /// Area and point-in-polygon work the same way on a `Polygon`, a `MultiPolygon`
    /// and a collection that happens to contain one, so they iterate this instead of
    /// repeating the flattening.
    pub fn polygons(&self) -> Vec<&Polygon> {
        let mut out = Vec::new();
        self.push_polygons(&mut out);
        out
    }

    fn push_polygons<'a>(&'a self, out: &mut Vec<&'a Polygon>) {
        match self {
            Geometry::Polygon(p) => out.push(p),
            Geometry::MultiPolygon(ps) => out.extend(ps.iter()),
            Geometry::GeometryCollection(gs) => gs.iter().for_each(|g| g.push_polygons(out)),
            _ => {}
        }
    }

    /// Every line chain in the geometry, flattening collections. Polygon rings are
    /// included, because a polygon's boundary is a set of lines and the linear
    /// predicates are defined against it.
    pub fn lines(&self) -> Vec<&LineString> {
        let mut out = Vec::new();
        self.push_lines(&mut out);
        out
    }

    fn push_lines<'a>(&'a self, out: &mut Vec<&'a LineString>) {
        match self {
            Geometry::LineString(l) => out.push(l),
            Geometry::MultiLineString(ls) => out.extend(ls.iter()),
            Geometry::Polygon(p) => {
                out.push(&p.exterior);
                out.extend(p.interiors.iter());
            }
            Geometry::MultiPolygon(ps) => {
                for p in ps {
                    out.push(&p.exterior);
                    out.extend(p.interiors.iter());
                }
            }
            Geometry::GeometryCollection(gs) => gs.iter().for_each(|g| g.push_lines(out)),
            Geometry::Point(_) | Geometry::MultiPoint(_) => {}
        }
    }

    /// Every position in the geometry, flattening collections.
    pub fn points(&self) -> Vec<Coord> {
        let mut out = Vec::new();
        self.push_points(&mut out);
        out
    }

    fn push_points(&self, out: &mut Vec<Coord>) {
        match self {
            Geometry::Point(p) => out.extend(p.iter().copied()),
            Geometry::MultiPoint(ps) => out.extend(ps.iter().flatten().copied()),
            Geometry::GeometryCollection(gs) => gs.iter().for_each(|g| g.push_points(out)),
            _ => {}
        }
    }

    /// The number of top-level members: 1 for a simple geometry, the member count for
    /// a multi-geometry or collection (PostGIS `ST_NumGeometries`).
    pub fn num_geometries(&self) -> usize {
        match self {
            Geometry::MultiPoint(ps) => ps.len(),
            Geometry::MultiLineString(ls) => ls.len(),
            Geometry::MultiPolygon(ps) => ps.len(),
            Geometry::GeometryCollection(gs) => gs.len(),
            _ => 1,
        }
    }

    /// The 1-based `n`-th member, or `None` when out of range. A simple geometry has
    /// exactly one member: itself.
    pub fn geometry_n(&self, n: usize) -> Option<Geometry> {
        if n == 0 {
            return None;
        }
        let i = n - 1;
        match self {
            Geometry::MultiPoint(ps) => ps.get(i).map(|p| Geometry::Point(*p)),
            Geometry::MultiLineString(ls) => ls.get(i).cloned().map(Geometry::LineString),
            Geometry::MultiPolygon(ps) => ps.get(i).cloned().map(Geometry::Polygon),
            Geometry::GeometryCollection(gs) => gs.get(i).cloned(),
            other => (i == 0).then(|| other.clone()),
        }
    }
}

/// True when the ring's first and last coordinates coincide in x and y.
pub fn is_closed(ring: &[Coord]) -> bool {
    match (ring.first(), ring.last()) {
        (Some(a), Some(b)) => a.x == b.x && a.y == b.y,
        _ => false,
    }
}

/// Close `ring` in place if it is not already closed and holds at least one point.
pub fn close_ring(ring: &mut LineString) {
    if let Some(&first) = ring.first() {
        if !is_closed(ring) {
            ring.push(first);
        }
    }
}

/// Twice the signed area of a ring (the shoelace sum).
///
/// Positive is counter-clockwise. Returned undoubled-and-unsigned by `ring_area`; the
/// raw value is what orientation tests want, and halving it first would only cost a
/// division on a quantity that is about to be compared against zero.
pub fn signed_area2(ring: &[Coord]) -> f64 {
    if ring.len() < 3 {
        return 0.0;
    }
    let mut acc = 0.0;
    for w in ring.windows(2) {
        acc += (w[1].x - w[0].x) * (w[1].y + w[0].y);
    }
    // The shoelace above accumulates the *clockwise*-positive trapezoid sum, so negate
    // to make counter-clockwise positive as OGC and GeoJSON both define it.
    -acc
}

/// Normalize a measurement so an empty one is `+0.0` rather than `-0.0`.
///
/// Rust's `Sum` for `f64` uses `-0.0` as its identity — correct for float addition, and
/// the reason `st_area` of a point came out as `-0.0`. That value compares equal to zero
/// and prints as `-0.0`, so it is invisible to a test and visible to a user. Every
/// measurement in this crate is a sum that can be empty, so every one of them normalizes.
pub fn measurement(v: f64) -> f64 {
    // `-0.0 == 0.0` is true, so this maps only the negative zero and leaves NaN alone.
    if v == 0.0 {
        0.0
    } else {
        v
    }
}

/// The unsigned area of a single ring.
pub fn ring_area(ring: &[Coord]) -> f64 {
    signed_area2(ring).abs() / 2.0
}

/// True when the ring winds counter-clockwise (positive signed area).
pub fn is_ccw(ring: &[Coord]) -> bool {
    signed_area2(ring) > 0.0
}

/// Reject a geometry whose coordinates are not all finite.
///
/// Every predicate in the crate is a chain of comparisons, and NaN makes each of them
/// false in both directions — so an unchecked NaN does not raise, it silently reports
/// "outside", "not equal", "does not intersect" for a row that has no answer.
pub fn require_finite(g: &Geom, op: &'static str) -> GeoResult<()> {
    if g.coords().iter().any(|c| c.is_nan()) {
        return Err(GeoError::parse(
            "WKB",
            format!("{op}: geometry contains a NaN coordinate"),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square() -> Polygon {
        Polygon {
            exterior: vec![
                Coord::new(0.0, 0.0),
                Coord::new(4.0, 0.0),
                Coord::new(4.0, 4.0),
                Coord::new(0.0, 4.0),
                Coord::new(0.0, 0.0),
            ],
            interiors: vec![],
        }
    }

    #[test]
    fn ring_area_is_orientation_independent() {
        let ccw = square().exterior;
        let mut cw = ccw.clone();
        cw.reverse();
        assert_eq!(ring_area(&ccw), 16.0);
        assert_eq!(ring_area(&cw), 16.0);
        assert!(is_ccw(&ccw));
        assert!(!is_ccw(&cw));
    }

    #[test]
    fn bbox_covers_every_member_of_a_collection() {
        let g = Geom::new(Geometry::GeometryCollection(vec![
            Geometry::Point(Some(Coord::new(-1.0, -2.0))),
            Geometry::Polygon(square()),
        ]));
        let b = g.bbox().unwrap();
        assert_eq!((b.xmin, b.ymin, b.xmax, b.ymax), (-1.0, -2.0, 4.0, 4.0));
    }

    #[test]
    fn empty_is_about_coordinates_not_members() {
        let g = Geometry::GeometryCollection(vec![Geometry::Point(None)]);
        assert!(g.is_empty());
        assert_eq!(g.num_geometries(), 1);
    }

    #[test]
    fn bbox_distance_is_zero_when_boxes_touch() {
        let a = Bbox {
            xmin: 0.0,
            ymin: 0.0,
            xmax: 1.0,
            ymax: 1.0,
        };
        let b = Bbox {
            xmin: 1.0,
            ymin: 0.0,
            xmax: 2.0,
            ymax: 1.0,
        };
        assert_eq!(a.distance(&b), 0.0);
        assert!(a.intersects(&b));
        let c = Bbox {
            xmin: 4.0,
            ymin: 0.0,
            xmax: 5.0,
            ymax: 1.0,
        };
        assert_eq!(a.distance(&c), 3.0);
        assert!(!a.intersects(&c));
    }

    #[test]
    fn geometry_n_is_one_based_and_simple_geometries_have_one_member() {
        let g = Geometry::MultiPoint(vec![
            Some(Coord::new(0.0, 0.0)),
            Some(Coord::new(1.0, 1.0)),
        ]);
        assert_eq!(g.geometry_n(0), None);
        assert_eq!(g.geometry_n(1), Some(Geometry::Point(Some(Coord::new(0.0, 0.0)))));
        assert_eq!(g.geometry_n(3), None);
        let p = Geometry::Point(Some(Coord::new(2.0, 2.0)));
        assert_eq!(p.geometry_n(1), Some(p.clone()));
    }
}
