//! Union and difference of polygons, by noding and edge classification.
//!
//! This exists for `buffer`, which is a union: a point grows into a disc, a chain into
//! the union of a capsule per segment, a polygon into itself plus a band around its
//! boundary. Taking the convex hull of all of that instead — which `buffer` used to do —
//! is exact only for a convex input and silently wrong otherwise: two points 10 apart
//! buffered by 1 became one 23-unit stadium instead of two 3.1-unit discs.
//!
//! The method is the textbook one, kept deliberately small:
//!
//! 1. **Orient** every piece so its interior is on the left of every edge (shells
//!    counter-clockwise, holes clockwise).
//! 2. **Node**: split every edge at every point where it meets another edge, so no two
//!    sub-edges cross. Intersection points are computed once and shared, and positions
//!    closer than a tolerance are merged, so the two sides of a crossing agree exactly
//!    on where it is.
//! 3. **Classify** each sub-edge by its midpoint. It is on the boundary of the union
//!    exactly when no *other* piece contains the midpoint; a sub-edge two pieces share
//!    is kept once when their interiors are on the same side and dropped when they are
//!    on opposite sides (the seam between two abutting pieces).
//! 4. **Trace** the kept sub-edges into rings, turning as far left as possible at a
//!    vertex several rings pass through, so two shapes touching at a point come out as
//!    two rings rather than one figure-eight.
//! 5. **Assemble**: counter-clockwise rings are shells, clockwise ones are holes, and
//!    each hole goes into the smallest shell containing it.
//!
//! Difference (`subject minus pieces`) is the same machinery with a different keep
//! rule, and is what an inward (negative) buffer is.
//!
//! Every step is exact up to the snapping tolerance, which is relative to the input's
//! extent (`1e-10` of it). When tracing cannot close a ring — which the tolerance is
//! there to prevent, and which would mean a lost piece of area — the operation reports
//! failure rather than returning the rings it did manage to close.

use std::collections::HashMap;

use crate::algo::primitive::{cross, point_segment_distance};
use crate::types::{is_ccw, signed_area2, Bbox, Coord, LineString, Polygon};

/// The relative snapping tolerance: positions closer than this fraction of the input's
/// extent are treated as one position.
const REL_TOL: f64 = 1e-10;

/// The union of `pieces`, as disjoint polygons. `None` when a ring could not be traced.
#[must_use]
pub fn union(pieces: &[Polygon]) -> Option<Vec<Polygon>> {
    Overlay::new(None, pieces)?.run()
}

/// `subject` minus the union of `pieces`, as disjoint polygons. `None` when a ring could
/// not be traced.
#[must_use]
pub fn difference(subject: &Polygon, pieces: &[Polygon]) -> Option<Vec<Polygon>> {
    Overlay::new(Some(subject), pieces)?.run()
}

fn sub(a: Coord, b: Coord) -> (f64, f64) {
    (a.x - b.x, a.y - b.y)
}

fn cross2(a: (f64, f64), b: (f64, f64)) -> f64 {
    a.0 * b.1 - a.1 * b.0
}

fn dot2(a: (f64, f64), b: (f64, f64)) -> f64 {
    a.0 * b.0 + a.1 * b.1
}

fn len2(a: (f64, f64)) -> f64 {
    a.0.hypot(a.1)
}

fn dist(a: Coord, b: Coord) -> f64 {
    len2(sub(a, b))
}

/// A polygon oriented so its interior is left of every edge, with an edge index for
/// point location.
struct Piece {
    bbox: Bbox,
    /// Every edge of every ring, as `(from, to)`.
    edges: Vec<(Coord, Coord)>,
    /// Horizontal strips over the bbox; each lists the edges whose y-range meets it.
    strips: Vec<Vec<u32>>,
}

/// Where a position is relative to a piece.
enum Loc {
    Inside,
    Outside,
    /// On the boundary, along an edge running in this direction.
    On((f64, f64)),
}

fn oriented(ring: &LineString, ccw: bool) -> LineString {
    let mut r = ring.clone();
    crate::types::close_ring(&mut r);
    if r.len() >= 4 && is_ccw(&r) != ccw {
        r.reverse();
    }
    r
}

impl Piece {
    fn new(p: &Polygon) -> Option<Self> {
        let shell = oriented(&p.exterior, true);
        if shell.len() < 4 {
            return None;
        }
        let mut bbox = Bbox::from_coord(shell[0]);
        let mut edges = Vec::new();
        for ring in std::iter::once(shell).chain(p.interiors.iter().map(|h| oriented(h, false))) {
            for w in ring.windows(2) {
                bbox.extend(w[0]);
                if w[0].x != w[1].x || w[0].y != w[1].y {
                    edges.push((w[0], w[1]));
                }
            }
        }
        let n = (edges.len() / 4).clamp(1, 4096);
        let mut strips = vec![Vec::new(); n];
        let h = (bbox.ymax - bbox.ymin).max(f64::MIN_POSITIVE);
        let strip = |y: f64| (((y - bbox.ymin) / h * n as f64) as usize).min(n - 1);
        for (i, (a, b)) in edges.iter().enumerate() {
            for s in strips
                .iter_mut()
                .take(strip(a.y.max(b.y)) + 1)
                .skip(strip(a.y.min(b.y)))
            {
                s.push(i as u32);
            }
        }
        Some(Piece {
            bbox,
            edges,
            strips,
        })
    }

    fn locate(&self, p: Coord, tol: f64) -> Loc {
        if p.x < self.bbox.xmin - tol
            || p.x > self.bbox.xmax + tol
            || p.y < self.bbox.ymin - tol
            || p.y > self.bbox.ymax + tol
        {
            return Loc::Outside;
        }
        let n = self.strips.len();
        let h = (self.bbox.ymax - self.bbox.ymin).max(f64::MIN_POSITIVE);
        let lo = (((p.y - tol - self.bbox.ymin) / h * n as f64).max(0.0) as usize).min(n - 1);
        let hi = (((p.y + tol - self.bbox.ymin) / h * n as f64).max(0.0) as usize).min(n - 1);
        for s in &self.strips[lo..=hi] {
            for &i in s {
                let (a, b) = self.edges[i as usize];
                if point_segment_distance(p, a, b) <= tol {
                    return Loc::On(sub(b, a));
                }
            }
        }
        let strip = (((p.y - self.bbox.ymin) / h * n as f64).max(0.0) as usize).min(n - 1);
        let mut inside = false;
        for &i in &self.strips[strip] {
            let (a, b) = self.edges[i as usize];
            if (a.y > p.y) != (b.y > p.y) {
                let x = (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x;
                if p.x < x {
                    inside = !inside;
                }
            }
        }
        if inside {
            Loc::Inside
        } else {
            Loc::Outside
        }
    }
}

/// A uniform grid over the pieces' bounding boxes, for "which pieces could contain p".
struct PieceGrid {
    bbox: Bbox,
    n: usize,
    cells: Vec<Vec<u32>>,
}

impl PieceGrid {
    fn new(pieces: &[Piece], bbox: Bbox) -> Self {
        let n = ((pieces.len() as f64).sqrt().ceil() as usize).clamp(1, 256);
        let mut g = PieceGrid {
            bbox,
            n,
            cells: vec![Vec::new(); n * n],
        };
        for (i, p) in pieces.iter().enumerate() {
            let (x0, y0) = g.cell(p.bbox.xmin, p.bbox.ymin);
            let (x1, y1) = g.cell(p.bbox.xmax, p.bbox.ymax);
            for cx in x0..=x1 {
                for cy in y0..=y1 {
                    g.cells[cy * n + cx].push(i as u32);
                }
            }
        }
        g
    }

    fn cell(&self, x: f64, y: f64) -> (usize, usize) {
        let w = (self.bbox.xmax - self.bbox.xmin).max(f64::MIN_POSITIVE);
        let h = (self.bbox.ymax - self.bbox.ymin).max(f64::MIN_POSITIVE);
        let f = |v: f64, lo: f64, span: f64| {
            (((v - lo) / span * self.n as f64).max(0.0) as usize).min(self.n - 1)
        };
        (f(x, self.bbox.xmin, w), f(y, self.bbox.ymin, h))
    }

    fn candidates(&self, p: Coord) -> &[u32] {
        let (cx, cy) = self.cell(p.x, p.y);
        &self.cells[cy * self.n + cx]
    }
}

/// Positions merged within the tolerance, by a hash grid.
struct Vertices {
    cell: f64,
    map: HashMap<(i64, i64), Vec<usize>>,
    pts: Vec<Coord>,
    tol: f64,
}

impl Vertices {
    fn id(&mut self, c: Coord) -> usize {
        let key = |v: f64| (v / self.cell).floor() as i64;
        let (kx, ky) = (key(c.x), key(c.y));
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(ids) = self.map.get(&(kx + dx, ky + dy)) {
                    for &id in ids {
                        if dist(self.pts[id], c) <= self.tol {
                            return id;
                        }
                    }
                }
            }
        }
        let id = self.pts.len();
        self.pts.push(c);
        self.map.entry((kx, ky)).or_default().push(id);
        id
    }
}

/// Which operand an edge came from: the subject of a difference, or a piece.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Owner {
    Subject,
    Piece(usize),
}

struct Seg {
    a: Coord,
    b: Coord,
    owner: Owner,
    splits: Vec<(f64, Coord)>,
}

impl Seg {
    fn add_split(&mut self, x: Coord, tol: f64) {
        if dist(x, self.a) <= tol || dist(x, self.b) <= tol {
            return;
        }
        if point_segment_distance(x, self.a, self.b) > tol {
            return;
        }
        let d = sub(self.b, self.a);
        let t = dot2(sub(x, self.a), d) / dot2(d, d);
        self.splits.push((t, x));
    }
}

struct Overlay {
    subject: Option<Piece>,
    pieces: Vec<Piece>,
    grid: PieceGrid,
    tol: f64,
}

impl Overlay {
    fn new(subject: Option<&Polygon>, pieces: &[Polygon]) -> Option<Self> {
        let subject = match subject {
            Some(s) => Some(Piece::new(s)?),
            None => None,
        };
        let pieces: Vec<Piece> = pieces.iter().filter_map(Piece::new).collect();
        let mut bbox: Option<Bbox> = subject.as_ref().map(|s| s.bbox);
        for p in &pieces {
            match &mut bbox {
                Some(b) => b.union(p.bbox),
                None => bbox = Some(p.bbox),
            }
        }
        let bbox = bbox?;
        let scale = [bbox.xmin, bbox.xmax, bbox.ymin, bbox.ymax]
            .iter()
            .fold(bbox.xmax - bbox.xmin + bbox.ymax - bbox.ymin, |m, v| {
                m.max(v.abs())
            });
        let tol = scale.max(f64::MIN_POSITIVE) * REL_TOL;
        let grid = PieceGrid::new(&pieces, bbox);
        Some(Overlay {
            subject,
            pieces,
            grid,
            tol,
        })
    }

    fn run(&self) -> Option<Vec<Polygon>> {
        let segs = self.node();
        let mut verts = Vertices {
            cell: 2.0 * self.tol,
            map: HashMap::new(),
            pts: Vec::new(),
            tol: self.tol,
        };
        // Sub-edges as (from, to, owner), after snapping.
        let mut kept: Vec<(usize, usize)> = Vec::new();
        for s in &segs {
            let mut pts: Vec<(f64, Coord)> = s.splits.clone();
            pts.sort_by(|x, y| x.0.total_cmp(&y.0));
            let mut chain = vec![s.a];
            chain.extend(pts.into_iter().map(|(_, c)| c));
            chain.push(s.b);
            let ids: Vec<usize> = chain.iter().map(|c| verts.id(*c)).collect();
            for w in ids.windows(2) {
                if w[0] == w[1] {
                    continue;
                }
                let (a, b) = (verts.pts[w[0]], verts.pts[w[1]]);
                if let Some(reverse) = self.keep(a, b, s.owner) {
                    kept.push(if reverse { (w[1], w[0]) } else { (w[0], w[1]) });
                }
            }
        }
        kept.sort_unstable();
        kept.dedup();
        let rings = trace(&kept, &verts.pts)?;
        Some(assemble(rings, self.tol))
    }

    /// Split every edge where it meets another.
    fn node(&self) -> Vec<Seg> {
        let mut segs: Vec<Seg> = Vec::new();
        let mut push = |piece: &Piece, owner: Owner| {
            for &(a, b) in &piece.edges {
                segs.push(Seg {
                    a,
                    b,
                    owner,
                    splits: Vec::new(),
                });
            }
        };
        if let Some(s) = &self.subject {
            push(s, Owner::Subject);
        }
        for (i, p) in self.pieces.iter().enumerate() {
            push(p, Owner::Piece(i));
        }
        let tol = self.tol;
        let mut order: Vec<usize> = (0..segs.len()).collect();
        let xmin = |s: &Seg| s.a.x.min(s.b.x);
        order.sort_by(|&i, &j| xmin(&segs[i]).total_cmp(&xmin(&segs[j])));
        for oi in 0..order.len() {
            let i = order[oi];
            let (ixmax, iymin, iymax) = {
                let s = &segs[i];
                (s.a.x.max(s.b.x), s.a.y.min(s.b.y), s.a.y.max(s.b.y))
            };
            for &j in &order[oi + 1..] {
                if xmin(&segs[j]) > ixmax + tol {
                    break;
                }
                let t = &segs[j];
                if t.a.y.min(t.b.y) > iymax + tol || t.a.y.max(t.b.y) < iymin - tol {
                    continue;
                }
                if let Some(points) = meet(&segs[i], &segs[j], tol) {
                    for (on_i, x) in points {
                        if on_i {
                            segs[i].add_split(x, tol);
                        } else {
                            segs[j].add_split(x, tol);
                        }
                    }
                }
            }
        }
        segs
    }

    /// Whether the sub-edge `a -> b` of `owner` is on the result's boundary, and if so
    /// whether it must be reversed to keep the result's interior on its left.
    fn keep(&self, a: Coord, b: Coord, owner: Owner) -> Option<bool> {
        let m = Coord::new(f64::midpoint(a.x, b.x), f64::midpoint(a.y, b.y));
        let dir = sub(b, a);
        let tol = self.tol;
        // Relative to the union of the pieces, excluding the edge's own piece.
        for &q in self.grid.candidates(m) {
            let q = q as usize;
            if owner == Owner::Piece(q) {
                continue;
            }
            match locate_along(&self.pieces[q], m, dir, tol) {
                Loc::Inside => return None,
                Loc::On(d) => {
                    let same_side = dot2(d, dir) > 0.0;
                    match owner {
                        // The piece's interior is on the subject's side here, so this
                        // stretch of the subject is removed. Opposite, the piece lies
                        // outside the subject and the subject's edge stands.
                        Owner::Subject if same_side => return None,
                        Owner::Subject => {}
                        // Two pieces sharing a stretch: interiors on opposite sides is
                        // a seam inside the union; on the same side, keep one copy (the
                        // lowest-numbered piece's).
                        Owner::Piece(p) if !same_side || q < p => return None,
                        Owner::Piece(_) => {}
                    }
                }
                Loc::Outside => {}
            }
        }
        match (&self.subject, owner) {
            (None, _) | (Some(_), Owner::Subject) => Some(false),
            // A piece edge becomes result boundary where it lies strictly inside the
            // subject, reversed: the result is on the piece's *outside*.
            (Some(s), Owner::Piece(_)) => match locate_along(s, m, dir, tol) {
                Loc::Inside => Some(true),
                _ => None,
            },
        }
    }
}

/// Locate `m` against a piece, treating it as on the boundary only when it lies along a
/// piece edge running (anti)parallel to `dir`. A sub-edge that merely crosses a boundary
/// near its midpoint is decided by the ray test instead.
fn locate_along(piece: &Piece, m: Coord, dir: (f64, f64), tol: f64) -> Loc {
    match piece.locate(m, tol) {
        Loc::On(d) => {
            let s = cross2(d, dir).abs() / (len2(d) * len2(dir)).max(f64::MIN_POSITIVE);
            if s < 1e-3 {
                Loc::On(d)
            } else {
                // Nudge off the boundary to the left of `dir` and ask again.
                let n = len2(dir).max(f64::MIN_POSITIVE);
                let off = Coord::new(m.x - dir.1 / n * 4.0 * tol, m.y + dir.0 / n * 4.0 * tol);
                match piece.locate(off, tol) {
                    Loc::Inside => Loc::Inside,
                    _ => Loc::Outside,
                }
            }
        }
        other => other,
    }
}

/// Where segments `s` and `t` meet, as `(belongs_to_s, point)` split requests.
fn meet(s: &Seg, t: &Seg, tol: f64) -> Option<Vec<(bool, Coord)>> {
    let r = sub(s.b, s.a);
    let v = sub(t.b, t.a);
    let (lr, lv) = (len2(r), len2(v));
    if lr == 0.0 || lv == 0.0 {
        return None;
    }
    let qp = sub(t.a, s.a);
    let rxv = cross2(r, v);
    if rxv.abs() <= 1e-12 * lr * lv {
        // Parallel: only a collinear overlap produces splits, at the other's endpoints.
        if cross2(qp, r).abs() / lr > tol {
            return None;
        }
        return Some(vec![(true, t.a), (true, t.b), (false, s.a), (false, s.b)]);
    }
    let tt = cross2(qp, v) / rxv;
    let uu = cross2(qp, r) / rxv;
    let (et, eu) = (tol / lr, tol / lv);
    if tt < -et || tt > 1.0 + et || uu < -eu || uu > 1.0 + eu {
        return None;
    }
    let mut x = Coord::new(s.a.x + r.0 * tt, s.a.y + r.1 * tt);
    // Prefer an existing vertex: a crossing within tolerance of an endpoint *is* that
    // endpoint, and computing a second, slightly different position for it is exactly
    // the disagreement snapping is there to prevent.
    if let Some(e) = [s.a, s.b, t.a, t.b]
        .into_iter()
        .find(|e| dist(*e, x) <= tol)
    {
        x = e;
    }
    Some(vec![(true, x), (false, x)])
}

/// Link directed sub-edges into closed rings.
fn trace(edges: &[(usize, usize)], pts: &[Coord]) -> Option<Vec<LineString>> {
    let mut out_of: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &(a, _)) in edges.iter().enumerate() {
        out_of.entry(a).or_default().push(i);
    }
    let mut used = vec![false; edges.len()];
    let mut rings = Vec::new();
    for start in 0..edges.len() {
        if used[start] {
            continue;
        }
        let origin = edges[start].0;
        let mut ring = vec![pts[origin]];
        let mut cur = start;
        loop {
            used[cur] = true;
            let (a, b) = edges[cur];
            ring.push(pts[b]);
            if b == origin {
                break;
            }
            let din = sub(pts[b], pts[a]);
            // Turn as far left as possible, so rings touching at a point separate. A
            // straight U-turn ranks last: it can only be a degenerate spike.
            let next = out_of
                .get(&b)?
                .iter()
                .copied()
                .filter(|&e| !used[e])
                .max_by(|&x, &y| {
                    let ang = |e: usize| {
                        let d = sub(pts[edges[e].1], pts[b]);
                        let t = cross2(din, d).atan2(dot2(din, d));
                        if t >= std::f64::consts::PI {
                            -t
                        } else {
                            t
                        }
                    };
                    ang(x).total_cmp(&ang(y))
                })?;
            cur = next;
        }
        rings.push(ring);
    }
    Some(rings)
}

/// Drop vertices that lie on the straight line between their neighbours, and rotate the
/// ring to a canonical starting vertex.
fn simplify_collinear(ring: &LineString, tol: f64) -> LineString {
    let n = ring.len() - 1; // closed: last == first
    let pts = &ring[..n];
    let mut out: Vec<Coord> = Vec::with_capacity(n + 1);
    for i in 0..n {
        let prev = pts[(i + n - 1) % n];
        let next = pts[(i + 1) % n];
        let c = pts[i];
        let span = dist(prev, next);
        let straight = cross(prev, c, next).abs() <= tol * span.max(tol)
            && dot2(sub(c, prev), sub(next, c)) > 0.0;
        if !straight {
            out.push(c);
        }
    }
    // Start at the lowest vertex (by x, then y), so the ring a union or difference
    // returns does not depend on which edge tracing happened to begin from.
    if let Some(start) = (0..out.len()).min_by(|&i, &j| {
        out[i]
            .x
            .total_cmp(&out[j].x)
            .then(out[i].y.total_cmp(&out[j].y))
    }) {
        out.rotate_left(start);
    }
    if let Some(&f) = out.first() {
        out.push(f);
    }
    out
}

/// Sort traced rings into shells and holes, and holes into shells.
fn assemble(rings: Vec<LineString>, tol: f64) -> Vec<Polygon> {
    let mut shells: Vec<(f64, Polygon)> = Vec::new();
    let mut holes: Vec<LineString> = Vec::new();
    for r in rings {
        let r = simplify_collinear(&r, tol);
        if r.len() < 4 {
            continue;
        }
        let a2 = signed_area2(&r);
        let perimeter: f64 = r.windows(2).map(|w| dist(w[0], w[1])).sum();
        if a2.abs() <= tol * perimeter {
            continue; // a sliver of rounding, not area
        }
        if a2 > 0.0 {
            shells.push((
                a2,
                Polygon {
                    exterior: r,
                    interiors: Vec::new(),
                },
            ));
        } else {
            holes.push(r);
        }
    }
    // Smallest shell first, so the first one containing a hole is the tightest.
    shells.sort_by(|x, y| x.0.total_cmp(&y.0));
    for h in holes {
        // An edge midpoint, not a vertex: a hole may touch its shell at a vertex.
        let probe = Coord::new(f64::midpoint(h[0].x, h[1].x), f64::midpoint(h[0].y, h[1].y));
        if let Some((_, shell)) = shells.iter_mut().find(|(_, s)| {
            crate::algo::primitive::point_in_ring(probe, &s.exterior)
                == crate::algo::primitive::PointRing::Inside
        }) {
            shell.interiors.push(h);
        }
    }
    shells.into_iter().map(|(_, p)| p).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algo::measure::polygon_area;

    fn square(x0: f64, y0: f64, s: f64) -> Polygon {
        Polygon {
            exterior: vec![
                Coord::new(x0, y0),
                Coord::new(x0 + s, y0),
                Coord::new(x0 + s, y0 + s),
                Coord::new(x0, y0 + s),
                Coord::new(x0, y0),
            ],
            interiors: Vec::new(),
        }
    }

    fn total(ps: &[Polygon]) -> f64 {
        ps.iter().map(polygon_area).sum()
    }

    #[test]
    fn overlapping_squares_union_to_their_combined_area() {
        let u = union(&[square(0.0, 0.0, 2.0), square(1.0, 1.0, 2.0)]).unwrap();
        assert_eq!(u.len(), 1);
        assert!((total(&u) - 7.0).abs() < 1e-9, "{}", total(&u));
    }

    #[test]
    fn disjoint_and_corner_touching_squares_stay_separate() {
        let u = union(&[square(0.0, 0.0, 1.0), square(5.0, 5.0, 1.0)]).unwrap();
        assert_eq!(u.len(), 2);
        let u = union(&[square(0.0, 0.0, 1.0), square(1.0, 1.0, 1.0)]).unwrap();
        assert_eq!(u.len(), 2, "touching at a point is two polygons");
        assert!((total(&u) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn edge_sharing_squares_merge_and_the_seam_disappears() {
        let u = union(&[square(0.0, 0.0, 1.0), square(1.0, 0.0, 1.0)]).unwrap();
        assert_eq!(u.len(), 1);
        assert_eq!(u[0].exterior.len(), 5, "{:?}", u[0].exterior);
        assert!((total(&u) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn a_ring_of_squares_encloses_a_hole() {
        let u = union(&[square(0.0, 0.0, 3.0), square(0.0, 0.0, 1.0)]).unwrap();
        assert!((total(&u) - 9.0).abs() < 1e-12);
        let frame = union(&[
            Polygon {
                exterior: vec![
                    Coord::new(0.0, 0.0),
                    Coord::new(3.0, 0.0),
                    Coord::new(3.0, 1.0),
                    Coord::new(0.0, 1.0),
                    Coord::new(0.0, 0.0),
                ],
                interiors: Vec::new(),
            },
            square(2.0, 0.0, 1.0),
            Polygon {
                exterior: vec![
                    Coord::new(2.0, 0.0),
                    Coord::new(3.0, 0.0),
                    Coord::new(3.0, 3.0),
                    Coord::new(2.0, 3.0),
                    Coord::new(2.0, 0.0),
                ],
                interiors: Vec::new(),
            },
            Polygon {
                exterior: vec![
                    Coord::new(0.0, 2.0),
                    Coord::new(3.0, 2.0),
                    Coord::new(3.0, 3.0),
                    Coord::new(0.0, 3.0),
                    Coord::new(0.0, 2.0),
                ],
                interiors: Vec::new(),
            },
            Polygon {
                exterior: vec![
                    Coord::new(0.0, 0.0),
                    Coord::new(1.0, 0.0),
                    Coord::new(1.0, 3.0),
                    Coord::new(0.0, 3.0),
                    Coord::new(0.0, 0.0),
                ],
                interiors: Vec::new(),
            },
        ])
        .unwrap();
        assert_eq!(frame.len(), 1);
        assert_eq!(frame[0].interiors.len(), 1, "{frame:?}");
        assert!((total(&frame) - 8.0).abs() < 1e-12, "{}", total(&frame));
    }

    #[test]
    fn difference_removes_the_covered_part() {
        // [0,4]^2 minus [2,6]x[-1,3]: the left half (8) plus the strip [2,4]x[3,4] (2).
        let d = difference(&square(0.0, 0.0, 4.0), &[square(2.0, -1.0, 4.0)]).unwrap();
        assert_eq!(d.len(), 1);
        assert!((total(&d) - 10.0).abs() < 1e-12, "{}", total(&d));
        let d = difference(&square(0.0, 0.0, 4.0), &[square(1.0, 1.0, 2.0)]).unwrap();
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].interiors.len(), 1);
        assert!((total(&d) - 12.0).abs() < 1e-12);
        let d = difference(&square(0.0, 0.0, 1.0), &[square(-1.0, -1.0, 3.0)]).unwrap();
        assert!(d.is_empty());
    }
}
