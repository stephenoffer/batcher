//! WKT — the human-readable geometry spelling, and the one users type.
//!
//! `ST_GeomFromText('POINT(1 2)')` is how a geometry gets into a query by hand and how
//! it appears in every test fixture, so the parser has to be forgiving about the things
//! humans vary (case, whitespace, `EMPTY`, the `SRID=4326;` prefix PostGIS prints) and
//! strict about the things that change meaning (ordinate counts, ring nesting).
//!
//! The writer is the inverse and is deliberately canonical: uppercase keywords, a
//! single space after each keyword, no trailing zeros. A stable rendering is what makes
//! `ST_AsText` usable as a group key and as a golden-test value.

use std::fmt::Write as _;

use crate::error::{GeoError, GeoResult};
use crate::types::{Coord, GeomType, Geometry, LineString, Polygon};
use crate::Geom;

/// A character cursor over WKT text.
struct Parser<'a> {
    s: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn skip_ws(&mut self) {
        while self.pos < self.s.len() && self.s[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_ws();
        self.s.get(self.pos).copied()
    }

    fn eat(&mut self, c: u8) -> bool {
        if self.peek() == Some(c) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, c: u8) -> GeoResult<()> {
        if self.eat(c) {
            Ok(())
        } else {
            Err(self.err(format!("expected '{}'", c as char)))
        }
    }

    fn err(&self, detail: impl Into<String>) -> GeoError {
        GeoError::parse("WKT", format!("{} at offset {}", detail.into(), self.pos))
    }

    /// The next bare word, uppercased. Empty when the next character is not a letter.
    fn word(&mut self) -> String {
        self.skip_ws();
        let start = self.pos;
        while self
            .s
            .get(self.pos)
            .is_some_and(|c| c.is_ascii_alphabetic() || *c == b'_')
        {
            self.pos += 1;
        }
        String::from_utf8_lossy(&self.s[start..self.pos]).to_uppercase()
    }

    fn number(&mut self) -> GeoResult<f64> {
        self.skip_ws();
        let start = self.pos;
        if matches!(self.s.get(self.pos), Some(b'+' | b'-')) {
            self.pos += 1;
        }
        while self
            .s
            .get(self.pos)
            .is_some_and(|c| c.is_ascii_digit() || *c == b'.')
        {
            self.pos += 1;
        }
        // Scientific notation: coordinates near the poles and in projected metres are
        // routinely printed as `1.2e6`, and rejecting it would fail on ordinary output.
        if matches!(self.s.get(self.pos), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.s.get(self.pos), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            while self.s.get(self.pos).is_some_and(u8::is_ascii_digit) {
                self.pos += 1;
            }
        }
        if start == self.pos {
            return Err(self.err("expected a number"));
        }
        std::str::from_utf8(&self.s[start..self.pos])
            .ok()
            .and_then(|t| t.parse::<f64>().ok())
            .ok_or_else(|| self.err("malformed number"))
    }

    /// One coordinate: two ordinates, plus a third when the geometry is tagged Z.
    ///
    /// A geometry tagged `M` (not `ZM`) also carries a third ordinate, which is the
    /// linear measure and is read and dropped — the crate has no operation on M, and
    /// refusing the input would be worse than ignoring the value.
    fn coord(&mut self, dims: Dims) -> GeoResult<Coord> {
        let x = self.number()?;
        let y = self.number()?;
        let mut z = 0.0;
        if dims.z {
            z = self.number()?;
        }
        if dims.m {
            self.number()?;
        }
        Ok(Coord { x, y, z })
    }

    fn coord_list(&mut self, dims: Dims) -> GeoResult<LineString> {
        self.expect(b'(')?;
        let mut out = Vec::new();
        loop {
            out.push(self.coord(dims)?);
            if !self.eat(b',') {
                break;
            }
        }
        self.expect(b')')?;
        Ok(out)
    }

    /// A parenthesized list of coordinate lists — polygon rings, or the
    /// `MULTIPOINT((1 2),(3 4))` spelling.
    fn ring_list(&mut self, dims: Dims) -> GeoResult<Vec<LineString>> {
        self.expect(b'(')?;
        let mut out = Vec::new();
        loop {
            out.push(self.coord_list(dims)?);
            if !self.eat(b',') {
                break;
            }
        }
        self.expect(b')')?;
        Ok(out)
    }

    /// `MULTIPOINT` accepts both `(1 2, 3 4)` and `((1 2),(3 4))`. Producers disagree
    /// about which to emit — PostGIS writes the parenthesized form, many hand-written
    /// fixtures use the bare one — so both are read.
    fn multipoint_body(&mut self, dims: Dims) -> GeoResult<Vec<Option<Coord>>> {
        self.expect(b'(')?;
        let mut out = Vec::new();
        loop {
            if self.eat(b'(') {
                out.push(Some(self.coord(dims)?));
                self.expect(b')')?;
            } else if self.word_is_empty() {
                out.push(None);
            } else {
                out.push(Some(self.coord(dims)?));
            }
            if !self.eat(b',') {
                break;
            }
        }
        self.expect(b')')?;
        Ok(out)
    }

    /// Consume the keyword `EMPTY` if it is next.
    fn word_is_empty(&mut self) -> bool {
        let save = self.pos;
        if self.word() == "EMPTY" {
            true
        } else {
            self.pos = save;
            false
        }
    }

    fn geometry(&mut self) -> GeoResult<(Geometry, bool)> {
        let kw = self.word();
        let (base, mut dims) = split_dimension_suffix(&kw);
        // The dimension can also be a separate token: `POINT Z (1 2 3)`.
        let save = self.pos;
        match self.word().as_str() {
            "Z" => dims.z = true,
            "M" => dims.m = true,
            "ZM" => {
                dims.z = true;
                dims.m = true;
            }
            _ => self.pos = save,
        }
        let ty = match base {
            "POINT" => GeomType::Point,
            "LINESTRING" | "LINEARRING" => GeomType::LineString,
            "POLYGON" => GeomType::Polygon,
            "MULTIPOINT" => GeomType::MultiPoint,
            "MULTILINESTRING" => GeomType::MultiLineString,
            "MULTIPOLYGON" => GeomType::MultiPolygon,
            "GEOMETRYCOLLECTION" => GeomType::GeometryCollection,
            other => return Err(self.err(format!("unknown geometry keyword {other:?}"))),
        };
        if self.word_is_empty() {
            return Ok((empty_of(ty), dims.z));
        }
        let g = match ty {
            GeomType::Point => {
                self.expect(b'(')?;
                let c = self.coord(dims)?;
                self.expect(b')')?;
                Geometry::Point(Some(c))
            }
            GeomType::LineString => Geometry::LineString(self.coord_list(dims)?),
            GeomType::Polygon => {
                let mut rings = self.ring_list(dims)?.into_iter();
                Geometry::Polygon(Polygon {
                    exterior: rings.next().unwrap_or_default(),
                    interiors: rings.collect(),
                })
            }
            GeomType::MultiPoint => Geometry::MultiPoint(self.multipoint_body(dims)?),
            GeomType::MultiLineString => Geometry::MultiLineString(self.ring_list(dims)?),
            GeomType::MultiPolygon => {
                self.expect(b'(')?;
                let mut out = Vec::new();
                loop {
                    if self.word_is_empty() {
                        out.push(Polygon::default());
                    } else {
                        let mut rings = self.ring_list(dims)?.into_iter();
                        out.push(Polygon {
                            exterior: rings.next().unwrap_or_default(),
                            interiors: rings.collect(),
                        });
                    }
                    if !self.eat(b',') {
                        break;
                    }
                }
                self.expect(b')')?;
                Geometry::MultiPolygon(out)
            }
            GeomType::GeometryCollection => {
                self.expect(b'(')?;
                let mut out = Vec::new();
                loop {
                    let (child, child_z) = self.geometry()?;
                    dims.z |= child_z;
                    out.push(child);
                    if !self.eat(b',') {
                        break;
                    }
                }
                self.expect(b')')?;
                Geometry::GeometryCollection(out)
            }
        };
        Ok((g, dims.z))
    }
}

/// Which optional ordinates each coordinate carries.
#[derive(Debug, Clone, Copy, Default)]
struct Dims {
    z: bool,
    m: bool,
}

/// Split a `POINTZ` / `POINTZM` keyword into its base name and dimension flags.
fn split_dimension_suffix(kw: &str) -> (&str, Dims) {
    for (suffix, dims) in [
        ("ZM", Dims { z: true, m: true }),
        ("Z", Dims { z: true, m: false }),
        ("M", Dims { z: false, m: true }),
    ] {
        if let Some(base) = kw.strip_suffix(suffix) {
            // `MULTIPOINT` ends in `T`, not `M`; only strip when what remains is still
            // a keyword, which every real base name satisfies by being ≥5 characters.
            if base.len() >= 5 && !base.ends_with('N') {
                return (base, dims);
            }
        }
    }
    (kw, Dims::default())
}

fn empty_of(t: GeomType) -> Geometry {
    match t {
        GeomType::Point => Geometry::Point(None),
        GeomType::LineString => Geometry::LineString(Vec::new()),
        GeomType::Polygon => Geometry::Polygon(Polygon::default()),
        GeomType::MultiPoint => Geometry::MultiPoint(Vec::new()),
        GeomType::MultiLineString => Geometry::MultiLineString(Vec::new()),
        GeomType::MultiPolygon => Geometry::MultiPolygon(Vec::new()),
        GeomType::GeometryCollection => Geometry::GeometryCollection(Vec::new()),
    }
}

/// Parse WKT, accepting an optional `SRID=<n>;` prefix (PostGIS EWKT).
pub fn read_wkt(text: &str) -> GeoResult<Geom> {
    let (srid, body) = match text.trim_start().strip_prefix("SRID=").or_else(|| {
        text.trim_start()
            .strip_prefix("srid=")
            .or_else(|| text.trim_start().strip_prefix("Srid="))
    }) {
        Some(rest) => {
            let (num, after) = rest.split_once(';').ok_or_else(|| {
                GeoError::parse("WKT", "SRID prefix is missing its terminating ';'")
            })?;
            let srid: i32 = num
                .trim()
                .parse()
                .map_err(|_| GeoError::parse("WKT", format!("SRID {num:?} is not an integer")))?;
            (srid, after)
        }
        None => (0, text),
    };
    let mut p = Parser {
        s: body.as_bytes(),
        pos: 0,
    };
    let (geometry, has_z) = p.geometry()?;
    p.skip_ws();
    if p.pos != body.len() {
        return Err(p.err("trailing text after the geometry"));
    }
    Ok(Geom {
        srid,
        has_z,
        geometry,
    })
}

/// Render one ordinate the way PostGIS does: shortest round-trippable form, so
/// `1.0` prints as `1` and `0.1` prints as `0.1` rather than `0.10000000000000001`.
fn num(out: &mut String, v: f64) {
    if v == v.trunc() && v.abs() < 1e15 {
        let _ = write!(out, "{}", v as i64);
    } else {
        let _ = write!(out, "{v}");
    }
}

fn coord(out: &mut String, c: Coord, has_z: bool) {
    num(out, c.x);
    out.push(' ');
    num(out, c.y);
    if has_z {
        out.push(' ');
        num(out, c.z);
    }
}

fn coord_list(out: &mut String, cs: &[Coord], has_z: bool) {
    out.push('(');
    for (i, c) in cs.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        coord(out, *c, has_z);
    }
    out.push(')');
}

fn polygon_body(out: &mut String, p: &Polygon, has_z: bool) {
    if p.exterior.is_empty() {
        out.push_str("EMPTY");
        return;
    }
    out.push('(');
    coord_list(out, &p.exterior, has_z);
    for r in &p.interiors {
        out.push_str(", ");
        coord_list(out, r, has_z);
    }
    out.push(')');
}

fn body(out: &mut String, g: &Geometry, has_z: bool) {
    if g.is_empty() && !matches!(g, Geometry::GeometryCollection(_)) {
        out.push_str("EMPTY");
        return;
    }
    match g {
        Geometry::Point(p) => {
            out.push('(');
            coord(out, p.expect("non-empty point"), has_z);
            out.push(')');
        }
        Geometry::LineString(l) => coord_list(out, l, has_z),
        Geometry::Polygon(p) => polygon_body(out, p, has_z),
        Geometry::MultiPoint(ps) => {
            out.push('(');
            for (i, p) in ps.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                match p {
                    Some(c) => {
                        out.push('(');
                        coord(out, *c, has_z);
                        out.push(')');
                    }
                    None => out.push_str("EMPTY"),
                }
            }
            out.push(')');
        }
        Geometry::MultiLineString(ls) => {
            out.push('(');
            for (i, l) in ls.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                coord_list(out, l, has_z);
            }
            out.push(')');
        }
        Geometry::MultiPolygon(ps) => {
            out.push('(');
            for (i, p) in ps.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                polygon_body(out, p, has_z);
            }
            out.push(')');
        }
        Geometry::GeometryCollection(gs) => {
            if gs.is_empty() {
                out.push_str("EMPTY");
                return;
            }
            out.push('(');
            for (i, child) in gs.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_into(out, child, has_z);
            }
            out.push(')');
        }
    }
}

fn write_into(out: &mut String, g: &Geometry, has_z: bool) {
    out.push_str(Geom::new(g.clone()).geom_type().name());
    if has_z {
        out.push_str(" Z");
    }
    // `POINT EMPTY` and `POINT(1 2)`: a space before the keyword, none before the paren,
    // which is the rendering PostGIS and GEOS both produce.
    let mut tail = String::new();
    body(&mut tail, g, has_z);
    if tail.starts_with('(') {
        out.push_str(&tail);
    } else {
        out.push(' ');
        out.push_str(&tail);
    }
}

/// Render canonical WKT.
pub fn write_wkt(g: &Geom) -> String {
    let mut out = String::with_capacity(16 + 24 * g.num_points());
    write_into(&mut out, &g.geometry, g.has_z);
    out
}

/// Render EWKT — WKT with PostGIS's `SRID=<n>;` prefix when an SRID is set.
pub fn write_ewkt(g: &Geom) -> String {
    if g.srid == 0 {
        return write_wkt(g);
    }
    format!("SRID={};{}", g.srid, write_wkt(g))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rt(text: &str) -> String {
        write_wkt(&read_wkt(text).expect(text))
    }

    #[test]
    fn canonical_forms_round_trip_unchanged() {
        for t in [
            "POINT(1 2)",
            "LINESTRING(0 0, 1 1, 2 2)",
            "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))",
            "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 2 1, 2 2, 1 1))",
            "MULTIPOINT((1 2), (3 4))",
            "MULTILINESTRING((0 0, 1 1), (2 2, 3 3))",
            "MULTIPOLYGON(((0 0, 1 0, 1 1, 0 0)))",
            "GEOMETRYCOLLECTION(POINT(1 2), LINESTRING(0 0, 1 1))",
        ] {
            assert_eq!(rt(t), t, "round trip of {t}");
        }
    }

    #[test]
    fn empty_geometries_round_trip() {
        for t in [
            "POINT EMPTY",
            "LINESTRING EMPTY",
            "POLYGON EMPTY",
            "MULTIPOINT EMPTY",
            "GEOMETRYCOLLECTION EMPTY",
        ] {
            assert_eq!(rt(t), t);
        }
    }

    #[test]
    fn human_spellings_are_accepted() {
        // Lowercase, extra whitespace, the bare MULTIPOINT form, and a leading sign.
        let g = read_wkt("  multipoint ( -1 2 , 3 4 )  ").unwrap();
        assert_eq!(write_wkt(&g), "MULTIPOINT((-1 2), (3 4))");
        assert_eq!(rt("point(1.5e2 -2)"), "POINT(150 -2)");
    }

    #[test]
    fn the_srid_prefix_is_read_and_written() {
        let g = read_wkt("SRID=4326;POINT(1 2)").unwrap();
        assert_eq!(g.srid, 4326);
        assert_eq!(write_ewkt(&g), "SRID=4326;POINT(1 2)");
        // Without a SRID the prefix is absent, not `SRID=0;`.
        assert_eq!(write_ewkt(&read_wkt("POINT(1 2)").unwrap()), "POINT(1 2)");
    }

    #[test]
    fn z_is_recognized_both_attached_and_detached() {
        for t in ["POINT Z(1 2 3)", "POINTZ(1 2 3)", "POINT Z (1 2 3)"] {
            let g = read_wkt(t).unwrap();
            assert!(g.has_z, "{t}");
            assert_eq!(write_wkt(&g), "POINT Z(1 2 3)");
        }
    }

    #[test]
    fn an_m_ordinate_is_read_and_dropped() {
        let g = read_wkt("POINT M(1 2 99)").unwrap();
        assert!(!g.has_z);
        assert_eq!(write_wkt(&g), "POINT(1 2)");
    }

    #[test]
    fn multipoint_keyword_is_not_mistaken_for_an_m_suffix() {
        let g = read_wkt("MULTIPOINT(1 2)").unwrap();
        assert_eq!(g.geometry.num_points(), 1);
        assert!(!g.has_z);
    }

    #[test]
    fn malformed_input_names_the_offset() {
        let err = read_wkt("POINT(1)").unwrap_err();
        assert!(format!("{err}").contains("offset"), "{err}");
        assert!(read_wkt("POINT(1 2) EXTRA").is_err());
        assert!(read_wkt("NOTAGEOMETRY(1 2)").is_err());
    }
}
