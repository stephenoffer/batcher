//! WKB — the binary encoding a geometry column is actually stored in.
//!
//! Batcher stores geometry as WKB in an Arrow `Binary` column. That is not an
//! arbitrary pick: it is what GeoParquet, PostGIS, GeoPackage, Sedona and DuckDB's
//! spatial extension all write, so a geometry column round-trips through any of them
//! without a conversion pass, and the Arrow-is-the-only-columnar-contract invariant
//! holds without inventing a geometry type.
//!
//! Three dialects exist in the wild and this reader accepts all three, because a
//! reader that accepted only one would reject roughly half of real files:
//!
//! * **OGC WKB** — byte order, then a 1..=7 type code.
//! * **ISO WKB** — the same, with `+1000` on the type code for Z, `+2000` for M,
//!   `+3000` for both. What GeoParquet and GDAL write for 3D.
//! * **EWKB** — PostGIS's, which flags Z/M/SRID in the *high* bits of the type code
//!   and appends a 4-byte SRID. What a `bytea` column out of PostGIS holds.
//!
//! The writer emits OGC/ISO WKB, upgrading to EWKB only when an SRID must be carried,
//! so output stays maximally portable. Endianness is little on write (every consumer
//! reads both; nothing benefits from big) and either on read.
//!
//! M values are parsed and discarded rather than rejected. A linear-referencing M is
//! meaningful to the producer and meaningless to every operation in this crate, and
//! erroring on it would make an ordinary PostGIS export unreadable.

use crate::error::{GeoError, GeoResult};
use crate::types::{Coord, GeomType, Geometry, LineString, Polygon};
use crate::Geom;

/// EWKB high bit marking a Z dimension.
const EWKB_Z: u32 = 0x8000_0000;
/// EWKB high bit marking an M dimension.
const EWKB_M: u32 = 0x4000_0000;
/// EWKB high bit marking that a 4-byte SRID follows the type code.
const EWKB_SRID: u32 = 0x2000_0000;

/// A cursor over a WKB byte stream that knows its own endianness.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
    little: bool,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Reader {
            buf,
            pos: 0,
            little: true,
        }
    }

    fn take(&mut self, n: usize) -> GeoResult<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or_else(|| {
            GeoError::parse("WKB", "length overflow while reading geometry".to_string())
        })?;
        if end > self.buf.len() {
            return Err(GeoError::parse(
                "WKB",
                format!(
                    "truncated: wanted {n} bytes at offset {} of {}",
                    self.pos,
                    self.buf.len()
                ),
            ));
        }
        let out = &self.buf[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn u8(&mut self) -> GeoResult<u8> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> GeoResult<u32> {
        let b: [u8; 4] = self.take(4)?.try_into().expect("take(4) yields 4 bytes");
        Ok(if self.little {
            u32::from_le_bytes(b)
        } else {
            u32::from_be_bytes(b)
        })
    }

    fn f64(&mut self) -> GeoResult<f64> {
        let b: [u8; 8] = self.take(8)?.try_into().expect("take(8) yields 8 bytes");
        Ok(if self.little {
            f64::from_le_bytes(b)
        } else {
            f64::from_be_bytes(b)
        })
    }

    /// A geometry header: byte order, type code, optional SRID.
    ///
    /// Returns the OGC type plus the dimension flags, which the caller needs to know
    /// how many ordinates each coordinate carries. Nested geometries re-read a full
    /// header (WKB is self-describing at every level), which is why this is a method
    /// rather than something the top-level call does once.
    fn header(&mut self) -> GeoResult<Header> {
        self.little = match self.u8()? {
            1 => true,
            0 => false,
            other => {
                return Err(GeoError::parse(
                    "WKB",
                    format!("byte-order flag must be 0 or 1, got {other}"),
                ))
            }
        };
        let raw = self.u32()?;
        let has_srid = raw & EWKB_SRID != 0;
        // EWKB puts Z/M in the high bits; ISO puts them in the thousands digit. Strip
        // the EWKB flags first so the remainder can be read as an ISO code.
        let mut code = raw & !(EWKB_Z | EWKB_M | EWKB_SRID);
        let mut has_z = raw & EWKB_Z != 0;
        let mut has_m = raw & EWKB_M != 0;
        match code / 1000 {
            0 => {}
            1 => has_z = true,
            2 => has_m = true,
            3 => {
                has_z = true;
                has_m = true;
            }
            other => {
                return Err(GeoError::parse(
                    "WKB",
                    format!("unknown ISO dimension prefix {other} in type code {code}"),
                ))
            }
        }
        code %= 1000;
        let srid = if has_srid { self.u32()? as i32 } else { 0 };
        Ok(Header {
            geom_type: GeomType::from_code(code)?,
            has_z,
            has_m,
            srid,
        })
    }

    fn coord(&mut self, h: &Header) -> GeoResult<Coord> {
        let x = self.f64()?;
        let y = self.f64()?;
        let z = if h.has_z { self.f64()? } else { 0.0 };
        if h.has_m {
            self.f64()?;
        }
        Ok(Coord { x, y, z })
    }

    /// A length-prefixed run of coordinates.
    ///
    /// The count is bounds-checked against the *remaining* buffer before allocating.
    /// Without that, a corrupt 4-byte count reserves gigabytes for a 40-byte input —
    /// the standard WKB decompression bomb, and the reason this is one helper rather
    /// than a `with_capacity` at each of the five call sites.
    fn coord_run(&mut self, h: &Header) -> GeoResult<LineString> {
        let n = self.checked_count(h.coord_width())?;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.coord(h)?);
        }
        Ok(out)
    }

    /// A length prefix, rejected when the buffer cannot hold that many `stride`-byte
    /// elements.
    fn checked_count(&mut self, stride: usize) -> GeoResult<usize> {
        let n = self.u32()? as usize;
        let remaining = self.buf.len() - self.pos;
        if stride > 0 && n.saturating_mul(stride) > remaining {
            return Err(GeoError::parse(
                "WKB",
                format!("element count {n} exceeds the {remaining} bytes remaining"),
            ));
        }
        Ok(n)
    }

    fn polygon(&mut self, h: &Header) -> GeoResult<Polygon> {
        // A ring costs at least its own 4-byte count, so that is the stride bound.
        let n = self.checked_count(4)?;
        let mut rings = Vec::with_capacity(n);
        for _ in 0..n {
            rings.push(self.coord_run(h)?);
        }
        let mut it = rings.into_iter();
        Ok(Polygon {
            exterior: it.next().unwrap_or_default(),
            interiors: it.collect(),
        })
    }

    fn geometry(&mut self, h: &Header) -> GeoResult<Geometry> {
        Ok(match h.geom_type {
            GeomType::Point => {
                let c = self.coord(h)?;
                // OGC has no POINT EMPTY encoding; every producer writes all-NaN
                // ordinates for one, and reading that back as a real point at NaN
                // would poison every predicate downstream.
                Geometry::Point((!c.x.is_nan() || !c.y.is_nan()).then_some(c))
            }
            GeomType::LineString => Geometry::LineString(self.coord_run(h)?),
            GeomType::Polygon => Geometry::Polygon(self.polygon(h)?),
            GeomType::MultiPoint => {
                let n = self.checked_count(WKB_POINT_MIN)?;
                let mut out = Vec::with_capacity(n);
                for _ in 0..n {
                    let child = self.header()?;
                    match self.geometry(&child)? {
                        Geometry::Point(p) => out.push(p),
                        other => return Err(member_mismatch("MULTIPOINT", &other)),
                    }
                }
                Geometry::MultiPoint(out)
            }
            GeomType::MultiLineString => {
                let n = self.checked_count(WKB_HEADER_MIN)?;
                let mut out = Vec::with_capacity(n);
                for _ in 0..n {
                    let child = self.header()?;
                    match self.geometry(&child)? {
                        Geometry::LineString(l) => out.push(l),
                        other => return Err(member_mismatch("MULTILINESTRING", &other)),
                    }
                }
                Geometry::MultiLineString(out)
            }
            GeomType::MultiPolygon => {
                let n = self.checked_count(WKB_HEADER_MIN)?;
                let mut out = Vec::with_capacity(n);
                for _ in 0..n {
                    let child = self.header()?;
                    match self.geometry(&child)? {
                        Geometry::Polygon(p) => out.push(p),
                        other => return Err(member_mismatch("MULTIPOLYGON", &other)),
                    }
                }
                Geometry::MultiPolygon(out)
            }
            GeomType::GeometryCollection => {
                let n = self.checked_count(WKB_HEADER_MIN)?;
                let mut out = Vec::with_capacity(n);
                for _ in 0..n {
                    let child = self.header()?;
                    out.push(self.geometry(&child)?);
                }
                Geometry::GeometryCollection(out)
            }
        })
    }
}

/// The smallest byte count a nested WKB geometry can occupy: order flag + type code.
const WKB_HEADER_MIN: usize = 5;
/// The smallest byte count a nested 2D WKB point can occupy.
const WKB_POINT_MIN: usize = WKB_HEADER_MIN + 16;

fn member_mismatch(parent: &'static str, got: &Geometry) -> GeoError {
    GeoError::parse(
        "WKB",
        format!(
            "{parent} member has the wrong type ({})",
            crate::Geom::new(got.clone()).geom_type().name()
        ),
    )
}

/// A decoded WKB geometry header.
struct Header {
    geom_type: GeomType,
    has_z: bool,
    has_m: bool,
    srid: i32,
}

impl Header {
    /// Bytes per coordinate under this header's dimensionality.
    fn coord_width(&self) -> usize {
        8 * (2 + usize::from(self.has_z) + usize::from(self.has_m))
    }
}

/// Parse a WKB / ISO WKB / EWKB byte string into a geometry.
pub fn read_wkb(bytes: &[u8]) -> GeoResult<Geom> {
    let mut r = Reader::new(bytes);
    let h = r.header()?;
    let geometry = r.geometry(&h)?;
    // Trailing bytes mean the producer and this reader disagree about the encoding,
    // which is exactly the case where silently accepting the prefix yields a geometry
    // that is wrong rather than absent.
    if r.pos != bytes.len() {
        return Err(GeoError::parse(
            "WKB",
            format!("{} trailing bytes after the geometry", bytes.len() - r.pos),
        ));
    }
    Ok(Geom {
        srid: h.srid,
        has_z: h.has_z,
        geometry,
    })
}

/// A little-endian WKB writer.
struct Writer {
    out: Vec<u8>,
    has_z: bool,
}

impl Writer {
    fn header(&mut self, t: GeomType, srid: Option<i32>) {
        self.out.push(1); // little-endian
        let mut code = t as u32;
        if self.has_z {
            // ISO spelling by default; EWKB's high bit only when an SRID rides along,
            // since the two flag schemes cannot both be set on one type code.
            if srid.is_some() {
                code |= EWKB_Z;
            } else {
                code += 1000;
            }
        }
        if let Some(s) = srid {
            code |= EWKB_SRID;
            self.out.extend_from_slice(&code.to_le_bytes());
            self.out.extend_from_slice(&(s as u32).to_le_bytes());
        } else {
            self.out.extend_from_slice(&code.to_le_bytes());
        }
    }

    fn coord(&mut self, c: Coord) {
        self.out.extend_from_slice(&c.x.to_le_bytes());
        self.out.extend_from_slice(&c.y.to_le_bytes());
        if self.has_z {
            self.out.extend_from_slice(&c.z.to_le_bytes());
        }
    }

    fn count(&mut self, n: usize) {
        self.out.extend_from_slice(&(n as u32).to_le_bytes());
    }

    fn coord_run(&mut self, cs: &[Coord]) {
        self.count(cs.len());
        for c in cs {
            self.coord(*c);
        }
    }

    fn polygon_body(&mut self, p: &Polygon) {
        if p.exterior.is_empty() {
            self.count(0);
            return;
        }
        self.count(1 + p.interiors.len());
        self.coord_run(&p.exterior);
        for r in &p.interiors {
            self.coord_run(r);
        }
    }

    /// Write `g`, tagging the SRID only on the outermost geometry (EWKB's rule).
    fn geometry(&mut self, g: &Geometry, srid: Option<i32>) {
        match g {
            Geometry::Point(p) => {
                self.header(GeomType::Point, srid);
                // POINT EMPTY has no OGC encoding; all-NaN ordinates is the universal
                // convention, and it is what `read_wkb` maps back to `Point(None)`.
                self.coord(p.unwrap_or(Coord {
                    x: f64::NAN,
                    y: f64::NAN,
                    z: f64::NAN,
                }));
            }
            Geometry::LineString(l) => {
                self.header(GeomType::LineString, srid);
                self.coord_run(l);
            }
            Geometry::Polygon(p) => {
                self.header(GeomType::Polygon, srid);
                self.polygon_body(p);
            }
            Geometry::MultiPoint(ps) => {
                self.header(GeomType::MultiPoint, srid);
                self.count(ps.len());
                for p in ps {
                    self.geometry(&Geometry::Point(*p), None);
                }
            }
            Geometry::MultiLineString(ls) => {
                self.header(GeomType::MultiLineString, srid);
                self.count(ls.len());
                for l in ls {
                    self.header(GeomType::LineString, None);
                    self.coord_run(l);
                }
            }
            Geometry::MultiPolygon(ps) => {
                self.header(GeomType::MultiPolygon, srid);
                self.count(ps.len());
                for p in ps {
                    self.header(GeomType::Polygon, None);
                    self.polygon_body(p);
                }
            }
            Geometry::GeometryCollection(gs) => {
                self.header(GeomType::GeometryCollection, srid);
                self.count(gs.len());
                for child in gs {
                    self.geometry(child, None);
                }
            }
        }
    }
}

/// Serialize to little-endian WKB, without an SRID (the portable spelling).
pub fn write_wkb(g: &Geom) -> Vec<u8> {
    let mut w = Writer {
        out: Vec::with_capacity(32 + 16 * g.num_points()),
        has_z: g.has_z,
    };
    w.geometry(&g.geometry, None);
    w.out
}

/// Serialize to little-endian EWKB, carrying the SRID when one is set.
pub fn write_ewkb(g: &Geom) -> Vec<u8> {
    let mut w = Writer {
        out: Vec::with_capacity(36 + 16 * g.num_points()),
        has_z: g.has_z,
    };
    w.geometry(&g.geometry, (g.srid != 0).then_some(g.srid));
    w.out
}

/// Lowercase hex of the EWKB encoding — the spelling PostGIS's text protocol uses,
/// and the one a geometry survives being pasted into a SQL client as.
pub fn write_hex_wkb(g: &Geom) -> String {
    let bytes = write_ewkb(g);
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(char::from_digit((b >> 4) as u32, 16).expect("nibble is < 16"));
        s.push(char::from_digit((b & 0xf) as u32, 16).expect("nibble is < 16"));
    }
    s
}

/// Parse hex-encoded WKB, accepting either case and an optional `0x` prefix.
pub fn read_hex_wkb(s: &str) -> GeoResult<Geom> {
    let s = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")).unwrap_or(s);
    if s.len() % 2 != 0 {
        return Err(GeoError::parse("WKB", "hex string has an odd length"));
    }
    let mut bytes = Vec::with_capacity(s.len() / 2);
    let raw = s.as_bytes();
    for pair in raw.chunks_exact(2) {
        let hi = (pair[0] as char)
            .to_digit(16)
            .ok_or_else(|| GeoError::parse("WKB", "non-hex character in hex WKB"))?;
        let lo = (pair[1] as char)
            .to_digit(16)
            .ok_or_else(|| GeoError::parse("WKB", "non-hex character in hex WKB"))?;
        bytes.push(((hi << 4) | lo) as u8);
    }
    read_wkb(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Coord;

    fn roundtrip(g: Geom) {
        let back = read_wkb(&write_wkb(&g)).expect("round trip");
        assert_eq!(back.geometry, g.geometry);
        assert_eq!(back.has_z, g.has_z);
    }

    #[test]
    fn every_geometry_type_round_trips() {
        let pt = Coord::new(1.5, -2.5);
        roundtrip(Geom::new(Geometry::Point(Some(pt))));
        roundtrip(Geom::new(Geometry::LineString(vec![pt, Coord::new(3.0, 4.0)])));
        roundtrip(Geom::new(Geometry::Polygon(Polygon {
            exterior: vec![
                Coord::new(0.0, 0.0),
                Coord::new(1.0, 0.0),
                Coord::new(1.0, 1.0),
                Coord::new(0.0, 0.0),
            ],
            interiors: vec![],
        })));
        roundtrip(Geom::new(Geometry::MultiPoint(vec![Some(pt), None])));
        roundtrip(Geom::new(Geometry::MultiLineString(vec![vec![pt, pt]])));
        roundtrip(Geom::new(Geometry::MultiPolygon(vec![Polygon::default()])));
        roundtrip(Geom::new(Geometry::GeometryCollection(vec![
            Geometry::Point(Some(pt)),
            Geometry::LineString(vec![pt, pt]),
        ])));
    }

    #[test]
    fn ewkb_carries_the_srid_and_plain_wkb_does_not() {
        let g = Geom::new(Geometry::Point(Some(Coord::new(1.0, 2.0)))).with_srid(4326);
        assert_eq!(read_wkb(&write_ewkb(&g)).unwrap().srid, 4326);
        assert_eq!(read_wkb(&write_wkb(&g)).unwrap().srid, 0);
    }

    #[test]
    fn three_dimensional_geometries_survive_both_spellings() {
        let mut g = Geom::new(Geometry::Point(Some(Coord::new_z(1.0, 2.0, 3.0))));
        g.has_z = true;
        roundtrip(g.clone());
        let back = read_wkb(&write_ewkb(&g.clone().with_srid(3857))).unwrap();
        assert!(back.has_z);
        assert_eq!(back.srid, 3857);
        assert_eq!(back.coords()[0].z, 3.0);
    }

    #[test]
    fn big_endian_input_reads_identically() {
        let g = Geom::new(Geometry::Point(Some(Coord::new(1.0, 2.0))));
        let mut be = vec![0u8];
        be.extend_from_slice(&1u32.to_be_bytes());
        be.extend_from_slice(&1.0f64.to_be_bytes());
        be.extend_from_slice(&2.0f64.to_be_bytes());
        assert_eq!(read_wkb(&be).unwrap().geometry, g.geometry);
    }

    #[test]
    fn a_corrupt_count_is_rejected_rather_than_allocated() {
        // A LINESTRING header claiming four billion points in a 9-byte buffer.
        let mut bad = vec![1u8];
        bad.extend_from_slice(&2u32.to_le_bytes());
        bad.extend_from_slice(&u32::MAX.to_le_bytes());
        let err = read_wkb(&bad).unwrap_err();
        assert!(matches!(err, GeoError::Parse { .. }), "got {err:?}");
    }

    #[test]
    fn trailing_bytes_are_an_error_not_a_silent_prefix() {
        let mut buf = write_wkb(&Geom::new(Geometry::Point(Some(Coord::new(0.0, 0.0)))));
        buf.push(0);
        assert!(read_wkb(&buf).is_err());
    }

    #[test]
    fn hex_round_trips_with_and_without_the_prefix() {
        let g = Geom::new(Geometry::Point(Some(Coord::new(30.0, 10.0)))).with_srid(4326);
        let hex = write_hex_wkb(&g);
        assert_eq!(read_hex_wkb(&hex).unwrap(), g);
        assert_eq!(read_hex_wkb(&format!("0x{}", hex.to_uppercase())).unwrap(), g);
    }

    #[test]
    fn empty_point_survives_the_nan_convention() {
        let g = Geom::new(Geometry::Point(None));
        assert_eq!(read_wkb(&write_wkb(&g)).unwrap().geometry, Geometry::Point(None));
    }

    #[test]
    fn iso_z_type_codes_are_understood() {
        // POINT Z (ISO): type code 1001.
        let mut buf = vec![1u8];
        buf.extend_from_slice(&1001u32.to_le_bytes());
        for v in [1.0f64, 2.0, 3.0] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let g = read_wkb(&buf).unwrap();
        assert!(g.has_z);
        assert_eq!(g.coords()[0].z, 3.0);
    }
}
