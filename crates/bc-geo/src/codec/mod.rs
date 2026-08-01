//! The three geometry encodings — binary on the wire, text for humans, JSON for maps.
//!
//! Every one of them decodes to the same `Geom`, which is what makes a geometry
//! column readable regardless of which system wrote it: a GeoParquet file holds WKB,
//! a hand-written filter holds WKT, and a web export holds GeoJSON, and none of the
//! algorithms above this module can tell which it came from.

pub mod geojson;
pub mod wkb;
pub mod wkt;
