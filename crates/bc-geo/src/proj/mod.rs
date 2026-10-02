//! Answers about the Earth rather than about the coordinate plane.
//!
//! `geodesy` measures on the globe and returns metres; `crs` moves coordinates between
//! reference systems. The split matters because they solve the same problem two ways: a
//! query can either measure geodesically in degrees, or project to metres once and use
//! the cheap planar functions everywhere after. The second is usually right for a whole
//! pipeline, the first for a handful of distances. `arc` is the edge model the geodesic
//! distances measure against: the shorter great-circle arc between two positions.

pub mod arc;
pub mod crs;
pub mod geodesy;
pub mod karney;
