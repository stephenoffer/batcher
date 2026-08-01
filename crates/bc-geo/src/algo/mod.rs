//! Planar geometry algorithms, grouped by what they answer.
//!
//! The layering is one-way and worth keeping that way: `primitive` knows about
//! coordinates and segments, `relate` builds point-location and noding on it,
//! `predicate` and `measure` answer questions using those, and `construct`, `affine`,
//! `linear` and `validity` build new geometries or verdicts on all of it. Nothing here
//! reaches back up.

pub mod affine;
pub mod construct;
pub mod linear;
pub mod measure;
pub mod predicate;
pub mod primitive;
pub mod relate;
pub mod validity;
