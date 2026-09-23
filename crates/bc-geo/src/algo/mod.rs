//! Planar geometry algorithms, grouped by what they answer.
//!
//! The layering is one-way and worth keeping that way: `primitive` knows about
//! coordinates and segments, `relate` builds point-location and noding on it,
//! `predicate` and `measure` answer questions using those, and `construct`, `affine`,
//! `linear` and `validity` build new geometries or verdicts on all of it. `overlay`
//! (polygon union and difference) sits beside them and `buffer` is built on it.
//! Nothing here reaches back up.

pub mod affine;
pub mod buffer;
pub mod construct;
pub mod linear;
pub mod measure;
pub mod overlay;
pub mod predicate;
pub mod primitive;
pub mod relate;
pub mod validity;
