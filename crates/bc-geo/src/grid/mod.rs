//! Discrete spatial grids — the bridge from continuous coordinates to a group key.
//!
//! Every module here answers the same problem. Latitude and longitude are floats, so no
//! two observations share a value and `GROUP BY lat, lon` returns the input. A grid
//! turns a position into a discrete cell, and the cell is an ordinary string or integer
//! the engine can hash, sort, join and shuffle at full speed with no spatial index.
//!
//! They differ in what else the cell id gives you, and that is how to choose:
//!
//! | Grid | Cell id | Also gives you |
//! |---|---|---|
//! | `geohash` | base-32 string | prefix containment: `LIKE 'u09%'` is a region filter |
//! | `tile` | `(z, x, y)` or quadkey | the exact grid map tiles are served on |
//! | `s2` | `Int64` Hilbert index | near-equal-area cells, and a region as a `BETWEEN` |
//! | `hexbin` | packed `Int64` | six equidistant neighbours, for unbiased density |

pub mod geohash;
pub mod hexbin;
pub mod s2;
pub mod tile;
