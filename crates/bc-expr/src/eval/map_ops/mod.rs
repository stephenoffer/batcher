//! `Map`-column **construction**, as the counterpart to the read-side accessors in
//! `eval/map.rs` (`map_keys`/`map_values`/`map_entries`/`element_at`).
//!
//! Its own package rather than more files in `eval/` for the reason `list_ops` gives: the
//! parent directory is at its file-count limit, and grouping by responsibility is the
//! sanctioned way past it. The map family has somewhere to grow into here — `map_concat`
//! and `map_from_entries` are the next two, and both are construction rather than access.

pub(crate) mod make_map;
