//! Build a Parquet [`ProjectionMask`] that selects **exactly** the requested columns.
//!
//! `ProjectionMask::columns` matches leaf paths with `starts_with`, which is right for the
//! nested case it was written for — asking for `addr` should bring `addr.city` and `addr.zip`
//! — and wrong for every flat schema where one column's name is a prefix of another's.
//!
//! On the benchmark's own scan corpus (`column0` … `column15`) asking for `column1` returns
//! **seven** columns: `column1`, `column10`, `column11`, `column12`, `column13`, `column14`,
//! `column15`. It reads and decodes all seven, which is exactly why that one column measured
//! 376.6 ms against 74.9 ms for `column0` over S3, on byte-identical chunks with identical
//! encodings — DuckDB reads the same column in 41.7 ms. It is also why `filter_agg`
//! (`column0` + `column1`) was the slowest shape in the scan suite: it read eight columns to
//! answer a two-column query.
//!
//! The shape is not exotic. `id` beside `id_hash`, `ts` beside `ts_utc`, `name` beside
//! `name_first` — any schema with a common stem hits it, and it is silent: the extra columns
//! are correct data, so nothing fails, the read is just several times wider than it asked to
//! be. (It also breaks the reader's stated PyArrow parity, since `reorder_to_projection`
//! declines a batch whose column count does not match the request.)
//!
//! So: match a leaf when its path **is** the requested name, or when it is a child of it
//! (`name.` as a prefix). That keeps the nested behaviour `ProjectionMask::columns` exists for
//! and drops the accidental sibling matches.

use parquet::arrow::ProjectionMask;
use parquet::schema::types::SchemaDescriptor;

/// The dotted leaf path of column `idx`, as `ProjectionMask::columns` spells it.
fn leaf_path(schema: &SchemaDescriptor, idx: usize) -> String {
    schema.column(idx).path().string()
}

/// Whether `path` is the column `name` names, or a leaf beneath it.
///
/// The `.`-terminated prefix is what separates a genuine nested child (`addr.city` under
/// `addr`) from a sibling that merely shares a stem (`column10` under `column1`).
fn selects(path: &str, name: &str) -> bool {
    path == name
        || (path.len() > name.len()
            && path.starts_with(name)
            && path[name.len()..].starts_with('.'))
}

/// A mask selecting exactly the leaves `names` refers to.
///
/// A name matching nothing contributes nothing, which is the same tolerance
/// `ProjectionMask::columns` has: the caller's projection is validated against the Arrow
/// schema before it gets here, and a reader that hard-failed on an absent name would turn a
/// schema-evolution read into an error where the engine wants a null column.
pub(crate) fn exact_columns<'a>(
    schema: &SchemaDescriptor,
    names: impl IntoIterator<Item = &'a str>,
) -> ProjectionMask {
    let paths: Vec<String> = (0..schema.num_columns())
        .map(|i| leaf_path(schema, i))
        .collect();
    let mut leaves: Vec<usize> = Vec::new();
    for name in names {
        for (idx, path) in paths.iter().enumerate() {
            if selects(path, name) && !leaves.contains(&idx) {
                leaves.push(idx);
            }
        }
    }
    ProjectionMask::leaves(schema, leaves)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bug, stated as a test: a name that is a prefix of a sibling must not pull it in.
    #[test]
    fn a_prefix_name_does_not_select_its_siblings() {
        assert!(selects("column1", "column1"));
        for sibling in ["column10", "column11", "column15"] {
            assert!(
                !selects(sibling, "column1"),
                "{sibling} must not be selected by asking for column1"
            );
        }
        assert!(!selects("id_hash", "id"));
        assert!(!selects("ts_utc", "ts"));
        assert!(!selects("name_first", "name"));
    }

    /// The behaviour `ProjectionMask::columns` exists for, which must survive: asking for a
    /// struct's root selects the leaves under it.
    #[test]
    fn a_nested_root_still_selects_its_leaves() {
        assert!(selects("addr.city", "addr"));
        assert!(selects("addr.zip", "addr"));
        assert!(selects("a.b.c", "a"));
        assert!(selects("a.b.c", "a.b"));
        // ...but not a sibling struct sharing the stem.
        assert!(!selects("address.city", "addr"));
    }

    /// A leaf is never selected twice, however many names reach it — a duplicated leaf index
    /// would be a duplicated column in the decoded batch.
    #[test]
    fn overlapping_names_select_each_leaf_once() {
        let paths = ["a.b", "a.c"];
        let mut leaves: Vec<usize> = Vec::new();
        for name in ["a", "a.b"] {
            for (idx, path) in paths.iter().enumerate() {
                if selects(path, name) && !leaves.contains(&idx) {
                    leaves.push(idx);
                }
            }
        }
        assert_eq!(leaves, vec![0, 1]);
    }
}
