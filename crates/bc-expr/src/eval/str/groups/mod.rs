//! `StrFunc::RegexpExtractGroups` — every capture group of one match, as a struct.
//!
//! Extracting three fields from a log line with three `extract` calls runs the regex three
//! times per row. This runs it once (`Regex::captures`) and keeps every group, which is the
//! shape Polars `extract_groups` and DuckDB's `regexp_extract(s, p, [names])` return.
//!
//! The struct's field names are part of the plan's static schema, so the control plane
//! derives them from the pattern before any row exists
//! (`plan/expr_ir/namespaces/_dialect.py::regex_group_names`). The names here come from the
//! compiled regex itself, and the two are held to each other by
//! `tests/differential/test_diff_str_unicode_regex.py`: a named group keeps its name and an
//! unnamed one is called by its 1-based index.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, StringArray, StringBuilder, StructArray};
use arrow::datatypes::{DataType, Field, Fields};

use crate::{ExprError, StrFunc};

/// Extract every capture group of the first match of `re` from each row.
///
/// `null_missing` selects what an absent group gives: `''` (DuckDB) or a null field
/// (Polars). A null input row is a null struct either way.
pub(crate) fn extract_groups(
    s: &StringArray,
    re: &regex::Regex,
    func: StrFunc,
    null_missing: bool,
) -> Result<ArrayRef, ExprError> {
    let names = group_names(re);
    if names.is_empty() {
        return Err(ExprError::InvalidArgument {
            func: format!("{func:?}"),
            reason: format!("pattern {:?} has no capture groups", re.as_str()),
        });
    }
    let mut builders: Vec<StringBuilder> = names.iter().map(|_| StringBuilder::new()).collect();
    let mut locs = re.capture_locations();
    for o in s {
        let matched = o.and_then(|v| re.captures_read(&mut locs, v).map(|_| v));
        for (g, b) in builders.iter_mut().enumerate() {
            // Group `g + 1`: group 0 is the whole match, which is not a field.
            let text = matched.and_then(|v| locs.get(g + 1).map(|(lo, hi)| &v[lo..hi]));
            match (o, text) {
                (None, _) => b.append_null(),
                (Some(_), Some(t)) => b.append_value(t),
                (Some(_), None) if null_missing => b.append_null(),
                (Some(_), None) => b.append_value(""),
            }
        }
    }
    let fields: Fields = names
        .iter()
        .map(|n| Field::new(n.as_str(), DataType::Utf8, true))
        .collect();
    let columns: Vec<ArrayRef> = builders
        .into_iter()
        .map(|mut b| Arc::new(b.finish()) as ArrayRef)
        .collect();
    Ok(Arc::new(StructArray::try_new(
        fields,
        columns,
        s.nulls().cloned(),
    )?))
}

/// The field name of every capture group after group 0: its name, or its 1-based index.
fn group_names(re: &regex::Regex) -> Vec<String> {
    re.capture_names()
        .enumerate()
        .skip(1)
        .map(|(i, name)| name.map_or_else(|| i.to_string(), str::to_string))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::AsArray;

    fn field(out: &ArrayRef, name: &str) -> Vec<Option<String>> {
        out.as_struct()
            .column_by_name(name)
            .unwrap()
            .as_string::<i32>()
            .iter()
            .map(|v| v.map(str::to_string))
            .collect()
    }

    #[test]
    fn names_follow_the_pattern() {
        let re = regex::Regex::new(r"(?P<key>\w+)=(\d+)(?<unit>[a-z]+)?").unwrap();
        assert_eq!(group_names(&re), ["key", "2", "unit"]);
        let s = StringArray::from(vec![Some("a=1kg"), Some("b=2"), Some("--"), None]);
        let out = extract_groups(&s, &re, StrFunc::RegexpExtractGroups, false).unwrap();
        assert_eq!(
            field(&out, "key"),
            [Some("a"), Some("b"), Some(""), None].map(|v| v.map(String::from))
        );
        assert_eq!(
            field(&out, "unit"),
            [Some("kg"), Some(""), Some(""), None].map(|v| v.map(String::from))
        );
        assert!(out.is_null(3));
        assert!(!out.is_null(2));
        let nulls = extract_groups(&s, &re, StrFunc::RegexpExtractGroupsOrNull, true).unwrap();
        assert_eq!(
            field(&nulls, "unit"),
            [Some("kg"), None, None, None].map(|v| v.map(String::from))
        );
    }

    #[test]
    fn a_pattern_without_groups_is_refused() {
        let re = regex::Regex::new("abc").unwrap();
        let s = StringArray::from(vec![Some("abc")]);
        assert!(extract_groups(&s, &re, StrFunc::RegexpExtractGroups, false).is_err());
    }
}
