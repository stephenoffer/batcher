//! Extended `List`-column operations beyond the per-row reductions in `eval/list.rs`:
//! set operations between two lists (`intersect`/`except`/`union`) and the
//! higher-order `transform`/`filter` over an element sub-expression. Grouped here to
//! keep `eval/` within its file-count limit.

pub(crate) mod list_hof;
pub(crate) mod list_set;

pub(crate) use list_hof::{eval_list_filter, eval_list_transform};
pub(crate) use list_set::eval_list_set;
