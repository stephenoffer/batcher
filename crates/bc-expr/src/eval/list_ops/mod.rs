//! Extended `List`-column operations beyond the per-row reductions in `eval/list.rs`:
//! set operations between two lists (`intersect`/`except`/`union`) and the
//! higher-order `transform`/`filter` over an element sub-expression, and the SimHash
//! LSH signature of an embedding, and the input coercion plus numeric inner loop the
//! vector-distance kernels share. Grouped here to
//! keep `eval/` within its file-count limit.

pub(crate) mod coerce;
pub(crate) mod list_hof;
pub(crate) mod list_reduce;
pub(crate) mod list_reshape;
pub(crate) mod list_set;
pub(crate) mod list_zip;
pub(crate) mod simhash;

pub(crate) use coerce::{accumulate_pair, as_var_list};
pub(crate) use list_hof::{eval_list_filter, eval_list_transform};
pub(crate) use list_reshape::eval_flatten;
pub(crate) use list_set::eval_list_set;
pub(crate) use list_zip::eval_list_zip;
pub(crate) use simhash::eval_list_simhash;
