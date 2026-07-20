//! `bc-ir` — the query intermediate representation.
//!
//! Two IR levels (per the architecture plan):
//!
//! * **Relational IR (`RelOp`)** — the physical-plan DAG the optimizer (Kyber)
//!   produces. This is what Python lowers into and ships across the FFI boundary
//!   as a JSON document, so the field/tag names here are the stable wire
//!   contract between the control plane and the engine.
//! * **Pipeline IR (`PipeOp`)** — the data-centric produce/consume program a
//!   pipeline lowers to for the interpreter and JIT. Introduced once breakers
//!   land; for the bootstrap engine the interpreter walks `RelOp` directly.
//!
//! Only streaming operators (Scan/Filter/Project) exist today; breakers
//! (HashJoin/HashAgg/Sort/Distinct/Window/Opaque) arrive with the runtime
//! library and are slotted into this same enum.

use bc_expr::Expr;
use serde::Deserialize;

mod engine_config;
mod error;
pub use engine_config::EngineConfig;
pub use error::IrError;

/// A node in the relational plan DAG.
///
/// Boxed children keep the enum a thin tree; the JSON tag is `op`.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum RelOp {
    /// Read an input relation. `source_id` indexes into the input relations
    /// supplied alongside the plan (an in-memory table today; a file/stream
    /// source once `io` lands).
    Scan { source_id: usize },

    /// Keep rows for which `predicate` evaluates to true.
    Filter { input: Box<RelOp>, predicate: Expr },

    /// Produce a new relation with the given output columns.
    Project {
        input: Box<RelOp>,
        exprs: Vec<ProjectionItem>,
    },

    /// Group by zero or more key expressions and compute aggregates per group.
    /// A pipeline breaker: it consumes all input before producing output.
    Aggregate {
        input: Box<RelOp>,
        /// Group-by keys (each becomes an output column). Empty = global aggregate.
        group_keys: Vec<ProjectionItem>,
        aggregates: Vec<AggregateItem>,
    },

    /// Order rows by one or more sort keys. A (full) pipeline breaker.
    /// `limit` (when set, from a fused `Limit`) turns this into a top-N: only the
    /// first `limit` rows are produced, via a partial sort.
    Sort {
        input: Box<RelOp>,
        keys: Vec<SortKey>,
        #[serde(default)]
        limit: Option<usize>,
    },

    /// Keep at most `n` rows after skipping `offset`.
    Limit {
        input: Box<RelOp>,
        n: usize,
        #[serde(default)]
        offset: usize,
    },

    /// Equi-join two relations on key columns. A pipeline breaker (builds a hash
    /// table on the right, probes with the left). The output column list is
    /// resolved by the planner (which knows both schemas), so the engine just
    /// gathers each named column from its side.
    HashJoin {
        left: Box<RelOp>,
        right: Box<RelOp>,
        left_keys: Vec<String>,
        right_keys: Vec<String>,
        join_type: JoinType,
        output: Vec<JoinOutputCol>,
        /// Physical join algorithm chosen by the planner (Kyber). All strategies
        /// produce identical results; only the data movement differs. Defaults to
        /// `hash` (shuffle hash join) when the planner does not specify one.
        #[serde(default)]
        strategy: JoinStrategy,
    },

    /// ASOF (nearest-match) join: each left row matched to the right row whose `on`
    /// key is nearest in `direction` within the same `by` group (DataFrame
    /// `join_asof` / SQL `ASOF JOIN`). Left-style — every left row is emitted, with
    /// null right columns when unmatched. A pipeline breaker (both sides materialized).
    AsofJoin {
        left: Box<RelOp>,
        right: Box<RelOp>,
        left_on: String,
        right_on: String,
        left_by: Vec<String>,
        right_by: Vec<String>,
        /// `true` = backward (largest right.on ≤ left.on); `false` = forward.
        backward: bool,
        output: Vec<JoinOutputCol>,
    },

    /// Deduplicate rows (DISTINCT over all columns). A pipeline breaker.
    Distinct { input: Box<RelOp> },

    /// Window functions: partition rows by `partition_keys`, order within each
    /// partition by `order_keys`, and append one output column per function
    /// (the input columns are preserved). A pipeline breaker.
    ///
    /// `partition_keys` may be empty (one partition over all rows). `order_keys`
    /// may be empty (only valid for whole-partition aggregates, not ranking
    /// functions). The whole-partition aggregates (`sum`/`avg`/`min`/`max`/
    /// `count`) emit the same value for every row in the partition.
    Window {
        input: Box<RelOp>,
        #[serde(default)]
        partition_keys: Vec<Expr>,
        #[serde(default)]
        order_keys: Vec<SortKey>,
        functions: Vec<WindowFunc>,
        /// Fused per-partition top-N (`QUALIFY <rank> <= k`): keep only rows whose
        /// ranking value is `<= rank_limit`. The optimizer sets this only when the
        /// window has a single ranking function (`row_number`/`rank`/`dense_rank`),
        /// so the bound applies to the one appended column — for `row_number` this is
        /// the top-`k` per partition, and for `rank`/`dense_rank` it correctly keeps
        /// peers tied at the boundary. `None` = no limit (a plain window).
        #[serde(default)]
        rank_limit: Option<usize>,
    },

    /// Concatenate relations with identical schemas. `distinct` makes it a
    /// set UNION (vs UNION ALL). Trivially mergeable: concat partitions.
    Union {
        inputs: Vec<RelOp>,
        #[serde(default)]
        distinct: bool,
    },

    /// Explode a list/array column into one row per element (SQL `UNNEST`,
    /// DataFrame `explode`). The named `column` is replaced in place by its element
    /// values bound to `alias`; every other column is repeated once per element.
    /// Null and empty lists produce no output rows (DuckDB `UNNEST` semantics).
    ///
    /// Stateless and streaming — each batch explodes independently, so it maps over
    /// morsels (and partitions) with no breaker, exactly like `Filter`/`Project`.
    Unnest {
        input: Box<RelOp>,
        /// Name of the list/array column to explode.
        column: String,
        /// Output column name for the exploded element (defaults to `column` on the
        /// control-plane side).
        alias: String,
        /// Keep a row whose list is null or empty, with a NULL element (Spark
        /// `explode_outer`, SQL `LEFT JOIN LATERAL … ON true`). Default `false` is the
        /// DuckDB `UNNEST` semantics: such a row contributes nothing.
        ///
        /// This matters for document pipelines: with the default, a document that
        /// chunked to nothing vanishes from the relation entirely, taking its id and
        /// metadata with it — invisible row loss rather than an error.
        #[serde(default)]
        outer: bool,
        /// When set, also emit this column holding each element's 0-based position
        /// within its list (Spark `posexplode`). NULL for a row kept only by `outer`,
        /// which has no element and therefore no position.
        ///
        /// 0-based to match `RowId`/`with_row_index`, not the 1-based SQL `WITH
        /// ORDINALITY` — one convention per engine beats matching each source dialect.
        #[serde(default)]
        index_alias: Option<String>,
    },

    /// Append a 0-based (plus `offset`) sequential row-index column over the input,
    /// in batch-arrival order (Polars `with_row_index`). The id is assigned by a
    /// single sequential counter, so it is identical on the sequential and parallel
    /// paths for an order-preserving pipeline.
    RowId {
        input: Box<RelOp>,
        /// Output column name for the index.
        alias: String,
        /// Starting value for the first row (default 0).
        #[serde(default)]
        offset: i64,
    },

    /// Reshape wide → long (SQL `UNPIVOT`, pandas `melt`, Polars `unpivot`). Each
    /// input row becomes one row per `on` column: the `index` columns repeat, a
    /// `variable_name` string column holds the source column's name, and a
    /// `value_name` column holds its value. The `on` columns must share a type.
    ///
    /// Stateless and streaming — each batch reshapes independently, so it maps over
    /// morsels (and partitions) with no breaker, like `Unnest`/`Project`.
    Unpivot {
        input: Box<RelOp>,
        /// Identifier columns that repeat once per `on` column.
        index: Vec<String>,
        /// The wide value columns being melted into rows.
        on: Vec<String>,
        /// Output column name holding each melted column's name.
        variable_name: String,
        /// Output column name holding each melted column's value.
        value_name: String,
    },

    /// Randomly keep a `fraction` of rows (DataFrame `sample`). Each row is kept iff
    /// a stable hash of its values (seeded by `seed`) falls below `fraction` — so the
    /// sample is *deterministic and partition-independent*: the same row is kept on
    /// one node or many, honoring the single-node == distributed invariant. Stateless
    /// and streaming (each batch samples independently, no breaker).
    Sample {
        input: Box<RelOp>,
        /// Fraction of rows to keep, in `[0.0, 1.0]` (streaming, per-batch).
        fraction: f64,
        /// Seed mixed into the per-row hash (baked at plan-build for cross-worker
        /// consistency).
        seed: u64,
        /// Fixed-count mode: when set, keep exactly the `n` rows with the smallest
        /// per-row hash instead of a fraction. Deterministic and partition-
        /// independent (a breaker). `#[serde(default)]` keeps older plans (no `n`)
        /// on the fraction path.
        #[serde(default)]
        n: Option<usize>,
    },
}

/// Join flavor. Wire names are the contract with the planner.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
    Semi,
    Anti,
}

/// Physical join algorithm. A planner hint, not a semantic change — every
/// strategy yields the same relation, so the engine may safely fall back to
/// `Hash` for any case a strategy does not support.
///
/// * `Hash` — shuffle hash join: partition both sides by key, join per bucket.
/// * `Broadcast` — replicate the (small) build side, partition only the probe
///   side; no shuffle of the large side. Spark's most impactful AQE join choice.
/// * `SortMerge` — sort both sides by key and merge; no hash table, suits two
///   large (or already-sorted) inputs (Spark's default large-join algorithm).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JoinStrategy {
    #[default]
    Hash,
    Broadcast,
    SortMerge,
}

/// One output column of a join: which side it comes from, its source name there,
/// and the output name.
#[derive(Debug, Clone, Deserialize)]
pub struct JoinOutputCol {
    pub side: JoinSide,
    pub name: String,
    pub alias: String,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JoinSide {
    Left,
    Right,
}

/// One sort key: an expression and its ordering.
#[derive(Debug, Clone, Deserialize)]
pub struct SortKey {
    pub expr: Expr,
    #[serde(default)]
    pub descending: bool,
    #[serde(default)]
    pub nulls_first: bool,
}

/// One aggregate in an `Aggregate`: a function over an optional input expression,
/// bound to an output name.
#[derive(Debug, Clone, Deserialize)]
pub struct AggregateItem {
    pub func: AggFunc,
    /// The argument expression. `None` is only valid for `count_star`.
    #[serde(default)]
    pub input: Option<Expr>,
    /// The second argument — the ordering key for `arg_min`/`arg_max`. `None` for
    /// every single-input aggregate. `#[serde(default)]` keeps the wire contract
    /// backward-compatible (older plans without it deserialize to `None`).
    #[serde(default)]
    pub input2: Option<Expr>,
    /// Function parameter (the quantile in [0,1] for `Quantile`); ignored otherwise.
    #[serde(default)]
    pub param: Option<f64>,
    pub alias: String,
}

/// Aggregate function tags. The wire names are the contract with the engine.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AggFunc {
    CountStar,
    Count,
    /// COUNT(DISTINCT x) — exact, mergeable (the per-group distinct value set is
    /// the partial state).
    CountDistinct,
    Sum,
    Min,
    Max,
    Mean,
    Var,
    Stddev,
    /// MEDIAN — exact, mergeable (the per-group value list is the partial state).
    Median,
    /// `percentile_cont` — continuous quantile at `param` ∈ [0,1] (same list state
    /// as `Median`). The quantile is carried in `AggregateItem::param`.
    Quantile,
    /// `array_agg` — collect each group's values into a `List`.
    ListAgg,
    /// `bool_and` — logical AND of a group's non-null booleans (mergeable).
    BoolAnd,
    /// `bool_or` — logical OR of a group's non-null booleans (mergeable).
    BoolOr,
    /// `approx_count_distinct` — bounded-memory distinct count via per-group HLL
    /// (mergeable; ~2% error). Skew-safe alternative to `CountDistinct`.
    ApproxCountDistinct,
    /// `approx_quantile` — bounded-memory quantile via per-group KLL. The quantile
    /// `p ∈ [0,1]` rides `AggregateItem::param` (as for `Quantile`). Skew-safe
    /// alternative to `Median`/`Quantile`.
    ApproxQuantile,
    /// `mode` — most frequent value per group (ties → smallest, so mergeable).
    Mode,
    /// `arg_min` / `arg_max` — the value (`input`) at the row with the min/max
    /// ordering key (`input2`). Two-input, 2-column-state, mergeable.
    ArgMin,
    ArgMax,
    /// `product` — product of a group's non-null values as Float64 (mergeable).
    Product,
    /// `bit_and`/`bit_or`/`bit_xor` — bitwise fold of a group's non-null Int64
    /// values (mergeable: each op associates and commutes).
    BitAnd,
    BitOr,
    BitXor,
    /// `covar_pop`/`covar_samp`/`corr` — two-input covariance/correlation (the
    /// second input rides `AggregateItem::input2`). 6-column sum-of-powers state.
    CovarPop,
    CovarSamp,
    Corr,
    /// `skewness`/`kurtosis` — single-input moment aggregates (5-column state).
    Skewness,
    Kurtosis,
    /// `histogram` — a `Map<value, count>` per group (same value-list state as
    /// `Median`; finalize counts).
    Histogram,
}

/// One window function in a `Window`: a function over an optional input
/// expression, bound to an output name.
#[derive(Debug, Clone, Deserialize)]
pub struct WindowFunc {
    pub func: WindowFn,
    /// The argument expression. `None` is valid for the ranking functions
    /// (`row_number`/`rank`/`dense_rank`); aggregates and value functions need it.
    #[serde(default)]
    pub input: Option<Expr>,
    /// Lag/lead distance (default 1); ignored by other functions.
    #[serde(default = "default_offset")]
    pub offset: i64,
    /// Explicit window frame (`ROWS BETWEEN …`). `None` is the SQL default frame
    /// (`RANGE UNBOUNDED PRECEDING TO CURRENT ROW`, peer-tie semantics). Applies to
    /// the aggregate functions; ignored by ranking/value functions.
    #[serde(default)]
    pub frame: Option<WindowFrame>,
    pub alias: String,
}

fn default_offset() -> i64 {
    1
}

/// An explicit window frame: the rows each output row aggregates over.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct WindowFrame {
    pub units: FrameUnits,
    pub start: FrameBound,
    pub end: FrameBound,
}

/// Frame units. `Rows` counts physical rows; `Range`/`Groups` count peer groups
/// (rows with an equal ORDER BY value). `Range` supports peer bounds (CURRENT ROW /
/// UNBOUNDED) only; a numeric `RANGE` offset is value-based and is *rejected* by the
/// interpreter rather than approximated, since substituting the peer frame would
/// silently return wrong rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrameUnits {
    Rows,
    Range,
    Groups,
}

/// One edge of a window frame. `n` is a non-negative row offset for the bounded
/// `preceding`/`following` cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FrameBound {
    UnboundedPreceding,
    Preceding { n: u64 },
    CurrentRow,
    Following { n: u64 },
    UnboundedFollowing,
}

/// Window function tags. The wire names are the contract with the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WindowFn {
    RowNumber,
    Rank,
    DenseRank,
    /// `(rank - 1) / (rows - 1)` over the ordered partition; `0` for a single row.
    PercentRank,
    /// Fraction of partition rows at or before the current row's peer group.
    CumeDist,
    /// Distribute the ordered partition into `offset` buckets numbered `1..=offset`,
    /// as evenly as possible (earlier buckets take the remainder).
    Ntile,
    Sum,
    Avg,
    Min,
    Max,
    Count,
    FirstValue,
    LastValue,
    Lag,
    Lead,
    /// `nth_value(expr, n)` — value of the `offset`-th row (1-based) of the
    /// partition in order; null if the partition has fewer than `offset` rows.
    NthValue,
    /// Carry the last non-null value forward along the ordered partition; rows before
    /// the first non-null stay null. SQL spells this
    /// `last_value(x IGNORE NULLS) OVER (… ROWS UNBOUNDED PRECEDING)`; because the
    /// frame is implied, no `frame` is needed here. Requires ORDER BY.
    ForwardFill,
    /// The mirror of `ForwardFill`: carry the next non-null value backward.
    BackwardFill,
}

/// One output column of a `Project`: an expression and the name it is bound to.
#[derive(Debug, Clone, Deserialize)]
pub struct ProjectionItem {
    pub expr: Expr,
    pub alias: String,
}

impl RelOp {
    /// This node's input plans, in the order an executor visits them.
    ///
    /// The order is the contract: it is the pre-order the executors and the Python control
    /// plane both number operators by, so `children()` and [`Self::node_count`] agree with
    /// the ids a recursive walk hands out.
    pub fn children(&self) -> Vec<&RelOp> {
        match self {
            RelOp::Scan { .. } => Vec::new(),
            RelOp::Filter { input, .. }
            | RelOp::Project { input, .. }
            | RelOp::Aggregate { input, .. }
            | RelOp::Sort { input, .. }
            | RelOp::Limit { input, .. }
            | RelOp::Distinct { input }
            | RelOp::Window { input, .. }
            | RelOp::Unnest { input, .. }
            | RelOp::RowId { input, .. }
            | RelOp::Unpivot { input, .. }
            | RelOp::Sample { input, .. } => vec![input],
            RelOp::HashJoin { left, right, .. } | RelOp::AsofJoin { left, right, .. } => {
                vec![left, right]
            }
            RelOp::Union { inputs, .. } => inputs.iter().collect(),
        }
    }

    /// Number of operators in this subtree — i.e. how many pre-order ids executing it consumes.
    ///
    /// Lets an executor know a subtree's id span *without running it*, so it can execute the
    /// children of a node out of order and still hand each the ids a plain recursive walk
    /// would. The fused join pipeline uses this to test its (cheap) build sides for
    /// streamability before committing to its (expensive) probe side.
    pub fn node_count(&self) -> u32 {
        1 + self.children().iter().map(|c| c.node_count()).sum::<u32>()
    }

    /// True if any expression anywhere in this plan is a library-backed media decode
    /// (`.image`/`.audio`/`.video`).
    ///
    /// A *scheduling* signal, not a semantic one: a media-decode kernel parallelizes
    /// *within* a single morsel (heavy per-row JPEG/audio/video decode — see
    /// [`Expr::contains_media_decode`]), so a plan carrying one can saturate every core
    /// even when its input is a single morsel. The parallel executor uses this to lift
    /// its morsel-count cap on pool width for such plans. Exhaustive by construction: a
    /// new `RelOp` variant is a compile error here until it is classified.
    pub fn contains_media_decode(&self) -> bool {
        match self {
            RelOp::Scan { .. } => false,
            RelOp::Filter { input, predicate } => {
                predicate.contains_media_decode() || input.contains_media_decode()
            }
            RelOp::Project { input, exprs } => {
                exprs.iter().any(|p| p.expr.contains_media_decode())
                    || input.contains_media_decode()
            }
            RelOp::Aggregate {
                input,
                group_keys,
                aggregates,
            } => {
                group_keys.iter().any(|p| p.expr.contains_media_decode())
                    || aggregates.iter().any(|a| {
                        a.input.as_ref().is_some_and(Expr::contains_media_decode)
                            || a.input2.as_ref().is_some_and(Expr::contains_media_decode)
                    })
                    || input.contains_media_decode()
            }
            RelOp::Sort { input, keys, .. } => {
                keys.iter().any(|k| k.expr.contains_media_decode()) || input.contains_media_decode()
            }
            RelOp::Limit { input, .. }
            | RelOp::Distinct { input }
            | RelOp::Unnest { input, .. }
            | RelOp::RowId { input, .. }
            | RelOp::Unpivot { input, .. }
            | RelOp::Sample { input, .. } => input.contains_media_decode(),
            RelOp::HashJoin { left, right, .. } | RelOp::AsofJoin { left, right, .. } => {
                left.contains_media_decode() || right.contains_media_decode()
            }
            RelOp::Window {
                input,
                partition_keys,
                order_keys,
                functions,
                ..
            } => {
                partition_keys.iter().any(Expr::contains_media_decode)
                    || order_keys.iter().any(|k| k.expr.contains_media_decode())
                    || functions
                        .iter()
                        .any(|f| f.input.as_ref().is_some_and(Expr::contains_media_decode))
                    || input.contains_media_decode()
            }
            RelOp::Union { inputs, .. } => inputs.iter().any(RelOp::contains_media_decode),
        }
    }

    /// Parse a plan from the JSON IR document emitted by the Python control plane.
    ///
    /// serde_json guards against stack overflow with a default 128-deep recursion limit,
    /// but a legitimately deep generated pipeline — a long `.filter(...)` chain, hundreds
    /// of interleaved operators — nests past it and would fail with "recursion limit
    /// exceeded". The Python control plane already caps plan depth at its own recursion
    /// limit (~1000), which the native stack comfortably handles, so we lift serde_json's
    /// guard here and rely on that bound — deep plans deserialize out of the box. `end()`
    /// still rejects trailing garbage, matching `serde_json::from_str`'s contract.
    pub fn from_json(s: &str) -> Result<Self, IrError> {
        let mut de = serde_json::Deserializer::from_str(s);
        de.disable_recursion_limit();
        let op = Self::deserialize(&mut de)?;
        de.end()?;
        Ok(op)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A well-formed node round-trips, and an *unknown* field is a hard parse error
    /// rather than a silently-ignored key. `deny_unknown_fields` is the boundary
    /// guard: a Python `to_ir()` that emits a stale/misspelled IR key (e.g. an
    /// `offset` that should be `skip`) fails loudly at deserialization instead of
    /// being dropped to its default and producing a silent wrong result.
    #[test]
    fn unknown_relop_field_is_rejected() {
        assert!(RelOp::from_json(r#"{"op":"scan","source_id":0}"#).is_ok());
        let drift = RelOp::from_json(r#"{"op":"scan","source_id":0,"bogus":1}"#);
        assert!(
            drift.is_err(),
            "an unknown RelOp field must be rejected, not ignored"
        );
    }

    /// The guard reaches `Expr` (the other half of the wire contract) too: an unknown
    /// key inside a nested predicate expression is rejected.
    #[test]
    fn unknown_expr_field_is_rejected() {
        let ok = r#"{"op":"filter","input":{"op":"scan","source_id":0},
                     "predicate":{"e":"col","name":"a"}}"#;
        assert!(RelOp::from_json(ok).is_ok());
        let drift = r#"{"op":"filter","input":{"op":"scan","source_id":0},
                       "predicate":{"e":"col","name":"a","bogus":true}}"#;
        assert!(
            RelOp::from_json(drift).is_err(),
            "an unknown Expr field must be rejected"
        );
    }
}
