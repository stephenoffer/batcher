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

mod depth;
mod engine_config;
mod error;
pub use depth::{json_max_depth, MAX_PLAN_DEPTH};
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
        /// Which side of the left key a match may come from.
        #[serde(default)]
        direction: AsofDirection,
        /// Cap on how far apart the two `on` keys may be, in the key's own units and in
        /// **microseconds** for any temporal key. `None` = no cap, which matches whatever
        /// the nearest row is however stale it has become. Requires a numeric or temporal
        /// `on` key, as does `direction = nearest`.
        #[serde(default)]
        tolerance: Option<f64>,
        /// Whether a right row whose `on` key *equals* the left row's may be the match.
        /// `false` is pandas' `allow_exact_matches=False`, the strict form that keeps a
        /// backtest honest: a quote stamped at the same instant as the trade is information
        /// the trade did not have.
        #[serde(default = "default_true")]
        allow_exact_matches: bool,
        output: Vec<JoinOutputCol>,
    },

    /// Join two relations on one or two **inequalities** (`l.x < r.y`, interval
    /// containment, a band join). A pipeline breaker.
    ///
    /// Without this node every such join lowers to a cartesian `HashJoin` with the
    /// predicate as a `Filter` above it, so the intermediate is `|left| x |right|` rows
    /// however few survive — quadratic time *and* memory. This node is executed by an
    /// output-sensitive algorithm instead: a sorted-suffix scan for one inequality, and
    /// IEJoin (Khayyat et al., the algorithm DuckDB's `PhysicalIEJoin` implements) for
    /// two.
    ///
    /// `conditions` holds one or two inequalities, each oriented `left_key OP right_key`.
    /// The planner is responsible for their key columns sharing a data type; anything
    /// further in the original predicate stays as a `Filter` above this node, which is
    /// what keeps the rewrite a *restriction* of the cartesian plan rather than a
    /// reinterpretation of it.
    RangeJoin {
        left: Box<RelOp>,
        right: Box<RelOp>,
        conditions: Vec<RangeCondition>,
        join_type: JoinType,
        output: Vec<JoinOutputCol>,
    },

    /// Deduplicate rows. A pipeline breaker.
    ///
    /// With no `keys`, this is `DISTINCT` over every column: rows that agree on all of them
    /// collapse, and there is no payload to choose between. With `keys` it is `DISTINCT ON`
    /// — the named columns decide which rows collapse, and the surviving row still carries
    /// every other column. `order` then names the row that survives (the minimum under it);
    /// empty `order` takes an arbitrary one.
    ///
    /// Both forms are one mergeable reduction, so the parallel and distributed paths are
    /// scheduling over the same operator rather than a second semantics — see
    /// `bc_runtime::agg::distinct_on`.
    Distinct {
        input: Box<RelOp>,
        /// The dedup key. Empty = every column (plain `DISTINCT`).
        #[serde(default)]
        keys: Vec<String>,
        /// Which row survives per key: the minimum under this ordering. Empty = any row.
        /// Only meaningful alongside `keys`; a whole-row `DISTINCT` has no payload to order.
        #[serde(default)]
        order: Vec<SortKey>,
        /// Stop once this many distinct rows exist, for a `LIMIT` fused in from above.
        ///
        /// Without it a `DISTINCT` under a `LIMIT` consumes its whole input before the limit
        /// discards nearly all of it, which is asymptotic rather than constant: on a
        /// high-cardinality key the dedup does work proportional to the *input* to answer a
        /// question about `k` rows. DuckDB fuses the same pair
        /// (`PhysicalLimitedDistinct`, whose `Sink` returns `FINISHED` once the hash table
        /// holds `limit` groups).
        ///
        /// **The rows kept are the first `k` in input order**, not an arbitrary `k`. That
        /// distinction is the whole reason this is sound here. SQL leaves the choice open,
        /// and DuckDB takes whichever `k` its threads happen to reach first — but invariant
        /// #7 requires a result identical on one node and on many, and "whichever `k` won the
        /// race" is not. First-in-input-order is deterministic, costs nothing extra to
        /// maintain (the dedup already visits morsels in order), and still permits the early
        /// exit, because a prefix that already holds `k` distinct rows determines the answer
        /// whatever follows it.
        ///
        /// Only set when the surviving order is genuinely unconstrained — no `Sort` between
        /// the `Distinct` and its `Limit`. `None` = no limit (a plain dedup).
        #[serde(default)]
        limit: Option<usize>,
    },

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

/// One inequality in a [`RelOp::RangeJoin`] condition, oriented `left_key OP right_key`.
#[derive(Debug, Clone, Deserialize)]
pub struct RangeCondition {
    pub left_key: String,
    pub right_key: String,
    pub op: RangeOp,
}

/// The comparison in a [`RangeCondition`]. Wire names are the contract with the planner.
///
/// Only the four ordering comparisons appear: `=` is a hash join and `<>` admits no
/// ordering structure at all, so both are left to the paths that already handle them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RangeOp {
    Lt,
    Le,
    Gt,
    Ge,
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
    /// `n50`/`n90` — assembly contiguity: the *length* such that pieces at least that long
    /// hold the `param` fraction of the total. The fraction rides `AggregateItem::param`
    /// (0.5 for N50, 0.9 for N90), as for `Quantile`. Base-weighted, which is what makes it
    /// different from a quantile over the same lengths. Mergeable (same value-list state).
    NLength,
    /// `l50`/`l90` — the *count* of pieces needed to reach `param` of the total. → Int64.
    LCount,
    /// `aun` — the area under the Nx curve, `Σ(l²)/Σ(l)`: the threshold-free contiguity
    /// statistic. Takes no param. Mergeable.
    ///
    /// Renamed explicitly because the derive would spell this `au_n`: the field's name is
    /// "auN" (one word, a capital N), and `snake_case` has no way to know that. The wire tag
    /// is what Python's `to_ir()` sends, so the two must agree — and they did not until the
    /// engine rejected the plan by name, which is the contract working.
    #[serde(rename = "aun")]
    AuN,
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
    /// `any_value`/`arbitrary` — *an* element of the group, unspecified which.
    /// Resolved to the minimum, because a mergeable combine has to be commutative:
    /// "the first row" is not a property a partition-order-independent fold can have.
    /// DuckDB leaves the choice unspecified, so the minimum is a conforming answer
    /// and, unlike scan order, it is the same on one node and on a hundred.
    AnyValue,
    /// `entropy` — the base-2 Shannon entropy of the group's value distribution
    /// (same value-list state as `Median`; finalize counts frequencies).
    Entropy,
    /// `mad` — the median absolute deviation, `median(|x - median(x)|)` (same
    /// value-list state as `Median`).
    Mad,
    /// `quantile_disc` — the *discrete* quantile at `param`: an element that is
    /// actually present, rather than the interpolation `Quantile` computes.
    QuantileDisc,
    /// `approx_top_k` — the `param` most frequent values as a `List`. Exact here (the
    /// value-list state carries every value), which is a stronger answer than the
    /// sketch DuckDB's name promises; the name is kept for portability.
    ApproxTopK,
    /// `kurtosis_pop` — *population* excess kurtosis (`m4/m2² - 3`), where `Kurtosis`
    /// is the sample-corrected one. Same 5-column moment state.
    KurtosisPop,
    /// `kahan_sum`/`fsum` — Neumaier-compensated summation. Same answer as `Sum` for a
    /// short or well-conditioned column, and a materially better one for a long float
    /// column whose addends differ wildly in magnitude.
    KahanSum,
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
    /// EWM smoothing factor in `(0, 1]` — required by `ewm_mean`/`ewm_var`/`ewm_std`
    /// and absent for every other function. Carried as the resolved alpha rather than
    /// as the `span`/`halflife`/`com` a caller may have spelled it with, so the engine
    /// has one number to honour and the conversion lives in one place.
    #[serde(default)]
    pub alpha: Option<f64>,
    /// EWM half-life in the ORDER BY key's own units, and in **microseconds** for a temporal
    /// key. Set instead of `alpha` when the smoother should decay by *elapsed key value*
    /// rather than by row position — the form an irregularly sampled series needs, where an
    /// hour's silence must not cost the same weight as a second's. Only `ewm_mean` takes it.
    #[serde(default)]
    pub half_life: Option<f64>,
    pub alias: String,
}

fn default_offset() -> i64 {
    1
}

/// `serde` default for a flag whose absence means "the permissive, pre-existing behaviour".
fn default_true() -> bool {
    true
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

/// Which side of the left key an ASOF match may come from (`RelOp::AsofJoin::direction`).
///
/// `Backward` is the default in pandas, Polars and DuckDB alike, and is the reading that
/// makes an ASOF join useful: the last known value at or before the left row. `Forward`
/// looks the other way, and `Nearest` takes whichever is closer, preferring the backward
/// one on an exact tie. `Nearest` needs to subtract two keys, so it requires a numeric or
/// temporal `on` key where the other two work on any ordered type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AsofDirection {
    #[default]
    Backward,
    Forward,
    Nearest,
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
    /// The aggregates beyond `sum`/`avg`/`min`/`max`/`count`. Every reference engine
    /// (DuckDB, Spark, Polars) allows any aggregate over a window; these are the ones
    /// whose running form is O(1) per row, which is what lets them share the same
    /// whole-partition and running machinery as the five above. Order statistics
    /// (`median`/`quantile`/`mode`) are deliberately absent — see `bc_runtime::window::agg`.
    Var,
    Stddev,
    Product,
    BoolAnd,
    BoolOr,
    BitAnd,
    BitOr,
    BitXor,
    CountDistinct,
    /// The whole-prefix series recurrences. Unlike every function above, each row's
    /// answer is a function of the entire ordered prefix carried in a running state, so
    /// they take no frame and, like the fills, require an ORDER BY.
    ///
    /// `ewm_mean`/`ewm_var`/`ewm_std` are the exponentially weighted moving statistics,
    /// smoothed by [`WindowFunc::alpha`]; `interpolate` draws a straight line across an
    /// interior null run; `rle_id` numbers the runs of equal values.
    EwmMean,
    EwmVar,
    EwmStd,
    Interpolate,
    RleId,
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
            | RelOp::Distinct { input, .. }
            | RelOp::Window { input, .. }
            | RelOp::Unnest { input, .. }
            | RelOp::RowId { input, .. }
            | RelOp::Unpivot { input, .. }
            | RelOp::Sample { input, .. } => vec![input],
            RelOp::HashJoin { left, right, .. }
            | RelOp::AsofJoin { left, right, .. }
            | RelOp::RangeJoin { left, right, .. } => {
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
    /// Walks an explicit worklist rather than recursing. [`Self::from_json`] bounds the
    /// depth of a plan that arrives over the wire, but a plan built in Rust — or one
    /// deserialized before this guard existed — has no such bound, and this used to be a
    /// second way to overflow the stack on a deep chain. A `Vec` grows on the heap.
    pub fn node_count(&self) -> u32 {
        let mut count = 0u32;
        let mut stack: Vec<&RelOp> = vec![self];
        while let Some(node) = stack.pop() {
            count += 1;
            stack.extend(node.children());
        }
        count
    }

    /// The `source_id`s this plan reads, each once however many times it is scanned.
    ///
    /// Sources cross the FFI boundary positionally — a call carries every source bound to the
    /// session, named by this plan or not — so any question of the form "how much data is this
    /// query holding" has to be asked of the scans rather than of that list. `bc-py`'s
    /// executor-routing guard is the caller that needs it, and it lives here because the
    /// question is about a `RelOp` and nothing else.
    pub fn scanned_source_ids(&self) -> std::collections::HashSet<usize> {
        let mut ids = std::collections::HashSet::new();
        let mut stack: Vec<&RelOp> = vec![self];
        while let Some(node) = stack.pop() {
            if let RelOp::Scan { source_id } = node {
                ids.insert(*source_id);
            }
            stack.extend(node.children());
        }
        ids
    }

    /// The expressions this node evaluates itself, ignoring its inputs.
    ///
    /// Split out of [`Self::contains_media_decode`] so that walk can be a flat worklist
    /// while this stays an exhaustive `match` — which is what keeps the guarantee that a
    /// new `RelOp` variant is a compile error until someone classifies its expressions.
    fn own_exprs(&self) -> Vec<&Expr> {
        match self {
            RelOp::Scan { .. } => Vec::new(),
            RelOp::Limit { .. }
            | RelOp::Unnest { .. }
            | RelOp::RowId { .. }
            | RelOp::Unpivot { .. }
            | RelOp::Sample { .. }
            | RelOp::HashJoin { .. }
            | RelOp::AsofJoin { .. }
            | RelOp::RangeJoin { .. }
            | RelOp::Union { .. } => Vec::new(),
            RelOp::Filter { predicate, .. } => vec![predicate],
            RelOp::Project { exprs, .. } => exprs.iter().map(|p| &p.expr).collect(),
            RelOp::Aggregate {
                group_keys,
                aggregates,
                ..
            } => group_keys
                .iter()
                .map(|p| &p.expr)
                .chain(
                    aggregates
                        .iter()
                        .flat_map(|a| a.input.iter().chain(a.input2.iter())),
                )
                .collect(),
            RelOp::Sort { keys, .. } => keys.iter().map(|k| &k.expr).collect(),
            // The dedup key is named by column, but the ordering that picks the surviving
            // row is a general expression, so it counts here exactly as `Sort`'s does.
            RelOp::Distinct { order, .. } => order.iter().map(|k| &k.expr).collect(),
            RelOp::Window {
                partition_keys,
                order_keys,
                functions,
                ..
            } => partition_keys
                .iter()
                .chain(order_keys.iter().map(|k| &k.expr))
                .chain(functions.iter().filter_map(|f| f.input.as_ref()))
                .collect(),
        }
    }

    /// True if any expression anywhere in this plan is a library-backed media decode
    /// (`.image`/`.audio`/`.video`).
    ///
    /// A *scheduling* signal, not a semantic one: a media-decode kernel parallelizes
    /// *within* a single morsel (heavy per-row JPEG/audio/video decode — see
    /// [`Expr::contains_media_decode`]), so a plan carrying one can saturate every core
    /// even when its input is a single morsel. The parallel executor uses this to lift
    /// its morsel-count cap on pool width for such plans. Exhaustive by construction: a
    /// new `RelOp` variant is a compile error in [`Self::own_exprs`] until it is
    /// classified.
    ///
    /// Walks a worklist rather than recursing, for the reason given on
    /// [`Self::node_count`]. Short-circuits on the first hit, so the common answer
    /// (`false`, after a full walk) is the only one that pays for the whole plan.
    pub fn contains_media_decode(&self) -> bool {
        let mut stack: Vec<&RelOp> = vec![self];
        while let Some(node) = stack.pop() {
            if node
                .own_exprs()
                .into_iter()
                .any(Expr::contains_media_decode)
            {
                return true;
            }
            stack.extend(node.children());
        }
        false
    }

    /// Parse a plan from the JSON IR document emitted by the Python control plane.
    ///
    /// serde_json guards against stack overflow with a default 128-deep recursion limit,
    /// but a legitimately deep generated pipeline — a long `.filter(...)` chain, hundreds
    /// of interleaved operators — nests past it and would fail with "recursion limit
    /// exceeded". So serde's guard stays lifted, and [`MAX_PLAN_DEPTH`] replaces it with a
    /// *measured* bound checked by a non-recursive scan before any frame is pushed.
    ///
    /// The distinction matters more than the number does. Lifting serde's limit without
    /// replacing it turned a catchable `Err` into a stack overflow, which Rust reports as
    /// an **uncatchable `SIGABRT`** — no Python exception, no message, and on a shuffle
    /// actor just an `ActorDiedError`. Checking first restores the `Err`.
    ///
    /// `end()` still rejects trailing garbage, matching `serde_json::from_str`'s contract.
    ///
    /// # Errors
    ///
    /// [`IrError::PlanTooDeep`] if the document nests past [`MAX_PLAN_DEPTH`], and
    /// [`IrError::Parse`] for malformed or contract-violating JSON.
    pub fn from_json(s: &str) -> Result<Self, IrError> {
        let found = depth::json_max_depth(s);
        if found > depth::MAX_PLAN_DEPTH {
            return Err(IrError::PlanTooDeep {
                depth: found,
                limit: depth::MAX_PLAN_DEPTH,
            });
        }
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

    /// A keyed dedup arrives with its key and ordering, and a whole-row one arrives without
    /// them — the `serde(default)` that makes the two forms one node on the wire.
    ///
    /// The defaulted half is what needs pinning. `deny_unknown_fields` would catch a *new*
    /// field Rust does not know, but nothing catches a field Rust knows that Python stops
    /// sending: it would quietly default, and a `distinct(["k"])` whose `keys` went missing is
    /// a whole-row DISTINCT that runs, returns rows, and returns the wrong ones.
    #[test]
    fn distinct_carries_its_key_and_ordering() {
        let whole = RelOp::from_json(r#"{"op":"distinct","input":{"op":"scan","source_id":0}}"#)
            .expect("a whole-row DISTINCT sends neither field");
        let RelOp::Distinct { keys, order, .. } = whole else {
            panic!("expected a Distinct")
        };
        assert!(keys.is_empty() && order.is_empty());

        let keyed = RelOp::from_json(
            r#"{"op":"distinct","input":{"op":"scan","source_id":0},"keys":["k"],
                "order":[{"expr":{"e":"col","name":"ts"},"descending":true,
                          "nulls_first":false}]}"#,
        )
        .expect("a keyed dedup sends both");
        let RelOp::Distinct { keys, order, .. } = keyed else {
            panic!("expected a Distinct")
        };
        assert_eq!(keys, vec!["k".to_string()]);
        assert_eq!(order.len(), 1);
        assert!(order[0].descending && !order[0].nulls_first);
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
