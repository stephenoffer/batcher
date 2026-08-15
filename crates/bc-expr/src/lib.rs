//! `bc-expr` — scalar expression IR and its evaluation.
//!
//! There is exactly ONE expression representation in Batcher (`Expr`), and it is
//! the single source consumed by *both* the Tier-0 interpreter and (later) the
//! JIT codegen backends. That shared source is what guarantees semantic parity
//! across execution tiers — the interpreter is the correctness oracle the
//! compiled tiers are differential-tested against, and they can only agree if
//! they evaluate the same IR.
//!
//! Evaluation here is vectorized: an `Expr` is evaluated over a whole Arrow
//! `RecordBatch` (a morsel) at once using arrow compute kernels. Literals are
//! currently materialized to full-length arrays for simplicity; a later pass
//! will switch to Arrow `Datum` scalars + selection vectors for true late
//! materialization.

use std::sync::Arc;

use arrow::array::{
    ArrayRef, BooleanArray, Date32Array, Float64Array, Int64Array, StringArray,
    TimestampMicrosecondArray,
};
use serde::Deserialize;

mod analyze;
mod error;
mod select;
mod supertype;
pub use error::ExprError;
pub use select::ConjunctOrder;
pub use supertype::common_supertype;

/// What a payload's leading bytes say it is, or `None` when nothing recognizes them.
///
/// Public so the IO layer can reach the *same* magic-number table the `.str.mime_type()`
/// expression uses, through a `bc-py` helper. A reader that kept its own copy would be a
/// second answer to "what is this file", and the two would drift the first time a format
/// was added to one of them.
pub fn sniff_mime(data: &[u8]) -> Option<&'static str> {
    eval::mime::sniff(data)
}

// The per-variant evaluation bodies (and `Expr::eval` itself) live in `eval`; the
// wire-contract enum definitions stay here in `lib.rs`.
mod eval;

/// One named field of a `MakeStruct` — a field name paired with the sub-expression
/// whose per-row value populates it.
#[derive(Debug, Clone, Deserialize)]
pub struct NamedExpr {
    pub name: String,
    pub value: Box<Expr>,
}

/// A scalar expression over the columns of a record batch.
///
/// Deserialized from the language-agnostic JSON IR emitted by the Python control
/// plane, so the variant tags (`e`, `op`) are the stable wire contract.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "e", rename_all = "snake_case", deny_unknown_fields)]
pub enum Expr {
    /// Reference to an input column by name.
    Col { name: String },
    /// A constant literal.
    Lit { value: Literal },
    /// A binary operation over two sub-expressions.
    Binary {
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    /// Logical negation of a boolean sub-expression.
    Not { input: Box<Expr> },

    /// Cast a sub-expression to a target Arrow type (by name). `try_cast` selects
    /// DuckDB `TRY_CAST` semantics: a value that cannot be converted yields NULL
    /// instead of erroring the query (arrow `safe` cast). The default (`false`)
    /// is a strict `CAST` that errors on an invalid value.
    Cast {
        input: Box<Expr>,
        dtype: String,
        #[serde(default)]
        try_cast: bool,
    },

    /// Null predicate (true where the argument is null).
    IsNull { input: Box<Expr> },

    /// Non-null predicate.
    IsNotNull { input: Box<Expr> },

    /// IEEE NaN predicate (true where a float value is NaN; null → null). A
    /// first-class op because the `!=` operator uses total ordering (NaN == NaN),
    /// so the `x != x` trick cannot detect NaN. The JIT falls back to interpret it.
    IsNan { input: Box<Expr> },

    /// IEEE infinity predicate (true where a float value is `+inf` or `-inf`;
    /// null → null). A first-class op because `±inf` literals do not survive the
    /// JSON IR, so a comparison against them cannot express this. The JIT falls back.
    IsInf { input: Box<Expr> },

    /// SQL CASE: the first branch whose `when` is true yields its `then`,
    /// otherwise `otherwise`.
    Case {
        branches: Vec<CaseBranch>,
        otherwise: Box<Expr>,
    },

    /// A string function over a Utf8 sub-expression.
    Str {
        #[serde(rename = "fn")]
        func: StrFunc,
        input: Box<Expr>,
        #[serde(default)]
        pattern: Option<String>,
        #[serde(default)]
        replacement: Option<String>,
        #[serde(default)]
        start: Option<i64>,
        #[serde(default)]
        length: Option<i64>,
    },

    /// A date/time field extraction over a Date/Timestamp sub-expression.
    Date {
        #[serde(rename = "fn")]
        func: DateFunc,
        input: Box<Expr>,
    },

    /// An image decode op over a binary (image-bytes) sub-expression. Decoding is
    /// library-backed (heavy), so the JIT falls back to this interpreter path.
    ///
    /// `mean`/`std`/`channels_first` apply only to `ToTensorF32`: per-channel
    /// normalization `(pixel/255 - mean) / std` and NCHW-vs-NHWC layout. They carry
    /// `#[serde(default)]`, so existing `to_tensor`/`decode`/`resize`/`dhash` IR — which
    /// never sets them — round-trips byte-for-byte.
    Image {
        #[serde(rename = "fn")]
        func: ImageFunc,
        input: Box<Expr>,
        #[serde(default)]
        width: Option<i64>,
        #[serde(default)]
        height: Option<i64>,
        #[serde(default)]
        mean: Option<Vec<f64>>,
        #[serde(default)]
        std: Option<Vec<f64>>,
        #[serde(default)]
        channels_first: bool,
        /// The container every bytes-out op re-encodes into (`png` when absent).
        ///
        /// It used to be `Encode`'s alone, with `Convert` borrowing the slot for a colour
        /// *mode*. That left every other bytes-out op — `resize`, `thumbnail`,
        /// `auto_orient`, and now fifteen more — hard-wired to PNG, which for photographic
        /// content is both several times slower to write and several times larger than the
        /// JPEG it was decoded from. A resize step that silently inflates a corpus is not
        /// a resize step anyone wants, so the slot is now universal and `Convert` has its
        /// own `mode`.
        #[serde(default)]
        format: Option<String>,
        /// `Convert` only: the target colour mode (`L`/`LA`/`RGB`/`RGBA`).
        #[serde(default)]
        mode: Option<String>,
        /// Encoder quality for the lossy containers, 1..=100 (JPEG's default is 75).
        /// Ignored by the lossless ones, where there is nothing to trade.
        #[serde(default)]
        quality: Option<i64>,
        /// The one scalar knob the photometric and hashing ops take, named per op:
        /// `rotate`'s degrees, `adjust_*`'s factor, `blur`/`sharpen`'s sigma/amount,
        /// `posterize`'s bit count, `solarize`'s threshold, `autocontrast`'s cutoff.
        ///
        /// One slot rather than six, because they are mutually exclusive by construction —
        /// an op reads exactly one — and six `Option<f64>`s on the wire would be five nulls
        /// in every image plan.
        #[serde(default)]
        factor: Option<f64>,
        /// `Letterbox`/`Pad` only: the byte value the leftover canvas is filled with.
        #[serde(default)]
        fill: Option<i64>,
    },

    /// An audio decode op over a binary (audio-bytes) sub-expression. Library-backed
    /// (symphonia), so the JIT falls back to this interpreter path (like `Image`).
    /// `rate` is the target sample rate for `AudioFunc::Resample` (ignored otherwise).
    Audio {
        #[serde(rename = "fn")]
        func: AudioFunc,
        input: Box<Expr>,
        #[serde(default)]
        rate: Option<i64>,
        /// `MelSpectrogram` only: STFT window size. `#[serde(default)]`, so the other audio
        /// ops' IR round-trips unchanged.
        #[serde(default)]
        n_fft: Option<i64>,
        /// `MelSpectrogram` only: STFT hop (stride) between frames.
        #[serde(default)]
        hop_length: Option<i64>,
        /// `MelSpectrogram`/`Mfcc`: number of mel filterbank bands.
        #[serde(default)]
        n_mels: Option<i64>,
        /// `Mfcc` only: number of cepstral coefficients to keep.
        #[serde(default)]
        n_mfcc: Option<i64>,
        /// `TrimSilence`/`SilenceRatio`: the silence floor in dBFS (negative), defaulting
        /// to -40. `RmsNormalize` reuses it as its *target* level, defaulting to -20.
        #[serde(default)]
        threshold_db: Option<i64>,
        /// The one fractional knob the newer ops take, named per op: `PreEmphasis`'s
        /// coefficient, `ClippingRatio`'s full-scale fraction, `SpectralRolloff`'s energy
        /// percentile. One slot rather than three, because an op reads exactly one.
        #[serde(default)]
        factor: Option<f64>,
        /// `Slice` only: where the window starts, in seconds.
        #[serde(default)]
        offset_secs: Option<f64>,
        /// `Slice`/`PadOrTrim`: how long the window is, in seconds.
        #[serde(default)]
        duration_secs: Option<f64>,
    },

    /// A crop over an image column whose window is **four sub-expressions rather than
    /// four constants**.
    ///
    /// Its own variant rather than four more `Option<i64>` on `Image`, because the
    /// distinction is real: every other image op's dimensions are part of its output
    /// *type* (a `to_tensor(224, 224)` column is `fixed_shape_tensor(uint8, [224,224,3])`
    /// and cannot be per-row), while a crop window is *data*. Cropping the bounding box a
    /// detector predicted is the operation a vision pipeline is built around, and with
    /// literal-only arguments it was the one thing this namespace could not express —
    /// leaving a per-row Python loop as the only way to cut objects out of frames.
    ///
    /// The bounds are evaluated to arrays once per batch and read per row, so a literal
    /// costs one broadcast array and needs no separate path. Output is PNG `Binary`; a
    /// window clipped by an edge yields the part that exists, one starting past the image
    /// yields null, and so does a null, negative, or empty window.
    ImageCrop {
        input: Box<Expr>,
        x: Box<Expr>,
        y: Box<Expr>,
        width: Box<Expr>,
        height: Box<Expr>,
    },

    /// A video decode op over a binary (video-bytes) sub-expression. Backed by the
    /// system FFmpeg behind the `video` feature; without it, evaluation errors. The
    /// JIT falls back to this interpreter path.
    Video {
        #[serde(rename = "fn")]
        func: VideoFunc,
        input: Box<Expr>,
        /// `Frames` only: how many evenly-spaced frames to sample from each clip.
        /// `#[serde(default)]` throughout, so `decode`'s IR round-trips byte-identical
        /// to what it was before the sampling ops existed.
        #[serde(default)]
        num_frames: Option<i64>,
        /// `Frames`/`Thumbnail`/`FrameAt`: the size every sampled frame is scaled to.
        /// A fixed size is what makes the output a fixed-shape tensor column rather than
        /// a ragged one, so it is required rather than defaulted to the clip's own size.
        #[serde(default)]
        width: Option<i64>,
        #[serde(default)]
        height: Option<i64>,
        /// `FrameAt` only: the timestamp to seek to, in seconds from the clip start —
        /// an **expression**, because a timestamp is data. The row that wants a still
        /// usually already carries the moment it wants (a detection, a caption, a scene
        /// boundary), so a constant here would make the one common case the one this
        /// cannot express. The other three arguments stay constants: they describe the
        /// output, not the row.
        #[serde(default)]
        second: Option<Box<Expr>>,
    },

    /// A biological-sequence op over a Utf8 (nucleotide, protein, or FASTQ-quality)
    /// sub-expression — the `.seq` namespace.
    ///
    /// Its own variant rather than more `StrFunc` arms because the argument shape is genuinely
    /// different: a `.str` function carries a pattern and a window, while these carry a k-mer
    /// length, a reading frame, an ASCII quality offset, and an alphabet. Folding them into
    /// `Str` would have meant six more `Option` slots on the wire shape every string function
    /// already pays for.
    ///
    /// Every field is `#[serde(default)]`, so each function's IR carries only the arguments it
    /// actually uses.
    Seq {
        #[serde(rename = "fn")]
        func: SeqFunc,
        input: Box<Expr>,
        /// `kmers`/`canonical_kmers`/`minimizers`: the k-mer length.
        #[serde(default)]
        k: Option<i64>,
        /// `minimizers`: how many consecutive k-mers one window spans.
        #[serde(default)]
        window: Option<i64>,
        /// `translate`: the reading frame, 0, 1, or 2.
        #[serde(default)]
        frame: Option<i64>,
        /// `phred_quality`/`mean_quality`/`expected_errors`: the FASTQ ASCII offset (33 for
        /// Sanger and Illumina 1.8+, 64 for the older Illumina pipelines). Stated rather than
        /// sniffed — the two ranges overlap, so the bytes carry no reliable signal and a wrong
        /// guess shifts every score by 31.
        #[serde(default)]
        offset: Option<i64>,
        /// `molecular_weight`/`is_valid`: which alphabet the column is written in.
        #[serde(default)]
        alphabet: Option<String>,
        /// `find_motif`/`count_motif`: the IUPAC-degenerate pattern.
        #[serde(default)]
        pattern: Option<String>,
        /// `translate`: stop at the first stop codon rather than running to the end.
        #[serde(default)]
        to_stop: bool,
    },

    /// First non-null among the sub-expressions, per row (SQL COALESCE).
    Coalesce { inputs: Vec<Expr> },

    /// `input IN (lit, …)` — true where the value is in the literal set, false where
    /// not, null where the input is null. Hash-set membership (`eval_in_list`), the
    /// O(1)-per-row form of an `(x = l0) OR (x = l1) OR …` chain a runtime join filter
    /// or the SQL `IN` list folds to.
    InList { input: Box<Expr>, set: Vec<Literal> },

    /// An array literal `[e0, e1, …]` — each row becomes a `List` of the
    /// per-row element values (all elements coerced to a common type).
    Array { elements: Vec<Expr> },

    /// Build a Date/Timestamp from integer inputs — the inverse of the `Date` field
    /// extractions.
    ///
    /// One variant covers both directions because they share every hard part (null
    /// propagation, range validation, the Arrow builder): `make_date`/`make_timestamp`
    /// assemble calendar *parts*, and the `from_unix_*` functions reinterpret a single
    /// *epoch count* at a stated unit. The unit has to be stated — an `Int64` column of
    /// epoch values carries no record of whether it counts seconds, millis, or micros,
    /// and a plain `CAST(x AS TIMESTAMP)` has to guess (it assumes microseconds), which
    /// silently turns epoch seconds into 1970.
    ///
    /// Arity is checked at evaluation: 3 args for `make_date`, 6 for `make_timestamp`,
    /// 1 for every `from_unix_*`. An out-of-range or non-existent date (month 13,
    /// February 30) yields null rather than erroring, so one bad row cannot abort a scan.
    MakeTemporal {
        #[serde(rename = "fn")]
        func: MakeTemporalFunc,
        args: Vec<Expr>,
    },

    /// `hash(e0, e1, …, seed)` — a deterministic 64-bit hash of the row's *values* → Int64.
    ///
    /// Typed, not textual: an integer hashes its bits, a float its (canonicalized) bits, a
    /// string its UTF-8. Order-sensitive, and null is a distinct positional value. Stable
    /// across partitions, runs, machines and versions (pinned by golden tests) — which is
    /// what a reproducible split, a surrogate key, and hash bucketing all rest on.
    Hash {
        inputs: Vec<Expr>,
        #[serde(default)]
        seed: i64,
    },

    /// `sequence(start, stop, step)` — the integer series from `start` to `stop`
    /// **inclusive**, stepping by `step` (Spark `sequence`). → `List<Int64>`.
    Sequence {
        start: Box<Expr>,
        stop: Box<Expr>,
        step: Box<Expr>,
    },

    /// A set operation between two `List` columns (`array_intersect`/`array_except`):
    /// the distinct left elements that are present in / absent from the right list.
    ListSet {
        #[serde(rename = "fn")]
        op: ListSetOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },

    /// Element-wise arithmetic between two equal-length numeric `List` columns
    /// (`list_add`/`list_subtract`/`list_multiply`): pairs elements positionally and
    /// returns a `List<Float64>`. The embedding-math primitive — sum two embedding
    /// columns, subtract a centroid, weight a vector — in the data plane.
    ListZip {
        #[serde(rename = "fn")]
        op: ListZipOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },

    /// `list.transform(func)` — apply the element sub-expression `func` (which reads
    /// the reserved `element` column) to every list element, preserving lengths.
    ListTransform { input: Box<Expr>, func: Box<Expr> },

    /// `list.filter(pred)` — keep the elements where the boolean element predicate
    /// `pred` (reading the reserved `element` column) is true.
    ListFilter { input: Box<Expr>, pred: Box<Expr> },

    /// Struct construction (SQL `struct_pack` / Spark `struct`) — each row becomes a
    /// `Struct` with the named fields, each field's value being the per-row value of
    /// its sub-expression. The read-side counterpart is `StructField`.
    MakeStruct { fields: Vec<NamedExpr> },

    /// A unary math function over a numeric sub-expression.
    Math {
        #[serde(rename = "fn")]
        func: MathFunc,
        input: Box<Expr>,
    },

    /// A scalar reduction over each row's `List` value (e.g. list length, sum).
    List {
        #[serde(rename = "fn")]
        func: ListFunc,
        input: Box<Expr>,
    },

    /// `NULLIF(left, right)`: null where `left == right`, else `left`.
    #[serde(rename = "nullif")]
    NullIf { left: Box<Expr>, right: Box<Expr> },

    /// `GREATEST(a, b, …)`: the largest argument per row, ignoring nulls.
    Greatest { inputs: Vec<Expr> },

    /// `LEAST(a, b, …)`: the smallest argument per row, ignoring nulls.
    Least { inputs: Vec<Expr> },

    /// A two-argument math function over numeric sub-expressions (→ Float64).
    Math2 {
        #[serde(rename = "fn")]
        func: Math2Func,
        left: Box<Expr>,
        right: Box<Expr>,
    },

    /// `list[index]` — the element at 0-based `index` of each row's `List`
    /// (null where the row is null or the index is out of range). Type-preserving.
    ListGet { input: Box<Expr>, index: i64 },

    /// A random-hyperplane (SimHash) signature of an embedding → `List<Int64>` of
    /// `num_bits` bits: the blocking key a vector similarity join needs, as
    /// `.str.minhash` is for Jaccard. See `eval::list_ops::simhash`.
    #[rustfmt::skip]
    ListSimhash { input: Box<Expr>, num_bits: i64, #[serde(default)] seed: i64 },

    /// `struct.field` — extract a named field from a `Struct` column
    /// (type-preserving; null where the struct row is null).
    StructField { input: Box<Expr>, field: String },

    /// `list.contains(value)` — true where any element equals the literal. → Bool.
    ListContains { input: Box<Expr>, value: Literal },

    /// `list.position(value)` — the 1-based index of the first element equal to the
    /// literal; null if absent (DuckDB `list_position`). → Int64.
    ListPosition { input: Box<Expr>, value: Literal },

    /// Map accessors over a `Map` column: `map_keys`/`map_values` (→ `List`) or
    /// `element_at` (per-row value for a literal `key`, null if absent).
    Map {
        #[serde(rename = "fn")]
        func: MapFunc,
        input: Box<Expr>,
        #[serde(default)]
        key: Option<Literal>,
    },

    /// `list.slice(offset, length)` — the 0-based sub-range of each row's `List`.
    ListSlice {
        input: Box<Expr>,
        offset: i64,
        #[serde(default)]
        length: Option<i64>,
    },

    /// `date_trunc(unit, ts)` — truncate a timestamp to the start of `unit`
    /// (year/month/day/hour/minute/second). → Timestamp(us).
    DateTrunc { input: Box<Expr>, unit: String },

    /// `strftime(ts, format)` — format a Date/Timestamp with a chrono/strftime
    /// `format` string (e.g. `%Y-%m-%d`). Null instants format to null. → Utf8.
    Strftime { input: Box<Expr>, format: String },

    /// `convert_timezone(from_tz, to_tz, ts)` — shift each naive timestamp's
    /// wall-clock from `from_tz` to `to_tz` (DST-aware). → Timestamp(us).
    ConvertTimezone {
        input: Box<Expr>,
        from_tz: String,
        to_tz: String,
    },

    /// `strptime(s, format)` — parse a Utf8 column into a Timestamp(microsecond)
    /// using a chrono/strftime `format`. Unparseable values → NULL (DuckDB
    /// `try_strptime`). The inverse of `Strftime`.
    Strptime { input: Box<Expr>, format: String },

    /// `offset_by` — shift a Date32/Timestamp by a calendar+fixed offset. `months`
    /// (incl. years×12) shift calendar months with end-of-month clamping; `days`
    /// (incl. weeks×7) and `micros` are exact. Months/days preserve a Date32;
    /// `micros != 0` on a Date32 errors (sub-day offset has no Date representation).
    /// Type-preserving (Date32→Date32, Timestamp→Timestamp). Null → null.
    DateOffset {
        input: Box<Expr>,
        #[serde(default)]
        months: i64,
        #[serde(default)]
        days: i64,
        #[serde(default)]
        micros: i64,
    },

    /// `list_join(list, sep)` — concatenate each row's `List` elements (cast to
    /// Utf8, nulls skipped) with `separator` → Utf8. Backs SQL `string_agg`.
    ListJoin { input: Box<Expr>, separator: String },

    /// `window_start(ts, width_micros, origin_micros)` — the start of the fixed-width
    /// tumbling window containing each instant: `origin + ⌊(t−origin)/width⌋·width`
    /// (floored, so negative instants land correctly). → Timestamp(us). Null → null.
    /// The event-time window-assignment expression: a windowed aggregation is a
    /// group-by on this key, so it reuses the existing mergeable aggregate.
    WindowStart {
        input: Box<Expr>,
        width_micros: i64,
        #[serde(default)]
        origin_micros: i64,
    },

    /// `window_buckets(ts, width_micros, slide_micros)` — the starts of every sliding
    /// window that contains each instant (`⌈width/slide⌉` of them) as a
    /// `List<Timestamp(us)>`. Fan it out with `Unnest` to one row per window, then
    /// group-by the start — sliding windows over the existing mergeable aggregate,
    /// no new stateful operator. Null → null.
    WindowBuckets {
        input: Box<Expr>,
        width_micros: i64,
        slide_micros: i64,
    },

    /// A pairwise reduction over two `List` columns of equal length per row
    /// (`dot`/`cosine_similarity`/`l2_distance`) → Float64. The vector-search
    /// primitives; the query vector is typically a broadcast `array(...)` literal.
    ListBinary {
        #[serde(rename = "fn")]
        func: ListBinaryFunc,
        left: Box<Expr>,
        right: Box<Expr>,
    },

    /// A geospatial function over zero or more sub-expressions.
    ///
    /// One variadic variant covers the whole `ST_*` surface rather than a node per
    /// arity, because every one of these functions has the same shape at the wire
    /// level — a name and an argument list — and the arities range from one to six.
    /// Splitting them would multiply the wire contract by nothing gained: the arity is
    /// checked against `GeoFunc::arity` at evaluation, which is where a mismatch has to
    /// be reported anyway.
    ///
    /// Geometry arguments and geometry results are **WKB in a `Binary` array**, which
    /// is what makes geospatial work fit the Arrow-only columnar contract without a new
    /// physical type. See `bc_geo` for the encoding and the algorithms.
    Geo {
        #[serde(rename = "fn")]
        func: GeoFunc,
        args: Vec<Expr>,
    },

    /// A rigid-body (SE(3)) function over numeric sub-expressions.
    ///
    /// Shaped like `Geo` and for the same reason — a name and an ordered argument list
    /// covers a whole family whose arities run from three to ten, and the count is
    /// checked against `SpatialFunc::arity` at evaluation.
    ///
    /// Unlike `Geo` this variant needs no new physical type at all: every argument and
    /// every result is a plain `Float64`, because a robotics log already stores poses
    /// and point clouds as scalar columns (`x`, `y`, `z`, `qx`, …). That is what lets a
    /// coordinate-frame transform be an ordinary projection — pushed down, spilled,
    /// shuffled and JIT-adjacent like any other arithmetic. See `bc_spatial` for the
    /// conventions and the mathematics.
    Spatial {
        #[serde(rename = "fn")]
        func: SpatialFunc,
        args: Vec<Expr>,
    },
}

/// The geospatial function vocabulary. Names mirror PostGIS so a ported query reads the
/// same, and the Python `fn` strings in `plan/expr_ir/fn_names.py::GEO_FNS` are these
/// serde tags exactly.
///
/// Grouped by what they take and return, because that is what a caller needs to know:
/// constructors take numbers or text and return a geometry, accessors take a geometry
/// and return a scalar, predicates take two geometries and return a boolean, and the
/// grid functions take plain numbers and return a cell id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeoFunc {
    // --- Constructors and codecs (→ geometry, unless noted) -----------------
    /// `st_point(x, y)`.
    StPoint,
    /// `st_point_z(x, y, z)`.
    StPointZ,
    /// `st_make_line(a, b)` — the two-position chain joining two points.
    StMakeLine,
    /// `st_make_polygon(ring)` — a polygon from a closed chain.
    StMakePolygon,
    /// `st_make_envelope(xmin, ymin, xmax, ymax)`.
    StMakeEnvelope,
    /// `st_geom_from_text(wkt)` — WKT, EWKT, GeoJSON or hex WKB, detected by content.
    StGeomFromText,
    /// `st_geom_from_wkb(bytes)` — re-validate a `Binary` column as geometry.
    StGeomFromWkb,
    /// `st_geom_from_geojson(json)`.
    StGeomFromGeojson,
    /// `st_geom_from_geohash(hash)` — the cell the hash names, as a rectangle.
    StGeomFromGeohash,
    /// `st_as_text(g)` → Utf8 WKT.
    StAsText,
    /// `st_as_ewkt(g)` → Utf8 WKT with an `SRID=` prefix.
    StAsEwkt,
    /// `st_as_binary(g)` → Binary WKB, without an SRID.
    StAsBinary,
    /// `st_as_ewkb(g)` → Binary EWKB, carrying the SRID.
    StAsEwkb,
    /// `st_as_hex_wkb(g)` → Utf8 hex EWKB.
    StAsHexWkb,
    /// `st_as_geojson(g)` → Utf8 RFC 7946 geometry object.
    StAsGeojson,

    // --- Accessors (geometry in, scalar out) --------------------------------
    /// `st_x(g)` → Float64; null unless `g` is a point.
    StX,
    /// `st_y(g)` → Float64; null unless `g` is a point.
    StY,
    /// `st_z(g)` → Float64; null unless `g` is a 3D point.
    StZ,
    /// `st_xmin(g)` → Float64.
    StXmin,
    /// `st_ymin(g)` → Float64.
    StYmin,
    /// `st_xmax(g)` → Float64.
    StXmax,
    /// `st_ymax(g)` → Float64.
    StYmax,
    /// `st_geometry_type(g)` → Utf8, the uppercase OGC name.
    StGeometryType,
    /// `st_dimension(g)` → Int64: 0 points, 1 lines, 2 areas.
    StDimension,
    /// `st_srid(g)` → Int64; 0 when unknown.
    StSrid,
    /// `st_set_srid(g, srid)` — relabel without moving a coordinate.
    StSetSrid,
    /// `st_num_points(g)` → Int64.
    StNumPoints,
    /// `st_num_geometries(g)` → Int64.
    StNumGeometries,
    /// `st_num_interior_rings(g)` → Int64.
    StNumInteriorRings,
    /// `st_geometry_n(g, n)` — the 1-based `n`-th member.
    StGeometryN,
    /// `st_point_n(g, n)` — the 1-based `n`-th position of a chain.
    StPointN,
    /// `st_start_point(g)`.
    StStartPoint,
    /// `st_end_point(g)`.
    StEndPoint,
    /// `st_exterior_ring(g)`.
    StExteriorRing,
    /// `st_interior_ring_n(g, n)`.
    StInteriorRingN,
    /// `st_is_empty(g)` → Boolean.
    StIsEmpty,
    /// `st_is_valid(g)` → Boolean.
    StIsValid,
    /// `st_is_valid_reason(g)` → Utf8; null when valid.
    StIsValidReason,
    /// `st_is_closed(g)` → Boolean.
    StIsClosed,
    /// `st_is_ring(g)` → Boolean.
    StIsRing,
    /// `st_is_simple(g)` → Boolean.
    StIsSimple,
    /// `st_is_collection(g)` → Boolean.
    StIsCollection,
    /// `st_has_z(g)` → Boolean.
    StHasZ,
    /// `st_coord_dim(g)` → Int64: 2 or 3.
    StCoordDim,

    // --- Planar measures (→ Float64, in coordinate units) -------------------
    /// `st_area(g)`.
    StArea,
    /// `st_length(g)` — chains only; a polygon reports 0.
    StLength,
    /// `st_perimeter(g)` — polygon boundaries only.
    StPerimeter,
    /// `st_distance(a, b)`.
    StDistance,
    /// `st_max_distance(a, b)`.
    StMaxDistance,
    /// `st_hausdorff_distance(a, b)`.
    StHausdorffDistance,
    /// `st_azimuth(a, b)` → radians clockwise from north.
    StAzimuth,

    // --- Geodesic measures (→ metres / square metres) -----------------------
    /// `st_distance_sphere(a, b)` — haversine metres between the nearest positions.
    StDistanceSphere,
    /// `st_distance_spheroid(a, b)` — Vincenty metres on WGS 84.
    StDistanceSpheroid,
    /// `st_area_spheroid(g)` — geodesic square metres.
    StAreaSpheroid,
    /// `st_length_spheroid(g)` — geodesic metres along chains.
    StLengthSpheroid,
    /// `st_perimeter_spheroid(g)` — geodesic metres around polygons.
    StPerimeterSpheroid,

    // --- Predicates (→ Boolean) ---------------------------------------------
    /// `st_intersects(a, b)`.
    StIntersects,
    /// `st_disjoint(a, b)`.
    StDisjoint,
    /// `st_contains(a, b)`.
    StContains,
    /// `st_within(a, b)`.
    StWithin,
    /// `st_covers(a, b)`.
    StCovers,
    /// `st_covered_by(a, b)`.
    StCoveredBy,
    /// `st_touches(a, b)`.
    StTouches,
    /// `st_crosses(a, b)`.
    StCrosses,
    /// `st_overlaps(a, b)`.
    StOverlaps,
    /// `st_equals(a, b)` — topological, not structural.
    StEquals,
    /// `st_dwithin(a, b, radius)`.
    StDwithin,
    /// `st_dwithin_sphere(a, b, metres)` — the geodesic radius test.
    StDwithinSphere,
    /// `st_intersects_extent(a, b)` — bounding boxes only. The cheap prefilter.
    StIntersectsExtent,
    /// `st_contains_extent(a, b)` — bounding boxes only.
    StContainsExtent,

    // --- Constructions (→ geometry) ------------------------------------------
    /// `st_centroid(g)`.
    StCentroid,
    /// `st_envelope(g)`.
    StEnvelope,
    /// `st_boundary(g)`.
    StBoundary,
    /// `st_convex_hull(g)`.
    StConvexHull,
    /// `st_point_on_surface(g)` — a position guaranteed to lie on the geometry.
    StPointOnSurface,
    /// `st_buffer(g, radius, quad_segs)` — an approximation; see `bc_geo::algo::construct`.
    StBuffer,
    /// `st_simplify(g, tolerance)`.
    StSimplify,
    /// `st_reverse(g)`.
    StReverse,
    /// `st_force_2d(g)`.
    StForce2d,
    /// `st_force_3d(g, z)`.
    StForce3d,
    /// `st_force_polygon_ccw(g)`.
    StForcePolygonCcw,
    /// `st_force_polygon_cw(g)`.
    StForcePolygonCw,
    /// `st_flip_coordinates(g)` — swap x and y.
    StFlipCoordinates,
    /// `st_translate(g, dx, dy)`.
    StTranslate,
    /// `st_scale(g, sx, sy)`.
    StScale,
    /// `st_rotate(g, radians)`.
    StRotate,
    /// `st_affine(g, a, b, d, e, xoff, yoff)`.
    StAffine,
    /// `st_snap_to_grid(g, size)`.
    StSnapToGrid,
    /// `st_segmentize(g, max_segment_length)`.
    StSegmentize,
    /// `st_expand(g, dx, dy)` — the bounding box grown on every side.
    StExpand,
    /// `st_collect(a, b)` — concatenate without an overlay.
    StCollect,
    /// `st_remove_repeated_points(g, tolerance)`.
    StRemoveRepeatedPoints,
    /// `st_line_interpolate_point(g, fraction)`.
    StLineInterpolatePoint,
    /// `st_line_locate_point(g, point)` → Float64 fraction.
    StLineLocatePoint,
    /// `st_line_substring(g, from, to)`.
    StLineSubstring,
    /// `st_closest_point(a, b)` — a position on `a`.
    StClosestPoint,
    /// `st_shortest_line(a, b)`.
    StShortestLine,
    /// `st_project(point, distance_m, azimuth_deg)` — the geodesic destination.
    StProject,
    /// `st_transform(g, from_srid, to_srid)`.
    StTransform,

    // --- Grid indexing (numbers in, cell id out) -----------------------------
    /// `st_geohash(g, precision)` → Utf8, from the geometry's centroid.
    StGeohash,
    /// `geohash_encode(lon, lat, precision)` → Utf8.
    GeohashEncode,
    /// `geohash_decode_lon(hash)` → Float64.
    GeohashDecodeLon,
    /// `geohash_decode_lat(hash)` → Float64.
    GeohashDecodeLat,
    /// `st_tile_x(lon, lat, zoom)` → Int64.
    StTileX,
    /// `st_tile_y(lon, lat, zoom)` → Int64.
    StTileY,
    /// `st_quadkey(lon, lat, zoom)` → Utf8.
    StQuadkey,
    /// `st_s2_cell(lon, lat, level)` → Int64.
    StS2Cell,
    /// `st_s2_cell_parent(cell, level)` → Int64.
    StS2CellParent,
    /// `st_hex_bin(x, y, size)` → Int64 packed axial key.
    StHexBin,
    /// `st_hex_center_x(key, size)` → Float64.
    StHexCenterX,
    /// `st_hex_center_y(key, size)` → Float64.
    StHexCenterY,
    /// `st_utm_zone(lon)` → Int64.
    StUtmZone,
    /// `st_utm_epsg(lon, lat)` → Int64.
    StUtmEpsg,
}

impl GeoFunc {
    /// The number of arguments this function takes.
    ///
    /// Checked at evaluation rather than at deserialization because the JSON IR carries
    /// an argument *list*: serde can prove it is a list of expressions, and only this
    /// table knows how long it should be.
    pub fn arity(self) -> usize {
        use GeoFunc::*;
        match self {
            // One geometry, or one text/number.
            StMakePolygon | StGeomFromText | StGeomFromWkb | StGeomFromGeojson
            | StGeomFromGeohash | StAsText | StAsEwkt | StAsBinary | StAsEwkb | StAsHexWkb
            | StAsGeojson | StX | StY | StZ | StXmin | StYmin | StXmax | StYmax
            | StGeometryType | StDimension | StSrid | StNumPoints | StNumGeometries
            | StNumInteriorRings | StStartPoint | StEndPoint | StExteriorRing | StIsEmpty
            | StIsValid | StIsValidReason | StIsClosed | StIsRing | StIsSimple | StIsCollection
            | StHasZ | StCoordDim | StArea | StLength | StPerimeter | StAreaSpheroid
            | StLengthSpheroid | StPerimeterSpheroid | StCentroid | StEnvelope | StBoundary
            | StConvexHull | StPointOnSurface | StReverse | StForce2d | StForcePolygonCcw
            | StForcePolygonCw | StFlipCoordinates | GeohashDecodeLon | GeohashDecodeLat
            | StUtmZone => 1,

            // Two: a pair of geometries, or a geometry and one parameter.
            StPoint
            | StMakeLine
            | StSetSrid
            | StGeometryN
            | StPointN
            | StInteriorRingN
            | StDistance
            | StMaxDistance
            | StHausdorffDistance
            | StAzimuth
            | StDistanceSphere
            | StDistanceSpheroid
            | StIntersects
            | StDisjoint
            | StContains
            | StWithin
            | StCovers
            | StCoveredBy
            | StTouches
            | StCrosses
            | StOverlaps
            | StEquals
            | StIntersectsExtent
            | StContainsExtent
            | StSimplify
            | StForce3d
            | StRotate
            | StSnapToGrid
            | StSegmentize
            | StCollect
            | StRemoveRepeatedPoints
            | StLineInterpolatePoint
            | StLineLocatePoint
            | StClosestPoint
            | StShortestLine
            | StGeohash
            | StHexCenterX
            | StHexCenterY
            | StS2CellParent
            | StUtmEpsg => 2,

            // Three.
            StPointZ | StDwithin | StDwithinSphere | StBuffer | StTranslate | StScale
            | StExpand | StLineSubstring | StProject | StTransform | GeohashEncode | StTileX
            | StTileY | StQuadkey | StS2Cell | StHexBin => 3,

            // Four and up.
            StMakeEnvelope => 4,
            StAffine => 7,
        }
    }

    /// True when the function's result is a geometry (a WKB `Binary` column).
    ///
    /// The one property callers outside the evaluator need: it decides the output
    /// Arrow type, and it is what lets the schema be known before a row is read.
    pub fn returns_geometry(self) -> bool {
        use GeoFunc::*;
        matches!(
            self,
            StPoint
                | StPointZ
                | StMakeLine
                | StMakePolygon
                | StMakeEnvelope
                | StGeomFromText
                | StGeomFromWkb
                | StGeomFromGeojson
                | StGeomFromGeohash
                | StSetSrid
                | StGeometryN
                | StPointN
                | StStartPoint
                | StEndPoint
                | StExteriorRing
                | StInteriorRingN
                | StCentroid
                | StEnvelope
                | StBoundary
                | StConvexHull
                | StPointOnSurface
                | StBuffer
                | StSimplify
                | StReverse
                | StForce2d
                | StForce3d
                | StForcePolygonCcw
                | StForcePolygonCw
                | StFlipCoordinates
                | StTranslate
                | StScale
                | StRotate
                | StAffine
                | StSnapToGrid
                | StSegmentize
                | StExpand
                | StCollect
                | StRemoveRepeatedPoints
                | StLineInterpolatePoint
                | StLineSubstring
                | StClosestPoint
                | StShortestLine
                | StProject
                | StTransform
        )
    }
}

/// The rigid-body function vocabulary — rotations and poses in three dimensions.
///
/// Every one of these takes `Float64` arguments and returns `Float64`, and the Python
/// `fn` strings in `plan/expr_ir/fn_names.py::SPATIAL_FNS` are these serde tags exactly.
///
/// # Why one function per output component
///
/// Rotating a point produces three numbers, and the name says which one:
/// `quat_rotate_x`, `quat_rotate_y`, `quat_rotate_z`. A single node returning a struct
/// or a fixed-size list would be one evaluation instead of three, and it would also put
/// a composite type in the middle of every pipeline that then has to be taken apart
/// again before a filter or a join key can touch it. Scalar in, scalar out keeps a
/// coordinate transform in the same class as `a * b + c`: projectable, pushable, and
/// spillable with no unpacking. It is the same choice `geohash_decode_lon` /
/// `geohash_decode_lat` already made.
///
/// The cost is real and is not claimed away: the three component functions each
/// normalize the quaternion and each build the same two cross products, so a full
/// transform does that work three times where a struct-returning node would do it once.
/// What buys it back is a column layout the rest of the engine already knows how to
/// move, and a plan of three nodes rather than the forty an equivalent arithmetic tree
/// expands to.
///
/// Whether the fused kernel is also *faster* than that arithmetic tree is a separate
/// question and is **not** settled here: the kernel is a scalar row loop and the tree is
/// a chain of vectorized arrow kernels. `benchmarks/scenarios/sweep_transform.py` is the
/// measurement, and it needs a release build to mean anything.
///
/// # Conventions
///
/// Quaternion arguments are always four separate values in `(x, y, z, w)` order, scalar
/// last. Poses are seven, translation first. See the `bc_spatial` crate documentation
/// for the full convention table and why each was chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpatialFunc {
    // --- Quaternion properties (4 args: qx, qy, qz, qw) ---------------------
    /// `quat_norm(qx, qy, qz, qw)` — the four-component length. One for a rotation;
    /// how far a logged quaternion has drifted from that is worth being able to ask.
    QuatNorm,
    /// `quat_normalize_x(...)` — the X component of the same rotation, unit length.
    QuatNormalizeX,
    /// `quat_normalize_y(...)`.
    QuatNormalizeY,
    /// `quat_normalize_z(...)`.
    QuatNormalizeZ,
    /// `quat_normalize_w(...)`.
    QuatNormalizeW,
    /// `quat_inverse_x(...)` — the X component of the inverse rotation.
    QuatInverseX,
    /// `quat_inverse_y(...)`.
    QuatInverseY,
    /// `quat_inverse_z(...)`.
    QuatInverseZ,
    /// `quat_inverse_w(...)`.
    QuatInverseW,
    /// `quat_angle(qx, qy, qz, qw)` — the rotation's magnitude in radians, on `[0, pi]`.
    QuatAngle,
    /// `quat_to_roll(...)` — rotation about X, in radians.
    QuatToRoll,
    /// `quat_to_pitch(...)` — rotation about Y, in radians.
    QuatToPitch,
    /// `quat_to_yaw(...)` — rotation about Z, in radians. The heading, and the one of
    /// the three a map-matching or planning query actually asks for.
    QuatToYaw,

    // --- Euler to quaternion (3 args: roll, pitch, yaw) ---------------------
    /// `quat_from_euler_x(roll, pitch, yaw)`.
    QuatFromEulerX,
    /// `quat_from_euler_y(roll, pitch, yaw)`.
    QuatFromEulerY,
    /// `quat_from_euler_z(roll, pitch, yaw)`.
    QuatFromEulerZ,
    /// `quat_from_euler_w(roll, pitch, yaw)`.
    QuatFromEulerW,

    // --- Rotation matrix to quaternion (9 args: row-major m00..m22) ---------
    /// `quat_from_rotmat_x(m00, m01, m02, m10, m11, m12, m20, m21, m22)`.
    QuatFromRotmatX,
    /// `quat_from_rotmat_y(...)`.
    QuatFromRotmatY,
    /// `quat_from_rotmat_z(...)`.
    QuatFromRotmatZ,
    /// `quat_from_rotmat_w(...)`.
    QuatFromRotmatW,

    // --- Quaternion composition (8 args: a then b, four each) ---------------
    /// `quat_multiply_x(ax, ay, az, aw, bx, by, bz, bw)` — the X component of `a * b`,
    /// the rotation that applies `b` first and then `a`.
    QuatMultiplyX,
    /// `quat_multiply_y(...)`.
    QuatMultiplyY,
    /// `quat_multiply_z(...)`.
    QuatMultiplyZ,
    /// `quat_multiply_w(...)`.
    QuatMultiplyW,
    /// `quat_angular_distance(ax, ay, az, aw, bx, by, bz, bw)` — the geodesic angle
    /// between two rotations, in radians on `[0, pi]`. The honest error metric for an
    /// orientation estimate; a component-wise difference is not, because `q` and `-q`
    /// are the same rotation.
    QuatAngularDistance,

    // --- Interpolation (9 args: a, b, t) ------------------------------------
    /// `quat_slerp_x(ax, ay, az, aw, bx, by, bz, bw, t)` — spherical interpolation,
    /// `a` at `t = 0` and `b` at `t = 1`, extrapolating outside that range.
    QuatSlerpX,
    /// `quat_slerp_y(...)`.
    QuatSlerpY,
    /// `quat_slerp_z(...)`.
    QuatSlerpZ,
    /// `quat_slerp_w(...)`.
    QuatSlerpW,

    // --- Rotating a vector (7 args: qx, qy, qz, qw, px, py, pz) -------------
    /// `quat_rotate_x(qx, qy, qz, qw, px, py, pz)` — the X component of the rotated
    /// vector. Rotation only: use the `se3_*` functions when there is a translation too.
    QuatRotateX,
    /// `quat_rotate_y(...)`.
    QuatRotateY,
    /// `quat_rotate_z(...)`.
    QuatRotateZ,
    /// `quat_inverse_rotate_x(...)` — rotated by the inverse, the direction a
    /// world-frame vector travels to reach a body frame.
    QuatInverseRotateX,
    /// `quat_inverse_rotate_y(...)`.
    QuatInverseRotateY,
    /// `quat_inverse_rotate_z(...)`.
    QuatInverseRotateZ,

    // --- Pose application (10 args: tx, ty, tz, qx, qy, qz, qw, px, py, pz) -
    /// `se3_transform_x(tx, ty, tz, qx, qy, qz, qw, px, py, pz)` — the X coordinate of
    /// `point` moved out of the pose's frame and into its parent. Rotate, then
    /// translate. The single most-run function in this family: it is what turns a lidar
    /// return into a world-frame point.
    Se3TransformX,
    /// `se3_transform_y(...)`.
    Se3TransformY,
    /// `se3_transform_z(...)`.
    Se3TransformZ,
    /// `se3_inverse_transform_x(...)` — the X coordinate of a parent-frame point
    /// expressed in the pose's own frame. Subtract, then rotate by the inverse.
    Se3InverseTransformX,
    /// `se3_inverse_transform_y(...)`.
    Se3InverseTransformY,
    /// `se3_inverse_transform_z(...)`.
    Se3InverseTransformZ,
}

impl SpatialFunc {
    /// The number of arguments this function takes.
    ///
    /// Checked at evaluation for the same reason `GeoFunc::arity` is: the JSON IR
    /// carries an argument *list*, so serde can prove it is a list of expressions and
    /// only this table knows how long it should be.
    pub fn arity(self) -> usize {
        use SpatialFunc::*;
        match self {
            // Euler angles in.
            QuatFromEulerX | QuatFromEulerY | QuatFromEulerZ | QuatFromEulerW => 3,

            // One quaternion.
            QuatNorm | QuatNormalizeX | QuatNormalizeY | QuatNormalizeZ | QuatNormalizeW
            | QuatInverseX | QuatInverseY | QuatInverseZ | QuatInverseW | QuatAngle
            | QuatToRoll | QuatToPitch | QuatToYaw => 4,

            // One quaternion and one vector.
            QuatRotateX | QuatRotateY | QuatRotateZ | QuatInverseRotateX | QuatInverseRotateY
            | QuatInverseRotateZ => 7,

            // Two quaternions.
            QuatMultiplyX | QuatMultiplyY | QuatMultiplyZ | QuatMultiplyW | QuatAngularDistance => {
                8
            }

            // Two quaternions and a parameter, or a 3x3 matrix.
            QuatSlerpX | QuatSlerpY | QuatSlerpZ | QuatSlerpW | QuatFromRotmatX
            | QuatFromRotmatY | QuatFromRotmatZ | QuatFromRotmatW => 9,

            // A pose and a point.
            Se3TransformX | Se3TransformY | Se3TransformZ | Se3InverseTransformX
            | Se3InverseTransformY | Se3InverseTransformZ => 10,
        }
    }
}

/// Pairwise list reductions over two equal-length numeric `List` columns (→ Float64).
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ListBinaryFunc {
    /// Dot product `Σ aᵢ·bᵢ` over the paired elements.
    Dot,
    /// Cosine similarity `dot(a,b) / (‖a‖·‖b‖)`; null if either vector has zero norm.
    CosineSimilarity,
    /// Euclidean distance `sqrt(Σ (aᵢ−bᵢ)²)` between the two vectors.
    L2Distance,
    /// Manhattan / L1 distance `Σ |aᵢ−bᵢ|` between the two vectors — the metric some
    /// embedding models (and sparse features) are trained under.
    L1Distance,
    /// Hamming distance: the number of positions where the two vectors differ. The metric
    /// for **binary / quantized embeddings** (each element 0/1 or a small int), where it is
    /// far cheaper than a float distance and is what a binary vector index ranks by.
    Hamming,
    /// The fraction of positions where the two lists hold the same value. Over a pair of
    /// `minhash` signatures this is the standard unbiased estimator of the documents'
    /// Jaccard similarity; over arbitrary lists it is simply the agreement rate.
    Jaccard,
    /// The clipped multiset intersection size `Σ_v min(count_left(v), count_right(v))` —
    /// how many of the left list's elements the right can account for, **counting
    /// repeats**. Unlike `array_intersect(...).len()` a value repeated four times on the
    /// left and once on the right contributes 1, not 4. That clip is the definition of
    /// BLEU's modified n-gram precision and of ROUGE-N's numerator, and it is what stops a
    /// degenerate `the the the the` from scoring a perfect unigram precision. Order-free
    /// and type-general (n-gram strings, token ids); a null row on either side → null, a
    /// null element matches nothing.
    MultisetOverlap,
    /// The length of the longest common **subsequence** of the two lists — the one overlap
    /// measure that reads order. `MultisetOverlap` cannot tell `the cat sat` from
    /// `sat cat the`; this scores the second far lower, which is the difference between
    /// ROUGE-N and ROUGE-L and why summarization is scored with the latter. The subsequence
    /// need not be contiguous.
    ///
    /// **`O(n·m)` per row**, against `O(n+m)` for every other list op here. Fine on tokenized
    /// sentences, a real cost on two thousand-token documents. A null row on either side →
    /// null; a null element matches nothing and cannot extend a subsequence.
    LcsLength,
}

/// Two-argument math functions (→ Float64).
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Math2Func {
    /// `pow(a, b)` = a raised to b.
    Pow,
    /// `atan2(y, x)`.
    Atan2,
    /// `round(x, digits)` — round to `digits` decimal places.
    Round,
    /// `gcd(a, b)` — greatest common divisor of two integers (DuckDB `gcd`).
    Gcd,
    /// `lcm(a, b)` — least common multiple of two integers (DuckDB `lcm`).
    Lcm,
    /// `hypot(a, b)` = sqrt(a² + b²), the Euclidean norm (DuckDB `hypot`).
    Hypot,
    /// `nextafter(a, b)` — the next representable `f64` after `a` in the direction of
    /// `b` (DuckDB `nextafter`). One ULP, which is what makes it useful for testing a
    /// boundary; `a + tiny` cannot express it.
    NextAfter,
}

/// Image decode operations for the `.image` namespace. `Decode` reads each
/// image's dimensions into a `{width, height}` struct; `ToTensor` decodes,
/// resizes to `(width, height)`, and flattens to a fixed-size RGB8 pixel list.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageFunc {
    /// `brightness()` → the mean luma of the image, normalized to `[0, 1]` (→ Float64).
    /// The blank-image detector: a placeholder tile, a blown-out scan, and the grey box a CDN
    /// serves for a missing asset all sit at an extreme, while a photograph of anything lands
    /// in the middle. Measured on a downsampled luma plane, so the cost is independent of the
    /// source resolution. Null/undecodable → null.
    Brightness,
    /// `sharpness()` → the variance of the Laplacian of the luma plane, normalized to
    /// `[0, 1]` (→ Float64). The standard focus measure: a sharp image has strong second
    /// derivatives at its edges, a blurred or empty one has almost none. Downsampled first,
    /// deliberately — full-resolution sensor noise reads as detail and makes a blurry large
    /// photograph score like a sharp one. It measures *detail*, not quality: a brick wall
    /// outscores a portrait. Null/undecodable → null.
    Sharpness,
    Decode,
    ToTensor,
    /// `to_grayscale(width, height)` → decode, resize to `(width, height)`, and convert to a
    /// single luminance channel (Rec.601), emitted `FixedSizeList<UInt8>` of shape
    /// `(height, width, 1)`. The color-convert step for models that take 1-channel input
    /// (many medical / document / depth models). Null/undecodable input → null.
    /// → FixedSizeList&lt;UInt8&gt;.
    ToGrayscale,
    /// `center_crop(width, height)` → decode and crop the centered `(height, width)` region,
    /// emitted `FixedSizeList<UInt8>` in HWC (RGB8). The second half of the standard vision
    /// inference transform (resize the short side, then center-crop to the model input); when
    /// the image is smaller than the crop it is zero-padded, matching torchvision `CenterCrop`.
    /// Null/undecodable input → null. → FixedSizeList&lt;UInt8&gt;.
    CenterCrop,
    /// `to_tensor_f32(width, height, mean, std, channels_first)` → the model-ready
    /// float tensor: decode, resize, scale to `[0, 1]` (`pixel / 255`), then optionally
    /// apply per-channel `(x - mean) / std`, emitted `FixedSizeList<Float32>` in HWC
    /// (default) or CHW layout. This is the step every torch/JAX vision model needs
    /// between `ToTensor` and the forward pass; doing it natively keeps the pipeline in
    /// the engine instead of exiting to a per-batch Python UDF (`x/255`, `Normalize`,
    /// `permute`). Null/undecodable input → null. → FixedSizeList&lt;Float32&gt;.
    ToTensorF32,
    /// `resize(width, height)` → re-encoded PNG bytes at the new size (Daft
    /// `image.resize`). Null/undecodable input → null. → Binary.
    Resize,
    /// `encode(format)` → the image re-encoded in `format` (`png`, `jpeg`, `bmp`, `gif`),
    /// pixels unchanged. Normalizes a mixed-format corpus to one codec, or trades a PNG
    /// for a smaller JPEG. Null/undecodable input → null. → Binary.
    Encode,
    /// `convert(mode)` → the image converted to color mode `L`, `LA`, `RGB`, or `RGBA`,
    /// re-encoded as PNG. The general form of `ToGrayscale`, which is `L` plus a resize;
    /// this changes only the channels, so it is the step for normalizing a corpus that
    /// mixes RGB and RGBA before a model that wants one of them. Reads the `format` slot,
    /// like `Encode`. Null/undecodable input → null. → Binary.
    Convert,
    /// `dhash()` → a 64-bit *difference hash*: the perceptual fingerprint that makes
    /// image near-duplicate detection expressible. Two visually similar images differ
    /// in few bits, so `bit_count(a ^ b)` is their Hamming distance and a threshold on
    /// it is a similarity join. Null/undecodable input → null. → Int64 (the 64 bits
    /// reinterpreted: the FFI boundary rejects a `u64` above `i64::MAX`).
    Dhash,
    /// `auto_orient()` → the image rotated/flipped per its Exif `Orientation` tag, as PNG
    /// bytes. A camera records which way up it was held rather than rotating the sensor
    /// data, so a portrait phone photo is *stored* landscape with a "rotate 90" note. Every
    /// viewer honours that note, and so does anything built on `PIL.ImageOps.exif_transpose`
    /// or `cv2.imread`; the decoder under this namespace does not. Without this, a corpus of
    /// phone photographs decodes a quarter turn from what the rest of the pipeline sees —
    /// right shape, real pixels, wrong image. Null/undecodable input → null. → Binary.
    AutoOrient,
    /// `exif_orientation()` → the Exif orientation code, 1..8 (Int32). The diagnostic half
    /// of `AutoOrient`, since whether a corpus needs orienting is otherwise invisible. `1`
    /// ("already upright") is reported both for an image carrying no tag and for a format
    /// that cannot carry one, because that is what the code means. Null input → null.
    ExifOrientation,
    /// `thumbnail(max_size)` → the image scaled so its **longest side** is `max_size`,
    /// as PNG bytes. The aspect-preserving counterpart of `Resize`, which takes both
    /// dimensions and therefore stretches anything not already at the target ratio — a
    /// distortion no shape assertion can see. Never upscales, matching Pillow's
    /// `Image.thumbnail`. Reads the `width` slot. Null/undecodable → null. → Binary.
    Thumbnail,
    /// `letterbox(width, height, fill)` → aspect-preserving fit onto a `(width, height)`
    /// canvas with the remainder filled, flattened to RGB8. The standard object-detection
    /// preprocessing: `ToTensor` stretches, which moves every predicted box off its
    /// object, and `CenterCrop` discards the border, which is where the missed detections
    /// live. `fill` defaults to 114, the YOLO family's grey. Null/undecodable → null.
    /// → FixedSizeList&lt;UInt8&gt; of `height * width * 3`.
    Letterbox,
    /// `rotate(degrees)` → the image turned by a multiple of 90 degrees, re-encoded.
    /// Only right angles: a free rotation resamples every pixel and leaves a border no
    /// caller asked for, while 90/180/270 is a transposition that is exact and lossless.
    /// Negative and >360 values are normalized (`-90` == `270`). Null/undecodable → null.
    Rotate,
    /// `flip_horizontal()` → the image mirrored left-to-right, re-encoded. The single
    /// most-used training-time augmentation, and the one that must happen *before* the
    /// tensor step so a detector's boxes can be flipped with it.
    FlipHorizontal,
    /// `flip_vertical()` → the image mirrored top-to-bottom, re-encoded.
    FlipVertical,
    /// `pad(width, height, fill)` → the image centered on a `(width, height)` canvas
    /// filled with `fill`, **without** scaling. The difference from `Letterbox` is that
    /// nothing is resampled: a canvas smaller than the image crops it. This is what makes
    /// a corpus of unequal-size crops batchable without touching a single pixel value.
    Pad,
    /// `adjust_brightness(factor)` → every channel scaled by `factor` and clamped.
    /// `1.0` is the identity, `0.0` black, `2.0` twice as bright — the same convention as
    /// `PIL.ImageEnhance.Brightness`, so an augmentation policy ports over unchanged.
    AdjustBrightness,
    /// `adjust_contrast(factor)` → each channel pushed away from (or toward) the image's
    /// mean luma by `factor`. `1.0` identity, `0.0` a flat grey field.
    /// Matches `PIL.ImageEnhance.Contrast`.
    AdjustContrast,
    /// `adjust_saturation(factor)` → each pixel interpolated between its grey (Rec.601)
    /// and its colour by `factor`. `0.0` is grayscale, `1.0` identity, `>1` more vivid.
    /// Matches `PIL.ImageEnhance.Color`.
    AdjustSaturation,
    /// `adjust_hue(degrees)` → every hue rotated by `degrees` around the colour wheel,
    /// saturation and value untouched. The colour-jitter axis the other three cannot
    /// express, and the one a robustness sweep varies.
    AdjustHue,
    /// `blur(sigma)` → a Gaussian blur of standard deviation `sigma` pixels. Both an
    /// augmentation and a curation tool: blurring a copy and comparing hashes separates
    /// images that carry fine detail from ones that are already soft.
    Blur,
    /// `sharpen(amount)` → an unsharp mask: the image plus `amount` times the difference
    /// between it and a Gaussian blur of it. `amount` 0 is the identity.
    Sharpen,
    /// `invert()` → the photographic negative of each colour channel (alpha untouched).
    Invert,
    /// `posterize(bits)` → each channel reduced to its top `bits` bits (1..=8). One of the
    /// AutoAugment/RandAugment primitives, and a cheap way to make a corpus's colour
    /// quantization uniform. `8` is the identity.
    Posterize,
    /// `solarize(threshold)` → every channel value at or above `threshold` (0..=255)
    /// inverted, the rest left alone. The other AutoAugment primitive.
    Solarize,
    /// `equalize()` → per-channel histogram equalization, so the tonal range is used
    /// evenly. What rescues an under-exposed scan without a model in the loop.
    Equalize,
    /// `autocontrast(cutoff)` → each channel linearly rescaled so its darkest and
    /// brightest surviving values hit 0 and 255, ignoring the bottom and top `cutoff`
    /// percent of the histogram. `PIL.ImageOps.autocontrast`. Gentler than `Equalize`:
    /// it stretches the range without redistributing within it.
    ///
    /// Spelled `autocontrast` on the wire rather than the derived `auto_contrast`: it is
    /// `PIL.ImageOps.autocontrast`'s own name, and the vocabulary is worth more than the
    /// consistency of a rename rule.
    #[serde(rename = "autocontrast")]
    AutoContrast,
    /// `phash()` → a 64-bit **DCT** perceptual hash (→ Int64, reinterpreted like
    /// [`ImageFunc::Dhash`]). Where `dhash` compares adjacent pixels, this keeps the 8x8
    /// lowest-frequency DCT coefficients of a 32x32 luma reduction and thresholds them at
    /// their median — the standard `pHash`. It survives rotation-free re-encoding, heavy
    /// rescaling, and moderate cropping far better than `dhash`, which is why a dedup pass
    /// over a scraped corpus usually wants this one and a *fast* pre-filter wants `dhash`.
    Phash,
    /// `ahash()` → a 64-bit **average** hash (→ Int64): an 8x8 luma reduction thresholded
    /// at its own mean. The cheapest of the three and the least discriminating; it exists
    /// because a Hamming pre-filter wants a hash that costs almost nothing.
    Ahash,
    /// `entropy()` → the Shannon entropy of the luma histogram in bits, 0..=8 (→ Float64).
    /// A blank tile is ~0, a photograph 6-8. Complementary to `Brightness`, which cannot
    /// tell a mid-grey placeholder from a scene, and cheaper than `Sharpness`.
    Entropy,
    /// `colorfulness()` → the Hasler-Süsstrunk colourfulness metric (→ Float64). It
    /// separates a genuinely colour image from a scan, a line drawing, or a sepia-toned
    /// duplicate, which no luma measure can see. Roughly 0 for grey, 15+ for vivid.
    Colorfulness,
    /// `mean_color()` → struct `{r, g, b}` of Float64 channel means in 0..=255. The
    /// cheapest colour summary there is: it makes "find the images on a white background"
    /// and "cluster a corpus by palette" ordinary predicates. Null/undecodable → null.
    MeanColor,
    /// `is_grayscale()` → whether every pixel has R == G == B (→ Boolean). A corpus is
    /// full of greyscale images *stored* as RGB, which no header field reports and which
    /// silently triples the cost of every downstream tensor.
    IsGrayscale,
    /// `aspect_ratio()` → width / height as Float64, from the **header** alone. The
    /// orientation and letterboxing decisions of a whole pipeline hang on it, and paying
    /// a full decode to learn it is what made people skip the check.
    AspectRatio,
    /// `has_alpha()` → whether the image's colour type carries an alpha channel
    /// (→ Boolean), from the header alone. The flag that decides whether a corpus needs
    /// flattening before a model that takes 3 channels.
    HasAlpha,
    /// `format()` → the container format's lowercase name (`"png"`, `"jpeg"`, `"gif"`,
    /// `"bmp"`, `"webp"`, …) as Utf8, sniffed from the magic bytes rather than from a file
    /// extension — which is how a corpus full of `.jpg` files that are really PNGs gets
    /// found. Null when the bytes match no known container.
    Format,
}

/// Element-wise arithmetic between two equal-length numeric `List` columns (the `.list`
/// vector-math methods). Wire tags are snake_case (`list_add`/`list_subtract`/`list_multiply`).
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ListZipOp {
    #[serde(rename = "list_add")]
    Add,
    #[serde(rename = "list_subtract")]
    Subtract,
    #[serde(rename = "list_multiply")]
    Multiply,
}

/// Set operations between two `List` columns (the `.list` set methods). Wire tags
/// are snake_case (`array_intersect` / `array_except` / `array_union`).
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ListSetOp {
    #[serde(rename = "array_intersect")]
    Intersect,
    #[serde(rename = "array_except")]
    Except,
    #[serde(rename = "array_union")]
    Union,
    /// `list_concat(a, b)` — the left list's elements followed by the right's, with
    /// **no** deduplication and **no** reordering. It rides `ListSetOp` because the two
    /// operands and the list result are the same shape, but it is not a set operation:
    /// duplicates survive, and a NULL list counts as empty rather than making the row
    /// null (DuckDB `list_concat(NULL, [1])` is `[1]`, where `list_union(NULL, [1])` is
    /// NULL).
    #[serde(rename = "array_concat")]
    Concat,
    /// `array_gather(values, indices)` — each row's elements at the positions its `indices`
    /// row names. Like `Concat` it rides this family for its shape (two lists in, one list
    /// out) rather than because it is a set operation. It is what makes `arg_sort` usable:
    /// the indices that rank a score vector are spent by gathering the candidates with them,
    /// so a rerank stays in the engine. A negative index counts from the end (as `list.get`
    /// does) and an out-of-range one yields a null element rather than an error, because a
    /// `head(k)` wider than the row is ordinary. A null row on either side → null row.
    #[serde(rename = "array_gather")]
    Gather,
}

/// Audio-decode operations for the `.audio` namespace. `Decode` reads each clip's
/// metadata into a struct; `ToWaveform` decodes to a mono `List<Float32>` signal;
/// `Resample` decodes then band-limited-resamples that signal to the `rate` on the
/// [`Expr::Audio`] node (the target sample rate), also a mono `List<Float32>`.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioFunc {
    Decode,
    ToWaveform,
    Resample,
    /// `mel_spectrogram(rate, n_fft, hop_length, n_mels)` → the mel **power** spectrogram
    /// that is the front end of every speech model (Whisper, wav2vec2, HuBERT): resample to
    /// `rate`, STFT with a periodic Hann window and centered reflect padding, power spectrum
    /// (`|.|²`), then an HTK-scale mel filterbank. Emitted as a `List<Float32>` of
    /// `n_mels * n_frames` in row-major `(n_mels, n_frames)` order (`n_frames` follows the
    /// clip length, so the row length varies across unequal clips — reshape by `n_mels`).
    /// Numerically matches `torchaudio.transforms.MelSpectrogram` defaults (`power=2.0`, `norm=None`,
    /// `mel_scale="htk"`, `center=True`, `pad_mode="reflect"`) — the log/normalization step
    /// varies by model, so it is applied downstream, not baked in. Null/undecodable → null.
    MelSpectrogram,
    /// `trim_silence(threshold_db)` → the decoded waveform with leading and trailing samples
    /// below the threshold removed, as a mono `List<Float32>`. dBFS relative to full scale;
    /// -40 (1% of full scale) is the conventional default. Only the *ends* are trimmed —
    /// interior pauses carry the timing an acoustic model reads. A clip quiet throughout
    /// trims to an empty list, which is what makes a silent-recording filter expressible.
    TrimSilence,
    /// `peak_normalize()` → the decoded waveform scaled so its loudest sample sits at full
    /// scale, as a mono `List<Float32>`. The level-matching step before batching clips from
    /// different sources. Peak, not loudness (LUFS): it equalizes the maximum, not the
    /// perceived level. An all-zero clip is returned unchanged rather than divided by zero.
    PeakNormalize,
    /// `zero_crossing_rate()` → the fraction of adjacent sample pairs that change sign, as
    /// Float64. The classic voiced/unvoiced descriptor: a vowel crosses zero rarely, a
    /// fricative constantly. A clip shorter than two samples yields null.
    ZeroCrossingRate,
    /// `mfcc(rate, n_fft, hop_length, n_mels, n_mfcc)` → the Mel-Frequency Cepstral
    /// Coefficients, the classic compact speech feature: mel power spectrogram →
    /// `AmplitudeToDB` → orthonormal DCT-II, keeping the first `n_mfcc` coefficients.
    /// Emitted as a `List<Float32>` row-major `(n_mfcc, n_frames)`. Numerically matches
    /// `torchaudio.transforms.MFCC` defaults. Null/undecodable → null.
    Mfcc,
    /// `rms()` → the root-mean-square amplitude of the clip, 0..=1 (→ Float64). The level
    /// measure that actually tracks perceived loudness, unlike the peak: a recording with
    /// one door slam has a peak of 1.0 and an RMS that still says "quiet".
    Rms,
    /// `dbfs()` → the RMS level in decibels relative to full scale (→ Float64, negative).
    /// The unit every audio tool states a level in, so a threshold ported from one is
    /// meaningful here. A digitally silent clip yields null rather than `-inf`, because an
    /// infinity silently passes every `< threshold` filter written to find quiet clips.
    Dbfs,
    /// `peak_dbfs()` → the loudest single sample in dBFS (→ Float64, negative). Paired with
    /// [`AudioFunc::Dbfs`] it is the crest factor, which is what separates a compressed,
    /// broadcast-loud recording from a natural one.
    PeakDbfs,
    /// `clipping_ratio()` → the fraction of samples at or above `factor` of full scale
    /// (→ Float64, default 0.99). The corpus-hygiene measure for audio: a clip recorded
    /// too hot is distorted in a way no level normalization can undo, and it is invisible
    /// to every other statistic here because normalizing makes it *look* well-levelled.
    ClippingRatio,
    /// `silence_ratio(threshold_db)` → the fraction of samples quieter than the threshold
    /// (→ Float64, default -40 dBFS). Where `trim_silence` removes the ends, this measures
    /// the whole clip, so a recording that is mostly dead air is one predicate away.
    SilenceRatio,
    /// `rms_normalize(threshold_db)` → the waveform scaled so its RMS sits at the target
    /// level (default -20 dBFS), as a mono `List<Float32>`. The loudness-matching
    /// counterpart of `PeakNormalize`, and usually the one you want: peak normalization
    /// equalizes the *maximum*, so a clip with one loud click stays quiet everywhere else.
    /// Clamped so the result cannot clip. A digitally silent clip is returned unchanged.
    RmsNormalize,
    /// `pre_emphasis(factor)` → the first-order high-pass `y[n] = x[n] − a·x[n−1]`
    /// (default `a = 0.97`), as a mono `List<Float32>`. The standard filter every classical
    /// ASR front end applies before framing, to flatten the spectral tilt of voiced speech.
    PreEmphasis,
    /// `pad_or_trim(duration_secs, rate)` → the waveform resampled to `rate` and forced to
    /// exactly `duration_secs` — truncated if longer, zero-padded if shorter — as a mono
    /// `List<Float32>`.
    ///
    /// The op that makes a clip corpus batchable. Whisper requires exactly 30 seconds at
    /// 16 kHz and every other fixed-input audio model requires something like it, so
    /// without this a pipeline either loops in Python or hands the model rows of unequal
    /// length. Because the length is a query parameter rather than a property of the data,
    /// the output column has a knowable fixed width.
    PadOrTrim,
    /// `slice(offset_secs, duration_secs)` → the region of the clip starting at
    /// `offset_secs` and running `duration_secs`, as a mono `List<Float32>`. A window past
    /// the end of the clip yields an empty list rather than null: an empty region is a fact
    /// about the window, not a failure to read the clip.
    Slice,
    /// `spectrogram(rate, n_fft, hop_length)` → the **linear** power spectrogram as a
    /// `List<Float32>` of `(n_fft/2+1) * n_frames` in row-major `(freq, frame)` order.
    /// The mel spectrogram's unwarped sibling: a mel filterbank is tuned to speech, and a
    /// music, bioacoustic or machine-fault model wants the frequencies themselves.
    Spectrogram,
    /// `spectral_centroid(rate, n_fft, hop_length)` → the energy-weighted mean frequency in
    /// Hz, averaged over frames (→ Float64). The standard "brightness" descriptor, and the
    /// cheapest way to separate speech from music from noise without a model.
    SpectralCentroid,
    /// `spectral_rolloff(rate, n_fft, hop_length, factor)` → the frequency below which
    /// `factor` of the spectral energy lies (default 0.85), averaged over frames
    /// (→ Float64). Where the centroid reports the middle of the spectrum, this reports its
    /// edge, which is what distinguishes a band-limited telephone recording from a
    /// full-band one — the single most useful thing to know about a scraped speech corpus.
    SpectralRolloff,
    /// `spectral_bandwidth(rate, n_fft, hop_length)` → the energy-weighted spread of
    /// frequencies about the centroid, in Hz, averaged over frames (→ Float64).
    SpectralBandwidth,
    /// `spectral_flatness()` → the ratio of the geometric to the arithmetic mean of the
    /// power spectrum, 0..=1, averaged over frames (→ Float64). The tonality measure: a
    /// pure tone is near 0, white noise near 1. It is what finds the dead channels and
    /// hiss-only recordings that every other measure reports as ordinary audio.
    SpectralFlatness,
    /// `encode_wav(rate)` → the (optionally resampled) mono waveform as a 16-bit PCM WAV
    /// container (→ Binary). The op that closes the loop: without it a trimmed, normalized
    /// or resampled clip can only leave the engine as a list of floats, so writing a
    /// cleaned corpus back to storage as audio meant a Python encode per row.
    EncodeWav,
}

/// Video-decode operations for the `.video` namespace. Requires the `video` cargo
/// feature (system FFmpeg).
///
/// The three sampling ops exist because a video pipeline's first step is always "turn a
/// clip into pixels a model or a person can look at", and doing that outside the engine
/// means a per-row Python decode loop — the one thing the control plane must never do.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VideoFunc {
    /// `decode()` → struct `{width, height, num_frames, duration_secs, fps}`, read from
    /// the container header without decoding a single frame.
    Decode,
    /// `frames(n, w, h)` → `FixedSizeList<UInt8>` of `n*h*w*3` RGB8 samples: `n`
    /// evenly-spaced frames, each scaled to `(w, h)`. The training-ingest kernel.
    Frames,
    /// `thumbnail(w, h)` → PNG bytes of the clip's middle frame, scaled to `(w, h)`.
    /// The middle rather than the first because the first frame of a real clip is very
    /// often a black or title frame.
    Thumbnail,
    /// `frame_at(second, w, h)` → PNG bytes of the frame at `second`, scaled to
    /// `(w, h)`. Seeks rather than decoding the whole clip, so the cost does not grow
    /// with how far into the clip the timestamp is.
    FrameAt,
}

/// Biological-sequence operations for the `.seq` namespace. Wire tags are snake_case (the
/// contract with the Python `.seq` namespace).
///
/// The alphabet is ASCII by construction, so every kernel indexes byte tables rather than
/// decoding UTF-8. Case is **preserved** by the transforms and **folded** by the measures:
/// lowercase is how a reference genome marks soft-masked repeats, so upper-casing in a
/// transform would destroy the mask while respecting it in a measure would report a
/// repeat-rich contig as mostly-unknown.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeqFunc {
    /// `complement()` → the base-for-base IUPAC complement, case preserved (→ Utf8).
    Complement,
    /// `reverse_complement()` → the complement read 3'→5', which is what the other strand
    /// says (→ Utf8). The single most-used operation in genomics.
    ReverseComplement,
    /// `transcribe()` → DNA to RNA, T→U, case preserved (→ Utf8).
    Transcribe,
    /// `back_transcribe()` → RNA to DNA, U→T, case preserved (→ Utf8).
    BackTranscribe,
    /// `gc_content()` → the (G+C) fraction of the *unambiguous* bases (→ Float64). `N` is
    /// excluded from the denominator rather than counted as non-GC, so a gap does not read as
    /// an AT-rich region. Null when the row has no unambiguous base.
    GcContent,
    /// `gc_skew()` → `(G−C)/(G+C)` (→ Float64), whose sign flips at a bacterial chromosome's
    /// replication origin and terminus. Null when the row has no G or C.
    GcSkew,
    /// `base_counts()` → struct `{a, c, g, t, u, n, other}` of Int64 counts, case-folded.
    BaseCounts,
    /// `translate(frame, to_stop)` → the protein encoded in reading frame `frame`, NCBI
    /// genetic code table 1 (→ Utf8). Ambiguous codons yield `X`, stops yield `*`, and a
    /// trailing partial codon is dropped rather than padded.
    Translate,
    /// `kmers(k)` → `List<Utf8>` of every length-`k` window, step 1, upper-cased.
    Kmers,
    /// `canonical_kmers(k)` → `List<Utf8>` of each window folded with its reverse complement
    /// (the lexicographic minimum), so a read and its other-strand copy agree.
    CanonicalKmers,
    /// `minimizers(k, window)` → `List<Utf8>`: the smallest canonical k-mer of each window of
    /// `window` consecutive k-mers, consecutive repeats collapsed. The seed-and-extend sketch.
    Minimizers,
    /// `melting_temp()` → duplex melting temperature in °C (→ Float64), SantaLucia (1998)
    /// nearest-neighbour at 50 mM Na⁺ and 500 nM strand. Null for anything but pure ACGT.
    MeltingTemp,
    /// `molecular_weight(alphabet)` → average molecular weight in daltons (→ Float64), for a
    /// **single** strand of `dna`/`rna` or a `protein` chain.
    MolecularWeight,
    /// `gravy()` → the Kyte-Doolittle grand average of hydropathy (→ Float64). Positive is
    /// hydrophobic, negative hydrophilic.
    Gravy,
    /// `isoelectric_point()` → the pH at which the peptide carries no net charge (→ Float64),
    /// solved by bisection over the Bjellqvist pKa set.
    IsoelectricPoint,
    /// `phred_quality(offset)` → `List<Int32>` of per-base FASTQ quality scores.
    PhredQuality,
    /// `mean_quality(offset)` → the arithmetic mean Phred score (→ Float64) — the "average
    /// quality" every FASTQ tool reports.
    MeanQuality,
    /// `expected_errors(offset)` → `Σ 10^(−Q/10)`, the expected number of miscalled bases in
    /// the read (→ Float64). The `fastq_maxee` filter, and additive where a mean is not.
    ExpectedErrors,
    /// `find_motif(pattern)` → `List<Int64>` of 1-based start positions of every (possibly
    /// overlapping) match of an IUPAC-degenerate motif.
    FindMotif,
    /// `count_motif(pattern)` → how many such matches the sequence contains (→ Int64).
    CountMotif,
    /// `max_homopolymer()` → the length of the longest single-base run (→ Int64), the
    /// nanopore and PacBio error signature a variant filter thresholds on.
    MaxHomopolymer,
    /// `is_valid(alphabet)` → whether every character is in the named alphabet (→ Boolean).
    IsValid,
}

/// Map-column accessors (over an Arrow `Map` column). Wire tags are snake_case (the
/// contract with the Python `.map` namespace).
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MapFunc {
    /// `map_keys(m)` → `List<K>` of each row's keys (DuckDB `map_keys`).
    MapKeys,
    /// `map_values(m)` → `List<V>` of each row's values (DuckDB `map_values`).
    MapValues,
    /// `map_entries(m)` → `List<Struct<key, value>>`, the row's entries as a list of
    /// pairs (DuckDB and Spark both spell it `map_entries`).
    ///
    /// This is the one accessor that keeps a key beside its value. `map_keys` and
    /// `map_values` each return a list, and pairing them back up relies on the two
    /// sharing an order — true here, but not a guarantee a caller should have to know.
    MapEntries,
    /// `element_at(m, key)` → the value for the literal `key` (null if absent).
    ElementAt,
}

/// Per-row scalar reductions over a `List` column. `len`/`n_unique` → Int64; the
/// numeric reductions (`sum`/`min`/`max`/`mean`) cast elements to Float64. Null
/// list rows stay null; empty lists reduce to null (no elements) except `len`
/// (0) and `n_unique` (0).
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ListFunc {
    Len,
    Sum,
    Min,
    Max,
    Mean,
    NUnique,
    /// Sort each row's list ascending → `List` (same element type).
    Sort,
    /// Sort each row's list descending → `List` (same element type), nulls **last**.
    ///
    /// Not `reverse(sort(x))`: ascending puts nulls last, so reversing moves them to the
    /// front, where DuckDB's `list_reverse_sort` leaves them at the back. The null
    /// placement is the whole reason this is its own kernel rather than a composition.
    SortDesc,
    /// Reverse each row's list → `List` (same element type).
    Reverse,
    /// Product of (non-null) elements → Float64; empty/null row → null.
    Product,
    /// Sample standard deviation `sqrt(Σ(x-mean)²/(n-1))` → Float64; null when
    /// fewer than 2 non-null elements.
    Std,
    /// Sample variance `Σ(x-mean)²/(n-1)` → Float64; null when n<2.
    Var,
    /// Distinct elements preserving first-occurrence order → `List` (same element
    /// type); null elements are dropped.
    Unique,
    /// Median of the (non-null) elements → Float64; for an even count the average
    /// of the two middle values; empty/null row → null.
    Median,
    /// 0-based index of the minimum non-null element (first on ties) → Int64;
    /// empty/null row → null.
    ArgMin,
    /// 0-based index of the maximum non-null element (first on ties) → Int64;
    /// empty/null row → null.
    ArgMax,
    /// The 0-based indices that sort each row's list **ascending** (stable; ties keep
    /// original order) → `List<Int64>`. Null-valued positions are placed last in their
    /// original order. `reverse` the result for a descending / top-k-first ranking — the
    /// standard way to turn a per-row score/logit vector into ranked positions in-engine.
    ArgSort,
    /// Euclidean (L2) norm `sqrt(Σ xᵢ²)` of the non-null elements → Float64;
    /// empty/null row → null. The vector magnitude used in similarity search.
    L2Norm,
    /// L1 (Manhattan) norm `Σ |xᵢ|` of the non-null elements → Float64; empty/null row →
    /// null. The scale used for L1 vector normalization (sparse / robust features).
    L1Norm,
    /// The maximum absolute value `max |xᵢ|` of the non-null elements → Float64; empty/null
    /// row → null. The divisor for MaxAbs feature scaling (scales into `[-1, 1]`).
    MaxAbs,
    /// L2-normalize each row to unit length: `xᵢ / sqrt(Σ xⱼ²)` → `List<Float64>`.
    /// A zero vector maps to all zeros (no division by zero); per-element nulls are
    /// preserved and excluded from the norm; a null/empty row stays null/empty. The
    /// standard preprocessing step before cosine/dot retrieval.
    Normalize,
    /// Cumulative sum over each row's list → `List<Float64>` of the same length: element `i`
    /// is `Σ_{j≤i} xⱼ`. A null element contributes 0 to the running total and stays null in
    /// the output (the prefix continues past it). The building block for a per-row cumulative
    /// distribution (`cumsum` then divide by the total) and prefix features.
    CumSum,
    /// Softmax over each row's list — `exp(xᵢ − max) / Σ exp(xⱼ − max)` → `List<Float64>`
    /// summing to 1. The logits→probabilities step (per-row, over a vector of scores):
    /// converts a classifier's raw output to a probability distribution in the data plane.
    /// The `− max` shift is the standard numerically-stable form. Per-element nulls are
    /// preserved and excluded; a null/empty row stays null/empty.
    Softmax,
    /// Concatenate a `List<List<T>>` into a `List<T>` per row, in order (DuckDB
    /// `flatten`; Polars `list.explode`-free flatten). Null inner lists are skipped;
    /// a null outer row stays null. Element type `T` is preserved.
    Flatten,
    /// `log_softmax(x)` — `xᵢ − max − ln Σ exp(xⱼ − max)` → `List<Float64>`. The log-domain
    /// sibling of `Softmax`, and not the same as taking its log: a probability that underflows
    /// to 0 in the linear form becomes `-inf` there, while here it stays a large negative
    /// finite number. That is the reason scoring and training pipelines carry
    /// log-probabilities, so the conversion has to happen in the log domain to be worth
    /// anything. Per-element nulls are preserved; a null/empty row stays null/empty.
    LogSoftmax,
    /// Shannon entropy of each row read as a distribution, in **nats**: `−Σ pᵢ ln pᵢ` after
    /// normalizing the row by its own sum → Float64. Works on a probability vector, a count
    /// vector, or unnormalized weights alike. 0 when all the mass is on one outcome, `ln n`
    /// when spread evenly over `n` — the per-row uncertainty of a classifier's output, a
    /// retrieval score distribution, or an attention row. A non-positive element is skipped
    /// (`p ln p` is undefined there); a row totalling zero has no distribution and yields null.
    Entropy,
    /// First difference over each row's list → `List<Float64>` of the **same length**:
    /// element `i` is `xᵢ − xᵢ₋₁`, with element 0 null (no predecessor). If either
    /// neighbor is null the difference is null (Polars `list.diff`). The delta-feature
    /// building block for audio (MFCC deltas) and time-series (returns / velocity);
    /// a null/empty row stays null/empty.
    Diff,
}

/// Unary math functions. `abs` preserves the input numeric type; the rest yield
/// Float64.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MathFunc {
    Abs,
    Round,
    Floor,
    Ceil,
    Sqrt,
    Ln,
    Log10,
    Log2,
    Exp,
    Sin,
    Cos,
    Tan,
    /// −1 / 0 / +1 by sign (0 maps to 0, matching DuckDB `sign`).
    Sign,
    /// Truncate toward zero.
    Trunc,
    /// Cube root.
    Cbrt,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    /// Radians → degrees.
    Degrees,
    /// Degrees → radians.
    Radians,
    /// Cotangent (1/tan).
    Cot,
    /// `n!` — factorial of a non-negative integer (DuckDB `factorial`). → Float64.
    Factorial,
    /// Population count: the number of set bits in the Int64 two's-complement value
    /// (DuckDB `bit_count`). → Float64 (integral-valued).
    BitCount,
    /// `even(x)` — round *away from zero* to the nearest even integer (DuckDB `even`):
    /// `2.1 → 4`, `-2.1 → -4`, `2.0 → 2`. Not `round`-then-adjust; the rounding
    /// direction is outward, which is why `3.0` is `4` and not `2`.
    Even,
    /// `gamma(x)` — the gamma function Γ(x) (DuckDB `gamma`), the continuous extension
    /// of the factorial: `Γ(n) = (n-1)!` for a positive integer.
    Gamma,
    /// `lgamma(x)` — the natural log of |Γ(x)| (DuckDB `lgamma`). Computed directly
    /// rather than as `ln(gamma(x))`, which overflows to `inf` above ~171.
    Lgamma,
    /// `sec(x)` = 1/cos(x) (Spark `sec`).
    Sec,
    /// `csc(x)` = 1/sin(x) (Spark `csc`).
    Csc,
    /// `rint(x)` — round half to **even** (Spark `rint`, IEEE-754 `roundTiesToEven`).
    /// Distinct from `round`, which is half away from zero here and in DuckDB:
    /// `rint(2.5)` is `2`, `round(2.5)` is `3`.
    Rint,
    /// `asinh(x)` — inverse hyperbolic sine, defined for every real.
    ///
    /// A node rather than the textbook `ln(x + sqrt(x*x + 1))` composition, because that
    /// composition is wrong at both ends of the range: `x*x` overflows above ~1.3e154, so
    /// `asinh(1e300)` returned `inf` instead of 691.47, and `-inf` produced
    /// `ln(-inf + inf)` = NaN instead of `-inf`.
    Asinh,
    /// `acosh(x)` — inverse hyperbolic cosine, defined for `x >= 1` (NaN below).
    ///
    /// A node for the same overflow reason as [`MathFunc::Asinh`]: `acosh(1e300)` came
    /// back `inf` from `ln(x + sqrt(x*x - 1))`.
    Acosh,
    /// `atanh(x)` — inverse hyperbolic tangent, defined for `|x| < 1`; `±1` gives `±inf`
    /// and beyond that NaN. A node beside its two siblings, and more accurate near zero
    /// than `0.5 * ln((1 + x) / (1 - x))`, which loses precision to cancellation there.
    Atanh,
}

/// String functions. `upper`/`lower` → Utf8; `len` → Int64; `contains`/
/// `starts_with`/`ends_with` → Boolean; `substr` (1-based, char-oriented) → Utf8.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StrFunc {
    Upper,
    Lower,
    Len,
    Contains,
    StartsWith,
    EndsWith,
    Substr,
    Replace,
    Trim,
    LTrim,
    RTrim,
    Reverse,
    /// Repeat the string `start` times (`start` reused as the count; ≤0 → empty).
    Repeat,
    /// Left-pad to `start` characters with `pattern` (cycled); truncates if longer.
    Lpad,
    /// Right-pad to `start` characters with `pattern` (cycled); truncates if longer.
    Rpad,
    /// 1-based position of `pattern` in the string (0 if absent). → Int64.
    Position,
    /// The last `start` characters (`start` reused as the count).
    Right,
    /// Unicode codepoint of the first character (0 for empty). → Int64.
    Ascii,
    /// Split on `pattern` → a `List<Utf8>` (null input → null list).
    Split,
    /// A MinHash signature of the text → `List<Int64>` of `length` (`num_perm`) values,
    /// each bounded to 32 bits; `start` is the character shingle width. The fraction of
    /// positions two signatures agree on estimates the documents' Jaccard similarity
    /// (`.list.jaccard`) — fuzzy dedup compares 128 integers, not two documents.
    ///
    /// `rename` pins the wire tag (`snake_case` would spell it `min_hash`).
    #[serde(rename = "minhash")]
    MinHash,
    /// Slice into fixed-size overlapping windows → a `List<Utf8>`: the document
    /// chunker a RAG ingest pipeline needs before embedding. `length` is the chunk
    /// size and `start` the overlap, both in **characters** (Unicode scalar values,
    /// as `Substr`/`Len` count them), so a chunk never splits a codepoint. Chunks
    /// start every `length - start` characters while a start remains inside the
    /// string, so the final chunk may be shorter. Empty string → empty list; null →
    /// null list.
    Chunk,
    /// Word n-grams → a `List<Utf8>`: split on whitespace, then join each window of
    /// `length` adjacent tokens with a single space. `length` carries `n` (reusing the
    /// scalar slot the way `chunk`/`repeat` do). A string with fewer than `n` tokens
    /// yields the single n-gram of all its tokens (never an empty list for non-empty
    /// input), so a short document still contributes. Empty string → empty list; null →
    /// null list. This is the token-level counterpart of `chunk`'s character windows,
    /// and the primitive the multiset generation metrics (BLEU/ROUGE-N/Distinct-n) build
    /// their token n-gram sets from.
    TokenNgrams,
    /// The SQuAD answer normalization every word-level text metric runs first: lowercase,
    /// drop the standalone articles `a`/`an`/`the`, delete punctuation, collapse whitespace,
    /// trim. → Utf8; null → null.
    ///
    /// It replaces a composition of `lower` and three `regexp_replace_all` passes, which cost
    /// ninety times a bare `len` over the same column and which every word metric paid twice.
    /// One pass, one allocation. `eval/str/squad.rs` documents how the five steps reduce to a
    /// scan over word and non-word runs, and pins the result against the composition.
    SquadNormalize,

    // --- Per-document text quality (the LLM pretraining-corpus filters) ---------------
    //
    // The Gopher (Rae et al. 2021), C4, and RefinedWeb heuristics, as *per-row* measures.
    // `plan/functions/metrics/text/` already scores the same properties across a corpus as
    // aggregates; these answer "which documents do I drop", which a filter needs and an
    // aggregate cannot express. Each ratio is in `[0, 1]`, and null where the document has
    // nothing to measure — an empty extraction must not pass a threshold by scoring 0.
    /// `word_count()` → whitespace-separated word count. → Int64.
    WordCount,
    /// `mean_word_length()` → mean word length in characters. Gopher keeps `[3, 10]`;
    /// below is a token list, above is usually a base64 blob or a URL dump. → Float64.
    MeanWordLength,
    /// `symbol_ratio()` → `(# + …) / words`. Gopher drops above 0.1: a stripped heading
    /// structure, or a listing page whose entries were truncated for display. → Float64.
    SymbolRatio,
    /// `alpha_word_ratio()` → the fraction of words containing a letter. Gopher drops below
    /// 0.8, which is a table that lost its structure. → Float64.
    AlphaWordRatio,
    /// `stopword_count()` → how many of Gopher's eight stop words appear, counted
    /// **distinctly**. Fewer than two is a keyword list, not prose. → Int64.
    StopwordCount,
    /// `bullet_line_ratio()` → the fraction of lines starting with a bullet. Gopher drops
    /// above 0.9 — a navigation menu. → Float64.
    BulletLineRatio,
    /// `ellipsis_line_ratio()` → the fraction of lines ending in an ellipsis. Gopher drops
    /// above 0.3 — a listing page of fixed-width teasers. → Float64.
    EllipsisLineRatio,
    /// `duplicate_line_ratio()` → the fraction of *characters* in repeated lines. Weighed by
    /// characters, following Gopher: one repeated footer and fifty repeated one-word lines
    /// are different documents that a line count cannot separate. → Float64.
    DuplicateLineRatio,
    /// `duplicate_paragraph_ratio()` → the same, over blank-line-separated paragraphs.
    /// → Float64.
    DuplicateParagraphRatio,
    /// `top_ngram_ratio(n)` → the fraction of characters covered by the single most frequent
    /// word n-gram (`n` rides the `length` slot). Gopher applies it for n of 2-4; it finds
    /// keyword-stuffed SEO pages and templated listings. → Float64.
    TopNgramRatio,
    /// `duplicate_ngram_ratio(n)` → the fraction of characters covered by *every* n-gram
    /// appearing more than once. Gopher applies it for n of 5-10; unlike `TopNgramRatio` it
    /// catches a page assembled from several boilerplate blocks. → Float64.
    DuplicateNgramRatio,
    /// `char_entropy()` → Shannon entropy of the character distribution, in bits. The
    /// gibberish and encoded-blob detector: prose sits near 4-5 bits, a base64 blob above,
    /// a repeated character at 0. The one measure here that is not Gopher's. → Float64.
    CharEntropy,
    /// True where `pattern` (a regex) matches anywhere in the string. → Boolean.
    RegexpMatches,
    /// Replace the first match of regex `pattern` with `replacement`. → Utf8.
    RegexpReplace,
    /// Replace *every* match of regex `pattern` with `replacement` (DuckDB
    /// `regexp_replace(..., 'g')`; Polars `replace_all`). → Utf8.
    RegexpReplaceAll,
    /// `split_part(string, delim, n)`: the `n`-th (1-based) field of the string
    /// split on `pattern` (the delimiter); `''` if `n` is out of range (DuckDB
    /// `split_part`; `start` carries `n`). → Utf8.
    SplitPart,
    /// Extract capture group `start` of regex `pattern` ('' if no match). → Utf8.
    RegexpExtract,
    /// Extract the string value at JSON `pattern` path (e.g. `$.a.b`); null if the
    /// input isn't valid JSON or the path is missing. → Utf8.
    JsonExtractString,
    /// Extract the integer value at JSON `pattern` path; null if the input isn't
    /// valid JSON, the path is missing, or the value isn't integral. → Int64.
    JsonExtractInt,
    /// Extract the numeric value at JSON `pattern` path as a float; null if absent
    /// or non-numeric. → Float64.
    JsonExtractFloat,
    /// Extract the boolean value at JSON `pattern` path; null if absent or
    /// non-boolean. → Boolean.
    JsonExtractBool,
    /// Number of elements in the JSON array at `pattern` path; null if the path is
    /// absent or the value there is not an array. Counted by structural skipping, so
    /// no element is parsed. → Int64.
    JsonArrayLength,
    /// The keys of the JSON object at `pattern` path, **in source order**; null if the
    /// path is absent or the value is not an object. → List<Utf8>.
    JsonObjectKeys,
    /// The elements of the JSON array at `pattern` path, each rendered as
    /// `json_extract_string` renders a leaf (string verbatim, container compacted, JSON
    /// null as a null element); null if absent or not an array. This is what turns a
    /// JSON array column into a list column that `explode` and `.list` can work on.
    /// → List<Utf8>.
    JsonArrayValues,
    /// The JSON type at `pattern` path: `object`, `array`, `string`, `number`,
    /// `boolean`, or `null`; null if the path is absent. → Utf8.
    JsonType,
    /// Whether a value exists at `pattern` path. A JSON `null` counts as present — the
    /// distinction `json_extract_*` cannot express, since both absent and null extract
    /// to null. → Boolean.
    JsonExists,
    /// The value at `pattern` path as text, **JSON-quoted** — DuckDB `json_value`.
    /// Unlike `JsonExtractString` (which unquotes a string and renders a container),
    /// this returns the raw JSON token for a scalar and **null for an object or an
    /// array**, which is the distinction DuckDB draws between the two functions. → Utf8.
    JsonValue,
    /// Whether the document contains `pattern` as a value, at the top level of an array
    /// or as any member of an object — DuckDB `json_contains`. → Boolean.
    JsonContains,
    /// The document re-rendered with two-space indentation (DuckDB `json_pretty`).
    /// Invalid JSON → null. → Utf8.
    JsonPretty,
    /// The document's *shape* with each leaf replaced by its type name (DuckDB
    /// `json_structure`), e.g. `{"a":1}` → `{"a":"UBIGINT"}`. Invalid JSON → null. → Utf8.
    JsonStructure,
    /// The single character at a Unicode code point (DuckDB/Spark `chr`). Takes an
    /// **integer** input, so it is handled before the Utf8 downcast. → Utf8.
    Chr,
    /// The integer written in base `start` (2..=36), no padding, `-` for a negative
    /// value (DuckDB `to_base`, and `bin` at base 2). Integer input. → Utf8.
    ToBase,
    /// A byte count as human-readable text with binary units — `1024` → `1.0 KiB`
    /// (DuckDB `format_bytes` / `formatReadableSize`). Integer input. → Utf8.
    FormatBytes,
    /// The same with decimal (SI) units — `1000` → `1.0 kB` (DuckDB
    /// `formatReadableDecimalSize`). Integer input. → Utf8.
    FormatBytesSi,
    /// Deterministic FNV-1a 64-bit hash of the UTF-8 bytes (→ Int64; the u64 digest
    /// reinterpreted as i64). Stable across partitions, runs, and machines — the
    /// building block for surrogate keys and slowly-changing-dimension change
    /// detection. Null → null.
    Hash64,
    /// Capitalize the first letter of each word, lowercasing the rest. A word is a
    /// maximal run of alphanumerics (DuckDB `initcap`). → Utf8.
    Initcap,
    /// Number of UTF-8 bytes in the string (`v.len()`; DuckDB `octet_length`). → Int64.
    OctetLength,
    /// Number of bits in the string (bytes × 8; DuckDB `bit_length`). → Int64.
    BitLength,
    /// Uppercase hex of the UTF-8 bytes, e.g. "abc" → "616263" (DuckDB `hex`). → Utf8.
    Hex,
    /// `translate(string, from, to)`: each char that appears at index i of `from`
    /// (`pattern`) is replaced by the char at index i of `to` (`replacement`); if
    /// `to` is shorter, chars in `from` beyond its length are deleted; chars not in
    /// `from` pass through (DuckDB `translate`). → Utf8.
    Translate,
    /// Standard base64 encoding of the UTF-8 bytes (DuckDB `to_base64`). → Utf8.
    Base64,
    /// Decode standard base64 to bytes, then interpret as UTF-8 (DuckDB
    /// `from_base64`). Invalid base64 or non-UTF-8 bytes → null. → Utf8 (nullable).
    FromBase64,
    /// Parse pairs of hex digits to bytes, then interpret as UTF-8 (DuckDB
    /// `unhex`). Odd length, non-hex, or non-UTF-8 bytes → null. → Utf8 (nullable).
    Unhex,
    /// SQL `LIKE`: anchored match where `pattern`'s `%` matches any run of chars,
    /// `_` matches exactly one char, every other char is literal. → Boolean.
    Like,
    /// SQL `ILIKE`: case-insensitive `LIKE`. → Boolean.
    Ilike,
    /// MD5 digest of the UTF-8 bytes as lowercase hex (DuckDB `md5`). → Utf8.
    Md5,
    /// SHA-1 digest of the UTF-8 bytes as lowercase hex (DuckDB `sha1`). → Utf8.
    Sha1,
    /// SHA-256 digest of the UTF-8 bytes as lowercase hex (DuckDB `sha256`). → Utf8.
    Sha256,
    /// CRC-32 (IEEE) checksum of the UTF-8 bytes (Spark `crc32`). → Int64.
    Crc32,
    /// `mime_type()` → what the value's leading bytes say it is (`image/png`,
    /// `video/mp4`, `application/pdf`, …), or **null** when nothing recognizes them.
    ///
    /// The byte-oriented sibling of the IO layer's `mime` column, for the bytes that never
    /// came from a file read — a `ds.ml.download`, a blob column in a Parquet table, a
    /// payload extracted from an archive. Those have no filename to guess from and, until
    /// this, no way to be identified at all, so routing a mixed blob corpus meant a Python
    /// UDF. Null rather than `application/octet-stream` because this reader knows only what
    /// the bytes say: "unrecognized" and "opaque binary" are different claims, and only a
    /// caller with a filename left to try can collapse them.
    MimeType,
    /// 64-bit xxHash of the UTF-8 bytes (the u64 digest reinterpreted as i64). The
    /// fast non-cryptographic hash for bucketing/sharding. Null → null. → Int64.
    #[serde(rename = "xxhash64")]
    XxHash64,
    /// `substring_index(s, delim, count)`: the substring before the `count`-th
    /// (1-based) occurrence of `pattern` (the delimiter). `count > 0` counts from the
    /// left, `count < 0` from the right (Spark `substring_index`; `start` carries
    /// `count`). → Utf8.
    SubstringIndex,
    /// `overlay(s, replacement, pos, len)`: replace `length` characters starting at
    /// 1-based `start` (`pos`) with `replacement` (SQL `OVERLAY`). `len` defaults to
    /// the replacement's length. → Utf8.
    Overlay,
    /// Every match of regex `pattern` (capture group 0) as a `List<Utf8>` (DuckDB
    /// `regexp_extract_all`; empty list if none, null input → null). → List<Utf8>.
    RegexpExtractAll,
    /// Number of non-overlapping matches of regex `pattern` (DuckDB `regexp_count`).
    /// → Int64.
    RegexpCount,
    /// Split on every match of regex `pattern` → a `List<Utf8>` of the pieces between
    /// matches. The regex counterpart of `Split`, whose delimiter is a literal. An empty
    /// string yields `[""]` and a null input a null list, matching `Split`. → List<Utf8>.
    RegexpSplit,
    /// Levenshtein edit distance to the literal string `pattern` (DuckDB
    /// `levenshtein` against a constant). → Int64.
    Levenshtein,
    /// Damerau-Levenshtein (Optimal String Alignment) distance to the literal `pattern`
    /// (DuckDB `damerau_levenshtein`): like `levenshtein` but an adjacent transposition
    /// costs 1, so it scores a swapped-letter typo (`teh`↔`the`) as one edit. → Int64.
    DamerauLevenshtein,
    /// Jaro similarity to the literal string `pattern` (DuckDB `jaro_similarity`): a
    /// `[0, 1]` fuzzy-match score based on matching characters and transpositions — the
    /// standard metric for entity resolution / record linkage on short strings like names.
    /// → Float64.
    JaroSimilarity,
    /// Jaro-Winkler similarity to the literal string `pattern` (DuckDB
    /// `jaro_winkler_similarity`): Jaro plus a common-prefix bonus, so strings that agree
    /// at the start (typical of names) score higher. `[0, 1]`. → Float64.
    JaroWinklerSimilarity,
    /// American Soundex phonetic code, a 4-character key (DuckDB `soundex`). → Utf8.
    Soundex,
    /// HMAC-SHA-256 keyed by `pattern` (the key's raw UTF-8 bytes), lowercase hex.
    /// The pseudonymization primitive — deterministic, so pseudonyms still join across
    /// tables, but irreversible and (unlike a bare `sha256`) not brute-forceable over a
    /// low-entropy domain such as email addresses. Null → null. → Utf8.
    HmacSha256,
    /// AES-256-GCM-SIV encryption under the 32-byte key in `pattern` (64 hex chars or
    /// base64); base64 of `ciphertext || tag`. Deterministic by design so the encrypted
    /// column stays joinable/groupable — which means it leaks equality; see
    /// `eval::security::crypto`. Null → null. → Utf8.
    AesEncrypt,
    /// Inverse of `AesEncrypt` under the same key. A value that is not base64, fails
    /// authentication (wrong key, tampered ciphertext), or is not UTF-8 → null, so one
    /// unreadable row cannot abort a scan. → Utf8 (nullable).
    AesDecrypt,
    /// Replace every character outside the first `start` and last `length` with the
    /// single character in `pattern` (default `X`). Character-length preserving; when
    /// the revealed windows overlap the value is returned unmasked. Null → null. → Utf8.
    Mask,
    /// Readable text of an HTML document: drops tags *and* `<script>`/`<style>` bodies
    /// and comments, decodes entities, collapses whitespace, and separates elements with
    /// a space. Lenient on malformed markup. Null → null. → Utf8. See `eval::str::html`.
    StripHtml,
    /// Compress the raw bytes with the codec named by `pattern` (`gzip`, `zlib`,
    /// `deflate`, `zstd`, `brotli`, or `lz4`). Accepts Utf8 (its UTF-8 bytes) or Binary.
    /// Null → null; an unknown codec is an error. → Binary. See `eval::str::compress`.
    Compress,
    /// Inverse of `Compress` under the codec named by `pattern`. Input that is not a valid
    /// frame for that codec yields **null** rather than erroring, matching `from_base64`
    /// and `unhex` — one corrupt blob in a scan is a bad row, not a bad query, which is
    /// why there is no separate `try_decompress`. → Binary (nullable).
    Decompress,
    /// Re-case an identifier into the style named by `pattern`: `snake`, `upper_snake`,
    /// `camel`, `pascal`, `kebab`, `upper_kebab`, `title`, `sentence`, `dot`, or `train`.
    /// One word splitter serves every style (separators, lower→upper transitions, and
    /// acronym runs), so the styles never disagree about where the words were. Null →
    /// null; an unknown style is an error, not a silent passthrough. → Utf8.
    /// See `eval::str::case`.
    ToCase,
    /// Percent-encode for use in a URL (DuckDB `url_encode`): everything outside the
    /// RFC 3986 unreserved set becomes `%XX` over the UTF-8 bytes, including `/` and
    /// `+` — this encodes a *component*, not a whole URL. Null → null. → Utf8.
    UrlEncode,
    /// Inverse of `UrlEncode` (DuckDB `url_decode`). A malformed escape (`%` not
    /// followed by two hex digits, or bytes that do not decode as UTF-8) is left
    /// **as written** rather than erroring or nulling the row — verified against DuckDB,
    /// which returns `'a%2'` for `url_decode('a%2')`. Null → null. → Utf8.
    UrlDecode,
    /// Escape the regex metacharacters in the value (DuckDB `regexp_escape`), so it can
    /// be embedded in a pattern as a literal. Null → null. → Utf8.
    RegexpEscape,
    /// The final component of a path (DuckDB `parse_filename`): everything after the
    /// last separator. → Utf8.
    ParseFilename,
    /// The directory part of a path (DuckDB `parse_dirname`) — the *first* component,
    /// which is `/` for an absolute POSIX path. Not the same as `ParseDirpath`, which is
    /// everything before the filename; DuckDB genuinely has both. → Utf8.
    ParseDirname,
    /// Everything before the last separator of a path (DuckDB `parse_dirpath`). → Utf8.
    ParseDirpath,
    /// A path split into its components (DuckDB `parse_path`), with a leading `/` kept
    /// as its own first element for an absolute POSIX path. → List<Utf8>.
    ParsePath,
    /// Hamming distance to the literal string `pattern` (DuckDB `hamming`/`mismatches`):
    /// the number of positions at which the two differ. Defined only for equal-length
    /// strings, which DuckDB enforces — an unequal length is an error, not a silent
    /// truncation. → Int64.
    Hamming,
    /// Jaccard similarity to the literal string `pattern` (DuckDB `jaccard`): the size of
    /// the intersection over the size of the union of the two strings' *character sets*.
    /// `[0, 1]`. Distinct from `.list.jaccard`, which is over list elements. → Float64.
    JaccardSimilarity,
    /// The value's UTF-8 bytes as a string of `0`/`1` (DuckDB `to_binary`), 8 characters
    /// per byte, most significant bit first. Null → null. → Utf8.
    ToBinary,
    /// Inverse of `ToBinary` (DuckDB `from_binary`). Input that is not a whole number of
    /// 8 `0`/`1` characters, or does not decode as UTF-8, yields **null**, matching
    /// `unhex`. → Utf8 (nullable).
    FromBinary,
}

/// Temporal *constructors* carried by [`Expr::MakeTemporal`] — the inverse direction of
/// [`DateFunc`]'s extractions. Wire tags are snake_case (the contract with Python).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MakeTemporalFunc {
    /// `(year, month, day)` → Date32. An impossible date is null, not an error.
    MakeDate,
    /// `(year, month, day, hour, minute, second)` → Timestamp(Microsecond).
    MakeTimestamp,
    /// Epoch **seconds** → Timestamp(Microsecond).
    FromUnixSeconds,
    /// Epoch **milliseconds** → Timestamp(Microsecond).
    FromUnixMillis,
    /// Epoch **microseconds** → Timestamp(Microsecond).
    FromUnixMicros,
    /// Epoch **nanoseconds** → Timestamp(Microsecond). Truncates toward negative
    /// infinity, so the result is the microsecond containing the instant.
    FromUnixNanos,
    /// Days since 1970-01-01 → Date32 (Spark `date_from_unix_date`).
    FromUnixDate,
}

/// Date/time field extractions (→ Int64). Wire tags are snake_case (the contract
/// with the Python `.dt` namespace).
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DateFunc {
    Year,
    Month,
    Day,
    Hour,
    Minute,
    Second,
    Quarter,
    /// ISO week of the year (1–53).
    Week,
    /// Day of week with Sunday = 0 (matches DuckDB `dayofweek`).
    DayOfWeek,
    /// Day of the year (1–366).
    DayOfYear,
    /// Seconds since the Unix epoch (DuckDB `epoch`). → Int64.
    Epoch,
    /// Full weekday name e.g. "Monday" (DuckDB `dayname`, chrono `%A`). → Utf8.
    Dayname,
    /// Full month name e.g. "January" (DuckDB `monthname`, chrono `%B`). → Utf8.
    Monthname,
    /// ISO day of week: Monday = 1 … Sunday = 7 (DuckDB `isodow`). → Int64.
    Isodow,
    /// The century, e.g. 2021 → 21, 1999 → 20 (DuckDB `century`). → Int64.
    Century,
    /// The decade, e.g. 2021 → 202 (DuckDB `decade`, `year/10`). → Int64.
    Decade,
    /// The millennium, e.g. 2021 → 3, 2000 → 2 (DuckDB `millennium`,
    /// `(Y-1)/1000 + 1`). → Int64.
    Millennium,
    /// The last day of the month of the instant, at 00:00:00 (DuckDB `last_day`).
    /// → Timestamp(Microsecond) (compare against `last_day(ts)::TIMESTAMP`).
    LastDay,
    /// Whether the instant's year is a leap year (DuckDB `isfinite`-style predicate
    /// `extract('isoyear')`-independent). → Boolean.
    IsLeapYear,
    /// Number of days in the instant's month, 28–31 (DuckDB
    /// `days_in_month`-equivalent). → Int64.
    DaysInMonth,
    /// ISO 8601 week-numbering year (DuckDB `isoyear`), which can differ from the
    /// calendar year near January 1st. → Int64.
    IsoYear,
}

/// One `WHEN condition THEN value` branch of a `Case`.
#[derive(Debug, Clone, Deserialize)]
pub struct CaseBranch {
    pub when: Expr,
    pub then: Expr,
}

/// A constant value. Kept deliberately small for the bootstrap engine; widened
/// as the type system grows.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Literal {
    Int(i64),
    /// A float literal. JSON has no NaN/Infinity tokens, so the Python control
    /// plane encodes a non-finite float as a name string (`"NaN"`/`"inf"`/
    /// `"-inf"`); a finite float stays a plain JSON number. Accept both.
    #[serde(deserialize_with = "de_float")]
    Float(f64),
    Bool(bool),
    Str(String),
    /// Microseconds since the Unix epoch (tz-naive Timestamp(Microsecond)).
    Timestamp(i64),
    /// Days since the Unix epoch (Date32).
    Date(i32),
}

/// Binary operators. Comparisons yield booleans; arithmetic yields the numeric
/// promotion arrow's kernels choose; boolean ops require boolean inputs.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BinaryOp {
    // comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    // arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    /// Floored division (`a // b`, wire tag `floor_div`) — the quotient rounded
    /// toward NEGATIVE INFINITY, i.e. Python/Polars `//`, not SQL's truncating
    /// integer division. `-7 // 3` is `-3` here, where `Div` gives `-2`.
    ///
    /// It is a distinct op rather than sugar over `floor(a / b)` because the
    /// result type depends on the *input* types, which the lazy Python builder
    /// cannot know at plan-build time: Int64 ÷ Int64 stays Int64 (exact past
    /// 2^53, where a Float64 round-trip silently loses precision), while Float64
    /// gives the IEEE `floor(a / b)`.
    FloorDiv,
    // boolean
    And,
    Or,
    // string
    Concat,
    // bitwise (Int64)
    BitAnd,
    BitOr,
    BitXor,
    ShiftLeft,
    ShiftRight,
    /// Add `right` calendar months to a Date32/Timestamp `left` (negative to
    /// subtract); used for `date + INTERVAL n MONTH/YEAR`.
    AddMonths,
}

/// Deserialize a float literal that may arrive as a JSON number (finite) or as a
/// name string for a non-finite value (`"NaN"`, `"inf"`/`"+inf"`/`"Infinity"`,
/// `"-inf"`/`"-Infinity"`). JSON cannot carry NaN/Infinity as numbers, so the
/// control plane spells them out; every other string is parsed as an f64.
fn de_float<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::{Error, Unexpected, Visitor};
    use std::fmt;

    struct FloatVisitor;

    impl Visitor<'_> for FloatVisitor {
        type Value = f64;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("a float number or a non-finite name string")
        }

        fn visit_f64<E: Error>(self, v: f64) -> Result<f64, E> {
            Ok(v)
        }
        fn visit_i64<E: Error>(self, v: i64) -> Result<f64, E> {
            Ok(v as f64)
        }
        fn visit_u64<E: Error>(self, v: u64) -> Result<f64, E> {
            Ok(v as f64)
        }
        fn visit_str<E: Error>(self, v: &str) -> Result<f64, E> {
            match v {
                "NaN" | "nan" => Ok(f64::NAN),
                "inf" | "+inf" | "Infinity" | "+Infinity" => Ok(f64::INFINITY),
                "-inf" | "-Infinity" => Ok(f64::NEG_INFINITY),
                other => other
                    .parse::<f64>()
                    .map_err(|_| E::invalid_value(Unexpected::Str(other), &self)),
            }
        }
    }

    deserializer.deserialize_any(FloatVisitor)
}

impl Literal {
    /// Materialize the literal as an array of length `n`.
    ///
    /// O(n) for now; replaced by `Datum` scalars once the kernels are threaded
    /// through selection vectors.
    pub(crate) fn to_array(&self, n: usize) -> ArrayRef {
        match self {
            Literal::Int(v) => Arc::new(Int64Array::from(vec![*v; n])),
            Literal::Float(v) => Arc::new(Float64Array::from(vec![*v; n])),
            Literal::Bool(v) => Arc::new(BooleanArray::from(vec![*v; n])),
            Literal::Str(v) => Arc::new(StringArray::from(vec![v.as_str(); n])),
            Literal::Timestamp(v) => Arc::new(TimestampMicrosecondArray::from(vec![*v; n])),
            Literal::Date(v) => Arc::new(Date32Array::from(vec![*v; n])),
        }
    }
}

#[cfg(test)]
mod float_literal_tests {
    use super::*;

    fn lit_float(json: &str) -> f64 {
        let e: Expr = serde_json::from_str(json).unwrap();
        match e {
            Expr::Lit {
                value: Literal::Float(v),
            } => v,
            other => panic!("expected float literal, got {other:?}"),
        }
    }

    #[test]
    fn float_literal_accepts_finite_number_and_nonfinite_names() {
        // Finite floats keep the plain-number wire form.
        assert_eq!(lit_float(r#"{"e":"lit","value":{"float":1.5}}"#), 1.5);
        assert_eq!(lit_float(r#"{"e":"lit","value":{"float":-0.0}}"#), 0.0);
        // Non-finite floats arrive as name strings (JSON has no NaN/Inf tokens).
        assert!(lit_float(r#"{"e":"lit","value":{"float":"NaN"}}"#).is_nan());
        assert_eq!(
            lit_float(r#"{"e":"lit","value":{"float":"inf"}}"#),
            f64::INFINITY
        );
        assert_eq!(
            lit_float(r#"{"e":"lit","value":{"float":"-inf"}}"#),
            f64::NEG_INFINITY
        );
        assert_eq!(
            lit_float(r#"{"e":"lit","value":{"float":"Infinity"}}"#),
            f64::INFINITY
        );
    }
}

#[cfg(test)]
mod str_date_tests {
    use super::*;
    use arrow::array::{Array, Date32Array, RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};

    fn batch_str() -> RecordBatch {
        let s = StringArray::from(vec![Some("Hello"), Some("wOrld"), None, Some("abcdef")]);
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)])),
            vec![Arc::new(s)],
        )
        .unwrap()
    }

    fn s(name: &str) -> Box<Expr> {
        Box::new(Expr::Col {
            name: name.to_string(),
        })
    }

    fn strf(func: StrFunc, pattern: Option<&str>, start: Option<i64>, length: Option<i64>) -> Expr {
        Expr::Str {
            func,
            input: s("s"),
            pattern: pattern.map(|p| p.to_string()),
            replacement: None,
            start,
            length,
        }
    }

    #[test]
    fn upper_lower_preserve_nulls() {
        let b = batch_str();
        let up = strf(StrFunc::Upper, None, None, None).eval(&b).unwrap();
        let up = up.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(up.value(0), "HELLO");
        assert_eq!(up.value(1), "WORLD");
        assert!(up.is_null(2));
    }

    #[test]
    fn len_counts_chars() {
        let b = batch_str();
        let l = strf(StrFunc::Len, None, None, None).eval(&b).unwrap();
        let l = l.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(l.value(0), 5);
        assert!(l.is_null(2));
    }

    #[test]
    fn contains_starts_ends() {
        let b = batch_str();
        let c = strf(StrFunc::Contains, Some("ell"), None, None)
            .eval(&b)
            .unwrap();
        let c = c.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(c.value(0) && !c.value(1) && c.is_null(2));
        let sw = strf(StrFunc::StartsWith, Some("abc"), None, None)
            .eval(&b)
            .unwrap();
        assert!(sw.as_any().downcast_ref::<BooleanArray>().unwrap().value(3));
    }

    #[test]
    fn substr_one_based() {
        let b = batch_str();
        let r = strf(StrFunc::Substr, None, Some(2), Some(3))
            .eval(&b)
            .unwrap();
        let r = r.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(r.value(0), "ell"); // "Hello"[2..5)
        assert_eq!(r.value(3), "bcd");
        // length to end
        let r2 = strf(StrFunc::Substr, None, Some(3), None).eval(&b).unwrap();
        assert_eq!(
            r2.as_any().downcast_ref::<StringArray>().unwrap().value(3),
            "cdef"
        );
    }

    #[test]
    fn like_ilike_semantics() {
        let s = StringArray::from(vec![
            Some("abc"),
            Some("a.b"),
            Some("axb"),
            None,
            Some("HELLO"),
        ]);
        let b = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)])),
            vec![Arc::new(s)],
        )
        .unwrap();
        // `a%` is anchored: matches anything starting with "a".
        let r = strf(StrFunc::Like, Some("a%"), None, None)
            .eval(&b)
            .unwrap();
        let r = r.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(r.value(0) && r.value(1) && r.value(2));
        assert!(r.is_null(3) && !r.value(4));
        // `a.b` literal-matches "a.b" only (the `.` is NOT a wildcard).
        let r = strf(StrFunc::Like, Some("a.b"), None, None)
            .eval(&b)
            .unwrap();
        let r = r.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(!r.value(0) && r.value(1) && !r.value(2));
        // `_` matches exactly one char: "a_b" matches "a.b" and "axb" but not "abc".
        let r = strf(StrFunc::Like, Some("a_b"), None, None)
            .eval(&b)
            .unwrap();
        let r = r.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(!r.value(0) && r.value(1) && r.value(2));
        // ILIKE is case-insensitive.
        let r = strf(StrFunc::Ilike, Some("hello"), None, None)
            .eval(&b)
            .unwrap();
        let r = r.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(r.value(4));
    }

    #[test]
    fn date_year_month_day() {
        // 2021-03-15 = day 18701 since epoch.
        let d = Date32Array::from(vec![Some(18701), None]);
        let b = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("d", DataType::Date32, true)])),
            vec![Arc::new(d)],
        )
        .unwrap();
        let year = Expr::Date {
            func: DateFunc::Year,
            input: Box::new(Expr::Col { name: "d".into() }),
        };
        let y = year.eval(&b).unwrap();
        let y = y.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(y.value(0), 2021);
        assert!(y.is_null(1));
    }

    #[test]
    fn case_coerces_int_then_against_float_otherwise() {
        // `when(true).then(0).otherwise(x)` over a Float64 column must coerce the
        // Int64 `then` to Float64 rather than erroring on mismatched zip types —
        // this is what makes `clip` / `fill_nan` / mixed when-then-otherwise work.
        use arrow::array::Float64Array;
        let x = Float64Array::from(vec![Some(1.0), Some(5.0)]);
        let b = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)])),
            vec![Arc::new(x)],
        )
        .unwrap();
        let case = Expr::Case {
            branches: vec![CaseBranch {
                when: Expr::Lit {
                    value: Literal::Bool(true),
                },
                then: Expr::Lit {
                    value: Literal::Int(0),
                },
            }],
            otherwise: Box::new(Expr::Col { name: "x".into() }),
        };
        let out = case.eval(&b).unwrap();
        let out = out.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(out.value(0), 0.0);
        assert_eq!(out.value(1), 0.0);
    }

    #[test]
    fn case_null_when_falls_through_to_else() {
        // SQL semantics: `CASE WHEN (x < 2) THEN 99 ELSE x` over x = [1, 5, null].
        // The null row's WHEN is null → not taken → ELSE (x stays null), it must NOT
        // pick the THEN branch.
        use arrow::array::Float64Array;
        let x = Float64Array::from(vec![Some(1.0), Some(5.0), None]);
        let b = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)])),
            vec![Arc::new(x)],
        )
        .unwrap();
        let lt = Expr::Binary {
            op: BinaryOp::Lt,
            left: Box::new(Expr::Col { name: "x".into() }),
            right: Box::new(Expr::Lit {
                value: Literal::Float(2.0),
            }),
        };
        let case = Expr::Case {
            branches: vec![CaseBranch {
                when: lt,
                then: Expr::Lit {
                    value: Literal::Float(99.0),
                },
            }],
            otherwise: Box::new(Expr::Col { name: "x".into() }),
        };
        let out = case.eval(&b).unwrap();
        let out = out.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(out.value(0), 99.0); // 1 < 2 → then
        assert_eq!(out.value(1), 5.0); // 5 < 2 false → else
        assert!(out.is_null(2)); // null when → else (null), not 99
    }
}
