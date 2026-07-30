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
pub use error::ExprError;
pub use select::ConjunctOrder;

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
        /// `Crop` only: the top-left corner of the window. `#[serde(default)]`, so every
        /// other image op's IR round-trips unchanged.
        #[serde(default)]
        x: Option<i64>,
        #[serde(default)]
        y: Option<i64>,
        /// `Encode` only: the target container format.
        #[serde(default)]
        format: Option<String>,
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
    },

    /// A video decode op over a binary (video-bytes) sub-expression. Backed by the
    /// system FFmpeg behind the `video` feature; without it, evaluation errors. The
    /// JIT falls back to this interpreter path.
    Video {
        #[serde(rename = "fn")]
        func: VideoFunc,
        input: Box<Expr>,
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
    /// `crop(x, y, width, height)` → the requested region, re-encoded as PNG bytes. The
    /// arbitrary-offset counterpart of `CenterCrop`, for pulling a detection's bounding box
    /// out of a frame and keeping it as an image rather than as a tensor. A window that
    /// runs past an edge is clipped to the image, so the output can be smaller than
    /// requested; a window entirely outside it is null. Null/undecodable input → null.
    /// → Binary.
    Crop,
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
    /// `mfcc(rate, n_fft, hop_length, n_mels, n_mfcc)` → the Mel-Frequency Cepstral
    /// Coefficients, the classic compact speech feature: mel power spectrogram →
    /// `AmplitudeToDB` → orthonormal DCT-II, keeping the first `n_mfcc` coefficients.
    /// Emitted as a `List<Float32>` row-major `(n_mfcc, n_frames)`. Numerically matches
    /// `torchaudio.transforms.MFCC` defaults. Null/undecodable → null.
    Mfcc,
}

/// Video-decode operations for the `.video` namespace. `Decode` reads each clip's
/// metadata into a struct. Requires the `video` cargo feature (system FFmpeg).
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VideoFunc {
    Decode,
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
