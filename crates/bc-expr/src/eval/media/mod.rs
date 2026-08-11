//! Library-backed multimodal decoders (image / audio / video) for the
//! `.image`/`.audio`/`.video` expression namespaces.
//!
//! These are the interpreter oracle for media decode: the JIT cannot compile
//! library-backed decode, so it falls back here (one implementation, no tier
//! divergence). Grouped under `media/` to keep `eval/` within its file-count limit.

use arrow::array::ArrayRef;
use rayon::prelude::*;

pub(crate) mod audio;
pub(crate) mod image;
pub(crate) mod mel;
pub(crate) mod video;

mod speech;

pub(crate) use audio::eval_audio;
pub(crate) use image::{eval_image, eval_image_crop, Bounds};
pub(crate) use video::{eval_video, VideoArgs};

/// Below this row count, decode serially: a handful of clips isn't worth rayon's
/// fan-out/join overhead, and it keeps the sub-second small-batch path allocation-free.
/// Above it, per-row library decode (milliseconds each) dwarfs the split cost, so the row
/// loop fans out across the shared rayon pool. See the crate-level note on nested rayon.
const PAR_ROW_THRESHOLD: usize = 8;

/// Map `f` over each row index `0..len`, order-preserving, in parallel once the batch
/// clears [`PAR_ROW_THRESHOLD`]. This is where media decode's row-level parallelism comes
/// from: the `bc-interp` morsel pool runs a whole morsel of clips on one thread, so a
/// corpus smaller than a morsel (16,384 rows) would otherwise decode fully serially — on
/// a single core of the box. Shared by the image / audio / video kernels.
pub(crate) fn map_rows<T, F>(len: usize, f: F) -> Vec<T>
where
    T: Send,
    F: (Fn(usize) -> T) + Sync + Send,
{
    if len >= PAR_ROW_THRESHOLD {
        (0..len).into_par_iter().map(f).collect()
    } else {
        (0..len).map(f).collect()
    }
}

/// Read an all-null column as an all-null **Binary** column of the same length.
///
/// An Arrow batch types a column of nothing but nulls as `Null`, not as `Binary` with the
/// null bits set — and that shape arrives constantly in a media pipeline: a download stage
/// where every fetch failed, an outer join that matched nothing, a partition whose rows
/// were all filtered out upstream. Every kernel here already answers null for a null row,
/// so the entire fix is to hand them a `Binary` array that is null everywhere; the output
/// type and the null bits then come out exactly right with no per-op special case.
///
/// Without it, each of these ops *failed the batch* with `expected a Binary argument, got
/// Null` — a type error about a column the caller never typed, on the one input where the
/// answer was never in doubt.
pub(crate) fn widen_null_column(arr: &ArrayRef) -> Option<ArrayRef> {
    use arrow::datatypes::DataType;

    matches!(arr.data_type(), DataType::Null)
        .then(|| arrow::array::new_null_array(&DataType::Binary, arr.len()))
}
