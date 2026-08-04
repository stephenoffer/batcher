//! Library-backed multimodal decoders (image / audio / video) for the
//! `.image`/`.audio`/`.video` expression namespaces.
//!
//! These are the interpreter oracle for media decode: the JIT cannot compile
//! library-backed decode, so it falls back here (one implementation, no tier
//! divergence). Grouped under `media/` to keep `eval/` within its file-count limit.

use rayon::prelude::*;

pub(crate) mod audio;
pub(crate) mod image;
pub(crate) mod mel;
pub(crate) mod video;

mod speech;

pub(crate) use audio::eval_audio;
pub(crate) use image::eval_image;
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
