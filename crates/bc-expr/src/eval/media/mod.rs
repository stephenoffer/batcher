//! Library-backed multimodal decoders (image / audio / video) for the
//! `.image`/`.audio`/`.video` expression namespaces.
//!
//! These are the interpreter oracle for media decode: the JIT cannot compile
//! library-backed decode, so it falls back here (one implementation, no tier
//! divergence). Grouped under `media/` to keep `eval/` within its file-count limit.

pub(crate) mod audio;
pub(crate) mod image;
pub(crate) mod video;

pub(crate) use audio::eval_audio;
pub(crate) use image::eval_image;
pub(crate) use video::eval_video;
