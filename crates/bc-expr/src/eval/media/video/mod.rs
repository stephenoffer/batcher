//! Video evaluation for `Expr::Video` (the `.video` namespace).
//!
//! Backed by the system FFmpeg behind the optional `video` cargo feature. Without the
//! feature the variants still deserialize (the wire contract is unconditional) but
//! evaluation returns a clear error. The JIT never compiles `Video`; this interpreter
//! path is the only one, so the tiers cannot diverge.
//!
//! Four ops, one shape: `decode` reads the header, and `frames`/`thumbnail`/`frame_at`
//! turn a clip into pixels. The sampling three are the reason this file is not just a
//! metadata probe — extracting frames is the *first* step of every video pipeline, and
//! doing it anywhere but here means a per-row Python decode loop in the control plane.
//!
//! FFmpeg reads from a path rather than a memory buffer, so each clip's bytes are written
//! to a short-lived temp file. That cost is paid once per row per op and is small next to
//! the decode itself; [`ClipFile`] makes sure it is also always cleaned up, including on
//! an early return.

#[cfg(feature = "video")]
mod sample;

use arrow::array::ArrayRef;

use crate::{ExprError, VideoFunc};

/// The scalar arguments a video function may carry, gathered into one struct.
///
/// Passed to every op even though only the sampling three read it, for the same reason
/// `image::ImageArgs` exists: four positional `Option`s at a call site is a place where
/// swapping `width` and `height` is a silent bug the compiler cannot see.
///
/// Without the `video` feature nothing reads the fields — the stub `eval_video` below
/// returns `FeatureDisabled` before looking at them — but the struct itself must still
/// exist, because `eval::dispatch` builds one unconditionally. That is what the
/// feature-gated `allow` is for; an unconditional one would hide a real dead field.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(not(feature = "video"), allow(dead_code))]
pub(crate) struct VideoArgs {
    pub num_frames: Option<i64>,
    pub width: Option<i64>,
    pub height: Option<i64>,
    pub second: Option<f64>,
}

#[cfg(feature = "video")]
pub(crate) fn eval_video(
    func: VideoFunc,
    arr: &ArrayRef,
    args: VideoArgs,
) -> Result<ArrayRef, ExprError> {
    let clips = Clips::new(func, arr)?;
    match func {
        VideoFunc::Decode => decode_meta(&clips),
        VideoFunc::Frames => sample::frames(&clips, args),
        VideoFunc::Thumbnail => sample::thumbnail(&clips, args),
        VideoFunc::FrameAt => sample::frame_at(&clips, args),
    }
}

/// A Binary or LargeBinary column of encoded clips, read through one accessor.
///
/// Both offset widths are accepted because a media source stores payloads as
/// `LargeBinary`: 32-bit offsets cap an array at 2 GB *in total*, which a batch of video
/// clips reaches immediately. Narrowing this to `Binary` would make the namespace fail on
/// exactly the inputs it exists for.
#[cfg(feature = "video")]
pub(crate) struct Clips<'a> {
    narrow: Option<&'a arrow::array::GenericBinaryArray<i32>>,
    wide: Option<&'a arrow::array::GenericBinaryArray<i64>>,
    len: usize,
}

#[cfg(feature = "video")]
impl<'a> Clips<'a> {
    fn new(func: VideoFunc, arr: &'a ArrayRef) -> Result<Self, ExprError> {
        use arrow::array::GenericBinaryArray;
        use arrow::datatypes::DataType;

        let (narrow, wide) = match arr.data_type() {
            DataType::Binary => (arr.as_any().downcast_ref::<GenericBinaryArray<i32>>(), None),
            DataType::LargeBinary => (None, arr.as_any().downcast_ref::<GenericBinaryArray<i64>>()),
            other => {
                return Err(ExprError::ExpectedBinary {
                    func: format!("{func:?}"),
                    got: other.to_string(),
                })
            }
        };
        Ok(Self {
            narrow,
            wide,
            len: arr.len(),
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// The bytes of row `i`, or `None` when the row is null.
    pub(crate) fn get(&self, i: usize) -> Option<&'a [u8]> {
        use arrow::array::Array;
        match (self.narrow, self.wide) {
            (Some(b), _) if !b.is_null(i) => Some(b.value(i)),
            (_, Some(b)) if !b.is_null(i) => Some(b.value(i)),
            _ => None,
        }
    }
}

/// One clip's header facts: `(width, height, num_frames, duration_secs, fps)`.
///
/// Named because it crosses two function signatures, and a bare five-tuple at both
/// ends is a place where two `f64`s can be swapped with nothing to notice.
#[cfg(feature = "video")]
type ClipMeta = (i32, i32, i64, f64, f64);

/// `decode()` → struct `{width, height, num_frames, duration_secs, fps}`.
#[cfg(feature = "video")]
fn decode_meta(clips: &Clips<'_>) -> Result<ArrayRef, ExprError> {
    use std::sync::Arc;

    use arrow::array::{Float64Array, Int32Array, Int64Array, StructArray};
    use arrow::buffer::NullBuffer;
    use arrow::datatypes::{DataType, Field};

    // Probe every clip in parallel (each opens its own temp file, so the rows are
    // independent), then fold into the column buffers serially. See `super::map_rows` —
    // without it a sub-morsel batch would probe on a single core.
    let metas: Vec<Option<ClipMeta>> =
        super::map_rows(clips.len(), |i| clips.get(i).and_then(probe_meta));
    let (mut w, mut h) = (Vec::new(), Vec::new());
    let (mut frames, mut dur, mut fps) = (Vec::new(), Vec::new(), Vec::new());
    let mut valid = Vec::with_capacity(clips.len());
    for meta in metas {
        match meta {
            Some((vw, vh, nf, d, f)) => {
                w.push(vw);
                h.push(vh);
                frames.push(nf);
                dur.push(d);
                fps.push(f);
                valid.push(true);
            }
            None => {
                // A struct's child arrays stay full length; the row's null bit is what
                // marks it absent, so these placeholders are never read.
                w.push(0);
                h.push(0);
                frames.push(0);
                dur.push(0.0);
                fps.push(0.0);
                valid.push(false);
            }
        }
    }
    let fields = vec![
        Arc::new(Field::new("width", DataType::Int32, false)),
        Arc::new(Field::new("height", DataType::Int32, false)),
        Arc::new(Field::new("num_frames", DataType::Int64, false)),
        Arc::new(Field::new("duration_secs", DataType::Float64, false)),
        Arc::new(Field::new("fps", DataType::Float64, false)),
    ];
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from(w)),
        Arc::new(Int32Array::from(h)),
        Arc::new(Int64Array::from(frames)),
        Arc::new(Float64Array::from(dur)),
        Arc::new(Float64Array::from(fps)),
    ];
    Ok(Arc::new(StructArray::new(
        fields.into(),
        cols,
        Some(NullBuffer::from(valid)),
    )))
}

/// Probe `(width, height, num_frames, duration_secs, fps)` from a clip's header.
#[cfg(feature = "video")]
fn probe_meta(data: &[u8]) -> Option<ClipMeta> {
    let clip = ClipFile::write(data)?;
    let ictx = open_input(clip.path())?;
    let stream = ictx.streams().best(ffmpeg_next::media::Type::Video)?;
    let ctx = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters()).ok()?;
    let dec = ctx.decoder().video().ok()?;
    let duration = if stream.duration() >= 0 {
        stream.duration() as f64 * rational(stream.time_base())
    } else {
        0.0
    };
    Some((
        dec.width() as i32,
        dec.height() as i32,
        stream.frames(),
        duration,
        rational(stream.avg_frame_rate()),
    ))
}

/// An FFmpeg `Rational` as an `f64`, with a zero denominator reading as 0 rather than NaN.
#[cfg(feature = "video")]
pub(crate) fn rational(r: ffmpeg_next::Rational) -> f64 {
    if r.denominator() != 0 {
        r.numerator() as f64 / r.denominator() as f64
    } else {
        0.0
    }
}

/// Initialize FFmpeg once per process and silence its logger.
///
/// The logger matters as much as the init: FFmpeg writes decode diagnostics straight to
/// the process's stderr, so without this a single corrupt clip in a corpus prints a block
/// of C library output *per row* — tens of thousands of lines interleaved into whatever
/// the caller was reading. A row that will not decode is already reported as a null; it
/// does not also need to be shouted.
#[cfg(feature = "video")]
fn ffmpeg_init() -> bool {
    use std::sync::OnceLock;

    static INIT: OnceLock<bool> = OnceLock::new();
    *INIT.get_or_init(|| {
        let ok = ffmpeg_next::init().is_ok();
        if ok {
            ffmpeg_next::util::log::set_level(ffmpeg_next::util::log::Level::Quiet);
        }
        ok
    })
}

/// Open a demuxer over a clip file, or `None` if FFmpeg cannot read it.
#[cfg(feature = "video")]
pub(crate) fn open_input(path: &std::path::Path) -> Option<ffmpeg_next::format::context::Input> {
    if !ffmpeg_init() {
        return None;
    }
    ffmpeg_next::format::input(path).ok()
}

/// A clip's bytes on disk for as long as FFmpeg needs a path to them.
///
/// FFmpeg's demuxer takes a filename, not a buffer, so the bytes have to land somewhere.
/// The `Drop` is the point: an ordinary `remove_file` at the end of the function leaks the
/// file on every early return, and each of these decoders has several — one per
/// unsupported codec, unreadable header, or truncated stream. A corpus with a corrupt tail
/// would otherwise leave one temp file per bad row behind for the life of the process.
#[cfg(feature = "video")]
pub(crate) struct ClipFile {
    path: std::path::PathBuf,
}

#[cfg(feature = "video")]
impl ClipFile {
    /// Write `data` to a uniquely-named temp file, or `None` if it cannot be written.
    pub(crate) fn write(data: &[u8]) -> Option<Self> {
        use std::io::Write;
        use std::sync::atomic::{AtomicU64, Ordering};

        // A unique path without an extra dependency (no rand, no clock): the process id
        // plus a monotonic counter, which is unique across threads within this process
        // and across processes by the pid.
        static CTR: AtomicU64 = AtomicU64::new(0);
        let n = CTR.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("bc_video_{}_{n}.bin", std::process::id()));
        let mut f = std::fs::File::create(&path).ok()?;
        // A partial write would be silently misread as a truncated clip, so a failure
        // here removes the file rather than leaving a half-written one behind.
        if f.write_all(data).and_then(|()| f.flush()).is_err() {
            let _ = std::fs::remove_file(&path);
            return None;
        }
        Some(Self { path })
    }

    pub(crate) fn path(&self) -> &std::path::Path {
        &self.path
    }
}

#[cfg(feature = "video")]
impl Drop for ClipFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

#[cfg(not(feature = "video"))]
pub(crate) fn eval_video(
    func: VideoFunc,
    _arr: &ArrayRef,
    _args: VideoArgs,
) -> Result<ArrayRef, ExprError> {
    Err(ExprError::FeatureDisabled {
        func: format!("video.{func:?}").to_lowercase(),
        feature: "video",
    })
}
