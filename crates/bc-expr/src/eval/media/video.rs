//! Video-decode evaluation for `Expr::Video` (the `.video` namespace).
//!
//! Backed by the system FFmpeg behind the optional `video` cargo feature. FFmpeg
//! reads from a path, not an in-memory buffer, so each clip's bytes are written to a
//! short-lived temp file and probed for metadata. Without the feature, the variant
//! still deserializes (wire contract) but evaluation returns a clear error. The JIT
//! never compiles `Video`; this interpreter path is the only one.

use arrow::array::ArrayRef;

use crate::{ExprError, VideoFunc};

#[cfg(feature = "video")]
pub(crate) fn eval_video(func: VideoFunc, arr: &ArrayRef) -> Result<ArrayRef, ExprError> {
    use std::sync::Arc;

    use arrow::array::{Array, BinaryArray, Float64Array, Int32Array, Int64Array, StructArray};
    use arrow::buffer::NullBuffer;
    use arrow::datatypes::{DataType, Field};

    let bytes =
        arr.as_any()
            .downcast_ref::<BinaryArray>()
            .ok_or_else(|| ExprError::ExpectedBinary {
                func: format!("{func:?}"),
                got: arr.data_type().to_string(),
            })?;
    let VideoFunc::Decode = func;

    let (mut w, mut h) = (Vec::new(), Vec::new());
    let (mut frames, mut dur, mut fps) = (Vec::new(), Vec::new(), Vec::new());
    let mut valid = Vec::with_capacity(bytes.len());
    for i in 0..bytes.len() {
        let meta = if bytes.is_null(i) {
            None
        } else {
            decode_video_meta(bytes.value(i))
        };
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

/// Probe video metadata `(width, height, num_frames, duration_secs, fps)` via FFmpeg
/// through a short-lived temp file; `None` on any failure.
#[cfg(feature = "video")]
fn decode_video_meta(data: &[u8]) -> Option<(i32, i32, i64, f64, f64)> {
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    // A unique temp path without an extra dep (no rand/clock): pid + an atomic counter.
    static CTR: AtomicU64 = AtomicU64::new(0);
    let n = CTR.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("bc_video_{}_{n}.bin", std::process::id()));
    {
        let mut f = std::fs::File::create(&path).ok()?;
        f.write_all(data).ok()?;
    }
    let result = (|| {
        ffmpeg_next::init().ok()?;
        let ictx = ffmpeg_next::format::input(&path).ok()?;
        let stream = ictx.streams().best(ffmpeg_next::media::Type::Video)?;
        let ctx =
            ffmpeg_next::codec::context::Context::from_parameters(stream.parameters()).ok()?;
        let dec = ctx.decoder().video().ok()?;
        let tb = stream.time_base();
        let to_f64 = |r: ffmpeg_next::Rational| {
            if r.denominator() != 0 {
                r.numerator() as f64 / r.denominator() as f64
            } else {
                0.0
            }
        };
        let duration = if stream.duration() >= 0 {
            stream.duration() as f64 * to_f64(tb)
        } else {
            0.0
        };
        Some((
            dec.width() as i32,
            dec.height() as i32,
            stream.frames(),
            duration,
            to_f64(stream.avg_frame_rate()),
        ))
    })();
    let _ = std::fs::remove_file(&path);
    result
}

#[cfg(not(feature = "video"))]
pub(crate) fn eval_video(func: VideoFunc, _arr: &ArrayRef) -> Result<ArrayRef, ExprError> {
    Err(ExprError::FeatureDisabled {
        func: format!("video.{func:?}").to_lowercase(),
        feature: "video",
    })
}
