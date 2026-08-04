//! Turning a clip into pixels: `frames`, `thumbnail`, and `frame_at`.
//!
//! These are the kernels that let a video pipeline stay in the data plane. Without them
//! the only way to get frames out of a clip is a per-row Python loop over PyAV, which is
//! a hot-path tuple touch and caps the whole pipeline at one interpreter.
//!
//! Two sampling strategies live here, and which one an op uses is a correctness decision
//! rather than a tuning one:
//!
//! * **`frames` decodes in order.** Uniform frame sampling means the *n*-th frame of the
//!   clip, and only a sequential decode knows which frame that is. It costs a decode of
//!   the clip up to the last wanted frame.
//! * **`thumbnail` and `frame_at` seek.** Both want one frame near a timestamp, and
//!   neither has a frame index to be exact about. Seeking lands on the keyframe at or
//!   before the target, which is what every video tool means by "the frame at 3.5s", and
//!   it makes the cost independent of how far in the timestamp is.
//!
//! Every failure mode — unreadable container, unsupported codec, truncated stream, a
//! clip with no video track — yields a null row rather than failing the batch, matching
//! the image and audio kernels.

use std::sync::Arc;

use arrow::array::{ArrayRef, BinaryArray, FixedSizeListArray, UInt8Array};
use arrow::buffer::NullBuffer;
use arrow::datatypes::{DataType, Field};
use ffmpeg_next as ff;

use super::{open_input, rational, ClipFile, Clips};
use crate::eval::media::map_rows;
use crate::ExprError;

/// What size a sampled frame comes out at, validated once for the whole batch.
///
/// Two shapes, and which one an op uses follows one rule the whole media surface obeys:
/// **an op that produces a tensor takes exact dimensions, and an op that produces an
/// encoded still takes a longest side and keeps the aspect ratio.** A tensor feeds a model
/// that needs one fixed input shape, so every row must agree; a still is looked at, and
/// squashing a 16:9 frame onto a square is a distortion no shape assertion can see. It is
/// also what makes `.image.thumbnail` and `.video.thumbnail` the same operation rather
/// than two methods that happen to share a name.
///
/// A plain `as u32` on the query's dimensions silently wraps: a negative value becomes a
/// ~4-billion dimension (an allocation that aborts the process) and one past `u32::MAX`
/// wraps to a small one (a silently wrong output size). Both are caller bugs to surface.
enum Size {
    /// `frames`: every row is this exact size, whatever the clip's own aspect ratio.
    Exact { width: u32, height: u32 },
    /// `thumbnail` / `frame_at`: the longest side, with the other following the frame.
    Fit { max: u32 },
}

impl Size {
    fn exact(func: &str, args: super::VideoArgs) -> Result<Self, ExprError> {
        Ok(Self::Exact {
            width: dim(func, "width", args.width)?,
            height: dim(func, "height", args.height)?,
        })
    }

    fn fit(func: &str, args: super::VideoArgs) -> Result<Self, ExprError> {
        Ok(Self::Fit {
            max: dim(func, "max_size", args.width)?,
        })
    }

    /// The output dimensions for a source frame of `(sw, sh)`.
    fn resolve(&self, sw: u32, sh: u32) -> (u32, u32) {
        match *self {
            Self::Exact { width, height } => (width, height),
            // Never upscale, matching `.image.thumbnail` and `PIL.Image.thumbnail`: a
            // frame enlarged to reach the target invents detail and costs bytes.
            Self::Fit { max } if sw <= max && sh <= max => (sw.max(1), sh.max(1)),
            Self::Fit { max } => {
                let scale = f64::from(max) / f64::from(sw.max(sh));
                let w = ((f64::from(sw) * scale).round() as u32).max(1);
                let h = ((f64::from(sh) * scale).round() as u32).max(1);
                (w.min(max), h.min(max))
            }
        }
    }

    /// Bytes of RGB8 in one frame, for the fixed-size case only.
    fn frame_bytes(&self) -> u64 {
        match *self {
            Self::Exact { width, height } => (width as u64) * (height as u64) * 3,
            // `Fit` is only used by the still-producing ops, which never pre-allocate a
            // per-row buffer — the size is not known until the clip's own is.
            Self::Fit { .. } => 0,
        }
    }
}

fn dim(func: &str, arg: &'static str, value: Option<i64>) -> Result<u32, ExprError> {
    let value = value.ok_or(ExprError::MissingImageArg {
        func: func.to_string(),
        arg,
    })?;
    u32::try_from(value)
        .ok()
        .filter(|&v| v > 0)
        .ok_or(ExprError::InvalidImageDim {
            func: func.to_string(),
            arg,
            value,
            max: u32::MAX,
        })
}

/// `frames(n, w, h)` → `FixedSizeList<UInt8>` of `n*h*w*3` RGB8 samples per row.
///
/// The output is a fixed-shape tensor column, which is what makes it usable directly as a
/// video model's input: every row has the same `(n, h, w, 3)` shape whatever the source
/// clip's resolution or length was.
pub(super) fn frames(clips: &Clips<'_>, args: super::VideoArgs) -> Result<ArrayRef, ExprError> {
    let size = Size::exact("frames", args)?;
    let n = args.num_frames.ok_or(ExprError::MissingImageArg {
        func: "frames".to_string(),
        arg: "num_frames",
    })?;
    let n = usize::try_from(n)
        .ok()
        .filter(|&v| v > 0)
        .ok_or_else(|| ExprError::InvalidArgument {
            func: "frames".to_string(),
            reason: format!("num_frames must be a positive integer, got {n}"),
        })?;
    // The per-row element count is also the `FixedSizeList` element length, an Arrow
    // `i32`, and it is driven by three query parameters that multiply. Computed in `u64`
    // so the multiply cannot itself overflow, then rejected above `i32::MAX` — where the
    // later `as i32` would wrap negative and the `len * per_row` allocation would become
    // a multi-gigabyte OOM bomb.
    let per_frame = size.frame_bytes();
    let per_row = per_frame * (n as u64);
    if per_row > i32::MAX as u64 {
        return Err(ExprError::InvalidArgument {
            func: "frames".to_string(),
            reason: format!(
                "{n} frames of {per_frame} bytes is {per_row} bytes per row, over the \
                 maximum element length of {}",
                i32::MAX
            ),
        });
    }
    let per_row = per_row as usize;

    let rows: Vec<Option<Vec<u8>>> = map_rows(clips.len(), |i| {
        let data = clips.get(i)?;
        sample_uniform(data, n, &size).filter(|buf| buf.len() == per_row)
    });

    // Assemble serially: this is a memcpy per row, cheap next to the decode above.
    let mut values: Vec<u8> = vec![0u8; clips.len() * per_row];
    let mut valid: Vec<bool> = Vec::with_capacity(clips.len());
    for (i, row) in rows.into_iter().enumerate() {
        match row {
            Some(buf) => {
                values[i * per_row..(i + 1) * per_row].copy_from_slice(&buf);
                valid.push(true);
            }
            None => valid.push(false),
        }
    }
    Ok(Arc::new(FixedSizeListArray::new(
        Arc::new(Field::new("item", DataType::UInt8, false)),
        per_row as i32,
        Arc::new(UInt8Array::from(values)),
        Some(NullBuffer::from(valid)),
    )))
}

/// `thumbnail(w, h)` → PNG bytes of the clip's middle frame.
pub(super) fn thumbnail(clips: &Clips<'_>, args: super::VideoArgs) -> Result<ArrayRef, ExprError> {
    let size = Size::fit("thumbnail", args)?;
    let out: Vec<Option<Vec<u8>>> = map_rows(clips.len(), |i| {
        let data = clips.get(i)?;
        // `None` for the fraction means "half the duration" — resolved inside, where the
        // duration is known.
        let (rgb, w, h) = seek_frame(data, None, &size)?;
        encode_png(&rgb, w, h)
    });
    Ok(Arc::new(BinaryArray::from_iter(out)))
}

/// `frame_at(second, w, h)` → PNG bytes of the frame at `second`.
pub(super) fn frame_at(clips: &Clips<'_>, args: super::VideoArgs) -> Result<ArrayRef, ExprError> {
    let size = Size::fit("frame_at", args)?;
    let second = args.second.ok_or(ExprError::MissingImageArg {
        func: "frame_at".to_string(),
        arg: "second",
    })?;
    if !second.is_finite() || second < 0.0 {
        return Err(ExprError::InvalidArgument {
            func: "frame_at".to_string(),
            reason: format!("second must be a finite, non-negative number of seconds, got {second}"),
        });
    }
    let out: Vec<Option<Vec<u8>>> = map_rows(clips.len(), |i| {
        let data = clips.get(i)?;
        let (rgb, w, h) = seek_frame(data, Some(second), &size)?;
        encode_png(&rgb, w, h)
    });
    Ok(Arc::new(BinaryArray::from_iter(out)))
}

/// Encode one `w*h*3` RGB8 buffer as PNG.
fn encode_png(rgb: &[u8], w: u32, h: u32) -> Option<Vec<u8>> {
    use std::io::Cursor;

    let img = image::RgbImage::from_raw(w, h, rgb.to_vec())?;
    let mut buf = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(img)
        .write_to(&mut buf, image::ImageFormat::Png)
        .ok()?;
    Some(buf.into_inner())
}

/// A decoder plus the scaler that lands its frames at the target size, built together.
///
/// The scaler must be created from the decoder's own pixel format and dimensions, which
/// are only known after the decoder exists — and creating one *per frame* (the obvious
/// mistake) rebuilds a swscale context tens of times per clip for no gain.
struct Decoding {
    decoder: ff::decoder::Video,
    scaler: ff::software::scaling::Context,
    stream_index: usize,
    /// Seconds per presentation timestamp unit, so a frame's `pts` can be compared
    /// against a wall-clock target.
    time_base: f64,
    /// The size frames come out at, resolved against *this clip's* own dimensions. For a
    /// `Fit` target that is not knowable until the decoder exists, which is why it is
    /// recorded here rather than recomputed at each use.
    out: (u32, u32),
}

impl Decoding {
    fn open(ictx: &ff::format::context::Input, size: &Size) -> Option<Self> {
        let stream = ictx.streams().best(ff::media::Type::Video)?;
        let stream_index = stream.index();
        let time_base = rational(stream.time_base());
        let ctx = ff::codec::context::Context::from_parameters(stream.parameters()).ok()?;
        let decoder = ctx.decoder().video().ok()?;
        if decoder.width() == 0 || decoder.height() == 0 {
            return None;
        }
        let out = size.resolve(decoder.width(), decoder.height());
        let scaler = ff::software::scaling::Context::get(
            decoder.format(),
            decoder.width(),
            decoder.height(),
            ff::format::Pixel::RGB24,
            out.0,
            out.1,
            // Bilinear matches the `.image` resize path's filter, so a frame scaled here
            // and an image scaled there agree about what downscaling means.
            ff::software::scaling::Flags::BILINEAR,
        )
        .ok()?;
        Some(Self {
            decoder,
            scaler,
            stream_index,
            time_base,
            out,
        })
    }

    /// A decoded frame's presentation time in seconds, when the container carries one.
    fn frame_secs(&self, frame: &ff::frame::Video) -> Option<f64> {
        (self.time_base > 0.0)
            .then(|| frame.pts().or_else(|| frame.timestamp()))
            .flatten()
            .map(|pts| pts as f64 * self.time_base)
    }

    /// Scale one decoded frame into a tightly packed `w*h*3` RGB8 buffer.
    ///
    /// The pack is not optional: FFmpeg pads each row out to a `stride(0)` that is `>=`
    /// `width * 3`, so handing `data(0)` straight to Arrow would ship the padding as
    /// pixels and shear the image progressively down the frame.
    fn to_rgb(&mut self, frame: &ff::frame::Video) -> Option<Vec<u8>> {
        let (w, h) = self.out;
        let mut rgb = ff::frame::Video::empty();
        self.scaler.run(frame, &mut rgb).ok()?;
        let row_bytes = (w as usize) * 3;
        let stride = rgb.stride(0);
        let data = rgb.data(0);
        let mut out = Vec::with_capacity(row_bytes * h as usize);
        for y in 0..h as usize {
            let start = y * stride;
            let end = start.checked_add(row_bytes)?;
            out.extend_from_slice(data.get(start..end)?);
        }
        Some(out)
    }
}

/// Sample `n` evenly-spaced frames from a clip, each scaled to `size`, concatenated.
///
/// Decodes in order and keeps only the wanted frames, so peak memory is the output plus
/// one frame rather than the whole clip — a minute of 1080p at 30fps is ~11 GB decoded,
/// for eight frames of output. The tail of the clip past the last wanted frame is never
/// decoded at all.
fn sample_uniform(data: &[u8], n: usize, size: &Size) -> Option<Vec<u8>> {
    let clip = ClipFile::write(data)?;
    let total = frame_total(clip.path())?;
    if total == 0 {
        return None;
    }
    let wanted = linspace(total, n);
    let last = *wanted.last()?;

    let mut ictx = open_input(clip.path())?;
    let mut dec = Decoding::open(&ictx, size)?;
    let frame_bytes = size.frame_bytes() as usize;
    let mut out = vec![0u8; frame_bytes * n];
    // Which output slots each wanted source index fills. A clip shorter than `n` frames
    // repeats frames rather than failing, so one source index can fill several slots.
    let mut filled = vec![false; n];
    let mut pos: usize = 0;
    let mut decoded = ff::frame::Video::empty();

    let mut place = |dec: &mut Decoding, frame: &ff::frame::Video, pos: usize| -> Option<()> {
        let rgb = dec.to_rgb(frame)?;
        if rgb.len() != frame_bytes {
            return None;
        }
        for (slot, &want) in wanted.iter().enumerate() {
            if want == pos {
                out[slot * frame_bytes..(slot + 1) * frame_bytes].copy_from_slice(&rgb);
                filled[slot] = true;
            }
        }
        Some(())
    };

    'outer: for (stream, packet) in ictx.packets() {
        if stream.index() != dec.stream_index {
            continue;
        }
        if dec.decoder.send_packet(&packet).is_err() {
            continue;
        }
        while dec.decoder.receive_frame(&mut decoded).is_ok() {
            if wanted.contains(&pos) {
                let frame = std::mem::replace(&mut decoded, ff::frame::Video::empty());
                place(&mut dec, &frame, pos)?;
                decoded = frame;
            }
            pos += 1;
            if pos > last {
                break 'outer;
            }
        }
    }
    // Flush whatever the decoder still holds — a clip whose last wanted frame is also its
    // last frame only produces it after EOF, so skipping this drops the final sample.
    if pos <= last && dec.decoder.send_eof().is_ok() {
        while dec.decoder.receive_frame(&mut decoded).is_ok() {
            if wanted.contains(&pos) {
                let frame = std::mem::replace(&mut decoded, ff::frame::Video::empty());
                place(&mut dec, &frame, pos)?;
                decoded = frame;
            }
            pos += 1;
            if pos > last {
                break;
            }
        }
    }
    // A slot that never got a frame would be black, and a black frame is indistinguishable
    // from a legitimately black one — exactly the silent corruption that puts blank samples
    // into a training set. Null the row instead.
    filled.iter().all(|&f| f).then_some(out)
}

/// The clip's frame count, from the header when it is there and by counting when it is not.
///
/// The header's count is free but only a hint: a stream copy, a truncated file, or a
/// fragmented MP4 all leave it zero or wrong. Falling back to `duration * fps` covers the
/// common case of a container that knows how long it is but not how many frames that was;
/// only when neither is available does this decode the clip to count it.
fn frame_total(path: &std::path::Path) -> Option<usize> {
    let ictx = open_input(path)?;
    let stream = ictx.streams().best(ff::media::Type::Video)?;
    let declared = stream.frames();
    if declared > 0 {
        return usize::try_from(declared).ok();
    }
    let duration = if stream.duration() > 0 {
        stream.duration() as f64 * rational(stream.time_base())
    } else {
        0.0
    };
    let fps = rational(stream.avg_frame_rate());
    let estimated = duration * fps;
    if estimated >= 1.0 {
        return Some(estimated as usize);
    }
    drop(ictx);
    count_frames(path)
}

/// Decode a clip and count its frames, without scaling any of them.
///
/// The last resort when neither a frame count nor a duration is in the header. It is a
/// full decode pass, which is why it is not the first choice — but returning `None` here
/// would drop every clip in a corpus of stream copies, which is worse than decoding twice.
fn count_frames(path: &std::path::Path) -> Option<usize> {
    let mut ictx = open_input(path)?;
    let stream = ictx.streams().best(ff::media::Type::Video)?;
    let index = stream.index();
    let ctx = ff::codec::context::Context::from_parameters(stream.parameters()).ok()?;
    let mut decoder = ctx.decoder().video().ok()?;
    let mut frame = ff::frame::Video::empty();
    let mut n = 0usize;
    for (stream, packet) in ictx.packets() {
        if stream.index() != index {
            continue;
        }
        if decoder.send_packet(&packet).is_err() {
            continue;
        }
        while decoder.receive_frame(&mut frame).is_ok() {
            n += 1;
        }
    }
    if decoder.send_eof().is_ok() {
        while decoder.receive_frame(&mut frame).is_ok() {
            n += 1;
        }
    }
    (n > 0).then_some(n)
}

/// `n` evenly-spaced indices over `0..total`, inclusive of both ends.
///
/// Matches `numpy.linspace(0, total - 1, n).astype(int)`, which is what the Python video
/// loader this replaces used and what every video model's reference preprocessing does —
/// so a pipeline moving onto the native kernel samples the same frames it did before.
fn linspace(total: usize, n: usize) -> Vec<usize> {
    if n == 1 || total == 1 {
        // A single sample takes the first frame, not the middle: `linspace(a, b, 1)` is
        // `[a]`, and matching numpy here is the whole point.
        return vec![0; n];
    }
    let last = (total - 1) as f64;
    (0..n)
        .map(|i| {
            let t = last * (i as f64) / ((n - 1) as f64);
            // Truncation, not rounding: `astype(int)` truncates, and a half-frame
            // disagreement would make the native kernel sample differently from the
            // reference preprocessing it is replacing.
            (t as usize).min(total - 1)
        })
        .collect()
}

/// How much a frame's timestamp may sit past the target and still count as *at* it.
///
/// Timestamps come back as `pts * time_base`, a rational-to-float conversion, so the
/// frame the caller means can land a hair either side of the number they asked for. A
/// microsecond is far below any real frame interval and far above that rounding.
const TS_EPSILON: f64 = 1e-6;

/// Decode the frame shown at a timestamp, scaled to `size`, as packed RGB8.
///
/// `second` of `None` means the middle of the clip (the thumbnail case).
///
/// Seeking alone is not enough, and getting this wrong is invisible: FFmpeg can only seek
/// to a *keyframe*, and a clip encoded with a long GOP — anything from a screen recording
/// to a `crf 0` archival encode — may have exactly one, at frame zero. Stopping at the
/// first frame after the seek then returns frame zero for every timestamp in the clip,
/// which looks like a working thumbnail column until someone notices every thumbnail is
/// the same black title frame. So: seek to the keyframe at or before the target, then
/// decode *forward* and keep the last frame whose presentation time is still at or before
/// it — the frame a player displays at that instant. The decode is bounded by the
/// keyframe interval, not by how far into the clip the target is.
fn seek_frame(data: &[u8], second: Option<f64>, size: &Size) -> Option<(Vec<u8>, u32, u32)> {
    let clip = ClipFile::write(data)?;
    let mut ictx = open_input(clip.path())?;
    let target_secs = {
        let stream = ictx.streams().best(ff::media::Type::Video)?;
        let duration = if stream.duration() > 0 {
            stream.duration() as f64 * rational(stream.time_base())
        } else {
            0.0
        };
        match second {
            // Past the end of a clip we know the length of, there is no frame to return.
            // Saying so beats handing back the last frame under a timestamp that lied.
            Some(s) if duration > 0.0 && s >= duration => return None,
            Some(s) => s,
            None => duration / 2.0,
        }
    };
    // `seek` takes AV_TIME_BASE (microsecond) units and a range; giving it `..ts` asks
    // for the last keyframe at or before the target rather than the next one after.
    let ts = (target_secs * f64::from(ff::ffi::AV_TIME_BASE)) as i64;
    if ts > 0 {
        // A container that cannot seek (a raw stream, a pipe-written file) is not an
        // error: decoding from the start still reaches the target, just more slowly.
        let _ = ictx.seek(ts, ..ts);
    }
    let mut dec = Decoding::open(&ictx, size)?;
    let mut decoded = ff::frame::Video::empty();
    let mut best: Option<Vec<u8>> = None;

    // `Some(())` from the closure means "stop": the frame past the target has been seen,
    // so `best` is the answer.
    let consider = |dec: &mut Decoding, frame: &ff::frame::Video, best: &mut Option<Vec<u8>>| {
        match dec.frame_secs(frame) {
            // No timestamp to compare against (a raw stream with no pts). The first
            // frame after the seek is the best available answer.
            None => {
                *best = dec.to_rgb(frame);
                true
            }
            Some(t) if t <= target_secs + TS_EPSILON => {
                *best = dec.to_rgb(frame);
                false
            }
            // Past the target. Keep the earlier frame if there is one; otherwise the seek
            // overshot (or the target precedes the first frame) and this is the closest.
            Some(_) => {
                if best.is_none() {
                    *best = dec.to_rgb(frame);
                }
                true
            }
        }
    };

    let mut stopped = false;
    'outer: for (stream, packet) in ictx.packets() {
        if stream.index() != dec.stream_index {
            continue;
        }
        if dec.decoder.send_packet(&packet).is_err() {
            continue;
        }
        while dec.decoder.receive_frame(&mut decoded).is_ok() {
            let frame = std::mem::replace(&mut decoded, ff::frame::Video::empty());
            stopped = consider(&mut dec, &frame, &mut best);
            decoded = frame;
            if stopped {
                break 'outer;
            }
        }
    }
    // Drain unless a frame past the target already settled it. The decoder buffers frames
    // it has not been asked for yet, and for a target near the end of a clip *every*
    // candidate can still be in that buffer when the packets run out — so skipping this
    // would return an earlier frame, or none at all.
    if !stopped && dec.decoder.send_eof().is_ok() {
        while dec.decoder.receive_frame(&mut decoded).is_ok() {
            let frame = std::mem::replace(&mut decoded, ff::frame::Video::empty());
            let stop = consider(&mut dec, &frame, &mut best);
            decoded = frame;
            if stop {
                break;
            }
        }
    }
    let (w, h) = dec.out;
    best.map(|rgb| (rgb, w, h))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `linspace` is the one piece of sampling policy that is pure arithmetic, and it is
    /// where a pipeline silently starts sampling different frames than its reference
    /// preprocessing did. These cases are `numpy.linspace(0, total-1, n).astype(int)`.
    #[test]
    fn linspace_matches_numpy() {
        assert_eq!(linspace(10, 1), vec![0]);
        assert_eq!(linspace(1, 4), vec![0, 0, 0, 0]);
        assert_eq!(linspace(10, 2), vec![0, 9]);
        assert_eq!(linspace(100, 8), vec![0, 14, 28, 42, 56, 70, 84, 99]);
        assert_eq!(linspace(8, 8), vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    /// A clip shorter than the requested sample count repeats frames rather than
    /// producing fewer, because the output is a fixed-shape tensor: a ragged row would
    /// not be one.
    #[test]
    fn linspace_repeats_when_the_clip_is_shorter_than_the_sample() {
        let idx = linspace(3, 6);
        assert_eq!(idx.len(), 6);
        assert_eq!(*idx.first().unwrap(), 0);
        assert_eq!(*idx.last().unwrap(), 2);
        assert!(idx.windows(2).all(|w| w[0] <= w[1]), "{idx:?}");
    }

    /// Every index is in range, for any combination — this is what indexes into the
    /// output buffer, so an off-by-one here is an out-of-bounds write.
    #[test]
    fn linspace_never_leaves_the_clip() {
        for total in 1..40usize {
            for n in 1..20usize {
                for &i in &linspace(total, n) {
                    assert!(i < total, "total={total} n={n} produced {i}");
                }
            }
        }
    }

    /// A temp file must not outlive the clip it was written for, including when the
    /// decode gives up early — which is the normal case for a corrupt row.
    #[test]
    fn a_clip_file_is_removed_when_dropped() {
        let path = {
            let clip = ClipFile::write(b"not a video").unwrap();
            assert!(clip.path().exists());
            clip.path().to_path_buf()
        };
        assert!(!path.exists(), "temp file outlived its ClipFile");
    }
}
