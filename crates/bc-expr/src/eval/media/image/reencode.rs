//! Bytes-to-bytes image ops: `resize`, `crop`, `encode`, `convert`.
//!
//! Split from the dispatch module because these four share a shape the tensor ops do not:
//! decode, transform, **re-encode**, and hand back `Binary`. They are what a pipeline
//! reaches for when the column should stay a compact blob through a shuffle, a spill, or a
//! write — as against `to_tensor`/`to_grayscale`, which produce a fixed-size numeric
//! column for a model and live next to the dispatcher.

use std::io::Cursor;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BinaryArray, GenericBinaryArray, OffsetSizeTrait};

use super::{decode_rgb_resized, dim, map_rows, rec601, ImageArgs};
use crate::ExprError;

/// `resize(w, h)` → re-encoded PNG bytes at the new size. Null/undecodable → null.
pub(super) fn resize<O: OffsetSizeTrait>(
    bytes: &GenericBinaryArray<O>,
    width: Option<i64>,
    height: Option<i64>,
    out: Output,
) -> Result<ArrayRef, ExprError> {
    let w = dim("resize", "width", width)?;
    let h = dim("resize", "height", height)?;
    // Each dimension is a valid `u32`, but the resize allocates a `w * h * 3`-byte RGB
    // buffer per row; an absurd product (e.g. 50_000²) is a multi-gigabyte allocation bomb
    // driven by a query parameter. Cap it at `i32::MAX` — no legitimate thumbnail approaches
    // 2 GiB — computed in `u64` so the multiply itself cannot overflow.
    if (w as u64) * (h as u64) * 3 > i32::MAX as u64 {
        return Err(ExprError::InvalidArgument {
            func: "resize".to_string(),
            reason: format!(
                "resize to {w}x{h}x3 bytes exceeds the maximum of {} bytes",
                i32::MAX
            ),
        });
    }
    let rows: Vec<Option<Vec<u8>>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            None
        } else {
            resize_one(bytes.value(i), w, h, out)
        }
    });
    Ok(Arc::new(BinaryArray::from_iter(rows)))
}

/// Decode, resize to `(w, h)`, and re-encode; `None` on any failure.
fn resize_one(data: &[u8], w: u32, h: u32, out: Output) -> Option<Vec<u8>> {
    let raw = decode_rgb_resized(data, w, h)?;
    let img = image::RgbImage::from_raw(w, h, raw)?;
    out.write(image::DynamicImage::ImageRgb8(img))
}

/// The container formats this namespace can write. Mirrored by `_IMAGE_FORMATS` in
/// `plan/expr_ir/image.py`, which rejects a typo at plan-build time.
///
/// WebP is decodable but not listed: `image` 0.25 reads WebP and does not write it, so
/// offering it would fail at run time on a name the plan accepted.
pub(super) const ENCODE_FORMATS: [&str; 4] = ["png", "jpeg", "bmp", "gif"];

fn encode_format(name: &str) -> Option<image::ImageFormat> {
    Some(match name {
        "png" => image::ImageFormat::Png,
        "jpeg" => image::ImageFormat::Jpeg,
        "bmp" => image::ImageFormat::Bmp,
        "gif" => image::ImageFormat::Gif,
        _ => return None,
    })
}

/// How every bytes-out image op writes its result: which container, and at what quality.
///
/// This exists because the container used to be a property of *one* op. `encode(format)`
/// took a name; every other bytes-out op wrote PNG unconditionally. For a photographic
/// corpus that is the wrong default twice over — PNG is slower to write than JPEG and
/// several times larger — so `resize`, `thumbnail` and `auto_orient` each turned a
/// compressed input into a much bigger output, and the only way back was a second
/// `.image.encode("jpeg")` pass that decoded and re-encoded the whole column again.
/// Resolving the container once, here, is what lets an op emit the format the caller
/// actually wants in the single decode it was already doing.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Output {
    fmt: image::ImageFormat,
    /// 1..=100 for the lossy containers; `None` means the encoder's own default (75).
    quality: Option<u8>,
}

impl Output {
    /// Validate the format name and quality once for the whole batch.
    ///
    /// Once, not per row: an unknown name is a caller's typo, and reporting it n times
    /// tells them nothing the first one did not. `func` names the op in the error, since
    /// `format` is now reachable from every bytes-out op rather than only `encode`.
    pub(super) fn resolve(
        func: crate::ImageFunc,
        format: Option<&str>,
        quality: Option<i64>,
    ) -> Result<Self, ExprError> {
        let fmt = match format {
            None => image::ImageFormat::Png,
            Some(name) => encode_format(name).ok_or_else(|| ExprError::InvalidArgument {
                func: format!("{func:?}"),
                reason: format!(
                    "unknown image format {name:?}; expected one of {}",
                    ENCODE_FORMATS.join(", ")
                ),
            })?,
        };
        let quality = match quality {
            None => None,
            Some(q) if (1..=100).contains(&q) => Some(q as u8),
            Some(q) => {
                return Err(ExprError::InvalidArgument {
                    func: format!("{func:?}"),
                    reason: format!("quality must be in 1..=100, got {q}"),
                })
            }
        };
        Ok(Self { fmt, quality })
    }

    /// Whether this container can carry an alpha channel.
    ///
    /// JPEG cannot, so an RGBA image is flattened rather than failing the row — the same
    /// decision `encode` always made, lifted here so all eighteen bytes-out ops make it.
    fn keeps_alpha(&self) -> bool {
        !matches!(self.fmt, image::ImageFormat::Jpeg)
    }

    /// Encode one image, or `None` if the encoder refuses it (the null-row convention).
    pub(super) fn write(&self, img: image::DynamicImage) -> Option<Vec<u8>> {
        let img = if self.keeps_alpha() {
            img
        } else {
            image::DynamicImage::ImageRgb8(img.into_rgb8())
        };
        let mut out = Vec::new();
        match (self.fmt, self.quality) {
            (image::ImageFormat::Jpeg, Some(q)) => {
                let enc =
                    image::codecs::jpeg::JpegEncoder::new_with_quality(Cursor::new(&mut out), q);
                img.write_with_encoder(enc).ok()?;
            }
            _ => img.write_to(&mut Cursor::new(&mut out), self.fmt).ok()?,
        }
        Some(out)
    }
}

/// `encode(format)` → the same pixels re-encoded in `format`.
pub(super) fn encode<O: OffsetSizeTrait>(
    bytes: &GenericBinaryArray<O>,
    out: Output,
) -> Result<ArrayRef, ExprError> {
    let rows: Vec<Option<Vec<u8>>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            return None;
        }
        // `Output::write` flattens for a container with no alpha channel, so an RGBA
        // source re-encodes as JPEG rather than failing the row.
        out.write(image::load_from_memory(bytes.value(i)).ok()?)
    });
    Ok(assemble_binary(rows))
}

/// The most decoded pixel data one image may produce before this namespace declines it.
///
/// Not a number chosen here: it is `image::Limits::default().max_alloc`, the ceiling the
/// `load_from_memory` path already applies to every other operation. Naming it keeps the
/// one op that has to check for itself agreeing with the fifteen that get it for free.
const MAX_DECODED_BYTES: u64 = 512 * 1024 * 1024;

/// `thumbnail(max_size)` → the image scaled so its **longest side** is `max_size`, as PNG.
///
/// The difference from `resize` is the whole point: `resize` takes both dimensions and
/// therefore stretches anything that is not already the target aspect ratio. A corpus of
/// mixed-orientation photographs run through `resize(256, 256)` comes out squashed, which
/// is invisible in a shape assertion and obvious to anyone who looks at it.
///
/// Never *up*scales. Enlarging a thumbnail to reach `max_size` invents detail and costs
/// bytes, and `Image.thumbnail` in Pillow — the operation this is named for — has always
/// behaved this way, so a corpus already normalized against it stays byte-comparable.
pub(super) fn thumbnail<O: OffsetSizeTrait>(
    bytes: &GenericBinaryArray<O>,
    max_size: Option<i64>,
    out: Output,
) -> Result<ArrayRef, ExprError> {
    let max = dim("thumbnail", "max_size", max_size)?;
    let rows: Vec<Option<Vec<u8>>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            return None;
        }
        let img = image::load_from_memory(bytes.value(i)).ok()?;
        let (w, h) = (img.width(), img.height());
        if w == 0 || h == 0 {
            return None;
        }
        let (tw, th) = fit_within(w, h, max, max);
        // Already inside the box: re-encode without resampling rather than round-tripping
        // the pixels through a filter that can only lose information.
        let scaled = if tw == w && th == h {
            img
        } else {
            image::DynamicImage::ImageRgb8(resize_rgb(&img.into_rgb8(), tw, th)?)
        };
        out.write(scaled)
    });
    Ok(assemble_binary(rows))
}

/// `letterbox(width, height, fill)` → aspect-preserving fit onto a `(width, height)`
/// canvas, the remainder filled, flattened to RGB8.
///
/// The standard object-detection preprocessing, and the reason neither `to_tensor` nor
/// `center_crop` covers it. `to_tensor` stretches, which moves every bounding box a model
/// predicts off its object; `center_crop` discards the border, which is where the missed
/// detections live. Letterboxing does neither: the whole image survives at its true aspect
/// ratio, and the leftover canvas is a constant the model learns to ignore.
///
/// The default fill is 114, the YOLO family's grey, so a model trained against that
/// preprocessing sees the padding it expects.
pub(super) fn letterbox<O: OffsetSizeTrait>(
    bytes: &GenericBinaryArray<O>,
    args: ImageArgs<'_>,
) -> Result<ArrayRef, ExprError> {
    let w = dim("letterbox", "width", args.width)?;
    let h = dim("letterbox", "height", args.height)?;
    let per_row = super::element_len_guard("letterbox", (w as u64) * (h as u64) * 3, "bytes")?;
    let fill = match args.fill {
        None => 114u8,
        Some(v) => u8::try_from(v).map_err(|_| ExprError::InvalidArgument {
            func: "letterbox".to_string(),
            reason: format!("fill must be a byte value in 0..=255, got {v}"),
        })?,
    };

    let rows: Vec<Option<Vec<u8>>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            return None;
        }
        let img = image::load_from_memory(bytes.value(i)).ok()?.into_rgb8();
        let (sw, sh) = img.dimensions();
        if sw == 0 || sh == 0 {
            return None;
        }
        let (tw, th) = fit_within(sw, sh, w, h);
        let scaled = resize_rgb(&img, tw, th)?;
        let mut out = vec![fill; per_row];
        // Centre the scaled image on the canvas, so the padding is split evenly between
        // the two sides. An off-centre paste would bias every coordinate a model predicts.
        let x0 = ((w - tw) / 2) as usize;
        let y0 = ((h - th) / 2) as usize;
        let src = scaled.as_raw();
        let row_bytes = tw as usize * 3;
        for y in 0..th as usize {
            let s = y * row_bytes;
            let d = ((y0 + y) * w as usize + x0) * 3;
            out[d..d + row_bytes].copy_from_slice(&src[s..s + row_bytes]);
        }
        Some(out)
    });
    Ok(super::assemble_u8_tensor(bytes.len(), per_row, rows))
}

/// The largest `(w, h)` with the source's aspect ratio that fits inside `(max_w, max_h)`.
///
/// Rounded rather than truncated, and floored at one pixel: truncation drops a very wide
/// image's short side to zero, and a zero-dimension resize is an error rather than a
/// degenerate image, so a panorama would fail the row instead of thumbnailing.
pub(super) fn fit_within(sw: u32, sh: u32, max_w: u32, max_h: u32) -> (u32, u32) {
    if sw <= max_w && sh <= max_h {
        return (sw, sh);
    }
    let scale = f64::from(max_w) / f64::from(sw);
    let scale = scale.min(f64::from(max_h) / f64::from(sh));
    let w = ((f64::from(sw) * scale).round() as u32).max(1);
    let h = ((f64::from(sh) * scale).round() as u32).max(1);
    (w.min(max_w), h.min(max_h))
}

/// SIMD bilinear resize of an RGB8 image, matching `decode_rgb_resized`'s filter.
///
/// Shared so a thumbnail and a `to_tensor` of the same image downscale identically —
/// two resamplers in one namespace disagreeing is a difference a caller finds by accident.
fn resize_rgb(img: &image::RgbImage, w: u32, h: u32) -> Option<image::RgbImage> {
    use fast_image_resize as fir;

    let (sw, sh) = img.dimensions();
    let src =
        fir::images::Image::from_vec_u8(sw, sh, img.as_raw().clone(), fir::PixelType::U8x3).ok()?;
    let mut dst = fir::images::Image::new(w, h, fir::PixelType::U8x3);
    let opts = fir::ResizeOptions::new()
        .resize_alg(fir::ResizeAlg::Convolution(fir::FilterType::Bilinear));
    fir::Resizer::new().resize(&src, &mut dst, &opts).ok()?;
    image::RgbImage::from_raw(w, h, dst.into_vec())
}

/// `auto_orient()` → the image rotated/flipped per its Exif orientation, as PNG bytes.
///
/// A camera almost never rotates its sensor data. It records which way up the camera was
/// held, in the Exif `Orientation` tag, and leaves the pixels as the sensor read them —
/// so a portrait phone photo is stored landscape with a "rotate 90" note attached. Every
/// viewer, phone gallery, and browser honours that note, and so does `OpenCV::imread` and
/// anything built on `PIL.ImageOps.exif_transpose`.
///
/// The decoder underneath this namespace does not, and neither did anything above it. So
/// a corpus of phone photographs decoded here came out rotated a quarter turn from what
/// every other tool in the pipeline saw, silently: the tensor is the right shape, the
/// pixels are real, and only a human looking at a sample would notice. This is the
/// operation that fixes it, and it is a separate op rather than a changed default because
/// flipping a default would rotate the output of pipelines that are already compensating.
///
/// The result is PNG, which carries no Exif, so the orientation cannot be applied twice.
pub(super) fn auto_orient<O: OffsetSizeTrait>(
    bytes: &GenericBinaryArray<O>,
    out: Output,
) -> Result<ArrayRef, ExprError> {
    let rows: Vec<Option<Vec<u8>>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            return None;
        }
        oriented(bytes.value(i), out)
    });
    Ok(assemble_binary(rows))
}

fn oriented(data: &[u8], out: Output) -> Option<Vec<u8>> {
    let mut decoder = image::ImageReader::new(Cursor::new(data))
        .with_guessed_format()
        .ok()?
        .into_decoder()
        .ok()?;
    // Refuse an image too large to decode, *before* decoding it.
    //
    // Every other op here reaches the pixels through `image::load_from_memory`, which
    // applies a 512 MiB allocation ceiling and fails cleanly — so an oversized image comes
    // back as a null row, the namespace's convention for anything it cannot read. Decoding
    // through a raw `ImageDecoder` does not inherit that, and `set_limits` does not supply
    // it either: its default implementation checks the *dimension* limits, which are unset,
    // and leaves `max_alloc` to whichever decoder chooses to honour it.
    //
    // The consequence is not a slow path, it is an abort. A 20000x20000 PNG is about a
    // megabyte on disk and 1.2 GB decoded, and `map_rows` fans this across every core — so
    // one gigapixel scan in a corpus takes the worker down rather than nulling its row.
    // Checking `total_bytes` against the same ceiling is explicit, decoder-independent, and
    // puts this op back on the same convention as its fifteen neighbours.
    if image::ImageDecoder::total_bytes(&decoder) > MAX_DECODED_BYTES {
        return None;
    }
    // A format that cannot carry orientation reports `NoTransforms`, so this is a no-op
    // re-encode for a PNG rather than a failure.
    let orientation = image::ImageDecoder::orientation(&mut decoder).ok()?;
    let mut img = image::DynamicImage::from_decoder(decoder).ok()?;
    img.apply_orientation(orientation);
    out.write(img)
}

/// `exif_orientation()` → the Exif orientation code, 1 through 8.
///
/// The diagnostic half of [`auto_orient`]: it says whether a corpus needs orienting at
/// all, which is otherwise invisible. `1` (no transform) is reported for an image that
/// carries no orientation and for a format that cannot carry one, because that is what
/// the tag means — not "absent", but "already upright".
pub(super) fn exif_orientation<O: OffsetSizeTrait>(
    bytes: &GenericBinaryArray<O>,
) -> Result<ArrayRef, ExprError> {
    use arrow::array::Int32Array;

    let codes: Vec<Option<i32>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            return None;
        }
        orientation_code(bytes.value(i))
    });
    Ok(Arc::new(Int32Array::from(codes)))
}

fn orientation_code(data: &[u8]) -> Option<i32> {
    let mut decoder = image::ImageReader::new(Cursor::new(data))
        .with_guessed_format()
        .ok()?
        .into_decoder()
        .ok()?;
    image::ImageDecoder::orientation(&mut decoder)
        .ok()
        .map(|o| i32::from(o.to_exif()))
}

/// The color modes `convert` can produce. Mirrors `_IMAGE_MODES` in
/// `plan/expr_ir/image.py` and the names `decode` reports, so a caller can read a mode off
/// `decode` and hand it straight back to `convert`.
pub(super) const COLOR_MODES: [&str; 4] = ["L", "LA", "RGB", "RGBA"];

/// `convert(mode)` → the image in `mode`, re-encoded as PNG.
pub(super) fn convert<O: OffsetSizeTrait>(
    bytes: &GenericBinaryArray<O>,
    mode: Option<&str>,
    out: Output,
) -> Result<ArrayRef, ExprError> {
    let mode = mode.ok_or(ExprError::MissingImageArg {
        func: "convert".to_string(),
        arg: "mode",
    })?;
    if !COLOR_MODES.contains(&mode) {
        return Err(ExprError::InvalidArgument {
            func: "convert".to_string(),
            reason: format!(
                "unknown color mode {mode:?}; expected one of {}",
                COLOR_MODES.join(", ")
            ),
        });
    }
    let rows: Vec<Option<Vec<u8>>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            return None;
        }
        let img = image::load_from_memory(bytes.value(i)).ok()?;
        // The luma channels are computed here rather than by `into_luma8`, which weights
        // Rec.709. `to_grayscale` and `dhash` both use Rec.601, and two functions in one
        // namespace disagreeing about what grey means is a bug a caller would find by
        // accident. On RGB(10, 200, 30) the two answers are 147 and 124.
        let converted = match mode {
            "L" => {
                let rgb = img.into_rgb8();
                let (w, h) = rgb.dimensions();
                let gray: Vec<u8> = rgb.pixels().map(|p| rec601(p.0)).collect();
                image::DynamicImage::ImageLuma8(image::GrayImage::from_raw(w, h, gray)?)
            }
            "LA" => {
                let rgba = img.into_rgba8();
                let (w, h) = rgba.dimensions();
                let la: Vec<u8> = rgba
                    .pixels()
                    .flat_map(|p| [rec601([p.0[0], p.0[1], p.0[2]]), p.0[3]])
                    .collect();
                image::DynamicImage::ImageLumaA8(image::GrayAlphaImage::from_raw(w, h, la)?)
            }
            "RGB" => image::DynamicImage::ImageRgb8(img.into_rgb8()),
            _ => image::DynamicImage::ImageRgba8(img.into_rgba8()),
        };
        out.write(converted)
    });
    Ok(assemble_binary(rows))
}

/// Collect per-row byte buffers into a `Binary` column (`None` → null).
pub(super) fn assemble_binary(rows: Vec<Option<Vec<u8>>>) -> ArrayRef {
    use arrow::array::BinaryBuilder;
    let mut b = BinaryBuilder::with_capacity(rows.len(), rows.len() * 512);
    for row in rows {
        match row {
            Some(v) => b.append_value(v),
            None => b.append_null(),
        }
    }
    Arc::new(b.finish())
}

/// `crop(x, y, width, height)` where each bound is a **column**, not a constant.
///
/// The operation a detection pipeline is built around: cut the box a model predicted out
/// of the frame it was predicted in. With literal-only bounds that was the one thing this
/// namespace could not express, so a pipeline had to leave the engine and loop in Python
/// to do it.
///
/// The four bounds arrive already evaluated, as full-length arrays. That is deliberately
/// uniform: a literal evaluates to a broadcast array like anything else, so there is no
/// "is it constant" branch to get subtly wrong, and the per-batch cost of the broadcast is
/// nothing next to a decode.
///
/// Per-row semantics match what the literal form always did, because they are the same
/// decisions: a window clipped by an edge yields the part that exists rather than padding
/// (an image someone will look at, and inventing black pixels invents data), and a window
/// starting past the image entirely yields null. A null or non-positive bound also yields
/// null for that row — a box the caller could not supply is a row with no answer, not a
/// reason to fail the batch, which matters when the boxes came from a model that declined
/// to predict on some frames.
pub(super) fn crop_dynamic<O: OffsetSizeTrait>(
    bytes: &GenericBinaryArray<O>,
    bounds: &Bounds<'_>,
) -> Result<ArrayRef, ExprError> {
    let rows: Vec<Option<Vec<u8>>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            return None;
        }
        let (x, y, w, h) = bounds.at(i)?;
        let img = image::load_from_memory(bytes.value(i)).ok()?;
        let (sw, sh) = (img.width(), img.height());
        if x >= sw || y >= sh {
            return None; // the window starts past the image entirely
        }
        let cw = w.min(sw - x);
        let ch = h.min(sh - y);
        let cropped = image::imageops::crop_imm(&img, x, y, cw, ch).to_image();
        let mut out = Vec::new();
        image::DynamicImage::ImageRgba8(cropped)
            .write_to(&mut Cursor::new(&mut out), image::ImageFormat::Png)
            .ok()?;
        Some(out)
    });
    Ok(assemble_binary(rows))
}

/// The four per-row crop bounds, each read from its own evaluated column.
pub(crate) struct Bounds<'a> {
    x: &'a arrow::array::Int64Array,
    y: &'a arrow::array::Int64Array,
    width: &'a arrow::array::Int64Array,
    height: &'a arrow::array::Int64Array,
}

impl<'a> Bounds<'a> {
    /// Read the four bound columns, requiring each to be Int64.
    ///
    /// Int64 rather than "any integer" because the FFI boundary already widens every
    /// narrower integer to it, so a bound column reaching here as anything else is a
    /// genuine type error rather than a width this should paper over.
    pub(crate) fn new(columns: [&'a ArrayRef; 4]) -> Result<Self, ExprError> {
        let mut typed = [None; 4];
        for (slot, arr) in typed.iter_mut().zip(columns) {
            *slot = Some(
                arr.as_any()
                    .downcast_ref::<arrow::array::Int64Array>()
                    .ok_or_else(|| ExprError::InvalidArgument {
                        func: "crop".to_string(),
                        reason: format!(
                            "crop bounds must be integers, got {}; cast the column first",
                            arr.data_type()
                        ),
                    })?,
            );
        }
        Ok(Self {
            x: typed[0].expect("set above"),
            y: typed[1].expect("set above"),
            width: typed[2].expect("set above"),
            height: typed[3].expect("set above"),
        })
    }

    /// Row `i`'s window as `(x, y, width, height)`, or `None` when it is unusable.
    ///
    /// A null bound, a negative offset, or a non-positive extent all yield `None` — the
    /// row simply has no window. Narrowing through `u32::try_from` is what makes a
    /// negative value a null rather than a ~4-billion offset that silently matches nothing.
    fn at(&self, i: usize) -> Option<(u32, u32, u32, u32)> {
        use arrow::array::Array;

        if self.x.is_null(i) || self.y.is_null(i) || self.width.is_null(i) || self.height.is_null(i)
        {
            return None;
        }
        let x = u32::try_from(self.x.value(i)).ok()?;
        let y = u32::try_from(self.y.value(i)).ok()?;
        let w = u32::try_from(self.width.value(i)).ok().filter(|&v| v > 0)?;
        let h = u32::try_from(self.height.value(i))
            .ok()
            .filter(|&v| v > 0)?;
        // A single row's window is still an allocation driven by the data, so bound it the
        // way the literal form bounds its constant.
        ((w as u64) * (h as u64) * 3 <= i32::MAX as u64).then_some((x, y, w, h))
    }
}
