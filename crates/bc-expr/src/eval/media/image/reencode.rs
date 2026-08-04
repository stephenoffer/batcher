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
    let out: Vec<Option<Vec<u8>>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            None
        } else {
            resize_png(bytes.value(i), w, h)
        }
    });
    Ok(Arc::new(BinaryArray::from_iter(out)))
}

/// Decode, resize to `(w, h)`, and re-encode as PNG; `None` on any failure.
fn resize_png(data: &[u8], w: u32, h: u32) -> Option<Vec<u8>> {
    let raw = decode_rgb_resized(data, w, h)?;
    let img = image::RgbImage::from_raw(w, h, raw)?;
    let mut buf = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(img)
        .write_to(&mut buf, image::ImageFormat::Png)
        .ok()?;
    Some(buf.into_inner())
}

/// The container formats `encode` can write. Mirrored by `_IMAGE_FORMATS` in
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

/// `encode(format)` → the same pixels re-encoded in `format`.
pub(super) fn encode<O: OffsetSizeTrait>(
    bytes: &GenericBinaryArray<O>,
    format: Option<&str>,
) -> Result<ArrayRef, ExprError> {
    let name = format.ok_or(ExprError::MissingImageArg {
        func: "encode".to_string(),
        arg: "format",
    })?;
    // Validate once, before any row: an unknown format is a plan error, and raising it
    // per row would emit the same message n times.
    let fmt = encode_format(name).ok_or_else(|| ExprError::InvalidArgument {
        func: "encode".to_string(),
        reason: format!(
            "unknown image format {name:?}; expected one of {}",
            ENCODE_FORMATS.join(", ")
        ),
    })?;
    let rows: Vec<Option<Vec<u8>>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            return None;
        }
        let img = image::load_from_memory(bytes.value(i)).ok()?;
        // JPEG has no alpha channel, so an RGBA source is flattened to RGB rather than
        // failing the row. Every other target keeps whatever the decoder produced.
        let img = if matches!(fmt, image::ImageFormat::Jpeg) {
            image::DynamicImage::ImageRgb8(img.into_rgb8())
        } else {
            img
        };
        let mut out = Vec::new();
        img.write_to(&mut Cursor::new(&mut out), fmt).ok()?;
        Some(out)
    });
    Ok(assemble_binary(rows))
}

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
        let mut buf = Vec::new();
        scaled
            .write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Png)
            .ok()?;
        Some(buf)
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
fn fit_within(sw: u32, sh: u32, max_w: u32, max_h: u32) -> (u32, u32) {
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
) -> Result<ArrayRef, ExprError> {
    let rows: Vec<Option<Vec<u8>>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            return None;
        }
        oriented_png(bytes.value(i))
    });
    Ok(assemble_binary(rows))
}

fn oriented_png(data: &[u8]) -> Option<Vec<u8>> {
    let mut decoder = image::ImageReader::new(Cursor::new(data))
        .with_guessed_format()
        .ok()?
        .into_decoder()
        .ok()?;
    // A format that cannot carry orientation reports `NoTransforms`, so this is a no-op
    // re-encode for a PNG rather than a failure.
    let orientation = image::ImageDecoder::orientation(&mut decoder).ok()?;
    let mut img = image::DynamicImage::from_decoder(decoder).ok()?;
    img.apply_orientation(orientation);
    let mut buf = Vec::new();
    img.write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Png)
        .ok()?;
    Some(buf)
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
        let out = match mode {
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
        let mut buf = Vec::new();
        out.write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Png)
            .ok()?;
        Some(buf)
    });
    Ok(assemble_binary(rows))
}

/// `crop(x, y, width, height)` → the requested region, re-encoded as PNG bytes.
///
/// The arbitrary-offset counterpart of `center_crop`, and the shape a detection pipeline
/// needs: pull a bounding box out of a frame and keep it as an *image*, not as a tensor.
///
/// A window that runs past an edge is **clipped** to the image rather than zero-padded,
/// which is the opposite of `center_crop`'s choice and deliberate. `center_crop` feeds a
/// model that needs a fixed input size, so padding preserves the shape contract; `crop`
/// produces an image a human or another tool will look at, and inventing black pixels
/// there would be inventing data. A window entirely outside the image yields null.
pub(super) fn crop<O: OffsetSizeTrait>(
    bytes: &GenericBinaryArray<O>,
    args: ImageArgs<'_>,
) -> Result<ArrayRef, ExprError> {
    let w = dim("crop", "width", args.width)?;
    let h = dim("crop", "height", args.height)?;
    // `x`/`y` may be zero (unlike a dimension), so they are read directly rather than
    // through `dim`, which rejects zero.
    let x = offset("crop", "x", args.x)?;
    let y = offset("crop", "y", args.y)?;
    if (w as u64) * (h as u64) * 3 > i32::MAX as u64 {
        return Err(ExprError::InvalidArgument {
            func: "crop".to_string(),
            reason: format!(
                "crop of {w}x{h}x3 bytes exceeds the maximum of {}",
                i32::MAX
            ),
        });
    }
    let rows: Vec<Option<Vec<u8>>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            return None;
        }
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

/// A non-negative pixel offset narrowed to `u32`.
///
/// Separate from [`dim`] because zero is a valid offset and an invalid dimension, and
/// because a negative offset must be rejected rather than wrapped to ~4 billion.
fn offset(func: &str, arg: &'static str, value: Option<i64>) -> Result<u32, ExprError> {
    let value = value.ok_or(ExprError::MissingImageArg {
        func: func.to_string(),
        arg,
    })?;
    u32::try_from(value).map_err(|_| ExprError::InvalidImageDim {
        func: func.to_string(),
        arg,
        value,
        max: u32::MAX,
    })
}

/// Collect per-row byte buffers into a `Binary` column (`None` → null).
fn assemble_binary(rows: Vec<Option<Vec<u8>>>) -> ArrayRef {
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
