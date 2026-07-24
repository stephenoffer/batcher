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
