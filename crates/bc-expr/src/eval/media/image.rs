//! Image-decode evaluation for `Expr::Image` (the `.image` namespace).
//!
//! This is the interpreter *oracle* for image decoding. The JIT cannot compile
//! library-backed decode, so `bc-codegen` marks `Expr::Image` unsupported and
//! falls back here — the two never diverge because there is only this one
//! implementation. Decode runs per row over the whole batch; a row whose bytes
//! are null or fail to decode yields a null result (corrupt inputs don't fail
//! the batch), matching the multimodal source's header-metadata convention.

use std::io::Cursor;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BinaryArray, FixedSizeListArray, Int32Array, StructArray, UInt8Array,
};
use arrow::buffer::NullBuffer;
use arrow::datatypes::{DataType, Field};

use super::map_rows;
use crate::{ExprError, ImageFunc};

/// Evaluate an image function over a Binary array of encoded image bytes.
pub(crate) fn eval_image(
    func: ImageFunc,
    arr: &ArrayRef,
    width: Option<i64>,
    height: Option<i64>,
) -> Result<ArrayRef, ExprError> {
    let bytes =
        arr.as_any()
            .downcast_ref::<BinaryArray>()
            .ok_or_else(|| ExprError::ExpectedBinary {
                func: format!("{func:?}"),
                got: arr.data_type().to_string(),
            })?;
    match func {
        ImageFunc::Decode => decode_dims(bytes),
        ImageFunc::ToTensor => to_tensor(bytes, width, height),
        ImageFunc::Resize => resize(bytes, width, height),
    }
}

/// Validate and narrow a target dimension to `u32`.
///
/// A plain `as u32` cast silently wraps an out-of-range `i64`: a negative value becomes a
/// ~4-billion dimension (an unbounded allocation that aborts the process), and a value past
/// `u32::MAX` wraps to a small one (a silently wrong output size). Both are rejected with a
/// clear error instead — the dimension is a query parameter, so a bad one is a caller bug to
/// surface, not a crash to suffer or a wrong answer to return.
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

/// `resize(w, h)` → re-encoded PNG bytes at the new size. Null/undecodable → null.
fn resize(
    bytes: &BinaryArray,
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

/// `decode` → struct `{width: Int32, height: Int32}` (header read only).
fn decode_dims(bytes: &BinaryArray) -> Result<ArrayRef, ExprError> {
    // Read each header in parallel (an undecodable/null row → `None`), then unzip into
    // the three column buffers. The header read is the cost; the unzip is a cheap memcpy.
    let dims: Vec<Option<(i32, i32)>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            None
        } else {
            image_dimensions(bytes.value(i)).map(|(w, h)| (w as i32, h as i32))
        }
    });
    let mut widths: Vec<i32> = Vec::with_capacity(bytes.len());
    let mut heights: Vec<i32> = Vec::with_capacity(bytes.len());
    let mut valid: Vec<bool> = Vec::with_capacity(bytes.len());
    for dim in dims {
        match dim {
            Some((w, h)) => {
                widths.push(w);
                heights.push(h);
                valid.push(true);
            }
            None => {
                widths.push(0);
                heights.push(0);
                valid.push(false);
            }
        }
    }
    let nulls = NullBuffer::from(valid);
    let fields = vec![
        Arc::new(Field::new("width", DataType::Int32, false)),
        Arc::new(Field::new("height", DataType::Int32, false)),
    ];
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from(widths)),
        Arc::new(Int32Array::from(heights)),
    ];
    let struct_arr = StructArray::new(fields.into(), columns, Some(nulls));
    Ok(Arc::new(struct_arr))
}

/// `to_tensor(w, h)` → `FixedSizeList<UInt8>` of length `w*h*3` (RGB8, resized).
fn to_tensor(
    bytes: &BinaryArray,
    width: Option<i64>,
    height: Option<i64>,
) -> Result<ArrayRef, ExprError> {
    let w = dim("to_tensor", "width", width)?;
    let h = dim("to_tensor", "height", height)?;
    // Each dimension is individually a valid `u32` (guarded by `dim`), but the *product*
    // `w * h * 3` is the per-row byte count — and it is also the `FixedSizeList` element
    // length, an Arrow `i32`. At ~26_755² it already exceeds `i32::MAX`: the `as i32` cast
    // below would wrap negative (an invalid array / panic), and `bytes.len() * per_row`
    // would pre-allocate a multi-gigabyte zeroed buffer (an OOM bomb driven by a query
    // parameter). Reject the request cleanly instead — computed in `u64` so the multiply
    // itself cannot overflow.
    let per_row = (w as u64) * (h as u64) * 3;
    if per_row > i32::MAX as u64 {
        return Err(ExprError::InvalidArgument {
            func: "to_tensor".to_string(),
            reason: format!(
                "tensor of {w}x{h}x3 = {per_row} bytes per row exceeds the maximum \
                 element length of {} bytes",
                i32::MAX
            ),
        });
    }
    let per_row = per_row as usize;
    // Decode + resize every row in parallel (this is the training-data hot path:
    // `read.images(decode=True)` lowers to exactly this kernel). Each row yields its
    // own `per_row`-byte RGB8 buffer, or `None` on null/undecodable/wrong-size input.
    let rows: Vec<Option<Vec<u8>>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            return None;
        }
        decode_rgb_resized(bytes.value(i), w, h).filter(|buf| buf.len() == per_row)
    });
    // Assemble the contiguous FixedSizeList child buffer serially — a straight memcpy
    // per row (undecodable rows leave their slot zeroed), cheap next to the decode.
    let mut values: Vec<u8> = vec![0u8; bytes.len() * per_row];
    let mut valid: Vec<bool> = Vec::with_capacity(bytes.len());
    for (i, row) in rows.into_iter().enumerate() {
        match row {
            Some(buf) => {
                values[i * per_row..(i + 1) * per_row].copy_from_slice(&buf);
                valid.push(true);
            }
            None => valid.push(false),
        }
    }
    let field = Arc::new(Field::new("item", DataType::UInt8, false));
    let arr = FixedSizeListArray::new(
        field,
        per_row as i32,
        Arc::new(UInt8Array::from(values)),
        Some(NullBuffer::from(valid)),
    );
    Ok(Arc::new(arr))
}

/// Read just the image header to get `(width, height)`; `None` on any failure.
fn image_dimensions(data: &[u8]) -> Option<(u32, u32)> {
    image::ImageReader::new(Cursor::new(data))
        .with_guessed_format()
        .ok()?
        .into_dimensions()
        .ok()
}

/// Decode, resize to `(w, h)`, and flatten to RGB8; `None` on any failure.
///
/// Two SIMD-accelerated stages, each the fast option for its job:
///   1. **Decode.** For a JPEG whose source is ≥2× the target, decode at a 1/2·1/4·1/8
///      **DCT scale** ([`decode_jpeg_scaled`]) — the physical-AI case (large camera frame
///      → small model input), where full-res decode wastes most of the pixels and memory
///      bandwidth. Everything else (small JPEGs, PNG/WebP/…) decodes via zune-jpeg / the
///      `image` crate, already SIMD.
///   2. **Resize** to the exact `(w, h)` with `fast_image_resize` (SIMD bilinear — the same
///      tent/linear filter the previous scalar `resize_exact(Triangle)` used).
///
/// Pixels are not bit-identical across decoder/resizer implementations; the multimodal
/// benchmark accounts for this by comparing counts/dims, not raw pixels.
fn decode_rgb_resized(data: &[u8], w: u32, h: u32) -> Option<Vec<u8>> {
    use fast_image_resize as fir;

    // Stage 1: decode to an RGB8 buffer at some `(sw, sh)` ≥ the target.
    let (rgb, sw, sh) = match decode_jpeg_scaled(data, w, h) {
        Some(scaled) => scaled,
        None => {
            let img = image::load_from_memory(data).ok()?;
            let (sw, sh) = (img.width(), img.height());
            (img.into_rgb8().into_raw(), sw, sh)
        }
    };
    if sw == 0 || sh == 0 {
        return None;
    }
    // Stage 2: exact resize (skip when the scaled decode already hit the target size).
    if sw == w && sh == h {
        return Some(rgb);
    }
    let src = fir::images::Image::from_vec_u8(sw, sh, rgb, fir::PixelType::U8x3).ok()?;
    let mut dst = fir::images::Image::new(w, h, fir::PixelType::U8x3);
    let opts = fir::ResizeOptions::new()
        .resize_alg(fir::ResizeAlg::Convolution(fir::FilterType::Bilinear));
    fir::Resizer::new().resize(&src, &mut dst, &opts).ok()?;
    Some(dst.into_vec())
}

/// DCT-scaled JPEG decode: decode at the smallest 1/1·1/2·1/4·1/8 scale whose output is
/// still ≥ `(w, h)`, returning `(rgb8, out_w, out_h)`. `None` (→ caller uses the full
/// decoder) unless the source is a baseline RGB JPEG at least 2× the target in both dims —
/// below that, `jpeg-decoder`'s slower per-pixel decode would lose to zune-jpeg's full one.
fn decode_jpeg_scaled(data: &[u8], w: u32, h: u32) -> Option<(Vec<u8>, u32, u32)> {
    use jpeg_decoder::{Decoder, PixelFormat};

    let mut dec = Decoder::new(Cursor::new(data));
    dec.read_info().ok()?;
    let info = dec.info()?;
    // Only a win when at least one DCT octave can be dropped, and only for RGB (the common
    // 3-channel photo case); grayscale/CMYK fall back so the pixel semantics stay simple.
    if info.pixel_format != PixelFormat::RGB24
        || u32::from(info.width) < w.saturating_mul(2)
        || u32::from(info.height) < h.saturating_mul(2)
    {
        return None;
    }
    // Request the target size; the decoder snaps up to the nearest DCT scale that still
    // covers it, so the subsequent resize only ever downsamples.
    let (out_w, out_h) = dec.scale(w as u16, h as u16).ok()?;
    let pixels = dec.decode().ok()?;
    let (out_w, out_h) = (u32::from(out_w), u32::from(out_h));
    if pixels.len() != (out_w as usize) * (out_h as usize) * 3 {
        return None; // unexpected layout — let the full decoder handle it
    }
    Some((pixels, out_w, out_h))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2×3 red PNG, encoded once so the test has no I/O.
    fn red_png(width: u32, height: u32) -> Vec<u8> {
        let buf = image::RgbImage::from_pixel(width, height, image::Rgb([255, 0, 0]));
        let mut out = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(buf)
            .write_to(&mut out, image::ImageFormat::Png)
            .unwrap();
        out.into_inner()
    }

    /// Manual throughput check (ignored). Run:
    ///   RAYON_NUM_THREADS=1 cargo test -p bc-expr to_tensor_bench -- --ignored --nocapture
    ///   cargo test -p bc-expr to_tensor_bench -- --ignored --nocapture
    #[test]
    #[ignore]
    fn to_tensor_bench() {
        use std::time::Instant;
        let n = 4000usize;
        let blobs: Vec<Vec<u8>> = (0..n)
            .map(|i| {
                // A noisy image so JPEG decode does real work (not a trivial solid fill).
                let mut img = image::RgbImage::new(256, 256);
                for (x, y, p) in img.enumerate_pixels_mut() {
                    let v = ((x * 7 + y * 13 + (i as u32) * 31) % 256) as u8;
                    *p = image::Rgb([v, v.wrapping_add(80), v.wrapping_add(160)]);
                }
                let mut out = Cursor::new(Vec::new());
                image::DynamicImage::ImageRgb8(img)
                    .write_to(&mut out, image::ImageFormat::Jpeg)
                    .unwrap();
                out.into_inner()
            })
            .collect();
        let arr: ArrayRef = Arc::new(BinaryArray::from_iter(
            blobs.iter().map(|b| Some(b.as_slice())),
        ));
        let _ = eval_image(ImageFunc::ToTensor, &arr, Some(224), Some(224)).unwrap(); // warm
        let t = Instant::now();
        let out = eval_image(ImageFunc::ToTensor, &arr, Some(224), Some(224)).unwrap();
        let dt = t.elapsed().as_secs_f64();
        assert_eq!(out.len(), n);
        println!(
            "rayon_threads={}  to_tensor {n} imgs: {:.1} ms ({:.0} img/s)",
            rayon::current_num_threads(),
            dt * 1000.0,
            n as f64 / dt
        );
    }

    #[test]
    fn decode_reads_dimensions() {
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![
            Some(red_png(2, 3).as_slice()),
            None,
            Some(b"not an image".as_slice()),
        ]));
        let out = eval_image(ImageFunc::Decode, &arr, None, None).unwrap();
        let s = out.as_any().downcast_ref::<StructArray>().unwrap();
        let w = s.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let h = s.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        assert!(s.is_valid(0) && w.value(0) == 2 && h.value(0) == 3);
        assert!(s.is_null(1)); // null bytes → null struct
        assert!(s.is_null(2)); // undecodable bytes → null struct
    }

    #[test]
    fn to_tensor_decodes_and_resizes() {
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![
            Some(red_png(8, 8).as_slice()),
            None,
        ]));
        let out = eval_image(ImageFunc::ToTensor, &arr, Some(4), Some(4)).unwrap();
        let fsl = out.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
        assert_eq!(fsl.value_length(), 4 * 4 * 3);
        assert!(fsl.is_valid(0));
        assert!(fsl.is_null(1));
        let row0 = fsl.value(0);
        let px = row0.as_any().downcast_ref::<UInt8Array>().unwrap();
        // Resized solid-red image stays red: first pixel ~ (255, 0, 0).
        assert_eq!(px.value(0), 255);
        assert_eq!(px.value(1), 0);
        assert_eq!(px.value(2), 0);
    }

    #[test]
    fn to_tensor_parallel_batch_preserves_order_and_nulls() {
        // A batch above `PAR_ROW_THRESHOLD` so the parallel row map runs. Every third
        // row is null and every fifth is corrupt; the rest are solid-color images whose
        // color encodes their index, so we can prove order is preserved across the fan-out.
        let n = 40usize;
        let mut rows: Vec<Option<Vec<u8>>> = Vec::with_capacity(n);
        for i in 0..n {
            if i % 3 == 0 {
                rows.push(None);
            } else if i % 5 == 0 {
                rows.push(Some(b"not an image".to_vec()));
            } else {
                let c = (i % 256) as u8;
                let img = image::RgbImage::from_pixel(6, 6, image::Rgb([c, 0, 0]));
                let mut out = Cursor::new(Vec::new());
                image::DynamicImage::ImageRgb8(img)
                    .write_to(&mut out, image::ImageFormat::Png)
                    .unwrap();
                rows.push(Some(out.into_inner()));
            }
        }
        let arr: ArrayRef = Arc::new(BinaryArray::from_iter(rows));
        let out = eval_image(ImageFunc::ToTensor, &arr, Some(4), Some(4)).unwrap();
        let fsl = out.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
        assert_eq!(fsl.len(), n);
        for i in 0..n {
            if i % 3 == 0 || i % 5 == 0 {
                assert!(fsl.is_null(i), "row {i} should be null");
            } else {
                assert!(fsl.is_valid(i), "row {i} should be valid");
                let row = fsl.value(i);
                let px = row.as_any().downcast_ref::<UInt8Array>().unwrap();
                // The solid color survives resize, proving this slot got row `i`'s pixels.
                assert_eq!(px.value(0), (i % 256) as u8, "row {i} decoded out of order");
                assert_eq!(px.value(1), 0);
            }
        }
    }

    #[test]
    fn scaled_decode_downscales_large_jpeg_correctly() {
        // A large solid-color JPEG (source 512×512 ≥ 2× target 128 → the DCT-scaled path
        // runs). The color must survive decode+resize, proving the scaled path is wired
        // correctly and produces the right shape.
        let src = image::RgbImage::from_pixel(512, 512, image::Rgb([40, 160, 200]));
        let mut jpg = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(src)
            .write_to(&mut jpg, image::ImageFormat::Jpeg)
            .unwrap();
        let jpg = jpg.into_inner();

        // The scaled path is taken and returns the right shape.
        let scaled = decode_jpeg_scaled(&jpg, 128, 128);
        assert!(scaled.is_some(), "512→128 JPEG should use the scaled path");
        let (_, sw, sh) = scaled.unwrap();
        assert!(
            sw >= 128 && sh >= 128,
            "scaled output must still cover the target"
        );

        // End to end: the tensor is 128×128×3 and stays the source color (± JPEG noise).
        let arr: ArrayRef = Arc::new(BinaryArray::from_iter(vec![Some(jpg.as_slice())]));
        let out = eval_image(ImageFunc::ToTensor, &arr, Some(128), Some(128)).unwrap();
        let fsl = out.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
        assert_eq!(fsl.value_length(), 128 * 128 * 3);
        assert!(fsl.is_valid(0));
        let row = fsl.value(0);
        let px = row.as_any().downcast_ref::<UInt8Array>().unwrap();
        assert!(
            (px.value(0) as i32 - 40).abs() < 20,
            "R ~40, got {}",
            px.value(0)
        );
        assert!(
            (px.value(1) as i32 - 160).abs() < 20,
            "G ~160, got {}",
            px.value(1)
        );
        assert!(
            (px.value(2) as i32 - 200).abs() < 20,
            "B ~200, got {}",
            px.value(2)
        );

        // A small JPEG (≤2× target) must NOT take the scaled path — full decode is faster.
        let small = image::RgbImage::from_pixel(130, 130, image::Rgb([10, 20, 30]));
        let mut sbuf = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(small)
            .write_to(&mut sbuf, image::ImageFormat::Jpeg)
            .unwrap();
        assert!(
            decode_jpeg_scaled(&sbuf.into_inner(), 128, 128).is_none(),
            "130→128 is not a ≥2× downscale; the scaled path should decline it"
        );
    }

    #[test]
    fn out_of_range_dimensions_are_rejected_not_wrapped() {
        // A dimension is `i64` in the IR. A plain `as u32` cast silently wraps:
        //   * a value past u32::MAX → a small one, silently producing a wrong-size tensor;
        //   * a negative value → ~4 billion, an unbounded allocation that aborts the process.
        // Both must be rejected with a clear error rather than wrapped.
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![Some(red_png(4, 4).as_slice())]));
        for &bad in &[-1_i64, 0, i64::from(u32::MAX) + 1, i64::MAX] {
            assert!(
                eval_image(ImageFunc::ToTensor, &arr, Some(bad), Some(4)).is_err(),
                "to_tensor width {bad} should be rejected"
            );
            assert!(
                eval_image(ImageFunc::ToTensor, &arr, Some(4), Some(bad)).is_err(),
                "to_tensor height {bad} should be rejected"
            );
            assert!(
                eval_image(ImageFunc::Resize, &arr, Some(bad), Some(4)).is_err(),
                "resize width {bad} should be rejected"
            );
        }
        // The error names the arg but never the (potentially huge) allocation it prevented.
        let err = eval_image(ImageFunc::ToTensor, &arr, Some(-1), Some(4)).unwrap_err();
        assert!(err.to_string().contains("width"), "{err}");
    }

    #[test]
    fn to_tensor_rejects_dimensions_whose_product_overflows() {
        // Each dimension is a valid `u32`, but `w * h * 3` exceeds `i32::MAX` (the
        // FixedSizeList element length). The old `(w as usize) * (h as usize) * 3` then
        // `per_row as i32` wrapped negative and pre-allocated a multi-GB buffer; it must
        // now be a clean error, returned *before* any allocation.
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![Some(red_png(4, 4).as_slice())]));
        // 40_000 * 40_000 * 3 = 4.8e12 > i32::MAX, but each dim is well within u32.
        let err = eval_image(ImageFunc::ToTensor, &arr, Some(40_000), Some(40_000)).unwrap_err();
        assert!(err.to_string().contains("exceeds"), "{err}");
        // Just over the boundary: 26_756 * 26_756 * 3 = 2_147_449_008 > i32::MAX (2_147_483_647)?
        // 26_756^2 * 3 = 2_147_608_... let's use a clearly-over value.
        assert!(eval_image(ImageFunc::ToTensor, &arr, Some(30_000), Some(30_000)).is_err());
        // A legitimate, bounded request still succeeds.
        assert!(eval_image(ImageFunc::ToTensor, &arr, Some(64), Some(64)).is_ok());
        // `resize` guards the same product (its per-row RGB buffer is `w * h * 3`).
        assert!(eval_image(ImageFunc::Resize, &arr, Some(40_000), Some(40_000)).is_err());
        assert!(eval_image(ImageFunc::Resize, &arr, Some(8), Some(8)).is_ok());
    }

    #[test]
    fn resize_reencodes_at_new_size() {
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![
            Some(red_png(8, 8).as_slice()),
            None,
            Some(b"not an image".as_slice()),
        ]));
        let out = eval_image(ImageFunc::Resize, &arr, Some(4), Some(2)).unwrap();
        let b = out.as_any().downcast_ref::<BinaryArray>().unwrap();
        assert!(b.is_valid(0));
        // The re-encoded PNG decodes back to the requested 4×2 dimensions.
        assert_eq!(image_dimensions(b.value(0)), Some((4, 2)));
        assert!(b.is_null(1)); // null input → null
        assert!(b.is_null(2)); // undecodable input → null
    }
}
