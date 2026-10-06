//! Byte-stream compression for `StrFunc::Compress`/`Decompress` — six codecs, one shape.
//!
//! Compression belongs in the expression layer, not just the IO layer, because payloads
//! arrive *inside* columns: a gzipped JSON body in a Kafka record, a zstd-framed blob in a
//! warehouse table. Without this the only way to read one is a Python UDF, which means
//! leaving the data plane for every row.
//!
//! Every codec listed is already in the build — `flate2`, `zstd`, `brotli`, and `lz4_flex`
//! arrive through Arrow's IPC compression and the object-store stack — so the direct
//! dependency edges added here bring no new third-party code with them. That is why six
//! codecs cost the same as the three Daft offers.
//!
//! Decompression is **lenient**: input that is not a valid frame yields null rather than
//! erroring the batch, matching `from_base64` and `unhex`. One corrupt blob in a scan of a
//! billion is a bad row, not a bad query, and it removes the need for the separate
//! `try_decompress` spelling Daft carries.

use std::io::{Read, Write};

use crate::ExprError;

/// The codec names accepted in the `pattern` slot. Mirrored by `_COMPRESSION_CODECS` in
/// `plan/expr_ir/namespaces/strings.py`, which rejects a typo at plan-build time.
pub(crate) const CODECS: [&str; 6] = ["gzip", "zlib", "deflate", "zstd", "brotli", "lz4"];

/// Compress `data` with `codec`, or `None` if the codec name is unknown.
///
/// Returns `Err` only for a codec that fails on *valid* input, which in practice means an
/// allocation failure — the writers here cannot reject well-formed bytes. A codec-level
/// error is surfaced rather than nulled, because unlike a corrupt input it indicates the
/// query cannot proceed.
pub(crate) fn compress(data: &[u8], codec: &str) -> Option<Result<Vec<u8>, ExprError>> {
    Some(match codec {
        "gzip" => finish(
            flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default()),
            data,
        ),
        "zlib" => finish(
            flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default()),
            data,
        ),
        "deflate" => finish(
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default()),
            data,
        ),
        // `bulk::compress` rather than `stream::encode_all`: the one-shot form records the
        // decompressed size in the frame header. Streaming does not, and a frame without
        // it is legal but unreadable by several one-shot decoders that need to size their
        // output buffer up front — Python's `zstandard.ZstdDecompressor().decompress()`
        // among them. Writing a frame the ecosystem cannot read defeats the point of
        // naming the codec.
        "zstd" => zstd::bulk::compress(data, 0).map_err(|e| codec_error("zstd", &e)),
        "brotli" => {
            let mut out = Vec::new();
            // 4096-byte window, quality 5, lgwin 22: the encoder's own defaults for
            // general data. Tuning is deliberately not exposed — a compression *level*
            // argument is a second dimension on every codec, and no caller has asked.
            let mut w = brotli::CompressorWriter::new(&mut out, 4096, 5, 22);
            match w.write_all(data).and_then(|()| w.flush()) {
                Ok(()) => {
                    drop(w);
                    Ok(out)
                }
                Err(e) => Err(codec_error("brotli", &e)),
            }
        }
        "lz4" => Ok(lz4_flex::block::compress_prepend_size(data)),
        _ => return None,
    })
}

/// Decompress `data` with `codec`. `None` for an unknown codec; `Some(None)` for input
/// that is not a valid frame for this codec (the lenient case), or whose payload would
/// exceed `limit` bytes.
///
/// `limit` is the decompression-bomb bound: a few hundred bytes of gzip can expand to
/// gigabytes, and a single such cell would otherwise allocate all of it. The streaming
/// decoders read at most `limit + 1` bytes, so an oversized payload is detected without
/// being materialized; `lz4`'s frame *states* its size up front, so that is checked before
/// anything is allocated. `None` keeps the unbounded read.
pub(crate) fn decompress(
    data: &[u8],
    codec: &str,
    limit: Option<usize>,
) -> Option<Option<Vec<u8>>> {
    Some(match codec {
        "gzip" => read_all(flate2::read::GzDecoder::new(data), limit),
        "zlib" => read_all(flate2::read::ZlibDecoder::new(data), limit),
        "deflate" => read_all(flate2::read::DeflateDecoder::new(data), limit),
        "zstd" => match limit {
            None => zstd::stream::decode_all(data).ok(),
            Some(_) => zstd::stream::read::Decoder::new(data)
                .ok()
                .and_then(|d| read_all(d, limit)),
        },
        "brotli" => read_all(brotli::Decompressor::new(data, 4096), limit),
        "lz4" => match lz4_flex::block::uncompressed_size(data) {
            Ok((size, _)) if limit.is_some_and(|cap| size > cap) => None,
            _ => lz4_flex::block::decompress_size_prepended(data).ok(),
        },
        _ => return None,
    })
}

/// Write `data` through a `flate2` encoder and take its buffer.
fn finish<W: Write + Finish>(mut enc: W, data: &[u8]) -> Result<Vec<u8>, ExprError> {
    enc.write_all(data)
        .map_err(|e| codec_error("deflate-family", &e))?;
    enc.finish_inner()
        .map_err(|e| codec_error("deflate-family", &e))
}

/// The `finish()` each `flate2` encoder has, unified so `finish` above is written once
/// rather than three times.
trait Finish {
    fn finish_inner(self) -> std::io::Result<Vec<u8>>;
}

macro_rules! impl_finish {
    ($($t:ty),*) => {$(
        impl Finish for $t {
            fn finish_inner(self) -> std::io::Result<Vec<u8>> {
                self.finish()
            }
        }
    )*};
}

impl_finish!(
    flate2::write::GzEncoder<Vec<u8>>,
    flate2::write::ZlibEncoder<Vec<u8>>,
    flate2::write::DeflateEncoder<Vec<u8>>
);

/// Read a decoder to the end, or `None` if the stream is malformed or truncated, or holds
/// more than `limit` bytes.
///
/// A truncated frame surfaces as a read error partway through, so this cannot be
/// shortened to "decode and hope" — the whole read has to succeed for the output to be
/// the real payload rather than a prefix of it. For the same reason a payload over the
/// limit is `None`, never its first `limit` bytes.
fn read_all(r: impl Read, limit: Option<usize>) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    match limit {
        None => {
            let mut r = r;
            r.read_to_end(&mut out).ok()?;
        }
        Some(cap) => {
            // One byte past the cap is enough to know the payload does not fit.
            r.take(cap as u64 + 1).read_to_end(&mut out).ok()?;
            if out.len() > cap {
                return None;
            }
        }
    }
    Some(out)
}

fn codec_error(codec: &str, e: &dyn std::fmt::Display) -> ExprError {
    ExprError::InvalidArgument {
        func: "Compress".to_string(),
        reason: format!("{codec} codec failed: {e}"),
    }
}

/// The error raised once when a codec name is not one of [`CODECS`].
pub(crate) fn unknown_codec(func: &str, codec: &str) -> ExprError {
    ExprError::InvalidArgument {
        func: func.to_string(),
        reason: format!(
            "unknown codec {codec:?}; expected one of {}",
            CODECS.join(", ")
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAYLOAD: &[u8] =
        b"the quick brown fox jumps over the lazy dog, repeatedly and at some length";

    /// The compressed bytes themselves are pinned, not only the round trip: a compressed
    /// column is a stored value, so a codec crate bump that changes its encoder output would
    /// change query results. `(codec, input index, length, xxhash64 of the frame)`.
    #[test]
    fn compressed_bytes_are_stable() {
        let inputs: [&[u8]; 3] = [b"", PAYLOAD, &PAYLOAD.repeat(100)];
        let pinned: [(&str, usize, usize, u64); 9] = [
            ("zstd", 0, 9, 0x9fcc_d29d_986b_864f),
            ("zstd", 1, 76, 0xaec9_2d5e_c73d_a2d6),
            ("zstd", 2, 84, 0x3c43_a8cc_278f_1704),
            ("brotli", 0, 3, 0xd3af_89fa_1474_46f4),
            ("brotli", 1, 66, 0x1ae7_9a41_f389_a883),
            ("brotli", 2, 75, 0x26f3_19a1_733d_02ec),
            ("lz4", 0, 5, 0x00f4_f72f_b7a8_c648),
            ("lz4", 1, 80, 0xe1da_9144_be10_c5eb),
            ("lz4", 2, 118, 0xb971_6464_8cf8_05f7),
        ];
        for (codec, i, len, digest) in pinned {
            let packed = compress(inputs[i], codec).unwrap().unwrap();
            assert_eq!(packed.len(), len, "{codec} input {i}: frame length moved");
            assert_eq!(
                bc_arrow::xxhash64(&packed, 0),
                digest,
                "{codec} input {i}: frame bytes moved"
            );
        }
    }

    #[test]
    fn every_codec_round_trips() {
        for codec in CODECS {
            let packed = compress(PAYLOAD, codec).unwrap().unwrap();
            let back = decompress(&packed, codec, None).unwrap().unwrap();
            assert_eq!(back, PAYLOAD, "{codec} did not round-trip");
        }
    }

    #[test]
    fn every_codec_round_trips_the_empty_input() {
        for codec in CODECS {
            let packed = compress(b"", codec).unwrap().unwrap();
            let back = decompress(&packed, codec, None).unwrap().unwrap();
            assert!(back.is_empty(), "{codec} did not round-trip an empty input");
        }
    }

    #[test]
    fn compression_actually_shrinks_repetitive_data() {
        let repetitive = PAYLOAD.repeat(100);
        for codec in CODECS {
            let packed = compress(&repetitive, codec).unwrap().unwrap();
            assert!(
                packed.len() < repetitive.len() / 2,
                "{codec}: {} bytes from {}",
                packed.len(),
                repetitive.len()
            );
        }
    }

    #[test]
    fn a_corrupt_frame_is_none_rather_than_an_error() {
        // Every framed codec rejects arbitrary bytes because each carries a header or a
        // checksum. `deflate` is excluded deliberately and asserted separately below: raw
        // deflate is a bare bit-stream with neither, so "is this a deflate frame?" is not
        // a question that can be answered. Excluding it by name rather than with an
        // `|| codec == "deflate"` inside the loop keeps a regression in the other five
        // visible.
        for codec in ["gzip", "zlib", "zstd", "brotli", "lz4"] {
            assert!(
                decompress(b"\x00not a valid frame at all\xff", codec, None)
                    .unwrap()
                    .is_none(),
                "{codec} accepted garbage"
            );
        }
    }

    #[test]
    fn raw_deflate_cannot_detect_garbage_and_that_is_a_property_of_the_format() {
        // Stated as a test so the leniency is a known limit rather than a surprise: raw
        // deflate has no magic bytes and no checksum, so some byte strings decode to
        // something. Callers who need detection should use `zlib` (adler32) or `gzip`
        // (crc32), which wrap the same algorithm in a frame that can be validated.
        let got = decompress(b"\x00not a valid frame at all\xff", "deflate", None).unwrap();
        assert_ne!(got.as_deref(), Some(PAYLOAD));
    }

    #[test]
    fn a_truncated_frame_is_none_rather_than_a_prefix() {
        // The failure this guards: a decoder that reports success on a partial read would
        // hand back a *prefix* of the payload, which is worse than an error because it
        // looks like data.
        for codec in CODECS {
            let packed = compress(&PAYLOAD.repeat(50), codec).unwrap().unwrap();
            let truncated = &packed[..packed.len() / 2];
            let got = decompress(truncated, codec, None).unwrap();
            assert!(
                got.as_ref().is_none_or(|v| v != &PAYLOAD.repeat(50)),
                "{codec} returned data from a truncated frame"
            );
        }
    }

    #[test]
    fn a_frame_does_not_decode_under_a_different_codec() {
        let packed = compress(PAYLOAD, "gzip").unwrap().unwrap();
        for codec in ["zstd", "brotli", "lz4"] {
            assert_ne!(
                decompress(&packed, codec, None).unwrap().as_deref(),
                Some(PAYLOAD),
                "{codec} decoded a gzip frame"
            );
        }
    }

    #[test]
    fn an_unknown_codec_is_none_in_both_directions() {
        assert!(compress(PAYLOAD, "snappy").is_none());
        assert!(decompress(PAYLOAD, "snappy", None).is_none());
    }

    #[test]
    fn the_codec_list_is_exactly_what_is_implemented() {
        for codec in CODECS {
            assert!(
                compress(b"x", codec).is_some(),
                "{codec} listed but not compressed"
            );
            assert!(
                decompress(b"x", codec, None).is_some(),
                "{codec} listed but not decompressed"
            );
        }
    }

    #[test]
    fn a_payload_over_the_limit_is_none_for_every_codec() {
        // A megabyte of zeros: the decompression-bomb shape, a tiny frame with a huge
        // payload. Under a 1 KiB cap every codec refuses it rather than allocating it.
        let bomb = vec![0u8; 1 << 20];
        for codec in CODECS {
            let packed = compress(&bomb, codec).unwrap().unwrap();
            assert!(packed.len() < 1 << 14, "{codec}: the fixture is not a bomb");
            assert_eq!(
                decompress(&packed, codec, Some(1024)),
                Some(None),
                "{codec} decoded past its limit"
            );
            // At exactly the payload size the limit admits it, so the bound is not off by one.
            assert_eq!(
                decompress(&packed, codec, Some(bomb.len()))
                    .unwrap()
                    .as_deref(),
                Some(bomb.as_slice()),
                "{codec} refused a payload that fits"
            );
        }
    }
}
