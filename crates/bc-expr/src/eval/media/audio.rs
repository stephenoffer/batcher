//! Audio-decode evaluation for `Expr::Audio` (the `.audio` namespace).
//!
//! Like `eval/image.rs`, this is the interpreter *oracle* for audio decoding: the
//! JIT can't compile library-backed decode, so `bc-codegen` marks `Expr::Audio`
//! unsupported and falls back here. Decode runs per row over the batch; a row whose
//! bytes are null or fail to decode yields a null result (corrupt inputs don't fail
//! the batch). `decode` returns metadata; `to_waveform` returns the mono PCM samples
//! as a `List<Float32>` — moving audio decode off the per-row Python `map_batches`
//! path and into the native data plane.

use std::io::Cursor;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, Float32Builder, Float64Array, GenericBinaryArray, Int32Array, Int64Array,
    ListBuilder, OffsetSizeTrait, StructArray,
};
use arrow::buffer::NullBuffer;
use arrow::datatypes::{DataType, Field};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use super::map_rows;
use crate::{AudioFunc, ExprError};

/// Evaluate an audio function over a Binary array of encoded audio bytes. `rate` is the
/// target sample rate for [`AudioFunc::Resample`]/[`AudioFunc::MelSpectrogram`] (ignored by
/// the others); `n_fft`/`hop`/`n_mels` parameterize the mel spectrogram only.
#[allow(clippy::too_many_arguments)]
pub(crate) fn eval_audio(
    func: AudioFunc,
    arr: &ArrayRef,
    rate: Option<i64>,
    n_fft: Option<i64>,
    hop: Option<i64>,
    n_mels: Option<i64>,
    n_mfcc: Option<i64>,
    threshold_db: Option<i64>,
) -> Result<ArrayRef, ExprError> {
    let mel = MelParams::resolve(func, rate, n_fft, hop, n_mels, n_mfcc)?;
    // An all-null column is typed `Null`, not `Binary`; see `widen_null_column`.
    if let Some(nulls) = super::widen_null_column(arr) {
        return eval_audio_sized::<i32>(func, &nulls, rate, mel, threshold_db);
    }
    match arr.data_type() {
        DataType::Binary => eval_audio_sized::<i32>(func, arr, rate, mel, threshold_db),
        DataType::LargeBinary => eval_audio_sized::<i64>(func, arr, rate, mel, threshold_db),
        other => Err(ExprError::ExpectedBinary {
            // Namespaced, because `Decode` alone is a name three namespaces share and the
            // error otherwise reported an audio failure as an image one.
            func: format!("audio.{func:?}"),
            got: other.to_string(),
        }),
    }
}

/// Validated spectral parameters (resolved for `MelSpectrogram` and `Mfcc`). `n_mfcc` is
/// `Some` only for `Mfcc`, and is required to be `<= n_mels`.
#[derive(Clone, Copy)]
struct MelParams {
    rate: u32,
    n_fft: usize,
    hop: usize,
    n_mels: usize,
    n_mfcc: Option<usize>,
}

impl MelParams {
    fn resolve(
        func: AudioFunc,
        rate: Option<i64>,
        n_fft: Option<i64>,
        hop: Option<i64>,
        n_mels: Option<i64>,
        n_mfcc: Option<i64>,
    ) -> Result<Option<Self>, ExprError> {
        let which = match func {
            AudioFunc::MelSpectrogram => "mel_spectrogram",
            AudioFunc::Mfcc => "mfcc",
            _ => return Ok(None),
        };
        let pos = |v: Option<i64>, what: &str| -> Result<usize, ExprError> {
            v.and_then(|x| usize::try_from(x).ok())
                .filter(|&x| x > 0)
                .ok_or_else(|| ExprError::InvalidArgument {
                    func: which.to_string(),
                    reason: format!("{what} must be a positive integer"),
                })
        };
        let rate = rate
            .and_then(|r| u32::try_from(r).ok())
            .filter(|&r| r > 0)
            .ok_or(ExprError::MissingAudioRate)?;
        let n_mels = pos(n_mels, "n_mels")?;
        let n_mfcc = if matches!(func, AudioFunc::Mfcc) {
            let c = pos(n_mfcc, "n_mfcc")?;
            if c > n_mels {
                return Err(ExprError::InvalidArgument {
                    func: "mfcc".to_string(),
                    reason: format!("n_mfcc ({c}) must not exceed n_mels ({n_mels})"),
                });
            }
            Some(c)
        } else {
            None
        };
        Ok(Some(Self {
            rate,
            n_fft: pos(n_fft, "n_fft")?,
            hop: pos(hop, "hop_length")?,
            n_mels,
            n_mfcc,
        }))
    }
}

/// `eval_audio` for one offset width. Both are supported because a media source stores
/// payloads as `LargeBinary` — 32-bit offsets cap an array at 2 GB total, which a batch of
/// audio files reaches.
fn eval_audio_sized<O: OffsetSizeTrait>(
    func: AudioFunc,
    arr: &ArrayRef,
    rate: Option<i64>,
    mel: Option<MelParams>,
    threshold_db: Option<i64>,
) -> Result<ArrayRef, ExprError> {
    let bytes = arr
        .as_any()
        .downcast_ref::<GenericBinaryArray<O>>()
        .ok_or_else(|| ExprError::ExpectedBinary {
            func: format!("{func:?}"),
            got: arr.data_type().to_string(),
        })?;
    match func {
        AudioFunc::Decode => decode_meta(bytes),
        AudioFunc::ToWaveform => to_waveform(bytes),
        AudioFunc::Resample => resample(bytes, rate),
        // `resolve` guaranteed `Some` for these variants.
        AudioFunc::MelSpectrogram => mel_spectrogram_col(bytes, mel.expect("mel params")),
        AudioFunc::Mfcc => mfcc_col(bytes, mel.expect("mfcc params")),
        AudioFunc::TrimSilence => super::speech::trim_silence(bytes, threshold_db),
        AudioFunc::PeakNormalize => super::speech::peak_normalize(bytes),
        AudioFunc::ZeroCrossingRate => super::speech::zero_crossing_rate(bytes),
    }
}

/// `mel_spectrogram(...)` → `List<Float32>` of `n_mels * n_frames` per row.
///
/// Decode + resample to the target rate + compute the mel power spectrogram, per row in
/// parallel. The per-row list length is `n_mels * n_frames`, and `n_frames` depends on the
/// clip length — so, like any variable-length list, rows can differ across a batch of
/// unequal clips. Null/undecodable → null.
fn mel_spectrogram_col<O: OffsetSizeTrait>(
    bytes: &GenericBinaryArray<O>,
    p: MelParams,
) -> Result<ArrayRef, ExprError> {
    use crate::eval::media::mel::mel_spectrogram;

    let rows: Vec<Option<Vec<f32>>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            return None;
        }
        let decoded = decode_pcm(bytes.value(i))?;
        let signal = resample_signal(&decoded.samples, decoded.sample_rate, p.rate);
        let (mel, _frames) = mel_spectrogram(&signal, p.rate as f64, p.n_fft, p.hop, p.n_mels);
        Some(mel)
    });

    Ok(build_f32_list_column(rows))
}

/// `mfcc(...)` → `List<Float32>` of `n_mfcc * n_frames` per row, `(n_mfcc, n_frames)`
/// row-major. Decode + resample + MFCC per row in parallel. Null/undecodable → null.
fn mfcc_col<O: OffsetSizeTrait>(
    bytes: &GenericBinaryArray<O>,
    p: MelParams,
) -> Result<ArrayRef, ExprError> {
    use crate::eval::media::mel::mfcc;

    let n_mfcc = p.n_mfcc.expect("mfcc n_mfcc");
    let rows: Vec<Option<Vec<f32>>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            return None;
        }
        let decoded = decode_pcm(bytes.value(i))?;
        let signal = resample_signal(&decoded.samples, decoded.sample_rate, p.rate);
        let (out, _frames) = mfcc(&signal, p.rate as f64, p.n_fft, p.hop, p.n_mels, n_mfcc);
        Some(out)
    });
    Ok(build_f32_list_column(rows))
}

/// Assemble per-row `Vec<f32>` (or `None`) into a `List<Float32>` column. Shared by the
/// spectral ops; a batch of equal-length clips gives every row the same length, which the
/// projection layer can tag as a fixed-shape tensor.
pub(super) fn build_f32_list_column(rows: Vec<Option<Vec<f32>>>) -> ArrayRef {
    let mut builder = ListBuilder::new(Float32Builder::new());
    for row in rows {
        match row {
            Some(vals) => {
                for v in vals {
                    builder.values().append_value(v);
                }
                builder.append(true);
            }
            None => builder.append(false),
        }
    }
    Arc::new(builder.finish())
}

/// A decoded mono signal: sample rate (Hz) and the channel-averaged f32 samples.
pub(super) struct Decoded {
    pub(super) sample_rate: u32,
    pub(super) channels: usize,
    pub(super) samples: Vec<f32>,
}

/// Decode WAV/FLAC bytes to a mono f32 signal; `None` on any failure.
pub(super) fn decode_pcm(data: &[u8]) -> Option<Decoded> {
    let mss = MediaSourceStream::new(Box::new(Cursor::new(data.to_vec())), Default::default());
    let probed = symphonia::default::get_probe()
        .format(
            &Hint::new(),
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .ok()?;
    let mut format = probed.format;
    let track = format.default_track()?;
    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate?;
    let channels = track.codec_params.channels?.count().max(1);
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .ok()?;

    let mut samples: Vec<f32> = Vec::new();
    while let Ok(packet) = format.next_packet() {
        if packet.track_id() != track_id {
            continue;
        }
        let Ok(decoded) = decoder.decode(&packet) else {
            break;
        };
        let spec = *decoded.spec();
        let mut buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        buf.copy_interleaved_ref(decoded);
        // Average the interleaved channels down to a mono sample per frame.
        for frame in buf.samples().chunks(channels) {
            samples.push(frame.iter().sum::<f32>() / channels as f32);
        }
    }
    Some(Decoded {
        sample_rate,
        channels,
        samples,
    })
}

/// `decode` → struct `{sample_rate: Int32, channels: Int32, num_frames: Int64,
/// duration_secs: Float64}`. Null/undecodable bytes → null struct.
fn decode_meta<O: OffsetSizeTrait>(bytes: &GenericBinaryArray<O>) -> Result<ArrayRef, ExprError> {
    // Decode every clip in parallel across the shared rayon pool (each is milliseconds of
    // symphonia work), then fold the results into the column buffers serially — the fold is
    // a cheap memcpy next to the decode. Without the row-level fan-out a batch smaller than
    // one morsel (16,384 rows) would decode on a single core; see `super::map_rows`.
    let decoded: Vec<Option<Decoded>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            None
        } else {
            decode_pcm(bytes.value(i))
        }
    });
    let (mut rate, mut chans) = (Vec::new(), Vec::new());
    let (mut frames, mut dur) = (Vec::new(), Vec::new());
    let mut valid = Vec::with_capacity(bytes.len());
    for d in decoded {
        match d {
            Some(a) => {
                rate.push(a.sample_rate as i32);
                chans.push(a.channels as i32);
                frames.push(a.samples.len() as i64);
                dur.push(a.samples.len() as f64 / a.sample_rate.max(1) as f64);
                valid.push(true);
            }
            None => {
                rate.push(0);
                chans.push(0);
                frames.push(0);
                dur.push(0.0);
                valid.push(false);
            }
        }
    }
    let fields = vec![
        Arc::new(Field::new("sample_rate", DataType::Int32, false)),
        Arc::new(Field::new("channels", DataType::Int32, false)),
        Arc::new(Field::new("num_frames", DataType::Int64, false)),
        Arc::new(Field::new("duration_secs", DataType::Float64, false)),
    ];
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from(rate)),
        Arc::new(Int32Array::from(chans)),
        Arc::new(Int64Array::from(frames)),
        Arc::new(Float64Array::from(dur)),
    ];
    Ok(Arc::new(StructArray::new(
        fields.into(),
        cols,
        Some(NullBuffer::from(valid)),
    )))
}

/// `to_waveform` → `List<Float32>` of mono samples per row. Null/undecodable → null.
fn to_waveform<O: OffsetSizeTrait>(bytes: &GenericBinaryArray<O>) -> Result<ArrayRef, ExprError> {
    // Decode every clip in parallel (see `decode_meta`); the `ListBuilder` append is
    // inherently serial, but it is a memcpy of already-decoded samples, not decode work.
    let decoded: Vec<Option<Decoded>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            None
        } else {
            decode_pcm(bytes.value(i))
        }
    });
    let mut builder = ListBuilder::new(Float32Builder::new());
    for d in decoded {
        match d {
            Some(a) => {
                for s in a.samples {
                    builder.values().append_value(s);
                }
                builder.append(true);
            }
            None => builder.append(false),
        }
    }
    Ok(Arc::new(builder.finish()))
}

/// `resample(rate)` → `List<Float32>` of mono samples resampled to `rate` Hz per row.
/// Decode + band-limited (sinc) resample per row in parallel; null/undecodable → null.
fn resample<O: OffsetSizeTrait>(
    bytes: &GenericBinaryArray<O>,
    rate: Option<i64>,
) -> Result<ArrayRef, ExprError> {
    // `u32::try_from` (not `as u32`) so a rate past u32::MAX is rejected rather than
    // silently wrapped down to a tiny — or zero — sample rate that empties the output.
    let target = rate
        .and_then(|r| u32::try_from(r).ok())
        .filter(|&r| r > 0)
        .ok_or(ExprError::MissingAudioRate)?;
    let waves: Vec<Option<Vec<f32>>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            None
        } else {
            decode_pcm(bytes.value(i)).map(|a| resample_signal(&a.samples, a.sample_rate, target))
        }
    });
    let mut builder = ListBuilder::new(Float32Builder::new());
    for w in waves {
        match w {
            Some(samples) => {
                for s in samples {
                    builder.values().append_value(s);
                }
                builder.append(true);
            }
            None => builder.append(false),
        }
    }
    Ok(Arc::new(builder.finish()))
}

/// Band-limited resample of a mono signal from `src` to `dst` Hz via a sinc interpolator
/// (librosa-comparable quality). The output is forced to the deterministic, resampler-
/// independent length `ceil(n * dst / src)` — the same length librosa produces — so a
/// decoded frame count agrees across engines and the result is reproducible. `src == dst`
/// (or an empty signal) is an exact passthrough.
fn resample_signal(samples: &[f32], src: u32, dst: u32) -> Vec<f32> {
    if samples.is_empty() || src == dst || src == 0 {
        return samples.to_vec();
    }
    use dasp_interpolate::sinc::Sinc;
    use dasp_signal::{self as signal, Signal};

    let want = (samples.len() as u128 * dst as u128).div_ceil(src as u128) as usize;
    let sinc = Sinc::new(dasp_ring_buffer::Fixed::from([0.0f32; 64]));
    let sig = signal::from_iter(samples.iter().copied());
    let mut out: Vec<f32> = sig
        .from_hz_to_hz(sinc, src as f64, dst as f64)
        .until_exhausted()
        .collect();
    // Sinc rounding can overshoot or (with warm-up) fall a few samples short of the exact
    // ratio length; truncate/pad to `want` so every clip resampled by the same ratio has a
    // predictable, resampler-independent size.
    out.truncate(want);
    out.resize(want, 0.0);
    out
}

#[cfg(test)]
mod tests {
    // `AsArray` and the primitive type tags drive the struct-column assertions below and are
    // used nowhere in the lib target, so they are imported here rather than at module scope
    // (where they are dead code under `-D warnings`).
    use arrow::array::{AsArray, BinaryArray, LargeBinaryArray};
    use arrow::datatypes::{Float32Type, Int32Type, Int64Type};

    use super::*;

    /// `eval_audio` for the non-mel ops, which ignore the mel params — keeps their tests
    /// readable at three arguments.
    fn ea(func: AudioFunc, arr: &ArrayRef, rate: Option<i64>) -> Result<ArrayRef, ExprError> {
        eval_audio(func, arr, rate, None, None, None, None, None)
    }

    /// Build a minimal mono 16-bit PCM WAV from `samples` at `sample_rate`.
    fn make_wav(sample_rate: u32, samples: &[i16]) -> Vec<u8> {
        let data_len = (samples.len() * 2) as u32;
        let byte_rate = sample_rate * 2;
        let mut w = Vec::new();
        w.extend_from_slice(b"RIFF");
        w.extend_from_slice(&(36 + data_len).to_le_bytes());
        w.extend_from_slice(b"WAVE");
        w.extend_from_slice(b"fmt ");
        w.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
        w.extend_from_slice(&1u16.to_le_bytes()); // PCM
        w.extend_from_slice(&1u16.to_le_bytes()); // mono
        w.extend_from_slice(&sample_rate.to_le_bytes());
        w.extend_from_slice(&byte_rate.to_le_bytes());
        w.extend_from_slice(&2u16.to_le_bytes()); // block align
        w.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        w.extend_from_slice(b"data");
        w.extend_from_slice(&data_len.to_le_bytes());
        for s in samples {
            w.extend_from_slice(&s.to_le_bytes());
        }
        w
    }

    #[test]
    fn decode_reads_metadata() {
        let wav = make_wav(8000, &[0, 16384, -16384, 0, 100, -100]);
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![
            Some(wav.as_slice()),
            None,
            Some(b"not audio".as_slice()),
        ]));
        let out = ea(AudioFunc::Decode, &arr, None).unwrap();
        let s = out.as_struct();
        let rate = s.column(0).as_primitive::<Int32Type>();
        let frames = s.column(2).as_primitive::<Int64Type>();
        assert!(s.is_valid(0) && rate.value(0) == 8000 && frames.value(0) == 6);
        assert!(s.is_null(1)); // null bytes → null
        assert!(s.is_null(2)); // undecodable → null
    }

    /// `LargeBinary` must decode identically to `Binary` — the layout a media source
    /// actually produces (32-bit offsets cap an array at 2 GB total).
    #[test]
    fn large_binary_matches_binary_for_every_audio_func() {
        let wav = make_wav(8000, &[0, 16384, -16384, 0, 100, -100]);
        let rows = vec![Some(wav.as_slice()), None];
        let narrow: ArrayRef = Arc::new(BinaryArray::from(rows.clone()));
        let wide: ArrayRef = Arc::new(LargeBinaryArray::from(rows));

        for func in [
            AudioFunc::Decode,
            AudioFunc::ToWaveform,
            AudioFunc::Resample,
        ] {
            let narrow_out = ea(func, &narrow, Some(8000)).unwrap();
            let wide_out = ea(func, &wide, Some(8000)).unwrap();
            assert_eq!(
                narrow_out.as_ref(),
                wide_out.as_ref(),
                "{func:?} disagreed between Binary and LargeBinary"
            );
        }
    }

    #[test]
    fn parallel_batch_preserves_order_and_nulls() {
        // A batch past PAR_ROW_THRESHOLD (8) with valid / null / undecodable rows
        // interleaved: the row-parallel decode must keep every result at its own index.
        let mut rows: Vec<Option<Vec<u8>>> = Vec::new();
        let mut expect_frames: Vec<Option<i64>> = Vec::new();
        for i in 0..20 {
            if i % 3 == 0 {
                rows.push(None); // null bytes
                expect_frames.push(None);
            } else if i % 3 == 1 {
                rows.push(Some(b"not audio".to_vec())); // undecodable
                expect_frames.push(None);
            } else {
                let n = (i % 5) + 1;
                rows.push(Some(make_wav(8000, &vec![100i16; n])));
                expect_frames.push(Some(n as i64));
            }
        }
        let arr: ArrayRef = Arc::new(BinaryArray::from(
            rows.iter().map(|o| o.as_deref()).collect::<Vec<_>>(),
        ));
        let out = ea(AudioFunc::Decode, &arr, None).unwrap();
        let s = out.as_struct();
        let frames = s.column(2).as_primitive::<Int64Type>();
        for (i, exp) in expect_frames.iter().enumerate() {
            match exp {
                Some(f) => assert!(s.is_valid(i) && frames.value(i) == *f, "row {i}"),
                None => assert!(s.is_null(i), "row {i} should be null"),
            }
        }
    }

    #[test]
    fn resample_signal_length_and_identity() {
        // src == dst → exact passthrough.
        let x: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.1).sin()).collect();
        assert_eq!(resample_signal(&x, 8000, 8000), x);

        // Downsample 8000 -> 4000: length is exactly ceil(n * dst / src).
        let down = resample_signal(&x, 8000, 4000);
        assert_eq!(down.len(), 500);
        // Upsample 8000 -> 16000: length doubles.
        let up = resample_signal(&x, 8000, 16000);
        assert_eq!(up.len(), 2000);
        // Odd ratio 44100 -> 16000.
        let odd = resample_signal(&vec![0.5f32; 44100], 44100, 16000);
        assert_eq!(odd.len(), 16000);
    }

    #[test]
    fn resample_signal_preserves_energy() {
        // A pure 1 kHz tone at 8 kHz, resampled to 4 kHz, keeps its ~0.5 mean power
        // (band-limited resampling preserves the signal, unlike a lossy decimation).
        let n = 8000;
        let tone: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 8000.0).sin())
            .collect();
        let out = resample_signal(&tone, 8000, 4000);
        // Skip the sinc warm-up region when measuring power.
        let body = &out[64..];
        let power: f32 = body.iter().map(|s| s * s).sum::<f32>() / body.len() as f32;
        assert!((power - 0.5).abs() < 0.1, "power {power} should be ~0.5");
    }

    #[test]
    fn resample_over_batch_and_nulls() {
        let wav = make_wav(8000, &[0, 16384, -16384, 0, 100, -100, 0, 200]); // 8 frames
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![
            Some(wav.as_slice()),
            None,
            Some(b"not audio".as_slice()),
        ]));
        let out = ea(AudioFunc::Resample, &arr, Some(4000)).unwrap();
        let list = out.as_list::<i32>();
        assert_eq!(list.value_length(0), 4); // 8 frames at 8k -> 4 at 4k
        assert!(list.is_null(1) && list.is_null(2)); // null + undecodable -> null
    }

    #[test]
    fn resample_requires_rate() {
        let wav = make_wav(8000, &[0, 1, 2]);
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![Some(wav.as_slice())]));
        assert!(ea(AudioFunc::Resample, &arr, None).is_err());
        assert!(ea(AudioFunc::Resample, &arr, Some(0)).is_err());
        assert!(ea(AudioFunc::Resample, &arr, Some(-8000)).is_err());
        // Past u32::MAX must be rejected, not wrapped down (e.g. 2^32 → 0 Hz, which would
        // silently empty every clip instead of erroring).
        assert!(ea(AudioFunc::Resample, &arr, Some(i64::from(u32::MAX) + 1)).is_err());
        assert!(ea(AudioFunc::Resample, &arr, Some(i64::MAX)).is_err());
    }

    #[test]
    fn to_waveform_decodes_mono_samples() {
        let wav = make_wav(8000, &[0, 16384, -16384]);
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![Some(wav.as_slice()), None]));
        let out = ea(AudioFunc::ToWaveform, &arr, None).unwrap();
        let list = out.as_list::<i32>();
        assert!(list.is_valid(0) && list.value_length(0) == 3);
        let row0 = list.value(0);
        let px = row0.as_primitive::<Float32Type>();
        // 16384/32768 ≈ 0.5 in normalized f32.
        assert!((px.value(1) - 0.5).abs() < 0.01);
        assert!(list.is_null(1)); // null bytes → null list
    }
}
