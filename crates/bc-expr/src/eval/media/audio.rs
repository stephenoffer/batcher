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
    Array, ArrayRef, BinaryArray, Float32Builder, Float64Array, Int32Array, Int64Array,
    ListBuilder, StructArray,
};
use arrow::buffer::NullBuffer;
use arrow::datatypes::{DataType, Field};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::{AudioFunc, ExprError};

/// Evaluate an audio function over a Binary array of encoded audio bytes.
pub(crate) fn eval_audio(func: AudioFunc, arr: &ArrayRef) -> Result<ArrayRef, ExprError> {
    let bytes =
        arr.as_any()
            .downcast_ref::<BinaryArray>()
            .ok_or_else(|| ExprError::ExpectedBinary {
                func: format!("{func:?}"),
                got: arr.data_type().to_string(),
            })?;
    match func {
        AudioFunc::Decode => decode_meta(bytes),
        AudioFunc::ToWaveform => to_waveform(bytes),
    }
}

/// A decoded mono signal: sample rate (Hz) and the channel-averaged f32 samples.
struct Decoded {
    sample_rate: u32,
    channels: usize,
    samples: Vec<f32>,
}

/// Decode WAV/FLAC bytes to a mono f32 signal; `None` on any failure.
fn decode_pcm(data: &[u8]) -> Option<Decoded> {
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
fn decode_meta(bytes: &BinaryArray) -> Result<ArrayRef, ExprError> {
    let (mut rate, mut chans) = (Vec::new(), Vec::new());
    let (mut frames, mut dur) = (Vec::new(), Vec::new());
    let mut valid = Vec::with_capacity(bytes.len());
    for i in 0..bytes.len() {
        let d = if bytes.is_null(i) {
            None
        } else {
            decode_pcm(bytes.value(i))
        };
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
fn to_waveform(bytes: &BinaryArray) -> Result<ArrayRef, ExprError> {
    let mut builder = ListBuilder::new(Float32Builder::new());
    for i in 0..bytes.len() {
        let d = if bytes.is_null(i) {
            None
        } else {
            decode_pcm(bytes.value(i))
        };
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

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float32Array, ListArray};

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
        let out = eval_audio(AudioFunc::Decode, &arr).unwrap();
        let s = out.as_any().downcast_ref::<StructArray>().unwrap();
        let rate = s.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let frames = s.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
        assert!(s.is_valid(0) && rate.value(0) == 8000 && frames.value(0) == 6);
        assert!(s.is_null(1)); // null bytes → null
        assert!(s.is_null(2)); // undecodable → null
    }

    #[test]
    fn to_waveform_decodes_mono_samples() {
        let wav = make_wav(8000, &[0, 16384, -16384]);
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![Some(wav.as_slice()), None]));
        let out = eval_audio(AudioFunc::ToWaveform, &arr).unwrap();
        let list = out.as_any().downcast_ref::<ListArray>().unwrap();
        assert!(list.is_valid(0) && list.value_length(0) == 3);
        let row0 = list.value(0);
        let px = row0.as_any().downcast_ref::<Float32Array>().unwrap();
        // 16384/32768 ≈ 0.5 in normalized f32.
        assert!((px.value(1) - 0.5).abs() < 0.01);
        assert!(list.is_null(1)); // null bytes → null list
    }
}
