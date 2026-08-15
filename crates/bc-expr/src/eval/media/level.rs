//! Level, hygiene and framing: what a clip's amplitude says, and how to make it batchable.
//!
//! Two families that share one decode. The **measures** (`rms`, `dbfs`, `peak_dbfs`,
//! `clipping_ratio`, `silence_ratio`) reduce a clip to one number, so filtering a corpus by
//! recording quality is an ordinary predicate rather than a per-file Python loop over a
//! million floats. The **shaping** ops (`rms_normalize`, `pre_emphasis`, `pad_or_trim`,
//! `slice`, `encode_wav`) put a clip into the form a model or a writer needs.
//!
//! `pad_or_trim` is the one that changes what is expressible. Every fixed-input audio model
//! — Whisper at exactly 30 seconds of 16 kHz, and everything shaped like it — needs each
//! row to be the same length, and a clip corpus never is. Without it a pipeline either
//! loops in Python or hands the model rows of unequal length; with it the column has a
//! knowable fixed width, because the length is a query parameter and not a property of the
//! data.
//!
//! Levels are stated in **dBFS** throughout, because that is the unit every audio tool
//! states one in: -40 dBFS is 1% of full scale wherever it is written.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BinaryBuilder, Float64Builder, GenericBinaryArray, OffsetSizeTrait,
};

use super::audio::{
    build_f32_list_column, decode_pcm, resample_signal, AudioArgs, Decoded, Signal,
};
use super::map_rows;
use crate::{AudioFunc, ExprError};

/// The amplitude a dBFS level corresponds to, relative to full scale (1.0).
pub(super) fn amplitude_for(db: f64) -> f32 {
    (10.0f64.powf(db / 20.0)) as f32
}

/// The dBFS level an amplitude corresponds to, or `None` for digital silence.
///
/// `None` rather than `-inf` on purpose. An infinity compares less than every threshold, so
/// a silent clip would silently pass every "quieter than X" filter *and* every "louder than
/// X" one written with a negated comparison; a null is the answer that propagates honestly.
fn db_for(amplitude: f64) -> Option<f64> {
    (amplitude > 0.0).then(|| 20.0 * amplitude.log10())
}

fn rms_of(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples.iter().map(|s| f64::from(*s) * f64::from(*s)).sum();
    (sum / samples.len() as f64).sqrt()
}

fn peak_of(samples: &[f32]) -> f64 {
    samples
        .iter()
        .fold(0.0f64, |m, s| m.max(f64::from(s.abs())))
}

fn build_f64(rows: Vec<Option<f64>>) -> ArrayRef {
    let mut b = Float64Builder::with_capacity(rows.len());
    for row in rows {
        match row {
            Some(v) => b.append_value(v),
            None => b.append_null(),
        }
    }
    Arc::new(b.finish())
}

/// The scalar level and hygiene measures, which all read the decoded samples once.
///
/// Written against [`Signal`] rather than a byte column: none of them needs a sample rate,
/// so none of them has a reason to insist on a container. That is what lets
/// `trim_silence().rms()` be one expression instead of an error.
pub(super) fn measure(
    func: AudioFunc,
    clips: &Signal<'_>,
    args: AudioArgs,
) -> Result<ArrayRef, ExprError> {
    // 0.99 of full scale: a sample that close to the rail was almost certainly clamped by
    // the recorder rather than reached by the signal.
    let clip_at = bounded(func, args.factor, 0.99, 0.0, 1.0)?;
    let quiet_below = f64::from(amplitude_for(args.threshold_db.unwrap_or(-40) as f64));
    let rows: Vec<Option<f64>> = map_rows(clips.len(), |i| {
        let samples = clips.samples(i)?;
        // A clip that decoded to no samples has no level; every measure here is a
        // statistic over samples, and a mean of nothing is not zero.
        if samples.is_empty() {
            return None;
        }
        let fraction = |pred: &dyn Fn(f64) -> bool| {
            samples.iter().filter(|s| pred(f64::from(s.abs()))).count() as f64
                / samples.len() as f64
        };
        match func {
            AudioFunc::Rms => Some(rms_of(&samples)),
            AudioFunc::Dbfs => db_for(rms_of(&samples)),
            AudioFunc::PeakDbfs => db_for(peak_of(&samples)),
            AudioFunc::ClippingRatio => Some(fraction(&|a| a >= clip_at)),
            AudioFunc::SilenceRatio => Some(fraction(&|a| a < quiet_below)),
            _ => None,
        }
    });
    Ok(build_f64(rows))
}

/// The waveform-shaping ops that keep the clip's length: level match and pre-emphasis.
///
/// Also written against [`Signal`], for the same reason and with the same payoff: applied
/// to an already-decoded waveform they cost no second decode.
pub(super) fn shape(
    func: AudioFunc,
    clips: &Signal<'_>,
    args: AudioArgs,
) -> Result<ArrayRef, ExprError> {
    // -20 dBFS for the target level (the broadcast convention for speech) and 0.97 for the
    // pre-emphasis coefficient (the value every classical ASR front end uses).
    let target = f64::from(amplitude_for(args.threshold_db.unwrap_or(-20) as f64));
    let coeff = bounded(func, args.factor, 0.97, 0.0, 1.0)? as f32;
    let rows: Vec<Option<Vec<f32>>> = map_rows(clips.len(), |i| {
        let mut samples = clips.samples(i)?;
        match func {
            AudioFunc::RmsNormalize => {
                let rms = rms_of(&samples);
                if rms > 0.0 {
                    // Cap the gain so a very quiet clip is lifted toward the target rather
                    // than driven into the rails: normalizing a whisper to -20 dBFS by
                    // brute force clips it, which is worse than leaving it quiet.
                    let peak = peak_of(&samples);
                    let gain = (target / rms).min(if peak > 0.0 { 1.0 / peak } else { f64::MAX });
                    for s in &mut samples {
                        *s = (f64::from(*s) * gain) as f32;
                    }
                }
                Some(samples)
            }
            AudioFunc::PreEmphasis => {
                // In place and backwards, so each step still reads the *original* previous
                // sample. Forward and in place would feed the filter its own output.
                for n in (1..samples.len()).rev() {
                    samples[n] -= coeff * samples[n - 1];
                }
                Some(samples)
            }
            _ => None,
        }
    });
    Ok(build_f32_list_column(rows))
}

/// `pad_or_trim(duration_secs, rate)` → exactly `duration_secs` of audio at `rate`.
pub(super) fn pad_or_trim<O: OffsetSizeTrait>(
    bytes: &GenericBinaryArray<O>,
    args: AudioArgs,
) -> Result<ArrayRef, ExprError> {
    let rate = target_rate("pad_or_trim", args.rate)?;
    let secs = positive_secs("pad_or_trim", "duration_secs", args.duration_secs)?;
    // The length is a query parameter, so it is knowable before a byte is read — which is
    // the whole point, and also why an absurd one must be refused rather than allocated.
    let want = length_guard("pad_or_trim", secs, rate)?;
    let rows: Vec<Option<Vec<f32>>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            return None;
        }
        let decoded = decode_pcm(bytes.value(i))?;
        let mut signal = resample_signal(&decoded.samples, decoded.sample_rate, rate);
        signal.resize(want, 0.0);
        Some(signal)
    });
    Ok(build_f32_list_column(rows))
}

/// `slice(offset_secs, duration_secs)` → the region of the clip that window names.
pub(super) fn slice<O: OffsetSizeTrait>(
    bytes: &GenericBinaryArray<O>,
    args: AudioArgs,
) -> Result<ArrayRef, ExprError> {
    let offset = args.offset_secs.unwrap_or(0.0);
    if !offset.is_finite() || offset < 0.0 {
        return Err(ExprError::InvalidArgument {
            func: "slice".to_string(),
            reason: format!("offset_secs must be a finite, non-negative number, got {offset}"),
        });
    }
    let secs = positive_secs("slice", "duration_secs", args.duration_secs)?;
    // The clip's own rate, not a target one: slicing is a question about time, and
    // resampling it first would change which samples the window names.
    let rows: Vec<Option<Vec<f32>>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            return None;
        }
        let Decoded {
            samples,
            sample_rate,
            ..
        } = decode_pcm(bytes.value(i))?;
        let rate = f64::from(sample_rate.max(1));
        let start = (offset * rate) as usize;
        let len = (secs * rate) as usize;
        // A window past the end of the clip is an empty region, not an unreadable clip:
        // the caller asked about time the recording does not cover, and saying so as an
        // empty list keeps `list.len() == 0` the test for it.
        let end = start.saturating_add(len).min(samples.len());
        Some(samples.get(start..end).unwrap_or_default().to_vec())
    });
    Ok(build_f32_list_column(rows))
}

/// `encode_wav(rate)` → a 16-bit PCM mono WAV container per row.
///
/// The op that closes the loop. Every other waveform op in this namespace hands back a
/// `List<Float32>`, which is what a model wants and what nothing else can read: writing a
/// trimmed, normalized corpus back to object storage as *audio* meant encoding each row in
/// Python. 16-bit PCM rather than float32 because it is the format every player, dataset
/// loader and annotation tool accepts.
pub(super) fn encode_wav<O: OffsetSizeTrait>(
    bytes: &GenericBinaryArray<O>,
    args: AudioArgs,
) -> Result<ArrayRef, ExprError> {
    // No `rate` means "leave it alone", which is what an encode-only step wants.
    let target = match args.rate {
        None => None,
        Some(_) => Some(target_rate("encode_wav", args.rate)?),
    };
    let rows: Vec<Option<Vec<u8>>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            return None;
        }
        let decoded = decode_pcm(bytes.value(i))?;
        let rate = target.unwrap_or(decoded.sample_rate.max(1));
        let signal = match target {
            Some(r) => resample_signal(&decoded.samples, decoded.sample_rate, r),
            None => decoded.samples,
        };
        Some(wav_pcm16(&signal, rate))
    });
    Ok(build_binary(rows))
}

/// `encode_wav(rate)` over a **waveform** column rather than encoded bytes.
///
/// Returns `Ok(None)` when this is not that case, so the caller falls through to the
/// ordinary container path. This is what closes the loop the rest of the namespace leaves
/// open: `to_waveform`, `trim_silence`, `rms_normalize` and the rest all hand back a
/// `List<Float32>`, which is what a model wants and what no player, loader or annotation
/// tool can read — so a cleaned corpus could be written back out only as a list of floats.
///
/// The samples are written **at `rate`, not resampled to it**. A waveform column carries no
/// sample rate, so there is nothing to resample *from*; `rate` therefore states what the
/// samples already are. Getting that wrong makes a clip play at the wrong speed, so it is
/// required rather than defaulted.
pub(super) fn encode_waveform(arr: &ArrayRef, args: AudioArgs) -> Result<ArrayRef, ExprError> {
    use arrow::array::{AsArray, ListArray};
    use arrow::datatypes::Float32Type;

    let rate = args.rate.ok_or_else(|| ExprError::InvalidArgument {
        func: "audio.encode_wav".to_string(),
        reason: "a waveform column carries no sample rate, so `rate` says what its samples \
                 already are and cannot be omitted"
            .to_string(),
    })?;
    let rate = target_rate("audio.encode_wav", Some(rate))?;
    let list: &ListArray = arr.as_list::<i32>();
    let rows: Vec<Option<Vec<u8>>> = map_rows(list.len(), |i| {
        if list.is_null(i) {
            return None;
        }
        let values = list.value(i);
        let samples = values.as_primitive::<Float32Type>();
        let signal: Vec<f32> = (0..samples.len())
            // A null *element* inside a waveform is silence: the alternative is nulling the
            // whole clip because one sample was absent, which no upstream op produces and
            // which would discard a recording over a single missing frame.
            .map(|k| {
                if samples.is_null(k) {
                    0.0
                } else {
                    samples.value(k)
                }
            })
            .collect();
        Some(wav_pcm16(&signal, rate))
    });
    Ok(build_binary(rows))
}

/// Collect per-row byte buffers into a `Binary` column (`None` → null).
fn build_binary(rows: Vec<Option<Vec<u8>>>) -> ArrayRef {
    let mut b = BinaryBuilder::with_capacity(rows.len(), rows.len() * 1024);
    for row in rows {
        match row {
            Some(v) => b.append_value(v),
            None => b.append_null(),
        }
    }
    Arc::new(b.finish())
}

/// A canonical 44-byte-header mono 16-bit PCM WAV file.
///
/// Written by hand rather than through an encoder crate: the header is four fixed chunks
/// and two sizes, and the alternative is a dependency whose only job here is to write
/// those. Samples are clamped before scaling, so a signal that a gain stage pushed past
/// full scale saturates rather than wrapping from +32767 to -32768 — which is a click.
fn wav_pcm16(samples: &[f32], rate: u32) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // format: PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // channels: mono
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * 2).to_le_bytes()); // byte rate
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        let v = (s.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// A required, positive target sample rate.
pub(super) fn target_rate(func: &str, rate: Option<i64>) -> Result<u32, ExprError> {
    rate.and_then(|r| u32::try_from(r).ok())
        .filter(|&r| r > 0)
        .ok_or(ExprError::MissingAudioRate)
        .map_err(|_| ExprError::InvalidArgument {
            func: func.to_string(),
            reason: "rate must be a positive sample rate in Hz".to_string(),
        })
}

/// A required, positive duration in seconds.
fn positive_secs(func: &str, what: &str, secs: Option<f64>) -> Result<f64, ExprError> {
    secs.filter(|s| s.is_finite() && *s > 0.0)
        .ok_or_else(|| ExprError::InvalidArgument {
            func: func.to_string(),
            reason: format!("{what} must be a finite, positive number of seconds"),
        })
}

/// A sample count that an Arrow list can actually hold.
///
/// `duration_secs * rate` is driven entirely by query parameters, so a mistyped duration is
/// an allocation of arbitrary size per row — computed in `f64` and checked against the
/// offset width before anything is allocated, the same guard the image tensors apply to
/// `width * height`.
fn length_guard(func: &str, secs: f64, rate: u32) -> Result<usize, ExprError> {
    let n = secs * f64::from(rate);
    if !(n >= 0.0 && n <= i32::MAX as f64) {
        return Err(ExprError::InvalidArgument {
            func: func.to_string(),
            reason: format!(
                "{secs} seconds at {rate} Hz is {n} samples, past the maximum of {}",
                i32::MAX
            ),
        });
    }
    Ok(n as usize)
}

/// A fractional knob, defaulted and range-checked once for the batch.
pub(super) fn bounded(
    func: AudioFunc,
    value: Option<f64>,
    default: f64,
    lo: f64,
    hi: f64,
) -> Result<f64, ExprError> {
    let v = value.unwrap_or(default);
    if !v.is_finite() || !(lo..=hi).contains(&v) {
        return Err(ExprError::InvalidArgument {
            func: format!("{func:?}"),
            reason: format!("factor must be in {lo}..={hi}, got {v}"),
        });
    }
    Ok(v)
}

/// The WAV writer, reachable from the sibling spectral tests.
///
/// Test-only re-export rather than a second hand-rolled header in `spectral`'s tests:
/// two encoders disagreeing about a chunk size is a test failure that looks like a
/// spectral bug.
#[cfg(test)]
pub(super) fn tests_wav_pcm16(samples: &[f32], rate: u32) -> Vec<u8> {
    wav_pcm16(samples, rate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{AsArray, BinaryArray};
    use arrow::datatypes::Float64Type;

    /// A mono 16-bit WAV of `n` samples from a callable, so the tests need no fixture.
    fn wav(rate: u32, n: usize, f: impl Fn(usize) -> f32) -> Vec<u8> {
        let samples: Vec<f32> = (0..n).map(&f).collect();
        wav_pcm16(&samples, rate)
    }

    fn column(clips: Vec<Vec<u8>>) -> BinaryArray {
        BinaryArray::from_iter(clips.iter().map(|c| Some(c.as_slice())))
    }

    fn scores(func: AudioFunc, clips: Vec<Vec<u8>>, args: AudioArgs) -> Vec<Option<f64>> {
        let arr = column(clips);
        let out = measure(func, &Signal::Narrow(&arr), args).unwrap();
        let a = out.as_primitive::<Float64Type>();
        (0..a.len())
            .map(|i| (!a.is_null(i)).then(|| a.value(i)))
            .collect()
    }

    fn waveform(func: AudioFunc, clip: Vec<u8>, args: AudioArgs) -> Vec<f32> {
        let arr = column(vec![clip]);
        let out = shape(func, &Signal::Narrow(&arr), args).unwrap();
        let list = out.as_list::<i32>();
        let vals = list.value(0);
        let f = vals.as_primitive::<arrow::datatypes::Float32Type>();
        (0..f.len()).map(|i| f.value(i)).collect()
    }

    /// The reason RMS exists next to the peak: one loud sample must not read as a loud clip.
    #[test]
    fn rms_tracks_level_where_the_peak_does_not() {
        let quiet_with_a_click = wav(8000, 800, |i| if i == 0 { 1.0 } else { 0.01 });
        let steady = wav(8000, 800, |_| 0.5);
        let got = scores(
            AudioFunc::Rms,
            vec![quiet_with_a_click.clone(), steady],
            AudioArgs::default(),
        );
        assert!(
            got[0].unwrap() < 0.05,
            "a click made a quiet clip look loud"
        );
        assert!((got[1].unwrap() - 0.5).abs() < 0.01);
        // The peak, by contrast, sees only the click.
        let peaks = scores(
            AudioFunc::PeakDbfs,
            vec![quiet_with_a_click],
            AudioArgs::default(),
        );
        assert!(peaks[0].unwrap() > -0.5);
    }

    /// Digital silence has no level; `-inf` would pass every threshold written to find it.
    #[test]
    fn a_silent_clip_has_a_null_level_rather_than_negative_infinity() {
        let silent = wav(8000, 400, |_| 0.0);
        for func in [AudioFunc::Dbfs, AudioFunc::PeakDbfs] {
            assert_eq!(
                scores(func, vec![silent.clone()], AudioArgs::default()),
                vec![None]
            );
        }
        // RMS is a plain amplitude, so silence is honestly zero there.
        assert_eq!(
            scores(AudioFunc::Rms, vec![silent], AudioArgs::default()),
            vec![Some(0.0)]
        );
    }

    #[test]
    fn clipping_and_silence_ratios_count_the_samples_they_name() {
        // Half the samples at full scale, half at digital silence.
        let half = wav(8000, 400, |i| if i % 2 == 0 { 1.0 } else { 0.0 });
        let clip = scores(
            AudioFunc::ClippingRatio,
            vec![half.clone()],
            AudioArgs::default(),
        );
        assert!((clip[0].unwrap() - 0.5).abs() < 0.01);
        let quiet = scores(AudioFunc::SilenceRatio, vec![half], AudioArgs::default());
        assert!((quiet[0].unwrap() - 0.5).abs() < 0.01);
    }

    #[test]
    fn rms_normalize_lifts_a_quiet_clip_without_clipping_it() {
        let quiet = wav(8000, 400, |_| 0.02);
        let out = waveform(AudioFunc::RmsNormalize, quiet, AudioArgs::default());
        let level = rms_of(&out);
        assert!(
            (level - 0.1).abs() < 0.01,
            "target -20 dBFS not reached: {level}"
        );
        assert!(out.iter().all(|s| s.abs() <= 1.0), "normalization clipped");
    }

    /// The bug an in-place forward filter has: it feeds itself its own output.
    #[test]
    fn pre_emphasis_reads_the_original_previous_sample() {
        let step = wav(8000, 4, |i| if i == 0 { 0.0 } else { 1.0 });
        let out = waveform(AudioFunc::PreEmphasis, step, AudioArgs::default());
        assert!((out[0] - 0.0).abs() < 1e-3);
        assert!(
            (out[1] - 1.0).abs() < 1e-3,
            "the edge should survive: {out:?}"
        );
        // Steady state after the edge is x - 0.97x = 0.03x, which only holds if each step
        // read the input rather than the freshly written output.
        assert!((out[2] - 0.03).abs() < 1e-3, "filter fed itself: {out:?}");
    }

    /// The property that makes a clip corpus batchable: every row the same length.
    #[test]
    fn pad_or_trim_gives_every_clip_exactly_the_length_asked_for() {
        let short = wav(8000, 100, |_| 0.5);
        let long = wav(8000, 8000, |_| 0.5);
        let arr = column(vec![short, long]);
        let args = AudioArgs {
            rate: Some(8000),
            duration_secs: Some(0.5),
            ..AudioArgs::default()
        };
        let out = pad_or_trim(&arr, args).unwrap();
        let list = out.as_list::<i32>();
        for i in 0..2 {
            assert_eq!(
                list.value(i).len(),
                4000,
                "row {i} was not padded/trimmed to length"
            );
        }
    }

    #[test]
    fn slice_reads_the_window_and_an_out_of_range_one_is_empty() {
        let clip = wav(8000, 8000, |i| i as f32 / 8000.0);
        let arr = column(vec![clip]);
        let inside = slice(
            &arr,
            AudioArgs {
                offset_secs: Some(0.25),
                duration_secs: Some(0.5),
                ..AudioArgs::default()
            },
        )
        .unwrap();
        assert_eq!(inside.as_list::<i32>().value(0).len(), 4000);
        let past = slice(
            &arr,
            AudioArgs {
                offset_secs: Some(10.0),
                duration_secs: Some(1.0),
                ..AudioArgs::default()
            },
        )
        .unwrap();
        assert_eq!(past.as_list::<i32>().value(0).len(), 0);
        assert!(
            !past.is_null(0),
            "an empty window is not an unreadable clip"
        );
    }

    /// The round trip that makes a cleaned corpus writable: decode, shape, re-encode, and
    /// the result must decode again to the same audio.
    #[test]
    fn encode_wav_round_trips_through_the_decoder() {
        let clip = wav(16000, 1600, |i| (i as f32 * 0.05).sin() * 0.5);
        let arr = column(vec![clip]);
        let out = encode_wav(&arr, AudioArgs::default()).unwrap();
        let bytes = out.as_binary::<i32>().value(0);
        let decoded = decode_pcm(bytes).expect("re-encoded WAV must decode");
        assert_eq!(decoded.sample_rate, 16000);
        assert_eq!(decoded.samples.len(), 1600);
        assert!((rms_of(&decoded.samples) - 0.35).abs() < 0.02);
    }

    #[test]
    fn encode_wav_resamples_when_asked() {
        let clip = wav(16000, 1600, |_| 0.25);
        let arr = column(vec![clip]);
        let out = encode_wav(
            &arr,
            AudioArgs {
                rate: Some(8000),
                ..AudioArgs::default()
            },
        )
        .unwrap();
        let decoded = decode_pcm(out.as_binary::<i32>().value(0)).unwrap();
        assert_eq!(decoded.sample_rate, 8000);
        assert!((decoded.samples.len() as i64 - 800).abs() <= 2);
    }

    /// A gain stage can push a signal past full scale; wrapping there is an audible click.
    #[test]
    fn encoding_saturates_rather_than_wrapping() {
        let hot = wav_pcm16(&[2.0, -2.0], 8000);
        let decoded = decode_pcm(&hot).unwrap();
        assert!(
            decoded.samples.iter().all(|s| s.abs() <= 1.0),
            "{:?}",
            decoded.samples
        );
        assert!(decoded.samples[0] > 0.9 && decoded.samples[1] < -0.9);
    }

    #[test]
    fn a_null_or_undecodable_row_is_null_rather_than_an_error() {
        let arr = BinaryArray::from_iter(vec![None, Some(b"not audio".as_slice())]);
        let signal = Signal::Narrow(&arr);
        assert_eq!(
            measure(AudioFunc::Rms, &signal, AudioArgs::default())
                .unwrap()
                .null_count(),
            2
        );
        assert_eq!(
            shape(AudioFunc::PreEmphasis, &signal, AudioArgs::default())
                .unwrap()
                .null_count(),
            2
        );
        assert_eq!(
            encode_wav(&arr, AudioArgs::default()).unwrap().null_count(),
            2
        );
    }
}
