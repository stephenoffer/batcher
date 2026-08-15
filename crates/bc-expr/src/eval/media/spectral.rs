//! Spectral descriptors and the linear spectrogram — what a clip's *frequencies* say.
//!
//! The mel spectrogram next door is a speech front end: its filterbank is warped to human
//! pitch perception and tuned to the models that consume it. These are the other half of
//! the audio story, and they answer questions the mel bands cannot.
//!
//! - The **linear spectrogram** keeps the frequencies themselves, which is what a music,
//!   bioacoustic or machine-fault model wants — a mel warp throws away exactly the
//!   high-frequency resolution those depend on.
//! - The four **descriptors** (centroid, rolloff, bandwidth, flatness) each reduce a whole
//!   clip to one number, so triaging a corpus by content is an ordinary predicate. Rolloff
//!   is the one worth knowing about: it says where a recording's usable band *ends*, which
//!   is how an 8 kHz telephone recording upsampled to 16 kHz is caught. Nothing else here
//!   can see that, and a model trained on the mixture learns the artifact.
//!
//! Every one of them reads the same [`Stft`] the mel front end does, so two features of one
//! clip cannot disagree about which samples they looked at or which window was applied.

use arrow::array::{Array, ArrayRef, Float64Builder, GenericBinaryArray, OffsetSizeTrait};
use std::sync::Arc;

use super::audio::{build_f32_list_column, decode_pcm, resample_signal, AudioArgs};
use super::level::{bounded, target_rate};
use super::map_rows;
use super::mel::Stft;
use crate::{AudioFunc, ExprError};

/// The framing every spectral op shares, validated once for the batch.
#[derive(Clone, Copy)]
struct Framing {
    rate: u32,
    n_fft: usize,
    hop: usize,
}

impl Framing {
    fn resolve(func: AudioFunc, args: AudioArgs) -> Result<Self, ExprError> {
        let name = format!("{func:?}");
        let rate = target_rate(&name, args.rate)?;
        let pos = |v: Option<i64>, what: &str, default: usize| -> Result<usize, ExprError> {
            match v {
                None => Ok(default),
                Some(x) => usize::try_from(x).ok().filter(|&x| x > 0).ok_or_else(|| {
                    ExprError::InvalidArgument {
                        func: name.clone(),
                        reason: format!("{what} must be a positive integer, got {x}"),
                    }
                }),
            }
        };
        // 400 / 160 is the 25 ms window and 10 ms hop the whole speech stack uses, and the
        // default the mel front end here is configured with everywhere it appears.
        Ok(Self {
            rate,
            n_fft: pos(args.n_fft, "n_fft", 400)?,
            hop: pos(args.hop, "hop_length", 160)?,
        })
    }

    /// The centre frequency, in Hz, of each power-spectrum bin.
    fn bin_hz(&self, n_freqs: usize) -> Vec<f64> {
        (0..n_freqs)
            .map(|k| k as f64 * f64::from(self.rate) / self.n_fft as f64)
            .collect()
    }
}

/// Decode, resample to the analysis rate, and hand back the signal.
fn analysis_signal(data: &[u8], rate: u32) -> Option<Vec<f32>> {
    let decoded = decode_pcm(data)?;
    Some(resample_signal(&decoded.samples, decoded.sample_rate, rate))
}

/// `spectrogram(rate, n_fft, hop_length)` → `List<Float32>` of `(n_fft/2+1) * n_frames`,
/// row-major `(freq, frame)`.
///
/// The same layout convention the mel spectrogram uses — the frequency axis is the slow one
/// — so a caller reshaping either by its first dimension is right about both.
pub(super) fn spectrogram<O: OffsetSizeTrait>(
    bytes: &GenericBinaryArray<O>,
    args: AudioArgs,
) -> Result<ArrayRef, ExprError> {
    let f = Framing::resolve(AudioFunc::Spectrogram, args)?;
    let rows: Vec<Option<Vec<f32>>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            return None;
        }
        let signal = analysis_signal(bytes.value(i), f.rate)?;
        let stft = Stft::new(f.n_fft, f.hop);
        let n_frames = stft.frame_count(&signal);
        let n_freqs = stft.n_freqs();
        let mut out = vec![0f32; n_freqs * n_frames];
        stft.frames(&signal, |t, power| {
            for (k, p) in power.iter().enumerate() {
                out[k * n_frames + t] = *p;
            }
        });
        Some(out)
    });
    Ok(build_f32_list_column(rows))
}

/// The four scalar descriptors, each a per-frame statistic averaged over the clip.
///
/// Averaged over frames rather than computed on one whole-clip spectrum, because a clip is
/// not stationary: a single transform of a recording that is half speech and half silence
/// describes neither half. Frames with no energy are skipped rather than counted as zero —
/// a centroid of "0 Hz" for a silent frame would drag the average toward DC and make a
/// mostly-quiet recording look band-limited, which is the exact confusion `rolloff` exists
/// to resolve.
pub(super) fn descriptor<O: OffsetSizeTrait>(
    func: AudioFunc,
    bytes: &GenericBinaryArray<O>,
    args: AudioArgs,
) -> Result<ArrayRef, ExprError> {
    let f = Framing::resolve(func, args)?;
    let percentile = bounded(func, args.factor, 0.85, 0.0, 1.0)?;
    let rows: Vec<Option<f64>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            return None;
        }
        let signal = analysis_signal(bytes.value(i), f.rate)?;
        let stft = Stft::new(f.n_fft, f.hop);
        let hz = f.bin_hz(stft.n_freqs());
        let mut total = 0.0f64;
        let mut counted = 0u64;
        stft.frames(&signal, |_, power| {
            if let Some(v) = frame_value(func, power, &hz, percentile) {
                total += v;
                counted += 1;
            }
        });
        (counted > 0).then(|| total / counted as f64)
    });
    let mut b = Float64Builder::with_capacity(rows.len());
    for row in rows {
        match row {
            Some(v) => b.append_value(v),
            None => b.append_null(),
        }
    }
    Ok(Arc::new(b.finish()))
}

/// One frame's descriptor, or `None` when the frame carries no energy to describe.
fn frame_value(func: AudioFunc, power: &[f32], hz: &[f64], percentile: f64) -> Option<f64> {
    let energy: f64 = power.iter().map(|p| f64::from(*p)).sum();
    // `<= 0.0` rather than a negated `> 0.0`, and it also catches a NaN energy — which a
    // frame cannot produce from finite samples, but which would otherwise divide into
    // every descriptor below and poison the clip's average.
    if !energy.is_finite() || energy <= 0.0 {
        return None;
    }
    match func {
        AudioFunc::SpectralCentroid => Some(centroid(power, hz, energy)),
        AudioFunc::SpectralBandwidth => {
            let c = centroid(power, hz, energy);
            let var: f64 = power
                .iter()
                .zip(hz)
                .map(|(p, f)| f64::from(*p) * (f - c) * (f - c))
                .sum::<f64>()
                / energy;
            Some(var.max(0.0).sqrt())
        }
        AudioFunc::SpectralRolloff => {
            let target = energy * percentile;
            let mut acc = 0.0;
            for (p, f) in power.iter().zip(hz) {
                acc += f64::from(*p);
                if acc >= target {
                    return Some(*f);
                }
            }
            hz.last().copied()
        }
        AudioFunc::SpectralFlatness => {
            // The geometric mean via logs, and floored: a bin that is exactly zero would
            // send the geometric mean to zero and report every real recording as perfectly
            // tonal, which is the classic way this measure is got wrong.
            const FLOOR: f64 = 1e-20;
            let n = power.len() as f64;
            let log_mean = power
                .iter()
                .map(|p| f64::from(*p).max(FLOOR).ln())
                .sum::<f64>()
                / n;
            Some((log_mean.exp() / (energy / n)).clamp(0.0, 1.0))
        }
        _ => None,
    }
}

fn centroid(power: &[f32], hz: &[f64], energy: f64) -> f64 {
    power
        .iter()
        .zip(hz)
        .map(|(p, f)| f64::from(*p) * f)
        .sum::<f64>()
        / energy
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{AsArray, BinaryArray};
    use arrow::datatypes::Float64Type;

    const RATE: u32 = 16_000;

    /// A mono WAV of a callable over `n` samples at [`RATE`].
    fn wav(n: usize, f: impl Fn(usize) -> f32) -> Vec<u8> {
        let samples: Vec<f32> = (0..n).map(&f).collect();
        super::super::level::tests_wav_pcm16(&samples, RATE)
    }

    fn tone(hz: f64, n: usize) -> Vec<u8> {
        wav(n, |i| {
            (2.0 * std::f64::consts::PI * hz * i as f64 / f64::from(RATE)).sin() as f32 * 0.5
        })
    }

    fn args() -> AudioArgs {
        AudioArgs {
            rate: Some(i64::from(RATE)),
            ..AudioArgs::default()
        }
    }

    fn score(func: AudioFunc, clip: Vec<u8>) -> Option<f64> {
        let arr = BinaryArray::from_iter(vec![Some(clip.as_slice())]);
        let out = descriptor(func, &arr, args()).unwrap();
        let a = out.as_primitive::<Float64Type>();
        (!a.is_null(0)).then(|| a.value(0))
    }

    /// The centroid must land on the tone that is actually there.
    #[test]
    fn the_centroid_of_a_pure_tone_is_its_frequency() {
        let got = score(AudioFunc::SpectralCentroid, tone(1000.0, 16_000)).unwrap();
        assert!(
            (got - 1000.0).abs() < 120.0,
            "centroid was {got} Hz, not ~1000"
        );
        let higher = score(AudioFunc::SpectralCentroid, tone(4000.0, 16_000)).unwrap();
        assert!(higher > got, "a higher tone must have a higher centroid");
    }

    /// The reason rolloff earns its place: it finds the band edge of an upsampled recording,
    /// which every other measure here reports as ordinary audio.
    #[test]
    fn rolloff_reports_where_the_band_ends() {
        let narrow = score(AudioFunc::SpectralRolloff, tone(500.0, 16_000)).unwrap();
        let wide = score(AudioFunc::SpectralRolloff, tone(6000.0, 16_000)).unwrap();
        assert!(narrow < 2000.0, "a 500 Hz tone rolled off at {narrow} Hz");
        assert!(wide > 4000.0, "a 6 kHz tone rolled off at {wide} Hz");
    }

    /// Flatness is the tonality axis: a tone is near 0, broadband noise near 1.
    #[test]
    fn flatness_separates_a_tone_from_noise() {
        let pure = score(AudioFunc::SpectralFlatness, tone(1000.0, 16_000)).unwrap();
        // A deterministic pseudo-noise, so the oracle and the parallel path agree.
        let noisy = score(
            AudioFunc::SpectralFlatness,
            wav(16_000, |i| {
                let x = (i as u64)
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .rotate_left(17);
                ((x % 2000) as f32 / 1000.0) - 1.0
            }),
        )
        .unwrap();
        assert!(pure < 0.05, "a pure tone scored {pure}");
        assert!(
            noisy > pure * 5.0,
            "noise ({noisy}) did not out-score a tone ({pure})"
        );
        assert!((0.0..=1.0).contains(&noisy));
    }

    #[test]
    fn bandwidth_is_narrow_for_a_tone_and_wide_for_noise() {
        let pure = score(AudioFunc::SpectralBandwidth, tone(2000.0, 16_000)).unwrap();
        let two_tones = score(
            AudioFunc::SpectralBandwidth,
            wav(16_000, |i| {
                let t = i as f64 / f64::from(RATE);
                let a = (2.0 * std::f64::consts::PI * 300.0 * t).sin();
                let b = (2.0 * std::f64::consts::PI * 6000.0 * t).sin();
                ((a + b) * 0.4) as f32
            }),
        )
        .unwrap();
        assert!(
            two_tones > pure,
            "a split spectrum must be wider: {two_tones} vs {pure}"
        );
    }

    /// The layout contract: `(n_fft/2+1) * n_frames`, frequency as the slow axis.
    #[test]
    fn the_spectrogram_has_the_shape_the_framing_implies() {
        let clip = tone(1000.0, 16_000);
        let arr = BinaryArray::from_iter(vec![Some(clip.as_slice())]);
        let out = spectrogram(&arr, args()).unwrap();
        let row = out.as_list::<i32>().value(0);
        let stft = Stft::new(400, 160);
        let signal: Vec<f32> = vec![0.0; 16_000];
        assert_eq!(row.len(), stft.n_freqs() * stft.frame_count(&signal));
    }

    /// A clip with no samples has no frames, so there is nothing to average and the answer
    /// is null rather than zero. Note what this does *not* say: because the framing is
    /// `center=True`, reflect padding gives even a 16-sample clip one full window, so
    /// "shorter than `n_fft`" is not the empty case — only "no samples at all" is.
    #[test]
    fn a_clip_with_no_samples_yields_null_rather_than_zero() {
        assert_eq!(score(AudioFunc::SpectralCentroid, wav(0, |_| 0.0)), None);
        assert!(score(AudioFunc::SpectralCentroid, wav(16, |_| 0.5)).is_some());
    }

    /// Digital silence frames carry no energy, so they are skipped rather than averaged in
    /// as "0 Hz" — which would drag a mostly-quiet recording's centroid toward DC and make
    /// it look band-limited, the exact confusion these measures exist to resolve.
    #[test]
    fn silent_frames_do_not_drag_the_average_down() {
        let half_silent = wav(32_000, |i| {
            if i < 16_000 {
                (2.0 * std::f64::consts::PI * 3000.0 * i as f64 / f64::from(RATE)).sin() as f32
                    * 0.5
            } else {
                0.0
            }
        });
        let all_tone = tone(3000.0, 16_000);
        let mixed = score(AudioFunc::SpectralCentroid, half_silent).unwrap();
        let pure = score(AudioFunc::SpectralCentroid, all_tone).unwrap();
        assert!(
            (mixed - pure).abs() < pure * 0.2,
            "silence moved the centroid from {pure} to {mixed}"
        );
    }

    #[test]
    fn a_null_or_undecodable_row_is_null_rather_than_an_error() {
        let arr = BinaryArray::from_iter(vec![None, Some(b"not audio".as_slice())]);
        assert_eq!(
            descriptor(AudioFunc::SpectralCentroid, &arr, args())
                .unwrap()
                .null_count(),
            2
        );
        assert_eq!(spectrogram(&arr, args()).unwrap().null_count(), 2);
    }
}
