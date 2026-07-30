//! Waveform conditioning for speech pipelines: silence trimming, peak normalization, and
//! the zero-crossing rate.
//!
//! These are the three things an ASR or speaker pipeline does to a clip between decoding it
//! and handing it to a model, and doing them anywhere but here means moving whole waveforms
//! into Python to loop over samples. A minute of 16 kHz audio is a million floats; a corpus
//! of them is the reason the decode already lives in the data plane.
//!
//! Each takes encoded audio bytes and decodes once, so `trim_silence().mel_spectrogram()`
//! costs two decodes today. That is the same shape every other op in this module has, and it
//! is the honest trade for keeping each one independently usable.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Builder, GenericBinaryArray, OffsetSizeTrait};

use super::audio::{build_f32_list_column, decode_pcm, Decoded};
use super::map_rows;
use crate::ExprError;

/// The amplitude a dBFS threshold corresponds to, relative to full scale (1.0).
///
/// dBFS is the unit every audio tool states a silence threshold in, and it is logarithmic
/// for the reason that matters here: -40 dBFS is 1% of full scale, which is roughly where
/// room tone sits and speech does not. A linear threshold would need a different value for
/// every recording level.
fn amplitude_for(threshold_db: f64) -> f32 {
    (10.0f64.powf(threshold_db / 20.0)) as f32
}

/// `trim_silence(threshold_db)` → the waveform with leading and trailing quiet removed.
///
/// Trims only the ends. Interior pauses are left alone, because they carry the timing an
/// acoustic model reads — an utterance with its pauses removed is not the same utterance.
/// A clip that is quiet throughout trims to empty rather than to itself, which is what makes
/// a silent-recording filter possible (`list.len() == 0`).
pub(crate) fn trim_silence<O: OffsetSizeTrait>(
    bytes: &GenericBinaryArray<O>,
    threshold_db: Option<i64>,
) -> Result<ArrayRef, ExprError> {
    // -40 dBFS: the conventional default, quiet enough to keep a soft consonant and loud
    // enough to drop room tone.
    let cutoff = amplitude_for(threshold_db.unwrap_or(-40) as f64);
    let rows: Vec<Option<Vec<f32>>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            return None;
        }
        let Decoded { samples, .. } = decode_pcm(bytes.value(i))?;
        let first = samples.iter().position(|s| s.abs() > cutoff);
        match first {
            None => Some(Vec::new()),
            Some(start) => {
                // `rposition` is over the reversed iterator, so it already counts from the end.
                let last = samples
                    .iter()
                    .rposition(|s| s.abs() > cutoff)
                    .unwrap_or(start);
                Some(samples[start..=last].to_vec())
            }
        }
    });
    Ok(build_f32_list_column(rows))
}

/// `peak_normalize()` → the waveform scaled so its loudest sample sits at full scale.
///
/// The level-matching step before batching clips from different sources: a model trained on
/// normalized audio sees a quiet recording as a different distribution, not a quieter one.
/// It is peak normalization, not loudness (LUFS) normalization — it equalizes the maximum,
/// not the perceived level, so a clip with one loud click stays quiet everywhere else.
/// An all-zero clip is returned unchanged rather than divided by zero.
pub(crate) fn peak_normalize<O: OffsetSizeTrait>(
    bytes: &GenericBinaryArray<O>,
) -> Result<ArrayRef, ExprError> {
    let rows: Vec<Option<Vec<f32>>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            return None;
        }
        let Decoded { mut samples, .. } = decode_pcm(bytes.value(i))?;
        let peak = samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        if peak > 0.0 {
            let gain = 1.0 / peak;
            for s in &mut samples {
                *s *= gain;
            }
        }
        Some(samples)
    });
    Ok(build_f32_list_column(rows))
}

/// `zero_crossing_rate()` → the fraction of adjacent sample pairs that change sign.
///
/// The cheapest useful descriptor of a waveform, and the classic voiced/unvoiced split:
/// a vowel is low-frequency and crosses zero rarely, a fricative or noise crosses constantly.
/// It also separates speech from silence-with-hiss without a spectrogram, which makes it a
/// good first-pass filter over a corpus that has not been curated.
///
/// A clip shorter than two samples has no adjacent pair and yields null.
pub(crate) fn zero_crossing_rate<O: OffsetSizeTrait>(
    bytes: &GenericBinaryArray<O>,
) -> Result<ArrayRef, ExprError> {
    let rows: Vec<Option<f64>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            return None;
        }
        let Decoded { samples, .. } = decode_pcm(bytes.value(i))?;
        if samples.len() < 2 {
            return None;
        }
        let crossings = samples
            .windows(2)
            .filter(|w| (w[0] < 0.0) != (w[1] < 0.0))
            .count();
        Some(crossings as f64 / (samples.len() - 1) as f64)
    });
    let mut builder = Float64Builder::with_capacity(rows.len());
    for row in rows {
        match row {
            Some(v) => builder.append_value(v),
            None => builder.append_null(),
        }
    }
    Ok(Arc::new(builder.finish()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// -40 dBFS is 1% of full scale, the conventional silence floor.
    #[test]
    fn a_db_threshold_converts_to_its_amplitude() {
        assert!((amplitude_for(-40.0) - 0.01).abs() < 1e-6);
        assert!((amplitude_for(0.0) - 1.0).abs() < 1e-6);
        assert!(amplitude_for(-20.0) > amplitude_for(-40.0));
    }

    /// The kernels take encoded bytes, so the row-level behaviour is exercised through the
    /// Python differential tests where real audio is available. What is checked here is the
    /// sample-level arithmetic each one is built on, isolated from the decoder.
    #[test]
    fn trimming_keeps_the_interior_and_drops_only_the_ends() {
        let samples = [0.0f32, 0.0, 0.5, 0.0, 0.6, 0.0, 0.0];
        let cutoff = 0.01f32;
        let start = samples.iter().position(|s| s.abs() > cutoff).unwrap();
        let last = samples.iter().rposition(|s| s.abs() > cutoff).unwrap();
        assert_eq!(&samples[start..=last], &[0.5, 0.0, 0.6]);
    }

    #[test]
    fn a_silent_clip_trims_to_nothing() {
        let samples = [0.0f32, 0.0001, -0.0001];
        assert!(samples.iter().position(|s| s.abs() > 0.01f32).is_none());
    }

    #[test]
    fn peak_normalization_puts_the_loudest_sample_at_full_scale() {
        let mut samples = [0.1f32, -0.25, 0.05];
        let peak = samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        for s in &mut samples {
            *s /= peak;
        }
        assert!((samples.iter().fold(0.0f32, |m, s| m.max(s.abs())) - 1.0).abs() < 1e-6);
        // The shape is preserved: ratios between samples are unchanged.
        assert!((samples[0] / samples[2] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn a_silent_clip_is_not_divided_by_zero() {
        let samples = [0.0f32, 0.0];
        let peak = samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        assert_eq!(peak, 0.0);
    }

    #[test]
    fn the_zero_crossing_rate_of_an_alternating_signal_is_one() {
        let samples = [1.0f32, -1.0, 1.0, -1.0];
        let crossings = samples
            .windows(2)
            .filter(|w| (w[0] < 0.0) != (w[1] < 0.0))
            .count();
        assert_eq!(crossings as f64 / (samples.len() - 1) as f64, 1.0);
    }

    #[test]
    fn a_constant_signal_never_crosses_zero() {
        let samples = [0.3f32, 0.3, 0.3];
        let crossings = samples
            .windows(2)
            .filter(|w| (w[0] < 0.0) != (w[1] < 0.0))
            .count();
        assert_eq!(crossings, 0);
    }
}
