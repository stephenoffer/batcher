//! Mel power-spectrogram kernel for `AudioFunc::MelSpectrogram`.
//!
//! This is the speech-model front end (Whisper / wav2vec2 / HuBERT). It is factored out of
//! `audio.rs` — both for the file-size budget and because the numeric core (STFT + mel
//! filterbank) is a pure function of a mono signal, so it is unit-testable without decoding
//! anything.
//!
//! **Conventions match `torchaudio.transforms.MelSpectrogram` defaults**, which is what
//! makes the output a drop-in for models trained on torchaudio: periodic Hann window,
//! `win_length == n_fft`, `center=True` with `reflect` padding of `n_fft/2` on each end,
//! power spectrum (`|.|²`), an HTK-scale mel filterbank with Slaney-free (`norm=None`)
//! triangular filters spanning `[0, sr/2]`. The log/normalization that turns this into a
//! model's exact input varies (Whisper log10 + clamp, others ln) and is applied downstream.

use std::f64::consts::PI;

use realfft::RealFftPlanner;

/// Hz → HTK mel: `2595 · log10(1 + f/700)`.
fn hz_to_mel(f: f64) -> f64 {
    2595.0 * (1.0 + f / 700.0).log10()
}

/// HTK mel → Hz, the inverse of [`hz_to_mel`].
fn mel_to_hz(m: f64) -> f64 {
    700.0 * (10f64.powf(m / 2595.0) - 1.0)
}

/// The mel filterbank as `n_mels` rows of `n_fft/2 + 1` weights.
///
/// Reproduces torchaudio's `melscale_fbanks` (HTK, `norm=None`): `n_mels + 2` points evenly
/// spaced on the mel scale between `f_min` and `f_max`, mapped back to Hz, then a triangular
/// filter per interior point built from the up/down slopes against the FFT bin centre
/// frequencies. A filter whose band is narrower than the FFT resolution can come out all
/// zeros — that is the documented torchaudio behavior (it warns), not an error here.
pub(crate) fn mel_filterbank(
    n_fft: usize,
    n_mels: usize,
    sample_rate: f64,
    f_min: f64,
    f_max: f64,
) -> Vec<Vec<f32>> {
    let n_freqs = n_fft / 2 + 1;
    // FFT bin centre frequencies: k · sr / n_fft.
    let all_freqs: Vec<f64> = (0..n_freqs)
        .map(|k| k as f64 * sample_rate / n_fft as f64)
        .collect();

    // n_mels + 2 mel points → Hz.
    let (m_min, m_max) = (hz_to_mel(f_min), hz_to_mel(f_max));
    let f_pts: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(m_min + (m_max - m_min) * i as f64 / (n_mels + 1) as f64))
        .collect();

    let mut fb = vec![vec![0f32; n_freqs]; n_mels];
    for m in 0..n_mels {
        let (left, center, right) = (f_pts[m], f_pts[m + 1], f_pts[m + 2]);
        for (k, &f) in all_freqs.iter().enumerate() {
            // torchaudio's slopes: down = (f - left)/(center - left), up = (right - f)/(right - center).
            let down = (f - left) / (center - left);
            let up = (right - f) / (right - center);
            let w = down.min(up).max(0.0);
            fb[m][k] = w as f32;
        }
    }
    fb
}

/// A periodic Hann window of length `n`: `0.5 − 0.5·cos(2πk/n)` — matching
/// `torch.hann_window(n)` (periodic, **not** symmetric; the `/n` rather than `/(n-1)` is the
/// difference, and getting it wrong is the classic STFT off-by-one).
fn hann_periodic(n: usize) -> Vec<f32> {
    (0..n)
        .map(|k| (0.5 - 0.5 * (2.0 * PI * k as f64 / n as f64).cos()) as f32)
        .collect()
}

/// Reflect-pad `signal` by `pad` samples on each end (`center=True`, `pad_mode="reflect"`).
///
/// Reflection excludes the edge sample itself: `[a, b, c]` padded by 2 → `[c, b, a, b, c, b, a]`.
/// A signal shorter than `pad + 1` cannot be reflect-padded (torchaudio errors); we clamp the
/// reflection index into range so a tiny clip degrades gracefully instead of panicking.
fn reflect_pad(signal: &[f32], pad: usize) -> Vec<f32> {
    let n = signal.len();
    if n == 0 {
        return Vec::new();
    }
    let refl = |i: isize| -> f32 {
        // Mirror index i (which may be negative or ≥ n) back into [0, n) by reflection.
        let mut idx = i;
        let m = n as isize;
        if m == 1 {
            return signal[0];
        }
        loop {
            if idx < 0 {
                idx = -idx; // reflect off the left edge (excludes sample 0)
            } else if idx >= m {
                idx = 2 * (m - 1) - idx; // reflect off the right edge
            } else {
                return signal[idx as usize];
            }
        }
    };
    let mut out = Vec::with_capacity(n + 2 * pad);
    for i in 0..pad {
        out.push(refl(-(pad as isize) + i as isize));
    }
    out.extend_from_slice(signal);
    for i in 0..pad {
        out.push(refl(n as isize + i as isize));
    }
    out
}

/// Compute the mel power spectrogram of a mono signal.
///
/// Returns the flattened `n_mels · n_frames` values in **row-major `(n_mels, n_frames)`**
/// order (mel band is the slow axis, matching torchaudio's `(…, n_mels, time)` shape) along
/// with `n_frames`, so the caller can tag the fixed-shape-tensor metadata.
pub(crate) fn mel_spectrogram(
    signal: &[f32],
    sample_rate: f64,
    n_fft: usize,
    hop: usize,
    n_mels: usize,
) -> (Vec<f32>, usize) {
    let n_freqs = n_fft / 2 + 1;
    let window = hann_periodic(n_fft);
    let fb = mel_filterbank(n_fft, n_mels, sample_rate, 0.0, sample_rate / 2.0);

    // center=True: reflect-pad n_fft/2 each side, then frames start at 0, hop apart.
    let padded = reflect_pad(signal, n_fft / 2);
    let n_frames = if padded.len() < n_fft {
        0
    } else {
        1 + (padded.len() - n_fft) / hop
    };

    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(n_fft);
    let mut input = r2c.make_input_vec();
    let mut spectrum = r2c.make_output_vec();

    // Output laid out (n_mels, n_frames) row-major: mel band m, frame t → m*n_frames + t.
    let mut out = vec![0f32; n_mels * n_frames];
    for t in 0..n_frames {
        let start = t * hop;
        for i in 0..n_fft {
            input[i] = padded[start + i] * window[i];
        }
        // realfft overwrites `input`; that's fine, it is rebuilt every frame.
        r2c.process(&mut input, &mut spectrum).expect("rfft length");
        // Power spectrum |X|² per bin, then project through the mel filterbank.
        let power: Vec<f32> = (0..n_freqs).map(|k| spectrum[k].norm_sqr()).collect();
        for (m, filt) in fb.iter().enumerate() {
            let mut acc = 0f32;
            for k in 0..n_freqs {
                acc += filt[k] * power[k];
            }
            out[m * n_frames + t] = acc;
        }
    }
    (out, n_frames)
}

/// `torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)` applied in place:
/// `10·log10(max(x, 1e-10))`, then clamped from below to `max_db − 80` (a per-spectrogram
/// floor on the dynamic range). The floor references the global max, which is why this runs
/// over the whole `(n_mels, n_frames)` buffer, not element-by-element.
fn amplitude_to_db(mel: &mut [f32]) {
    const AMIN: f32 = 1e-10;
    const TOP_DB: f32 = 80.0;
    let mut max_db = f32::NEG_INFINITY;
    for v in mel.iter_mut() {
        let db = 10.0 * v.max(AMIN).log10();
        *v = db;
        if db > max_db {
            max_db = db;
        }
    }
    let floor = max_db - TOP_DB;
    for v in mel.iter_mut() {
        if *v < floor {
            *v = floor;
        }
    }
}

/// The orthonormal DCT-II basis (`n_mfcc × n_mels`), matching `scipy`/`torchaudio`'s
/// `create_dct(norm="ortho")`: `D[k][n] = s_k · cos(π·k·(2n+1) / (2·N))` with
/// `s_0 = sqrt(1/N)` and `s_k = sqrt(2/N)` for `k > 0`.
fn dct2_ortho(n_mfcc: usize, n_mels: usize) -> Vec<Vec<f32>> {
    let n = n_mels as f64;
    (0..n_mfcc)
        .map(|k| {
            let scale = if k == 0 {
                (1.0 / n).sqrt()
            } else {
                (2.0 / n).sqrt()
            };
            (0..n_mels)
                .map(|m| (scale * (PI * k as f64 * (2 * m + 1) as f64 / (2.0 * n)).cos()) as f32)
                .collect()
        })
        .collect()
}

/// Compute the MFCCs of a mono signal: mel power spectrogram → `AmplitudeToDB` → DCT-II
/// (orthonormal), keeping the first `n_mfcc` coefficients. Numerically matches
/// `torchaudio.transforms.MFCC` defaults. Returns the flattened `n_mfcc · n_frames` values
/// row-major `(n_mfcc, n_frames)` with `n_frames`.
pub(crate) fn mfcc(
    signal: &[f32],
    sample_rate: f64,
    n_fft: usize,
    hop: usize,
    n_mels: usize,
    n_mfcc: usize,
) -> (Vec<f32>, usize) {
    let (mut mel, n_frames) = mel_spectrogram(signal, sample_rate, n_fft, hop, n_mels);
    amplitude_to_db(&mut mel); // in place: mel is now log-power dB, still (n_mels, n_frames)
    let dct = dct2_ortho(n_mfcc, n_mels);

    let mut out = vec![0f32; n_mfcc * n_frames];
    for t in 0..n_frames {
        for (k, basis) in dct.iter().enumerate() {
            let mut acc = 0f32;
            for m in 0..n_mels {
                acc += basis[m] * mel[m * n_frames + t];
            }
            out[k * n_frames + t] = acc;
        }
    }
    (out, n_frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hz_mel_round_trip() {
        for &f in &[0.0, 100.0, 1000.0, 8000.0] {
            assert!((mel_to_hz(hz_to_mel(f)) - f).abs() < 1e-6);
        }
    }

    #[test]
    fn dct2_ortho_basis_is_orthonormal() {
        // The ortho DCT basis rows must be orthonormal: <D[k], D[k]> = 1, <D[j], D[k]> = 0.
        let n = 12;
        let d = dct2_ortho(n, n);
        for j in 0..n {
            for k in 0..n {
                let dot: f64 = (0..n).map(|m| d[j][m] as f64 * d[k][m] as f64).sum();
                let expected = if j == k { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-5, "rows {j},{k} dot={dot}");
            }
        }
    }

    #[test]
    fn amplitude_to_db_floors_at_max_minus_80() {
        let mut x = vec![1.0f32, 100.0, 1e-12, 0.0];
        amplitude_to_db(&mut x);
        // max is 100 → 20 dB; floor = 20 - 80 = -60. Tiny/zero values clamp up to -60.
        assert!((x[1] - 20.0).abs() < 1e-4);
        assert!((x[0] - 0.0).abs() < 1e-4); // 10*log10(1)=0
        assert!((x[2] - (-60.0)).abs() < 1e-4);
        assert!((x[3] - (-60.0)).abs() < 1e-4);
    }

    #[test]
    fn mfcc_shape_matches_n_mfcc_and_frames() {
        let sig = vec![0.1f32; 16000];
        let (out, n_frames) = mfcc(&sig, 16000.0, 400, 160, 80, 13);
        assert_eq!(n_frames, 101);
        assert_eq!(out.len(), 13 * 101);
    }

    #[test]
    fn filterbank_shape_and_partition() {
        let fb = mel_filterbank(400, 80, 16000.0, 0.0, 8000.0);
        assert_eq!(fb.len(), 80);
        assert_eq!(fb[0].len(), 201);
        // Every weight is a valid triangular coefficient in [0, 1].
        for row in &fb {
            for &w in row {
                assert!((0.0..=1.0).contains(&w));
            }
        }
    }

    #[test]
    fn periodic_hann_first_and_symmetric_pair() {
        let w = hann_periodic(4);
        assert!((w[0] - 0.0).abs() < 1e-6); // periodic Hann starts at 0
        assert!((w[2] - 1.0).abs() < 1e-6); // and peaks at n/2
    }

    #[test]
    fn reflect_pad_excludes_edge() {
        let s = [1.0f32, 2.0, 3.0];
        // pad 2: reflect → [3, 2, | 1, 2, 3 | 2, 1]
        assert_eq!(reflect_pad(&s, 2), vec![3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn mel_spectrogram_shape_matches_frame_math() {
        // 16000 samples, n_fft=400, hop=160, center → n_frames = 1 + 16000/160 = 101.
        let sig = vec![0.1f32; 16000];
        let (out, n_frames) = mel_spectrogram(&sig, 16000.0, 400, 160, 80);
        assert_eq!(n_frames, 101);
        assert_eq!(out.len(), 80 * 101);
    }

    #[test]
    fn pure_tone_concentrates_energy_in_expected_band() {
        // A 1 kHz sine at 16 kHz should light up the mel band covering 1 kHz far more than a
        // band far away (say the very top). This validates the STFT+filterbank end to end
        // without an external oracle.
        let sr = 16000.0;
        let sig: Vec<f32> = (0..16000)
            .map(|i| (2.0 * PI * 1000.0 * i as f64 / sr).sin() as f32)
            .collect();
        let (out, n_frames) = mel_spectrogram(&sig, sr, 400, 160, 80);
        // Mel band containing 1 kHz.
        let target_hz = 1000.0;
        let mut band_1k = 0usize;
        let fpts: Vec<f64> = (0..82)
            .map(|i| mel_to_hz(hz_to_mel(0.0) + (hz_to_mel(8000.0)) * i as f64 / 81.0))
            .collect();
        for m in 0..80 {
            if fpts[m] <= target_hz && target_hz <= fpts[m + 2] {
                band_1k = m;
                break;
            }
        }
        // Average energy in the 1 kHz band vs the top band, at a mid frame.
        let t = n_frames / 2;
        let e_1k = out[band_1k * n_frames + t];
        let e_top = out[79 * n_frames + t];
        assert!(
            e_1k > 10.0 * (e_top + 1e-6),
            "1kHz band {e_1k} vs top {e_top}"
        );
    }
}
